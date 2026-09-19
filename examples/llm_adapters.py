"""Example LLM adapters — NOT part of the ``state_projection_loop`` package.

The core package is intentionally LLM-agnostic: it only defines the
``LLMAdapter`` Protocol (``async complete(messages, tools) -> Decision``) and a
scripted test double (``ScriptedLLM``). Talking to any real provider —
authentication, request shaping, retries, streaming, billing — is entirely
the integrator's responsibility and concern, not the library's.

They are provided here purely as *reference implementations* so the
examples and integration tests have something to run against. Copy
them into your own project and adapt freely; there is no supported
"upgrade path" contract for this file the way there is for the package.

Requires the corresponding optional client library:
    pip install openai                             # OpenAICompatAdapter
    pip install llama-cpp-python huggingface-hub   # LlamaCppEmbedding
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Callable, Optional, Sequence

from state_projection_loop.llm import extract_finish, parse_text_tool_calls
from state_projection_loop.messages import Decision, Message, ToolCall, Usage

Vector = list[float]


# ---------------------------------------------------------------------------
# Completion adapters
# ---------------------------------------------------------------------------

class OpenAICompatAdapter:
    """Any OpenAI-compatible chat-completion API — OpenAI itself, DeepSeek,
    Groq, a local vLLM/Ollama server, or anything else speaking the same
    wire format. There is no per-provider subclass: every such provider
    differs only in ``base_url``/``model``/``api_key``. Point this at
    DeepSeek with::

        OpenAICompatAdapter(model="deepseek-flash", api_key=..., base_url="https://api.deepseek.com")

    Native function calling is used when tool schemas are supplied; if the
    provider returns fenced ``tool_call`` JSON in plain text instead, the
    text protocol parser (``parse_text_tool_calls``) picks it up as a
    fallback.
    """

    @classmethod
    def from_env(cls, **kwargs: Any) -> "OpenAICompatAdapter":
        """The adapter every example script uses: ``LLM_MODEL`` / ``LLM_API_KEY``
        / ``LLM_BASE_URL`` from the environment (a ``.env`` file is loaded when
        python-dotenv is installed), defaulting to DeepSeek."""
        try:
            from dotenv import find_dotenv, load_dotenv

            load_dotenv(find_dotenv(usecwd=True))
        except ImportError:
            pass
        return cls(
            model=os.environ.get("LLM_MODEL", "deepseek-flash"),
            api_key=os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
            **kwargs,
        )

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        timeout: float = 120.0,
        client: Any = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body
        self._client_args = dict(api_key=api_key, base_url=base_url, timeout=timeout)
        self._client, self._client_loop = client, None

    def _async_client(self) -> Any:
        """The async client — so that Session.interrupt() cancelling a call
        really abandons the request instead of waiting on a thread — made
        per event loop, because the sync API (session.send) runs each turn
        in a fresh asyncio.run() and an httpx client cannot outlive its loop."""
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("pip install openai") from exc
            self._client, self._client_loop = AsyncOpenAI(**self._client_args), loop
        return self._client

    @staticmethod
    def _to_api(message: Message) -> dict[str, Any]:
        if message.role == "assistant" and message.tool_calls:
            return {
                "role": "assistant",
                "content": message.text() or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False, default=str),
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id or "",
                "content": message.text(),
            }
        return {"role": message.role, "content": message.content}

    async def complete(
        self, messages: list[Message], tools: Optional[list[dict]] = None, *,
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> Decision:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [self._to_api(m) for m in messages],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        if on_delta is None:
            response = await self._async_client().chat.completions.create(**kwargs)
            choice = response.choices[0].message
            text, thought = choice.content or "", getattr(choice, "reasoning_content", None) or ""
            raw_calls = [(tc.id, tc.function.name, tc.function.arguments or "{}") for tc in choice.tool_calls or []]
            usage = getattr(response, "usage", None)
        else:
            text, thought, raw_calls, usage = await self._stream(kwargs, on_delta)
            response = None

        calls: list[ToolCall] = []
        for call_id, name, raw in raw_calls:
            try:
                arguments = json.loads(raw)
                if not isinstance(arguments, dict):
                    raise ValueError
                calls.append(ToolCall(name=name, arguments=arguments, id=call_id))
            except (json.JSONDecodeError, ValueError):
                calls.append(ToolCall(name=name, arguments={}, id=call_id, raw_arguments=raw))
        if not calls and "```tool_call" in text:
            text, calls = parse_text_tool_calls(text)
        return extract_finish(Decision(
            text=text, calls=calls, thought=thought,
            usage=Usage(prompt_tokens=usage.prompt_tokens or 0, completion_tokens=usage.completion_tokens or 0)
            if usage is not None else None,
            raw=response,
        ))

    async def _stream(self, kwargs: dict[str, Any], on_delta: Callable[[str], None]) -> tuple[str, str, list, Any]:
        """The streaming form of the same call: text chunks go to ``on_delta``
        as they arrive; tool calls are assembled from their indexed deltas."""
        text: list[str] = []
        thought: list[str] = []
        calls: dict[int, list] = {}  # index -> [id, name, arguments]
        usage = None
        stream = await self._async_client().chat.completions.create(
            **kwargs, stream=True, stream_options={"include_usage": True})
        async for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                text.append(delta.content)
                on_delta(delta.content)
            if getattr(delta, "reasoning_content", None):
                thought.append(delta.reasoning_content)
            for tc in delta.tool_calls or []:
                slot = calls.setdefault(tc.index, ["", "", ""])
                if tc.id:
                    slot[0] = tc.id
                if tc.function is not None:
                    slot[1] = slot[1] or (tc.function.name or "")
                    slot[2] += tc.function.arguments or ""
        return "".join(text), "".join(thought), [tuple(c) for _, c in sorted(calls.items())], usage


# ---------------------------------------------------------------------------
# GGUF (llama-cpp-python) local embedding — offline, no API calls, but
# still a concrete backend implementation, so it lives here rather than in
# the package: requires pip install llama-cpp-python huggingface-hub
# ---------------------------------------------------------------------------

GEMMA_QUERY_PREFIX = "task: search result | query: "
GEMMA_DOC_PREFIX = "title: none | text: "
DEFAULT_GGUF_REPO = "ggml-org/embeddinggemma-300M-GGUF"
DEFAULT_GGUF_FILE = "embeddinggemma-300M-Q8_0.gguf"


class LlamaCppEmbedding:
    """GGUF embedding via llama-cpp-python. Resolution order for the model
    file: explicit ``model_path`` -> ``SPAL_EMBED_GGUF`` env var ->
    download ``repo_id``/``filename`` from Hugging Face."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        repo_id: str = DEFAULT_GGUF_REPO,
        filename: str = DEFAULT_GGUF_FILE,
        n_ctx: int = 2048,
        query_prefix: str = GEMMA_QUERY_PREFIX,
        doc_prefix: str = GEMMA_DOC_PREFIX,
        verbose: bool = False,
    ) -> None:
        import os

        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        path = model_path or os.environ.get("SPAL_EMBED_GGUF") or ""
        if not path:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("pip install huggingface-hub, or set SPAL_EMBED_GGUF") from exc
            path = hf_hub_download(repo_id=repo_id, filename=filename)
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install llama-cpp-python") from exc
        self._llama = Llama(model_path=path, embedding=True, n_ctx=n_ctx, verbose=verbose)

    def _embed_one(self, text: str) -> Vector:
        out: Any = self._llama.embed(text)
        if out and isinstance(out[0], (list, tuple)):
            dim = len(out[0])
            pooled = [sum(tok[i] for tok in out) / len(out) for i in range(dim)]
            return [float(x) for x in pooled]
        return [float(x) for x in out]

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed_one(self.doc_prefix + t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self._embed_one(self.query_prefix + text)
