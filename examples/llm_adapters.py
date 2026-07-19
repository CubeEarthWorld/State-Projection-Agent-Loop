"""Example LLM adapters — NOT part of the ``state_projection_loop`` package.

The core package is intentionally LLM-agnostic: it only defines the
``LLMAdapter`` Protocol (``complete(messages, tools) -> Decision``) and a
scripted test double (``ScriptedLLM``). Talking to any real provider —
authentication, request shaping, retries, streaming, billing — is entirely
the integrator's responsibility and concern, not the library's.

These two adapters are provided here purely as *reference implementations*
so the examples and integration tests have something to run against. Copy
them into your own project and adapt freely; there is no supported
"upgrade path" contract for this file the way there is for the package.

Requires the corresponding optional client library:
    pip install openai       # OpenAICompatAdapter, OpenAICompatEmbedding
    pip install anthropic    # AnthropicAdapter
"""
from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from state_projection_loop.llm import extract_finish, parse_text_tool_calls
from state_projection_loop.messages import ASSISTANT, Decision, Message, OBSERVATION, SYSTEM, ToolCall, Usage

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

        OpenAICompatAdapter(model="deepseek-v4-flash", api_key=..., base_url="https://api.deepseek.com")

    Native function calling is used when tool schemas are supplied; if the
    provider returns fenced ``tool_call`` JSON in plain text instead, the
    text protocol parser (``parse_text_tool_calls``) picks it up as a
    fallback.
    """

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
        if client is not None:
            self._client = client
        else:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("pip install openai") from exc
            self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

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

    def complete(self, messages: list[Message], tools: Optional[list[dict]] = None) -> Decision:
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

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0].message

        calls: list[ToolCall] = []
        for tc in choice.tool_calls or []:
            raw = tc.function.arguments or "{}"
            try:
                arguments = json.loads(raw)
                if not isinstance(arguments, dict):
                    raise ValueError
                calls.append(ToolCall(name=tc.function.name, arguments=arguments, id=tc.id))
            except (json.JSONDecodeError, ValueError):
                calls.append(ToolCall(name=tc.function.name, arguments={}, id=tc.id, raw_arguments=raw))

        text = choice.content or ""
        if not calls and "```tool_call" in text:
            text, calls = parse_text_tool_calls(text)

        usage = None
        if getattr(response, "usage", None) is not None:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
            )
        return extract_finish(Decision(
            text=text, calls=calls, thought=getattr(choice, "reasoning_content", None) or "",
            usage=usage, raw=response,
        ))


class AnthropicAdapter:
    """Adapter for the Anthropic Messages API — shows that ``LLMAdapter`` is
    a real Protocol, not an OpenAI-shaped abstraction with one
    implementation: message roles, tool-result framing, and native tool
    schemas all differ from the OpenAI wire format and are translated here,
    entirely behind the same ``complete(messages, tools) -> Decision``
    boundary every adapter uses.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        client: Any = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        if client is not None:
            self._client = client
        else:
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("pip install anthropic") from exc
            self._client = Anthropic(api_key=api_key, base_url=base_url, timeout=timeout)

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str, list[Message]]:
        system_parts = [m.text() for m in messages if m.role == SYSTEM]
        rest = [m for m in messages if m.role != SYSTEM]
        return "\n\n".join(p for p in system_parts if p), rest

    @staticmethod
    def _to_api(message: Message) -> dict[str, Any]:
        if message.role == ASSISTANT and message.tool_calls:
            content: list[dict[str, Any]] = []
            if message.text():
                content.append({"type": "text", "text": message.text()})
            for tc in message.tool_calls:
                content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
            return {"role": "assistant", "content": content}
        if message.role == OBSERVATION:
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result", "tool_use_id": message.tool_call_id or "", "content": message.text(),
                }],
            }
        role = "assistant" if message.role == ASSISTANT else "user"
        return {"role": role, "content": message.text()}

    @staticmethod
    def _to_api_tools(tools: list[dict]) -> list[dict[str, Any]]:
        out = []
        for t in tools:
            fn = t.get("function", t)
            out.append({
                "name": fn["name"], "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        return out

    def complete(self, messages: list[Message], tools: Optional[list[dict]] = None) -> Decision:
        system, rest = self._split_system(messages)
        kwargs: dict[str, Any] = {
            "model": self.model, "messages": [self._to_api(m) for m in rest],
            "temperature": self.temperature, "max_tokens": self.max_tokens,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._to_api_tools(tools)

        response = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                calls.append(ToolCall(name=block.name, arguments=dict(block.input or {}), id=block.id))

        text = "\n".join(text_parts)
        if not calls and "```tool_call" in text:
            text, calls = parse_text_tool_calls(text)

        usage = None
        if getattr(response, "usage", None) is not None:
            usage = Usage(
                prompt_tokens=response.usage.input_tokens or 0,
                completion_tokens=response.usage.output_tokens or 0,
            )
        return extract_finish(Decision(text=text, calls=calls, usage=usage, raw=response))


# ---------------------------------------------------------------------------
# Embedding backend
# ---------------------------------------------------------------------------

class OpenAICompatEmbedding:
    """Any OpenAI-compatible ``/embeddings`` endpoint (OpenAI, DeepSeek, a
    local server). Implements the package's ``EmbeddingBackend`` Protocol
    (``embed_documents``/``embed_query``); pass an instance to
    ``Session(embedder=...)``."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        client: Any = None,
    ) -> None:
        self.model = model
        if client is not None:
            self._client = client
        else:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("pip install openai") from exc
            self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def _embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=list(texts))
        return [list(item.embedding) for item in response.data]

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return self._embed(texts)

    def embed_query(self, text: str) -> Vector:
        return self._embed([text])[0]


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
