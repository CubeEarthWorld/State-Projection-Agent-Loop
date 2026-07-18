"""Adapter for OpenAI-compatible chat-completion APIs (OpenAI, DeepSeek,
most local servers). Requires the optional ``openai`` extra.

Native function calling is used when tool schemas are supplied; if the
provider returns fenced ``tool_call`` JSON in plain text instead, the text
protocol parser picks it up as a fallback.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from ..llm import parse_text_tool_calls
from ..messages import Decision, Message, ToolCall, Usage


class OpenAICompatAdapter:
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
                raise RuntimeError(
                    "The 'openai' package is required: pip install state-projection-loop[openai]"
                ) from exc
            self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    # -- message conversion --------------------------------------------------

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

    # -- completion ----------------------------------------------------------

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
                # malformed args → validation will bounce it with the spec (§6)
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
        return Decision(
            text=text,
            calls=calls,
            thought=getattr(choice, "reasoning_content", None) or "",
            usage=usage,
            raw=response,
        )


class DeepSeekAdapter(OpenAICompatAdapter):
    """DeepSeek chat API (OpenAI-compatible). Reads DEEPSEEK_* env vars."""

    def __init__(self, model: Optional[str] = None, *, api_key: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(
            model=model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=kwargs.pop("base_url", None) or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            **kwargs,
        )
