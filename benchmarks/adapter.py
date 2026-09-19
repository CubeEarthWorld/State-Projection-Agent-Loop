"""Anthropic (Claude) LLMAdapter for the benchmark.

Not part of the package — same status as ``examples/llm_adapters.py``: a
reference implementation of the ``LLMAdapter`` Protocol, here so the
benchmark has a real provider to talk to.

    pip install anthropic
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import anthropic

from state_projection_loop.llm import extract_finish
from state_projection_loop.messages import Decision, Message, ToolCall, Usage

# Haiku 4.5: $1.00 / $5.00 per MTok. Cache read 0.1x, cache write 1.25x.
PRICING = {"in": 1.00, "out": 5.00, "cache_read": 0.10, "cache_write": 1.25}


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Neutral tool specs -> Anthropic tool blocks.

    The runtime emits ``{name, description, parameters}``; Anthropic wants
    the same three fields with ``parameters`` renamed to ``input_schema``.
    That rename is exactly the kind of thing that belongs in an adapter
    rather than in the core.
    """
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters") or {"type": "object", "properties": {}},
        }
        for tool in tools
    ]


def _convert(messages: list[Message]) -> tuple[str, list[dict]]:
    """Split off the system prompt and build the Anthropic messages array.

    Anthropic pairs tool_use/tool_result strictly: a tool_result must sit in
    a user message and every tool_use must be answered. The projection is
    free to compress history and break that pairing, so orphans on either
    side degrade to plain text instead of 400-ing the request.
    """
    system_parts: list[str] = []
    answered: set[str] = set()
    for m in messages:
        if m.role == "tool" and m.tool_call_id:
            answered.add(m.tool_call_id)

    out: list[dict] = []
    seen_use: set[str] = set()

    def push(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for m in messages:
        if m.role == "system":
            system_parts.append(m.text())
        elif m.role == "assistant":
            blocks: list[dict] = []
            text = m.text().strip()
            unanswered = [tc for tc in m.tool_calls if tc.id not in answered]
            if unanswered:
                text = "\n".join([text] + [
                    f"(called {tc.name} with {json.dumps(tc.arguments, ensure_ascii=False, default=str)})"
                    for tc in unanswered
                ]).strip()
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in m.tool_calls:
                if tc.id in answered:
                    seen_use.add(tc.id)
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name,
                                   "input": tc.arguments or {}})
            push("assistant", blocks)
        elif m.role == "tool":
            cid = m.tool_call_id or ""
            if cid in seen_use:
                push("user", [{"type": "tool_result", "tool_use_id": cid, "content": m.text()}])
            else:
                push("user", [{"type": "text", "text": f"(result of {m.name or 'tool'}) {m.text()}"}])
        else:
            push("user", [{"type": "text", "text": m.text()}])

    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "(begin)"}]})
    return "\n\n".join(p for p in system_parts if p), out


class AnthropicAdapter:
    """One model call against the Messages API.

    ``self.seen_calls`` records every capability the model asked for across
    the run — the benchmark's tool-recall signal, taken here because the
    adapter is the one place that sees every decision.
    """

    def __init__(self, model: str = "claude-haiku-4-5", *, max_tokens: int = 2048,
                 cache: bool = True, client: Any = None) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.cache = cache
        if client is None:
            try:  # ANTHROPIC_API_KEY from .env, same convention as examples/
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
        self._client = client or anthropic.Anthropic(max_retries=4)
        self.seen_calls: list[str] = []
        self.usage = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}
        self.api_calls = 0

    @property
    def prompt_total(self) -> int:
        """Every token the provider read, cached or not. Anthropic reports
        the three buckets disjointly, so they add."""
        u = self.usage
        return u["in"] + u["cache_read"] + u["cache_write"]

    @property
    def cost(self) -> float:
        u = self.usage
        return (u["in"] * PRICING["in"] + u["out"] * PRICING["out"]
                + u["cache_read"] * PRICING["cache_read"]
                + u["cache_write"] * PRICING["cache_write"]) / 1_000_000

    async def complete(self, messages: list[Message], tools: Optional[list[dict]] = None) -> Decision:
        system, msgs = _convert(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": msgs,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)
        if self.cache:
            # Renders tools -> system -> messages; caching the last cacheable
            # block covers the whole stable prefix for both arms alike.
            kwargs["cache_control"] = {"type": "ephemeral"}

        resp = await asyncio.to_thread(lambda: self._client.messages.create(**kwargs))
        self.api_calls += 1

        u = resp.usage
        self.usage["in"] += u.input_tokens or 0
        self.usage["out"] += u.output_tokens or 0
        self.usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        self.usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0

        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(name=block.name, arguments=dict(block.input or {}), id=block.id))
        self.seen_calls.extend(c.name for c in calls)

        return extract_finish(Decision(
            text="\n".join(text_parts),
            calls=calls,
            usage=Usage(prompt_tokens=(u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0),
                        completion_tokens=u.output_tokens or 0),
            raw=resp,
        ))


class DeepSeekAdapter:
    """Same measurement surface as :class:`AnthropicAdapter`, but delegating
    to the reference ``OpenAICompatAdapter`` the repo already ships.

    DeepSeek reports cache hits as ``prompt_cache_hit_tokens``; those are
    included in ``prompt_tokens``, so they are recorded separately rather
    than added again.
    """

    def __init__(self, model: str = "deepseek-flash", *, max_tokens: int = 2048,
                 cache: bool = True, client: Any = None) -> None:
        import os

        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        from examples.llm_adapters import OpenAICompatAdapter

        self.model = model
        self.cache = cache  # DeepSeek caches server-side; nothing to opt into.
        self._inner = OpenAICompatAdapter(
            model=model,
            api_key=os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=180.0,
            client=client,
        )
        self.seen_calls: list[str] = []
        self.usage = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}
        self.api_calls = 0

    @property
    def prompt_total(self) -> int:
        """DeepSeek's ``prompt_tokens`` already includes cache hits, so the
        cache figure is a subset here, not an addend."""
        return self.usage["in"]

    @property
    def cost(self) -> float:
        return 0.0  # DeepSeek rates are not hard-coded; tokens are the metric.

    async def complete(self, messages: list[Message], tools: Optional[list[dict]] = None) -> Decision:
        decision = await self._inner.complete(messages, tools)
        self.api_calls += 1
        if decision.usage is not None:
            self.usage["in"] += decision.usage.prompt_tokens
            self.usage["out"] += decision.usage.completion_tokens
        raw_usage = getattr(decision.raw, "usage", None)
        self.usage["cache_read"] += getattr(raw_usage, "prompt_cache_hit_tokens", 0) or 0
        self.seen_calls.extend(c.name for c in decision.calls)
        if decision.finish:
            self.seen_calls.append("finish")
        return decision


PROVIDERS = {"anthropic": AnthropicAdapter, "deepseek": DeepSeekAdapter}
