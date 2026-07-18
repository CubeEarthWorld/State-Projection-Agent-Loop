"""LLM adapter protocol (spec §15: vendor-independent) and test helpers.

An adapter turns a rendered projection (list of Messages) plus optional
native tool schemas into a :class:`Decision`. Real adapters live in
``adapters/``; :class:`ScriptedLLM` drives deterministic tests.

For providers without native function calling, ``parse_text_tool_calls``
implements a fenced-JSON text protocol::

    ```tool_call
    {"name": "web_search", "arguments": {"query": "..."}}
    ```
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional, Protocol, Union, runtime_checkable

from .messages import Decision, Message, ToolCall


@runtime_checkable
class LLMAdapter(Protocol):
    def complete(self, messages: list[Message], tools: Optional[list[dict]] = None) -> Decision: ...


_FENCE = re.compile(r"```tool_call\s*\n(.*?)```", re.DOTALL)


def parse_text_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Extract fenced tool_call JSON blocks from plain text output."""
    calls: list[ToolCall] = []

    def _consume(match: "re.Match[str]") -> str:
        body = match.group(1).strip()
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            m = re.search(r'"(?:name|tool)"\s*:\s*"([^"]+)"', body)
            if m:
                calls.append(ToolCall(name=m.group(1), arguments={}, raw_arguments=body))
            return ""
        name = data.get("name") or data.get("tool")
        if not name:
            return ""
        args = data.get("arguments") or data.get("args") or {}
        if not isinstance(args, dict):
            calls.append(ToolCall(name=name, arguments={}, raw_arguments=json.dumps(args)))
        else:
            calls.append(ToolCall(name=name, arguments=args))
        return ""

    cleaned = _FENCE.sub(_consume, text).strip()
    return cleaned, calls


Step = Union[str, Decision, Callable[[list[Message], Optional[list[dict]]], Union[str, Decision]]]


class ScriptedLLM:
    """Deterministic adapter for tests: replays a fixed list of steps.

    A step may be a string (text-only decision), a Decision, or a callable
    ``(messages, tools) -> str | Decision`` for dynamic assertions. Every
    request (messages + tools) is recorded in ``self.requests``.
    """

    def __init__(self, steps: list[Step], *, strict: bool = True) -> None:
        self._steps = list(steps)
        self._i = 0
        self.strict = strict
        self.requests: list[dict[str, Any]] = []

    @staticmethod
    def call(name: str, /, _text: str = "", **arguments: Any) -> Decision:
        return Decision(text=_text, calls=[ToolCall(name=name, arguments=arguments)])

    @staticmethod
    def calls(*specs: tuple[str, dict[str, Any]], text: str = "") -> Decision:
        return Decision(text=text, calls=[ToolCall(name=n, arguments=a) for n, a in specs])

    def complete(self, messages: list[Message], tools: Optional[list[dict]] = None) -> Decision:
        self.requests.append({"messages": list(messages), "tools": list(tools or [])})
        if self._i >= len(self._steps):
            if self.strict:
                raise AssertionError(
                    f"ScriptedLLM exhausted after {len(self._steps)} steps; "
                    "the loop asked for another decision"
                )
            return Decision(text="(script exhausted)")
        step = self._steps[self._i]
        self._i += 1
        if callable(step) and not isinstance(step, Decision):
            step = step(messages, tools)
        if isinstance(step, str):
            return Decision(text=step)
        return step
