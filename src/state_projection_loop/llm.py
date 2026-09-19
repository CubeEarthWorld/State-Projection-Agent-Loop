"""LLM adapter protocol and test helpers.

An adapter turns a rendered projection (list of Messages) plus optional
native tool schemas into a :class:`Decision`. Real adapters live in
``adapters/``; :class:`ScriptedLLM` drives deterministic tests.

For providers without native function calling, ``parse_text_tool_calls``
implements a fenced-JSON text protocol::

    ```tool_call
    {"name": "web_search", "arguments": {"query": "..."}}
    ```

Completion is a formal property of a :class:`~state_projection_loop.messages.Decision`
(``finish``/``result``), not a capability the runtime executes like any
other. A model signals completion by calling the reserved
``finish(result)`` function — every adapter routes that call through
:func:`extract_finish` at the end of ``complete()`` so the rest of the
system only ever has to check ``decision.finish``.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional, Protocol, Union, runtime_checkable

from .messages import Decision, Message, ToolCall
from .serialization import dumps

FINISH_NAME = "finish"

FINISH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": FINISH_NAME,
        "description": (
            "Finish the job and return the final result. Call this ALONE — never combined with "
            "other tool calls in the same decision; a decision that does both is rejected."
        ),
        "parameters": {
            "type": "object",
            "properties": {"result": {"description": "The final result: string, object, or artifact reference."}},
            "required": ["result"],
        },
    },
}


@runtime_checkable
class LLMAdapter(Protocol):
    """One model call.

    Async because the session loop awaits it: a provider round-trip is the
    longest wait in a turn, and a synchronous adapter would block the host
    application's event loop for its whole duration. An adapter wrapping a
    blocking SDK should hand the call to ``asyncio.to_thread``.
    """

    async def complete(
        self, messages: list[Message], tools: Optional[list[dict]] = None, *,
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> Decision:
        """``on_delta``, when given, receives the assistant text as it
        streams in; the returned Decision is still the whole turn. An
        adapter that cannot stream simply ignores it."""
        ...


class FallbackAdapter:
    """Try each adapter in turn; the first that answers wins. A retry never
    reaches the tools — it happens before any Decision exists."""

    def __init__(self, adapters: list[LLMAdapter]) -> None:
        if not adapters:
            raise ValueError("FallbackAdapter needs at least one adapter")
        self.adapters = list(adapters)

    async def complete(
        self, messages: list[Message], tools: Optional[list[dict]] = None, *,
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> Decision:
        error: Optional[Exception] = None
        for adapter in self.adapters:
            try:
                return await adapter.complete(messages, tools, on_delta=on_delta)
            except Exception as exc:  # noqa: BLE001 — the next adapter gets its turn
                error = exc
        assert error is not None
        raise error


def extract_finish(decision: Decision) -> Decision:
    """Pull a ``finish(result)`` call (if present) out of ``decision.calls``
    and into ``decision.finish``/``decision.result``.

    Any *other* calls made in the same decision are deliberately left in
    ``decision.calls`` rather than dropped, so the session's validator can
    reject the mixed decision explicitly and tell the model why, instead of
    silently discarding side effects it asked for.
    """
    remaining: list[ToolCall] = []
    finished = False
    result: Any = None
    for call in decision.calls:
        if call.name == FINISH_NAME:
            finished = True
            result = call.arguments.get("result") if isinstance(call.arguments, dict) else None
        else:
            remaining.append(call)
    if finished:
        decision.finish = True
        decision.result = result
        decision.calls = remaining
    return decision


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
            calls.append(ToolCall(name=name, arguments={}, raw_arguments=dumps(args)))
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
        return extract_finish(Decision(text=_text, calls=[ToolCall(name=name, arguments=arguments)]))

    @staticmethod
    def calls(*specs: tuple[str, dict[str, Any]], text: str = "") -> Decision:
        return extract_finish(Decision(text=text, calls=[ToolCall(name=n, arguments=a) for n, a in specs]))

    @staticmethod
    def finish(result: Any = None, *, text: str = "") -> Decision:
        return Decision(text=text, finish=True, result=result)

    async def complete(
        self, messages: list[Message], tools: Optional[list[dict]] = None, *,
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> Decision:
        self.requests.append({"messages": list(messages), "tools": list(tools or [])})
        decision = await self._next(messages, tools)
        if on_delta is not None and decision.text:
            on_delta(decision.text)  # one chunk: enough to test a streaming host
        return decision

    async def _next(self, messages: list[Message], tools: Optional[list[dict]]) -> Decision:
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
        return extract_finish(step)
