"""Offline check for the benchmark harness - no API key, no spend.

Exercises the two pieces that can silently lie: the Anthropic message
converter (orphan tool_use/tool_result pairing) and the arm wiring
(pinned vs discovered), plus one full end-to-end loop against a fake
client so a live run cannot be the first time the path is tried.

    python -m benchmarks.test_bench
"""
from __future__ import annotations

from types import SimpleNamespace

from state_projection_loop.capability import build_capability_from_function
from state_projection_loop.messages import ASSISTANT, OBSERVATION, SYSTEM, USER, Message, ToolCall

from benchmarks.adapter import AnthropicAdapter, _convert, _to_anthropic_tools
from benchmarks.bench import ARMS, REAL, TASKS, filler_capabilities, run_one


def test_convert_pairs_and_orphans() -> None:
    answered = ToolCall(name="a__b__c", arguments={"x": 1}, id="call_ok")
    orphan = ToolCall(name="d__e__f", arguments={"y": 2}, id="call_orphan")
    messages = [
        Message(role=SYSTEM, content="kernel"),
        Message(role=USER, content="question"),
        Message(role=ASSISTANT, content="thinking", tool_calls=[answered, orphan]),
        Message(role=OBSERVATION, content="result-a", tool_call_id="call_ok", name="a.b.c"),
        # A result whose tool_use was compressed out of the projection.
        Message(role=OBSERVATION, content="result-z", tool_call_id="call_gone", name="z.z.z"),
    ]
    system, out = _convert(messages)
    assert system == "kernel"
    assert out[0]["role"] == "user"

    uses = {b["id"] for m in out for b in m["content"] if b["type"] == "tool_use"}
    results = {b["tool_use_id"] for m in out for b in m["content"] if b["type"] == "tool_result"}
    assert uses == {"call_ok"}, f"orphan tool_use survived: {uses}"
    assert results == {"call_ok"}, f"orphan tool_result survived: {results}"

    # The dropped orphans must still be visible to the model as text.
    blob = " ".join(b.get("text", "") for m in out for b in m["content"] if b["type"] == "text")
    assert "d__e__f" in blob and "result-z" in blob, blob

    for m in out:
        assert m["role"] in ("user", "assistant")
        assert m["content"], "empty content block would 400"


def test_tool_schema_conversion() -> None:
    # The runtime hands adapters the neutral spec; only the adapter knows
    # that Anthropic calls the JSON Schema "input_schema".
    neutral = [{"name": "inventory__stock__get", "description": "d",
                "parameters": {"type": "object", "properties": {"warehouse": {"type": "string"}}}}]
    out = _to_anthropic_tools(neutral)
    assert out == [{"name": "inventory__stock__get", "description": "d",
                    "input_schema": {"type": "object",
                                     "properties": {"warehouse": {"type": "string"}}}}]


def test_arms_differ_only_in_exposure() -> None:
    caps = REAL + filler_capabilities(20)
    for arm, spec in ARMS.items():
        built = [build_capability_from_function(fn, pinned=spec["pinned"], **meta)
                 for fn, meta in caps]
        pinned = sum(c.discovery.pinned for c in built)
        assert pinned == (len(built) if arm == "preload" else 0), (arm, pinned)
    assert len(filler_capabilities(200)) == 200
    assert len({m["name"] for _, m in filler_capabilities(200)}) == 200


# --------------------------------------------------------------------------
# End-to-end against a fake Anthropic client.
# --------------------------------------------------------------------------

def _usage(i: int = 100, o: int = 20) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=i, output_tokens=o,
                           cache_read_input_tokens=0, cache_creation_input_tokens=0)


class FakeClient:
    """Calls the warehouse tool once, then finishes."""

    def __init__(self) -> None:
        self.n = 0
        self.messages = self

    def create(self, **kwargs):
        self.n += 1
        names = {t["name"] for t in kwargs.get("tools", [])}
        assert "finish" in names, "finish schema must always be offered"
        if self.n == 1:
            assert "inventory__stock__get" in names, sorted(names)[:8]
            return SimpleNamespace(content=[SimpleNamespace(
                type="tool_use", id="call_1", name="inventory__stock__get",
                input={"warehouse": "tokyo"})], usage=_usage())
        return SimpleNamespace(content=[SimpleNamespace(
            type="tool_use", id="call_2", name="finish",
            input={"result": "Tokyo has 42 units on hand."})], usage=_usage())


def test_end_to_end_offline(monkeypatched: bool = True) -> None:
    import benchmarks.adapter as adapter_mod

    real_init = AnthropicAdapter.__init__

    def patched(self, model="claude-haiku-4-5", **kw):
        real_init(self, model=model, client=FakeClient(), **kw)

    AnthropicAdapter.__init__ = patched  # type: ignore[method-assign]
    try:
        task = next(t for t in TASKS if t.key == "stock")
        r = run_one("preload", 20, task, 0, "fake")
        assert not r.error, r.error
        assert r.ok, "deterministic check failed on a known-good answer"
        assert r.recall, r.calls
        assert r.api_calls == 2, r.api_calls
        assert r.cost > 0
    finally:
        AnthropicAdapter.__init__ = real_init  # type: ignore[method-assign]
    assert adapter_mod is not None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
