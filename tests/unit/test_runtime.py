"""Runtime (§6, §8): validation & self-repair, require_spec gate, parallel
execution, retries/timeouts, output policy, budget arithmetic."""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from state_projection_loop import Config, Registry, ToolCall, ValueStore
from state_projection_loop.hooks import Hooks
from state_projection_loop.logger import SessionLogger
from state_projection_loop.projection import TurnContext
from state_projection_loop.runtime import (
    BudgetState,
    Runtime,
    _mini_validate,
    apply_defaults,
    validate_args,
)
from state_projection_loop.tooldef import ToolContext

from _util import echo_handler, tool_dict


def make_runtime(registry: Registry, config: Config | None = None) -> tuple[Runtime, TurnContext, ToolContext]:
    config = config or Config()
    store = ValueStore()
    runtime = Runtime(registry, store, config, Hooks(), SessionLogger())
    turn = TurnContext(config=config, registry=registry, conversation=[], summary=[], store=store)
    ctx = ToolContext(registry=registry, store=store, config=config)
    return runtime, turn, ctx


def echo_registry(**overrides: Any) -> Registry:
    reg = Registry()
    reg.register(
        tool_dict("echo", description="Echo the text back.",
                  properties={"text": {"type": "string"}}, required=["text"], **overrides),
        handler=echo_handler,
    )
    return reg


class TestValidation:
    def test_happy_path(self):
        runtime, turn, ctx = make_runtime(echo_registry())
        [result] = asyncio.run(runtime.execute([ToolCall(name="echo", arguments={"text": "hi"})], turn, ctx))
        assert result.ok and result.observation == "echo: hi"

    def test_type_error_attaches_spec(self):
        runtime, turn, ctx = make_runtime(echo_registry())
        [result] = asyncio.run(runtime.execute([ToolCall(name="echo", arguments={"text": 42})], turn, ctx))
        assert not result.ok
        assert "Validation error" in result.observation
        assert "### echo" in result.observation  # full spec attached (§6 self-repair)
        assert "NOT executed" in result.observation

    def test_missing_required(self):
        runtime, turn, ctx = make_runtime(echo_registry())
        [result] = asyncio.run(runtime.execute([ToolCall(name="echo", arguments={})], turn, ctx))
        assert not result.ok and "required" in result.observation

    def test_consecutive_failures_cut_off(self):
        config = Config()
        config.limits.max_validation_retries = 2
        runtime, turn, ctx = make_runtime(echo_registry(), config)
        bad = lambda: ToolCall(name="echo", arguments={"text": 1})
        r1 = asyncio.run(runtime.execute([bad()], turn, ctx))[0]
        r2 = asyncio.run(runtime.execute([bad()], turn, ctx))[0]
        r3 = asyncio.run(runtime.execute([bad()], turn, ctx))[0]
        assert "### echo" in r1.observation and "### echo" in r2.observation
        assert "giving up" in r3.observation and "### echo" not in r3.observation

    def test_success_resets_failure_streak(self):
        runtime, turn, ctx = make_runtime(echo_registry())
        bad = ToolCall(name="echo", arguments={"text": 1})
        good = ToolCall(name="echo", arguments={"text": "ok"})
        asyncio.run(runtime.execute([bad], turn, ctx))
        asyncio.run(runtime.execute([good], turn, ctx))
        asyncio.run(runtime.execute([ToolCall(name="echo", arguments={"text": 2})], turn, ctx))
        result = asyncio.run(runtime.execute([ToolCall(name="echo", arguments={"text": 3})], turn, ctx))[0]
        assert "### echo" in result.observation  # streak restarted, not cut off

    def test_malformed_raw_arguments(self):
        runtime, turn, ctx = make_runtime(echo_registry())
        call = ToolCall(name="echo", arguments={}, raw_arguments='{"text": broken')
        [result] = asyncio.run(runtime.execute([call], turn, ctx))
        assert not result.ok and "not valid JSON" in result.observation

    def test_defaults_applied(self):
        reg = Registry()
        received = {}

        def handler(text: str, times: int = 0) -> str:
            received["times"] = times
            return text * times

        reg.register(
            tool_dict("rep", properties={"text": {"type": "string"},
                                         "times": {"type": "integer", "default": 2}},
                      required=["text"]),
            handler=handler,
        )
        runtime, turn, ctx = make_runtime(reg)
        [result] = asyncio.run(runtime.execute([ToolCall(name="rep", arguments={"text": "ab"})], turn, ctx))
        assert result.ok and received["times"] == 2 and result.observation == "abab"

    def test_unknown_tool_mentions_find_tools(self):
        runtime, turn, ctx = make_runtime(echo_registry())
        [result] = asyncio.run(runtime.execute([ToolCall(name="nope", arguments={})], turn, ctx))
        assert not result.ok
        assert "find_tools" in result.observation and "misc(1)" in result.observation


class TestRequireSpec:
    def test_first_call_bounced_second_runs(self):
        reg = Registry()
        reg.register(
            tool_dict("danger", require_spec=True,
                      properties={"target": {"type": "string"}}, required=["target"]),
            handler=lambda target: f"deleted {target}",
        )
        runtime, turn, ctx = make_runtime(reg)
        call = ToolCall(name="danger", arguments={"target": "tmp"})
        [first] = asyncio.run(runtime.execute([call], turn, ctx))
        assert not first.ok and "### danger" in first.observation
        [second] = asyncio.run(runtime.execute([call], turn, ctx))
        assert second.ok and second.observation == "deleted tmp"


class TestExecution:
    def test_parallel_safe_calls_run_concurrently(self):
        reg = Registry()

        async def slow(**kwargs: Any) -> str:
            await asyncio.sleep(0.15)
            return "done"

        for name in ("p1", "p2", "p3"):
            reg.register(tool_dict(name, parallel_safe=True), handler=slow)
        runtime, turn, ctx = make_runtime(reg)
        calls = [ToolCall(name=n, arguments={}) for n in ("p1", "p2", "p3")]
        start = time.perf_counter()
        results = asyncio.run(runtime.execute(calls, turn, ctx))
        elapsed = time.perf_counter() - start
        assert all(r.ok for r in results)
        assert elapsed < 0.4  # 3 × 0.15s would be 0.45s serially

    def test_results_keep_call_order(self):
        reg = Registry()
        reg.register(tool_dict("fast", parallel_safe=True), handler=lambda: "fast")
        reg.register(tool_dict("slow_serial"), handler=lambda: "slow")
        runtime, turn, ctx = make_runtime(reg)
        calls = [ToolCall(name="slow_serial", arguments={}), ToolCall(name="fast", arguments={})]
        results = asyncio.run(runtime.execute(calls, turn, ctx))
        assert [r.call.name for r in results] == ["slow_serial", "fast"]

    def test_timeout_becomes_observation(self):
        reg = Registry()

        async def sleeper() -> str:
            await asyncio.sleep(1.0)
            return "never"

        reg.register(tool_dict("sleepy", timeout_s=0.1), handler=sleeper)
        runtime, turn, ctx = make_runtime(reg)
        [result] = asyncio.run(runtime.execute([ToolCall(name="sleepy", arguments={})], turn, ctx))
        assert not result.ok and "timed out" in result.observation

    def test_retry_then_success(self):
        reg = Registry()
        attempts = {"n": 0}

        def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ConnectionError("transient")
            return "recovered"

        reg.register(tool_dict("flaky", retries=1), handler=flaky)
        runtime, turn, ctx = make_runtime(reg)
        [result] = asyncio.run(runtime.execute([ToolCall(name="flaky", arguments={})], turn, ctx))
        assert result.ok and result.observation == "recovered" and attempts["n"] == 2

    def test_exception_becomes_observation(self):
        reg = Registry()

        def boom() -> str:
            raise RuntimeError("kaboom")

        reg.register(tool_dict("boom"), handler=boom)
        runtime, turn, ctx = make_runtime(reg)
        [result] = asyncio.run(runtime.execute([ToolCall(name="boom", arguments={})], turn, ctx))
        assert not result.ok and "kaboom" in result.observation

    def test_ctx_injection(self):
        reg = Registry()

        def with_ctx(ctx: ToolContext, key: str) -> str:
            return f"state[{key}]={ctx.state.get(key)}"

        reg.register(tool_dict("st", properties={"key": {"type": "string"}}, required=["key"]),
                     handler=with_ctx)
        runtime, turn, ctx = make_runtime(reg)
        ctx.state["hp"] = 10
        [result] = asyncio.run(runtime.execute([ToolCall(name="st", arguments={"key": "hp"})], turn, ctx))
        assert result.observation == "state[hp]=10"


class TestOutputPolicy:
    def test_large_result_becomes_handle(self):
        reg = Registry()
        reg.register(tool_dict("big", max_inline_tokens=50),
                     handler=lambda: "data " * 500)
        runtime, turn, ctx = make_runtime(reg)
        [result] = asyncio.run(runtime.execute([ToolCall(name="big", arguments={})], turn, ctx))
        assert result.ok and result.handle == "$h1"
        assert "$h1" in result.observation and "peek" in result.observation
        assert ctx.store.get("$h1") == "data " * 500

    def test_truncate_policy(self):
        reg = Registry()
        reg.register(tool_dict("cut", max_inline_tokens=50, overflow="truncate"),
                     handler=lambda: "data " * 500)
        runtime, turn, ctx = make_runtime(reg)
        [result] = asyncio.run(runtime.execute([ToolCall(name="cut", arguments={})], turn, ctx))
        assert result.handle is None and "[truncated by output_policy]" in result.observation

    def test_handle_argument_resolution(self):
        reg = Registry()
        reg.register(tool_dict("length", properties={"data": {}}, required=["data"]),
                     handler=lambda data: f"len={len(data)}")
        runtime, turn, ctx = make_runtime(reg)
        record = ctx.store.put([1, 2, 3, 4])
        [result] = asyncio.run(
            runtime.execute([ToolCall(name="length", arguments={"data": record.id})], turn, ctx)
        )
        assert result.observation == "len=4"


class TestMiniValidator:
    """The dependency-free fallback (used when jsonschema is absent)."""

    SCHEMA = {
        "type": "object",
        "properties": {
            "q": {"type": "string", "minLength": 2},
            "n": {"type": "integer", "minimum": 1, "maximum": 10},
            "mode": {"enum": ["a", "b"]},
            "items": {"type": "array", "items": {"type": "string"}},
            "opt": {"type": ["string", "null"]},
        },
        "required": ["q"],
        "additionalProperties": False,
    }

    def test_accepts_valid(self):
        assert _mini_validate(self.SCHEMA, {"q": "ok", "n": 5, "mode": "a",
                                            "items": ["x"], "opt": None}) is None

    @pytest.mark.parametrize("args,fragment", [
        ({}, "required"),
        ({"q": "ok", "n": "5"}, "expected type"),
        ({"q": "ok", "n": 0}, "minimum"),
        ({"q": "ok", "n": 11}, "maximum"),
        ({"q": "x"}, "minLength"),
        ({"q": "ok", "mode": "c"}, "not one of"),
        ({"q": "ok", "items": ["x", 1]}, "expected type"),
        ({"q": "ok", "zzz": 1}, "unexpected properties"),
        ({"q": "ok", "n": True}, "expected type"),  # bool is not an integer
    ])
    def test_rejects_invalid(self, args, fragment):
        assert fragment in _mini_validate(self.SCHEMA, args)

    def test_validate_args_agrees(self):
        assert validate_args(self.SCHEMA, {"q": "ok"}) is None
        assert validate_args(self.SCHEMA, {"q": 1}) is not None
        assert validate_args(self.SCHEMA, "not a dict") is not None

    def test_apply_defaults(self):
        schema = {"type": "object", "properties": {"k": {"type": "integer", "default": 7}}}
        assert apply_defaults(schema, {}) == {"k": 7}
        assert apply_defaults(schema, {"k": 1}) == {"k": 1}


class TestBudgetState:
    def test_steps_and_tokens(self):
        cfg = Config.from_dict({"budget": {"max_steps": 2, "max_tokens": 100}})
        b = BudgetState()
        assert b.exceeded(cfg) is None
        b.steps = 2
        assert "max_steps" in b.exceeded(cfg)
        b.steps = 0
        b.note_usage(80, 30, cfg)
        assert "max_tokens" in b.exceeded(cfg)

    def test_cost_accounting(self):
        cfg = Config.from_dict({"budget": {"max_steps": 99, "max_cost": 0.01,
                                           "cost_per_1k_input": 0.001, "cost_per_1k_output": 0.002}})
        b = BudgetState()
        b.note_usage(5000, 2500, cfg)  # 0.005 + 0.005 = 0.01
        assert "max_cost" in b.exceeded(cfg)

    def test_max_seconds(self):
        cfg = Config.from_dict({"budget": {"max_steps": 99, "max_seconds": 0.0}})
        assert "max_seconds" in BudgetState().exceeded(cfg)
