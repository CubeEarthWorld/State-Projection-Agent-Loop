"""Runtime: validation & self-repair, require_spec gate, ordering (P0-1),
retry-safety-gated retries and OUTCOME_UNKNOWN (P0-2), output policy,
budget arithmetic."""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from state_projection_loop import Config, Registry, ToolCall
from state_projection_loop.artifacts import ArtifactStore, ref
from state_projection_loop.capability import ToolContext
from state_projection_loop.events import InMemoryLedger
from state_projection_loop.policy import PolicyEngine
from state_projection_loop.projection import TurnContext
from state_projection_loop.run import Run
from state_projection_loop.runtime import (
    BudgetState,
    Runtime,
    _mini_validate,
    apply_defaults,
    validate_args,
)

from _util import capability_dict, echo_handler


def make_runtime(registry: Registry, config: Config | None = None, *, allow_all: bool = True):
    config = config or Config()
    store = ArtifactStore("run_test")
    runtime = Runtime(registry, store, config)
    ledger = InMemoryLedger()
    run = Run("run_test", "ses_test", ledger)
    policy = PolicyEngine(default_decision="allow" if allow_all else "require_approval")
    turn = TurnContext(config=config, registry=registry, ledger=ledger, run_id="run_test", store=store)
    ctx = ToolContext(registry=registry, store=store, config=config, ledger=ledger, run=run)
    return runtime, turn, ctx, run, policy


def run_batch(runtime, calls, turn, ctx, run, policy):
    return asyncio.run(runtime.execute(calls, turn, ctx, run, policy))


def echo_registry(**overrides: Any) -> Registry:
    reg = Registry()
    reg.register(
        capability_dict("demo.echo", description="Echo the text back.",
                         properties={"text": {"type": "string"}}, required=["text"], **overrides),
        handler=echo_handler,
    )
    return reg


class TestValidation:
    def test_happy_path(self):
        runtime, turn, ctx, run, policy = make_runtime(echo_registry())
        batch = run_batch(runtime, [ToolCall(name="demo.echo", arguments={"text": "hi"})], turn, ctx, run, policy)
        [result] = batch.results
        assert result.ok and result.observation == "echo: hi"
        assert result.outcome == "ok"

    def test_type_error_attaches_spec(self):
        runtime, turn, ctx, run, policy = make_runtime(echo_registry())
        batch = run_batch(runtime, [ToolCall(name="demo.echo", arguments={"text": 42})], turn, ctx, run, policy)
        [result] = batch.results
        assert not result.ok
        assert "Validation error" in result.observation
        assert "### demo.echo" in result.observation
        assert "NOT executed" in result.observation

    def test_missing_required(self):
        runtime, turn, ctx, run, policy = make_runtime(echo_registry())
        batch = run_batch(runtime, [ToolCall(name="demo.echo", arguments={})], turn, ctx, run, policy)
        assert not batch.results[0].ok and "required" in batch.results[0].observation

    def test_malformed_raw_arguments(self):
        runtime, turn, ctx, run, policy = make_runtime(echo_registry())
        call = ToolCall(name="demo.echo", arguments={}, raw_arguments='{"text": broken')
        batch = run_batch(runtime, [call], turn, ctx, run, policy)
        assert not batch.results[0].ok and "not valid JSON" in batch.results[0].observation

    def test_unknown_capability_mentions_find_tools(self):
        runtime, turn, ctx, run, policy = make_runtime(echo_registry())
        batch = run_batch(runtime, [ToolCall(name="nope.nope", arguments={})], turn, ctx, run, policy)
        assert not batch.results[0].ok
        assert "find_tools" in batch.results[0].observation


class TestRequireSpec:
    def test_first_call_bounced_second_runs(self):
        reg = Registry()
        reg.register(
            capability_dict("demo.danger", require_spec=True,
                             properties={"target": {"type": "string"}}, required=["target"]),
            handler=lambda target: f"deleted {target}",
        )
        runtime, turn, ctx, run, policy = make_runtime(reg)
        call = ToolCall(name="demo.danger", arguments={"target": "tmp"})
        first = run_batch(runtime, [call], turn, ctx, run, policy).results[0]
        assert not first.ok and "### demo.danger" in first.observation
        second = run_batch(runtime, [call], turn, ctx, run, policy).results[0]
        assert second.ok and second.observation == "deleted tmp"


class TestOrdering:
    """P0-1: calls execute in the model's stated order; only a contiguous
    run of read-only capabilities may run concurrently."""

    def test_write_then_read_preserves_order(self):
        reg = Registry()
        log: list[str] = []

        def write(value: str) -> str:
            log.append(f"write:{value}")
            return "written"

        def read() -> str:
            log.append("read")
            return "".join(log)

        reg.register(capability_dict("fs.write", properties={"value": {"type": "string"}},
                                      required=["value"], effects=[("write", "workspace:*")]),
                     handler=write)
        reg.register(capability_dict("fs.read", effects=[("read", "workspace:*")]), handler=read)
        runtime, turn, ctx, run, policy = make_runtime(reg)
        calls = [ToolCall(name="fs.write", arguments={"value": "x"}), ToolCall(name="fs.read", arguments={})]
        batch = run_batch(runtime, calls, turn, ctx, run, policy)
        assert log == ["write:x", "read"]
        assert [r.call.name for r in batch.results] == ["fs.write", "fs.read"]

    def test_adjacent_read_only_calls_run_concurrently(self):
        reg = Registry()

        async def slow(**kwargs: Any) -> str:
            await asyncio.sleep(0.15)
            return "done"

        for name in ("demo.p1", "demo.p2", "demo.p3"):
            reg.register(capability_dict(name, effects=[("read", "workspace:*")]), handler=slow)
        runtime, turn, ctx, run, policy = make_runtime(reg)
        calls = [ToolCall(name=n, arguments={}) for n in ("demo.p1", "demo.p2", "demo.p3")]
        start = time.perf_counter()
        batch = run_batch(runtime, calls, turn, ctx, run, policy)
        elapsed = time.perf_counter() - start
        assert all(r.ok for r in batch.results)
        assert elapsed < 0.4  # 3 x 0.15s would be ~0.45s serially

    def test_write_breaks_the_parallel_streak(self):
        reg = Registry()
        reg.register(capability_dict("demo.read1", effects=[("read", "workspace:*")]), handler=lambda: "r1")
        reg.register(capability_dict("demo.write1", effects=[("write", "workspace:*")]), handler=lambda: "w1")
        reg.register(capability_dict("demo.read2", effects=[("read", "workspace:*")]), handler=lambda: "r2")
        runtime, turn, ctx, run, policy = make_runtime(reg)
        calls = [ToolCall(name=n, arguments={}) for n in ("demo.read1", "demo.write1", "demo.read2")]
        batch = run_batch(runtime, calls, turn, ctx, run, policy)
        assert [r.call.name for r in batch.results] == ["demo.read1", "demo.write1", "demo.read2"]
        assert [r.value for r in batch.results] == ["r1", "w1", "r2"]


class TestRetrySafety:
    """P0-2: retries are only permitted for pure/idempotent capabilities;
    a timeout is OUTCOME_UNKNOWN, never silently 'failed'."""

    def test_timeout_is_outcome_unknown_not_failed(self):
        reg = Registry()

        async def sleeper() -> str:
            await asyncio.sleep(1.0)
            return "never"

        reg.register(capability_dict("demo.sleepy", timeout_s=0.1), handler=sleeper)
        runtime, turn, ctx, run, policy = make_runtime(reg)
        batch = run_batch(runtime, [ToolCall(name="demo.sleepy", arguments={})], turn, ctx, run, policy)
        result = batch.results[0]
        assert not result.ok
        assert result.outcome == "unknown"
        assert "UNKNOWN" in result.observation

    def test_timeout_on_never_retry_does_not_retry(self):
        reg = Registry()
        attempts = {"n": 0}

        async def sleeper() -> str:
            attempts["n"] += 1
            await asyncio.sleep(1.0)
            return "never"

        reg.register(capability_dict("demo.sleepy", timeout_s=0.05, retry_safety="never_retry", retries=0),
                     handler=sleeper)
        runtime, turn, ctx, run, policy = make_runtime(reg)
        run_batch(runtime, [ToolCall(name="demo.sleepy", arguments={})], turn, ctx, run, policy)
        assert attempts["n"] == 1

    def test_idempotent_retry_then_success(self):
        reg = Registry()
        attempts = {"n": 0}

        def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ConnectionError("transient")
            return "recovered"

        reg.register(capability_dict("demo.flaky", retries=1, retry_safety="idempotent"), handler=flaky)
        runtime, turn, ctx, run, policy = make_runtime(reg)
        batch = run_batch(runtime, [ToolCall(name="demo.flaky", arguments={})], turn, ctx, run, policy)
        result = batch.results[0]
        assert result.ok and result.observation == "recovered" and attempts["n"] == 2

    def test_command_id_stable_across_retries(self):
        reg = Registry()
        seen_ids: list[str] = []

        def flaky(ctx: ToolContext) -> str:
            seen_ids.append(ctx.command_id)
            if len(seen_ids) == 1:
                raise ConnectionError("transient")
            return "ok"

        reg.register(capability_dict("demo.flaky", retries=1, retry_safety="idempotent"), handler=flaky)
        runtime, turn, ctx, run, policy = make_runtime(reg)
        run_batch(runtime, [ToolCall(name="demo.flaky", arguments={})], turn, ctx, run, policy)
        assert len(seen_ids) == 2 and seen_ids[0] == seen_ids[1]

    def test_exception_becomes_failed_observation(self):
        reg = Registry()

        def boom() -> str:
            raise RuntimeError("kaboom")

        reg.register(capability_dict("demo.boom"), handler=boom)
        runtime, turn, ctx, run, policy = make_runtime(reg)
        batch = run_batch(runtime, [ToolCall(name="demo.boom", arguments={})], turn, ctx, run, policy)
        result = batch.results[0]
        assert not result.ok and result.outcome == "failed" and "kaboom" in result.observation

    def test_ctx_injection(self):
        reg = Registry()

        def with_ctx(ctx: ToolContext, key: str) -> str:
            return f"state[{key}]={ctx.working_state.extra.get(key)}"

        reg.register(capability_dict("demo.st", properties={"key": {"type": "string"}}, required=["key"]),
                     handler=with_ctx)
        runtime, turn, ctx, run, policy = make_runtime(reg)
        from state_projection_loop.working_state import WorkingState

        ctx.working_state = WorkingState(extra={"hp": 10})
        batch = run_batch(runtime, [ToolCall(name="demo.st", arguments={"key": "hp"})], turn, ctx, run, policy)
        assert batch.results[0].value == "state[hp]=10"


class TestPolicyGating:
    def test_deny_prevents_execution(self):
        reg = Registry()
        called = {"n": 0}

        def handler():
            called["n"] += 1
            return "ran"

        reg.register(capability_dict("demo.risky", effects=[("external", "*")]), handler=handler)
        runtime, turn, ctx, run, policy = make_runtime(reg, allow_all=False)
        policy.default_decision = "deny"
        batch = run_batch(runtime, [ToolCall(name="demo.risky", arguments={})], turn, ctx, run, policy)
        assert not batch.results[0].ok
        assert batch.results[0].outcome == "denied"
        assert called["n"] == 0

    def test_require_approval_halts_batch_and_preserves_pending(self):
        reg = Registry()
        reg.register(capability_dict("demo.risky", effects=[("external", "*")]), handler=lambda: "ran")
        reg.register(capability_dict("demo.after"), handler=lambda: "after")
        runtime, turn, ctx, run, policy = make_runtime(reg, allow_all=False)
        calls = [ToolCall(name="demo.risky", arguments={}), ToolCall(name="demo.after", arguments={})]
        batch = run_batch(runtime, calls, turn, ctx, run, policy)
        assert batch.halted is True
        assert run.state == "WAITING_FOR_APPROVAL"
        assert [c.name for c in run.pending_calls] == ["demo.risky", "demo.after"]


class TestOutputPolicy:
    def test_large_result_becomes_artifact(self):
        reg = Registry()
        reg.register(capability_dict("demo.big", max_inline_tokens=50), handler=lambda: "data " * 500)
        runtime, turn, ctx, run, policy = make_runtime(reg)
        batch = run_batch(runtime, [ToolCall(name="demo.big", arguments={})], turn, ctx, run, policy)
        result = batch.results[0]
        assert result.ok and result.artifact_id is not None
        assert result.artifact_id in result.observation
        assert ctx.store.get(result.artifact_id) == "data " * 500

    def test_truncate_policy(self):
        reg = Registry()
        reg.register(capability_dict("demo.cut", max_inline_tokens=50, overflow="truncate"),
                     handler=lambda: "data " * 500)
        runtime, turn, ctx, run, policy = make_runtime(reg)
        batch = run_batch(runtime, [ToolCall(name="demo.cut", arguments={})], turn, ctx, run, policy)
        result = batch.results[0]
        assert result.artifact_id is None and "[truncated by output_policy]" in result.observation

    def test_artifact_reference_resolution(self):
        reg = Registry()
        reg.register(capability_dict("demo.length", properties={"data": {}}, required=["data"]),
                     handler=lambda data: f"len={len(data)}")
        runtime, turn, ctx, run, policy = make_runtime(reg)
        record = ctx.store.put([1, 2, 3, 4])
        batch = run_batch(
            runtime, [ToolCall(name="demo.length", arguments={"data": ref(record.id)})], turn, ctx, run, policy,
        )
        assert batch.results[0].value == "len=4"

    def test_bare_string_matching_artifact_id_not_resolved(self):
        reg = Registry()
        reg.register(capability_dict("demo.echo2", properties={"data": {}}, required=["data"]),
                     handler=lambda data: data)
        runtime, turn, ctx, run, policy = make_runtime(reg)
        record = ctx.store.put([1, 2, 3])
        batch = run_batch(
            runtime, [ToolCall(name="demo.echo2", arguments={"data": record.id})], turn, ctx, run, policy,
        )
        assert batch.results[0].value == record.id  # literal string passed through


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
        ({"q": "ok", "n": True}, "expected type"),
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
