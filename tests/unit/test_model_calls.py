"""The model call and what surrounds it: streaming deltas, the timeout /
retry / fallback envelope, cancellation, hooks around tool calls, and
content parts in the user message."""
from __future__ import annotations

import asyncio

import pytest

from state_projection_loop import Config, FallbackAdapter, Hooks, Registry, ScriptedLLM, Session
from state_projection_loop.compaction import FOLD_INSTRUCTIONS
from state_projection_loop.messages import Decision
from state_projection_loop.tokens import IMAGE_TOKENS, estimate_tokens

from _util import allow_all, capability_dict


def events(session: Session, kind: str) -> list[dict]:
    return [e.data for e in session.ledger.iter_run(session.run.id) if e.type == kind]


class TestStreaming:
    def test_model_text_and_tool_progress_reach_on_delta_but_not_the_ledger(self):
        registry = Registry()
        registry.register(capability_dict("demo.slow.work"), handler=lambda ctx: ctx.emit("50%") or "done")
        deltas: list[tuple[str, str]] = []
        session = Session(ScriptedLLM([ScriptedLLM.call("demo.slow.work"), "all done"]),
                          registry=registry, policy=allow_all(), on_delta=lambda s, t: deltas.append((s, t)))
        assert session.send("go") == "all done"
        assert deltas == [("tool", "50%"), ("model", "all done")]
        assert "50%" not in "".join(str(e.data) for e in session.ledger.iter_run(session.run.id))

    def test_an_adapter_without_streaming_is_not_asked_to_stream(self):
        class Plain:
            async def complete(self, messages, tools=None):
                return Decision(text="ok")

        assert Session(Plain()).send("hi") == "ok"


class TestModelCallEnvelope:
    def test_a_timeout_is_retried_then_recorded_and_raised(self):
        class Hanging:
            async def complete(self, messages, tools=None, *, on_delta=None):
                await asyncio.sleep(10)

        session = Session(Hanging(), config=Config.from_dict(
            {"mode": "job", "model": {"timeout_s": 0.01, "retries": 2, "backoff_s": 0}}))
        with pytest.raises(asyncio.TimeoutError):
            session.run_job("task")
        assert [e["attempt"] for e in events(session, "model_call_failed")] == [1, 2, 3]
        assert session.run.state == "FAILED"

    def test_a_provider_error_that_clears_up_on_retry_is_invisible_to_the_caller(self):
        attempts = {"n": 0}

        class Flaky:
            async def complete(self, messages, tools=None, *, on_delta=None):
                attempts["n"] += 1
                if attempts["n"] == 1:
                    raise ConnectionError("429")
                return Decision(text="fine")

        session = Session(Flaky(), config=Config.from_dict({"model": {"retries": 1, "backoff_s": 0}}))
        assert session.send("hi") == "fine"
        assert [e["error"] for e in events(session, "model_call_failed")] == ["ConnectionError: 429"]

    def test_fallback_adapter_moves_on_when_the_first_raises(self):
        class Broken:
            async def complete(self, messages, tools=None, *, on_delta=None):
                raise RuntimeError("down")

        session = Session(FallbackAdapter([Broken(), ScriptedLLM(["from the backup"])]))
        assert session.send("hi") == "from the backup"

    def test_interrupt_cancels_a_model_call_that_is_still_waiting(self):
        started = asyncio.Event()

        class Waiting:
            async def complete(self, messages, tools=None, *, on_delta=None):
                started.set()
                await asyncio.sleep(10)

        session = Session(Waiting())

        async def scenario():
            turn = asyncio.create_task(session.asend("go"))
            await started.wait()
            session.interrupt()
            return await asyncio.wait_for(turn, timeout=2)

        assert asyncio.run(scenario()) == "[interrupted]"
        assert [e["reason"] for e in events(session, "run_state_changed")][-1] == "interrupted"

    def test_interrupt_during_a_compaction_fold_returns_like_any_other_interrupt(self):
        """The fold's model call is a model call like any other: interrupting
        it returns the last answer instead of raising CancelledError at the
        host, and nothing after the fold runs."""
        started = asyncio.Event()

        class FoldHangs:
            def __init__(self) -> None:
                self.replies = 0

            async def complete(self, messages, tools=None, *, on_delta=None):
                if messages[0].content == FOLD_INSTRUCTIONS:
                    started.set()
                    await asyncio.sleep(10)
                self.replies += 1
                return Decision(text=f"r{self.replies}")

        llm = FoldHangs()
        session = Session(llm, config=Config.from_dict({"compaction": {"trigger_ratio": 0.01}}))

        async def scenario():
            for i in range(12):  # the verbatim point steps on the 13th turn
                await session.asend(f"m{i}")
            turn = asyncio.create_task(session.asend("m12"))
            await started.wait()
            session.interrupt()
            return await asyncio.wait_for(turn, timeout=2)

        assert asyncio.run(scenario()) == "r12"
        assert llm.replies == 12  # the interrupted turn asked for no decision
        assert events(session, "state_folded") == []


class TestHooks:
    @staticmethod
    def registry(seen: list) -> Registry:
        registry = Registry()
        registry.register(capability_dict("demo.write", properties={"path": {"type": "string"}}, required=["path"],
                                          effects=[("write", "workspace:*")]),
                          handler=lambda path: seen.append(path) or f"wrote {path}")
        return registry

    def test_before_call_may_rewrite_arguments_and_the_rewrite_is_validated(self):
        seen: list[str] = []
        hooks = Hooks(before_call=lambda cap, args, ctx: {"path": "sandbox/" + args["path"]})
        session = Session(ScriptedLLM([ScriptedLLM.call("demo.write", path="a.txt"), "ok"]),
                          registry=self.registry(seen), policy=allow_all(), hooks=hooks)
        session.send("go")
        assert seen == ["sandbox/a.txt"]
        assert events(session, "hook_intervened") == [
            {"command_id": session.run.commands and next(iter(session.run.commands)), "stage": "before",
             "arguments": {"path": "sandbox/a.txt"}}]

    def test_before_call_may_reject_and_an_invalid_rewrite_is_a_rejection(self):
        seen: list[str] = []
        hooks = Hooks(before_call=lambda cap, args, ctx: "lint failed" if args["path"] == "a.txt" else {"path": 3})
        session = Session(ScriptedLLM([ScriptedLLM.calls(("demo.write", {"path": "a.txt"}), ("demo.write", {"path": "b"})),
                                       "ok"]), registry=self.registry(seen), policy=allow_all(), hooks=hooks)
        session.send("go")
        assert seen == []
        observations = [e["text"] for e in events(session, "observation")]
        assert observations[0] == "Rejected by hook: lint failed"
        assert observations[1].startswith("Rejected by hook: hook returned invalid arguments")
        assert [c.outcome for c in session.run.commands.values()] == ["failed", "failed"]

    def test_after_call_may_replace_the_observation(self):
        seen: list[str] = []
        hooks = Hooks(after_call=lambda cap, args, result, ctx: result.observation.replace("a.txt", "[redacted]"))
        session = Session(ScriptedLLM([ScriptedLLM.call("demo.write", path="a.txt"), "ok"]),
                          registry=self.registry(seen), policy=allow_all(), hooks=hooks)
        session.send("go")
        assert [e["text"] for e in events(session, "observation")] == ["wrote [redacted]"]
        assert events(session, "hook_intervened")[0]["stage"] == "after"

    def test_a_throwing_hook_rejects_its_own_call_without_losing_the_batch(self):
        """A host hook that raises must not take the whole batch with it: the
        calls that already ran are recorded, and the broken one is rejected
        exactly like a hook that returned a rejection string."""
        seen: list[str] = []

        def boom(cap, args, ctx):
            if args["path"] == "b.txt":
                raise RuntimeError("hook exploded")
            return None

        calls = ScriptedLLM.calls(("demo.write", {"path": "a.txt"}), ("demo.write", {"path": "b.txt"}),
                                  ("demo.write", {"path": "c.txt"}))
        session = Session(ScriptedLLM([calls, "ok"]), registry=self.registry(seen), policy=allow_all(),
                          hooks=Hooks(before_call=boom))
        assert session.send("go") == "ok"
        assert seen == ["a.txt", "c.txt"]
        observations = [e["text"] for e in events(session, "observation")]
        assert observations == ["wrote a.txt", "Rejected by hook: RuntimeError: hook exploded", "wrote c.txt"]

    def test_a_throwing_after_hook_keeps_the_real_observation(self):
        seen: list[str] = []

        def boom(cap, args, result, ctx):
            raise RuntimeError("redactor exploded")

        session = Session(ScriptedLLM([ScriptedLLM.call("demo.write", path="a.txt"), "ok"]),
                          registry=self.registry(seen), policy=allow_all(), hooks=Hooks(after_call=boom))
        assert session.send("go") == "ok"
        assert [e["text"] for e in events(session, "observation")] == ["wrote a.txt"]
        assert events(session, "hook_intervened")[0]["error"] == "RuntimeError: redactor exploded"


class TestContentParts:
    def test_a_user_message_may_carry_image_parts(self):
        parts = [{"type": "text", "text": "what is this?"},
                 {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 40000}}]
        llm = ScriptedLLM(["a cat"])
        session = Session(llm)
        assert session.send(parts) == "a cat"
        user = [m for m in llm.requests[0]["messages"] if m.role == "user"][0]
        assert user.content == parts
        assert estimate_tokens(parts[1]) == IMAGE_TOKENS
        assert estimate_tokens(user) < 2 * IMAGE_TOKENS, "the base64 body must not be counted as text"
