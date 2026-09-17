"""Standard agent features: ask (pause for the user), loop guard, result
schema, observers, compaction, skills, toolkits."""
from __future__ import annotations

import time

import pytest

from state_projection_loop import (
    FOLD_INSTRUCTIONS, Config, PendingQuestion, PolicyEngine, Registry, Run, ScriptedLLM, Session,
    install_toolkits, skill_capability,
)
from state_projection_loop.policy import Rule

from _util import capability_dict


def allow_all() -> PolicyEngine:
    return PolicyEngine(default_decision="allow")


def observations(session: Session) -> list[tuple[str, str]]:
    return [(e.data["name"], e.data["text"]) for e in session.ledger.iter_run(session.run.id) if e.type == "observation"]


def notices(session: Session) -> list[str]:
    return [e.data["text"] for e in session.ledger.iter_run(session.run.id) if e.type == "notice"]


class TestAsk:
    def test_pauses_survives_snapshot_and_resumes_with_the_answer(self):
        session = Session(
            ScriptedLLM([ScriptedLLM.call("meta.user.ask", question="Which colour?"), "you said blue"]),
            builtins=["meta", "ask"], policy=allow_all(),
        )
        paused = session.send("pick a colour for me")
        assert isinstance(paused, PendingQuestion)
        assert session.run.state == "WAITING_FOR_USER"
        assert observations(session) == []

        restored = Run.from_snapshot_state(session.run.id, session.ledger, session.run.to_snapshot_state())
        assert restored.pending_question.text == "Which colour?"

        session.answer("blue")
        assert session.run.state == "RUNNING"
        assert session.resume() == "you said blue"
        assert observations(session) == [("meta.user.ask", "blue")]
        types = {e.type for e in session.ledger.iter_run(session.run.id)}
        assert {"question_asked", "question_answered"} <= types
        assert [c.outcome for c in session.run.commands.values()] == ["ok"]

    def test_calls_parked_behind_a_question_still_face_the_policy(self):
        sent = []
        registry = Registry()
        registry.register(capability_dict("mail.message.send", effects=[("external", "smtp:*")]),
                          handler=lambda: sent.append(True) or "sent")
        policy = PolicyEngine(default_decision="allow")
        policy.add_rule("admin", Rule(decision="deny", capability_pattern="mail.*"))
        session = Session(
            ScriptedLLM([ScriptedLLM.calls(("meta.user.ask", {"question": "Send it?"}), ("mail.message.send", {})),
                         "ok"]),
            registry=registry, builtins=["ask"], policy=policy,
        )
        session.send("mail the report")
        session.answer("yes")
        session.resume()
        assert sent == [], "answering a question must not wave the next call past the policy"
        assert ("mail.message.send", "Denied by policy (admin): ") == tuple(
            (n, t[:len("Denied by policy (admin): ")]) for n, t in observations(session) if n == "mail.message.send")[0]

    def test_default_policy_lets_the_model_ask_without_approval(self):
        session = Session(ScriptedLLM([]), builtins=["ask"])
        ask = session.registry.get("meta.user.ask")
        assert session.policy.evaluate(ask, {"question": "q"}).decision == "allow"


class TestLoopGuard:
    @staticmethod
    def registry() -> Registry:
        reg = Registry()

        def fail(x: int) -> str:
            raise RuntimeError(f"boom {time.time_ns()}")

        reg.register(capability_dict("demo.fail", properties={"x": {"type": "integer"}}), handler=fail)
        reg.register(capability_dict("demo.same", properties={"x": {"type": "integer"}}, effects=[("write", "w:*")]),
                     handler=lambda x: "same")
        reg.register(capability_dict("demo.poll", retry_safety="pure", effects=[("read", "r:*")]),
                     handler=lambda: "pending")
        return reg

    def drive(self, tool: str, times: int, **config) -> Session:
        steps = [ScriptedLLM.call(tool, x=1) if tool != "demo.poll" else ScriptedLLM.call(tool) for _ in range(times)]
        session = Session(ScriptedLLM(steps + ["done"]), registry=self.registry(), policy=allow_all(), builtins=(),
                          config=Config.from_dict(config) if config else None)
        session.send("go")
        return session

    def test_identically_failing_call_is_refused_after_max_repeats(self):
        texts = [t for _, t in observations(self.drive("demo.fail", 4))]
        assert all("boom" in t for t in texts[:3])
        assert "Loop guard" in texts[3]

    def test_identical_non_read_result_refused_but_pure_read_may_poll(self):
        assert "Loop guard" in observations(self.drive("demo.same", 4))[-1][1]
        assert all(t == "pending" for _, t in observations(self.drive("demo.poll", 5)))

    def test_zero_disables_the_guard(self):
        session = self.drive("demo.same", 5, limits={"max_repeats": 0})
        assert all(t == "same" for _, t in observations(session))


class TestResultSchema:
    def test_failing_result_is_bounced_back(self):
        session = Session(
            ScriptedLLM([ScriptedLLM.finish("oops"), ScriptedLLM.finish({"answer": 42})]),
            config=Config.from_dict({"mode": "job", "result_schema": {"type": "object", "required": ["answer"]}}),
            policy=allow_all(),
        )
        assert session.run_job("compute") == {"answer": 42}
        assert session.run.state == "COMPLETED"
        assert len(notices(session)) == 1 and "finish(result) rejected" in notices(session)[0]


class TestObservers:
    def test_every_append_reaches_the_observer_and_a_raising_observer_is_harmless(self):
        seen: list[str] = []

        def observer(event):
            seen.append(event.type)
            raise RuntimeError("observer bug")

        session = Session(ScriptedLLM(["hi"]), on_event=observer, policy=allow_all())
        assert session.send("hello") == "hi"
        assert {"run_state_changed", "user_input", "projection_compiled", "model_response"} <= set(seen)


class TestCompaction:
    def test_folds_history_older_than_the_full_window(self):
        folds: list[int] = []

        def fold(messages, tools):
            assert messages[0].content == FOLD_INSTRUCTIONS
            assert not tools
            folds.append(1)
            return '```json\n{"facts_add": ["user likes blue"], "next_actions": ["ship"]}\n```'

        session = Session(
            ScriptedLLM(["r1", "r2", "r3", fold, "r4"]),
            config=Config.from_dict({"compaction": {"trigger_ratio": 0.01}}), policy=allow_all(),
        )
        for m in ("m1", "m2", "m3"):
            session.send(m)
        assert folds == []
        assert session.send("m4") == "r4"
        assert folds == [1]
        assert session.working_state.confirmed_facts == ["user likes blue"]
        assert session.working_state.next_actions == ["ship"]
        assert session.working_state.folded_sequence > 0
        folded = [e for e in session.ledger.iter_run(session.run.id) if e.type == "state_folded"]
        assert len(folded) == 1 and folded[0].data["before"]["confirmed_facts"] == []

    def test_invalid_delta_is_skipped_and_logged(self):
        session = Session(
            ScriptedLLM(["r1", "r2", "r3", '{"facts_add": "not a list"}', "r4"]),
            config=Config.from_dict({"compaction": {"trigger_ratio": 0.01}}), policy=allow_all(),
        )
        for m in ("m1", "m2", "m3", "m4"):
            session.send(m)
        assert session.working_state.confirmed_facts == []
        assert len(notices(session)) == 1 and "compaction skipped" in notices(session)[0]


class TestSkills:
    def test_a_skill_is_a_capability_the_model_discovers_and_loads(self):
        session = Session(ScriptedLLM([]), policy=allow_all())
        session.registry.register(skill_capability("deploy", "Step 1: build. Step 2: tag.", summary="How to deploy"))
        assert session.registry.categories()["skill"] == (1, 0)
        assert session.invoke("skill.deploy.load") == "Step 1: build. Step 2: tag."
        assert session.search.search("how do I deploy")[0].tool.name == "skill.deploy.load"


class TestToolkits:
    def test_filesystem_confined_to_root_and_shell_runs_there(self, tmp_path):
        registry = Registry()
        install_toolkits(registry, tmp_path)
        session = Session(ScriptedLLM([]), registry=registry, policy=allow_all(), builtins=())
        assert "wrote 5" in session.invoke("filesystem.file.write", path="a/b.txt", content="hello")
        assert session.invoke("filesystem.file.read", path="a/b.txt") == "hello"
        assert session.invoke("filesystem.file.list") == ["a/b.txt"]
        with pytest.raises(RuntimeError):
            session.invoke("filesystem.file.read", path="../outside.txt")
        out = session.invoke("shell.command.run", command="echo hi")
        assert out.startswith("exit=0") and "hi" in out

    def test_shell_can_be_left_out(self, tmp_path):
        registry = Registry()
        install_toolkits(registry, tmp_path, shell=False)
        assert "shell.command.run" not in registry and "filesystem.file.read" in registry
