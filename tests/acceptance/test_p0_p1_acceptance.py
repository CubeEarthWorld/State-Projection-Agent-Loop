"""Acceptance tests for the Codex-review-driven redesign (P0-1..P0-6,
P1-1..P1-3). Each test class corresponds to one item of the redesign's
"minimum acceptance test" checklist:

01. execution order      -- write -> read executes in that order
02. completion state      -- finish() combined with side-effecting calls is rejected
03. idempotency            -- a non-idempotent timeout is OUTCOME_UNKNOWN, never auto-retried
04. concurrency            -- concurrent input to one session never interleaves state
05. artifact references    -- a literal string is never misread as a reference
06. budget                 -- total send size including tool schemas stays inside the window
07. approval                -- WAITING_FOR_APPROVAL survives a simulated process restart
08. policy                 -- a higher layer's deny cannot be relaxed by a lower one
09. reproducibility        -- Run state is fully recoverable from Events + Snapshot
10. rewind                 -- branching never deletes or mutates past events
11. external effects        -- rewinding surfaces effects it cannot undo
12. logging                 -- each command's proposal -> validation -> authorization ->
                              start -> completion/unknown is traceable in the ledger
"""
from __future__ import annotations

import asyncio

import pytest

from state_projection_loop import Config, Registry, ScriptedLLM, Session
from state_projection_loop.artifacts import ref
from state_projection_loop.policy import PolicyEngine, Rule
from state_projection_loop.session import ConcurrencyError

from _util import capability_dict


def job_session(registry=None, policy=None, **cfg_overrides):
    cfg = Config.from_dict({"mode": "job", **cfg_overrides})
    return cfg, registry or Registry(), policy


class Test01_ExecutionOrder:
    def test_write_then_read_executes_in_stated_order(self):
        log: list[str] = []
        reg = Registry()
        reg.register(capability_dict("fs.write", properties={"v": {"type": "string"}}, required=["v"],
                                      effects=[("write", "workspace:*")]),
                     handler=lambda v: log.append(f"write:{v}") or "ok")
        reg.register(capability_dict("fs.read", effects=[("read", "workspace:*")]),
                     handler=lambda: log.append("read") or "ok")

        llm = ScriptedLLM([
            ScriptedLLM.calls(("fs.write", {"v": "x"}), ("fs.read", {})),
            ScriptedLLM.finish(result="done"),
        ])
        session = Session(llm, registry=reg, config=Config.from_dict({"mode": "job"}),
                          policy=PolicyEngine(default_decision="allow"))
        session.run_job("write then read")
        assert log == ["write:x", "read"], "write must complete before read starts"


class Test02_CompletionState:
    def test_finish_with_side_effecting_calls_is_rejected_and_nothing_runs(self):
        from state_projection_loop.messages import Decision, ToolCall

        executed = []
        reg = Registry()
        reg.register(capability_dict("fs.delete", effects=[("write", "workspace:*")]),
                     handler=lambda: executed.append(True) or "deleted")

        mixed = Decision(text="", calls=[ToolCall(name="fs.delete", arguments={})],
                          finish=True, result="claiming done")
        llm = ScriptedLLM([mixed, ScriptedLLM.finish(result="actually done")])
        session = Session(llm, registry=reg, config=Config.from_dict({"mode": "job"}),
                          policy=PolicyEngine(default_decision="allow"))
        result = session.run_job("try to sneak a delete in with finish")

        assert result == "actually done"
        assert executed == [], "the side-effecting call must never run alongside finish()"
        assert session.run.state == "COMPLETED"


class Test03_Idempotency:
    def test_timeout_on_never_retry_capability_is_unknown_and_not_retried(self):
        attempts = {"n": 0}

        async def charge_card() -> str:
            attempts["n"] += 1
            await asyncio.sleep(1.0)
            return "charged"

        reg = Registry()
        reg.register(capability_dict("billing.charge", timeout_s=0.05, retry_safety="never_retry",
                                      effects=[("external", "payment_gateway:*")]),
                     handler=charge_card)
        llm = ScriptedLLM([ScriptedLLM.call("billing.charge"), ScriptedLLM.finish(result="x")])
        session = Session(llm, registry=reg, config=Config.from_dict({"mode": "job"}),
                          policy=PolicyEngine(default_decision="allow"))
        session.run_job("charge the card")

        assert attempts["n"] == 1, "a never_retry capability must not be retried after a timeout"
        obs = next(m for m in session.conversation if m.role == "tool" and "UNKNOWN" in str(m.content))
        assert "UNKNOWN" in obs.content


class Test04_Concurrency:
    async def test_concurrent_input_never_interleaves(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow() -> str:
            started.set()
            await release.wait()
            return "done"

        reg = Registry()
        reg.register(capability_dict("demo.slow"), handler=slow)
        llm = ScriptedLLM([ScriptedLLM.call("demo.slow"), "finished"])
        session = Session(llm, registry=reg, policy=PolicyEngine(default_decision="allow"))

        task = asyncio.create_task(session.asend("go"))
        await started.wait()
        with pytest.raises(ConcurrencyError):
            await session.asend("interleave me")
        release.set()
        assert await task == "finished"
        # exactly the one turn's worth of messages made it into the conversation
        assert len([m for m in session.conversation if m.role == "user"]) == 1


class Test05_ArtifactReferences:
    def test_literal_string_matching_an_artifact_id_is_never_resolved(self):
        reg = Registry()
        record_holder = {}

        def make_big() -> str:
            return "x" * 5000  # forced into the artifact store by output_policy

        def echo_arg(data) -> str:
            return f"got:{data!r}"

        reg.register(capability_dict("demo.make_big", max_inline_tokens=10), handler=make_big)
        reg.register(capability_dict("demo.echo_arg", properties={"data": {}}, required=["data"]),
                     handler=echo_arg)

        llm = ScriptedLLM([ScriptedLLM.call("demo.make_big"), "made it"])
        session = Session(llm, registry=reg, policy=PolicyEngine(default_decision="allow"))
        session.send("make something big")
        artifact_id = next(
            m.content.split("[", 1)[1].split(" ", 1)[0]
            for m in session.conversation if m.role == "tool" and m.content.startswith("[art_")
        )

        # A tool called with the LITERAL id string (not the structured ref)
        # must receive it as plain text, never the resolved payload.
        resolved = session.store.resolve_args({"data": artifact_id})
        assert resolved == {"data": artifact_id}
        # Only the structured form resolves.
        resolved_ref = session.store.resolve_args({"data": ref(artifact_id)})
        assert resolved_ref == {"data": "x" * 5000}


class Test06_Budget:
    def test_total_send_size_including_schemas_stays_inside_window(self):
        from state_projection_loop.tokens import estimate_tokens

        reg = Registry()
        for i in range(30):
            reg.register(capability_dict(f"demo.tool_{i}", properties={
                "a": {"type": "string", "description": "x" * 60},
                "b": {"type": "string", "description": "y" * 60},
            }))

        captured = {}

        def snapshot(messages, tools):
            captured["messages"] = messages
            captured["tools"] = tools
            return "ok"

        cfg = Config.from_dict({"projection": {"window_tokens": 2000, "reserved_output_tokens": 200}})
        session = Session(ScriptedLLM([snapshot]), registry=reg, config=cfg)
        session.send("do something with tool_5 and tool_12")

        message_tokens = estimate_tokens(captured["messages"])
        schema_tokens = session.projection.schema_tokens(captured["tools"])
        assert message_tokens + schema_tokens + 200 <= 2000


class Test07_ApprovalSurvivesRestart:
    def test_waiting_for_approval_resumes_after_simulated_restart(self, tmp_path):
        written = {}

        def write_file(path: str, content: str) -> str:
            written[path] = content
            return f"wrote {len(content)} bytes"

        def make_registry():
            reg = Registry()
            reg.register(capability_dict("fs.write", properties={
                "path": {"type": "string"}, "content": {"type": "string"},
            }, required=["path", "content"], effects=[("write", "workspace:*")],
                retry_safety="never_retry"), handler=write_file)
            return reg

        # A real deployment reconstructs its PolicyEngine from its own
        # config on every boot; here that means a fresh engine with the
        # same (empty) rule set, so its revision number lines up with what
        # was recorded at approval-request time.
        def make_policy():
            return PolicyEngine(default_decision="require_approval")

        cfg = Config.from_dict({"mode": "job", "persistence": {"ledger_directory": str(tmp_path)}})
        llm1 = ScriptedLLM([ScriptedLLM.call("fs.write", path="a.txt", content="hello")])
        session1 = Session(llm1, registry=make_registry(), config=cfg, policy=make_policy())
        result1 = session1.run_job("write a.txt")
        assert session1.run.state == "WAITING_FOR_APPROVAL"
        run_id = session1.run.id
        assert written == {}  # never executed before approval

        # --- simulate a process restart: build a brand new Session purely
        # from what's on disk, with no reference to session1/run1 ---
        llm2 = ScriptedLLM([ScriptedLLM.finish(result="all done")])
        restored = Session.resume_from_ledger(llm2, run_id, config=cfg, registry=make_registry(),
                                              policy=make_policy())
        assert restored.run.state == "WAITING_FOR_APPROVAL"
        assert [c.name for c in restored.run.pending_calls] == ["fs.write"]

        restored.resolve_approval("approved")
        result2 = restored.resume()

        assert result2 == "all done"
        assert written == {"a.txt": "hello"}
        assert restored.run.state == "COMPLETED"


class Test08_PolicyLayering:
    def test_higher_layer_deny_cannot_be_relaxed_by_lower_layer(self):
        reg = Registry()
        executed = []
        reg.register(capability_dict("fs.write", effects=[("write", "workspace:*")]),
                     handler=lambda: executed.append(True) or "ok")

        policy = PolicyEngine(default_decision="allow")
        policy.add_rule("admin", Rule(decision="deny", capability_pattern="fs.*"))
        # A lower layer (session/workspace) tries to allow it anyway.
        policy.add_rule("session", Rule(decision="allow", capability_pattern="fs.*"))

        llm = ScriptedLLM([ScriptedLLM.call("fs.write"), "could not write"])
        session = Session(llm, registry=reg, policy=policy)
        reply = session.send("please write")
        assert reply == "could not write"
        assert executed == []


class Test09_Reproducibility:
    def test_run_state_recoverable_from_events_and_snapshot(self, tmp_path):
        reg = Registry()
        reg.register(capability_dict("demo.echo", properties={"text": {"type": "string"}},
                                      required=["text"]), handler=lambda text: f"echo:{text}")
        cfg = Config.from_dict({"mode": "job", "persistence": {"ledger_directory": str(tmp_path)}})
        llm = ScriptedLLM([ScriptedLLM.call("demo.echo", text="hi"), ScriptedLLM.finish(result="done")])
        session = Session(llm, registry=reg, config=cfg, policy=PolicyEngine(default_decision="allow"))
        session.run_job("echo hi then finish")
        run_id = session.run.id

        events = list(session.ledger.iter_run(run_id))
        event_types = {e.type for e in events}
        assert {"user_input", "projection_compiled", "model_response", "decision_validated",
                "command_started", "command_completed", "run_state_changed"} <= event_types

        snapshot = session.ledger.load_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.state["state"] == "COMPLETED"

        restored_llm = ScriptedLLM([], strict=False)
        restored = Session.resume_from_ledger(restored_llm, run_id, config=cfg, registry=reg)
        assert restored.run.state == "COMPLETED"
        assert restored.run.result == "done"
        assert [m.role for m in restored.conversation] == [m.role for m in session.conversation]


class Test10_Rewind:
    def test_branch_never_mutates_or_deletes_parent_events(self):
        reg = Registry()
        session = Session(ScriptedLLM(["one", "two", "three"]), registry=reg)
        session.send("a")
        session.send("b")
        session.send("c")
        parent_events_before = list(session.ledger.iter_run(session.run.id))

        branch, _irreversible = session.branch(at_message=2)

        parent_events_after = list(session.ledger.iter_run(session.run.id))
        assert [e.id for e in parent_events_before] == [e.id for e in parent_events_after]
        assert branch.run.id != session.run.id
        assert len(branch.conversation) == 2
        assert len(session.conversation) == 6  # parent untouched


class Test11_ExternalEffectsSurfaced:
    def test_rewind_reports_effects_it_cannot_undo(self):
        reg = Registry()
        reg.register(capability_dict("mail.send", effects=[("external", "smtp:*")]),
                     handler=lambda: "sent")
        llm = ScriptedLLM([ScriptedLLM.call("mail.send"), "sent the email"])
        session = Session(llm, registry=reg, policy=PolicyEngine(default_decision="allow"))
        session.send("send the email")

        _branch, irreversible = session.branch()
        assert any("mail.send" in note for note in irreversible)


class Test12_CommandTraceability:
    def test_each_command_traceable_start_to_finish(self):
        reg = Registry()
        reg.register(capability_dict("demo.echo", properties={"text": {"type": "string"}},
                                      required=["text"]), handler=lambda text: f"echo:{text}")
        session = Session(ScriptedLLM([ScriptedLLM.call("demo.echo", text="hi"), "done"]),
                          registry=reg, policy=PolicyEngine(default_decision="allow"))
        session.send("echo hi")

        events = list(session.ledger.iter_run(session.run.id))
        started = next(e for e in events if e.type == "command_started")
        completed = next(e for e in events if e.type == "command_completed")
        assert started.data["command_id"] == completed.data["command_id"]
        assert started.data["capability"] == "demo.echo@1"  # qualified (versioned) name
        # the full pipeline is visible in order: decision -> command -> outcome
        types_in_order = [e.type for e in events]
        assert types_in_order.index("decision_validated") < types_in_order.index("command_started")
        assert types_in_order.index("command_started") < types_in_order.index("command_completed")
