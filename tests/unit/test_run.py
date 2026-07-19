"""Run state machine: transitions, terminal-state guard, approval
lifecycle including the "premise changed" staleness check, snapshot
round-trip."""
from __future__ import annotations

import pytest

from state_projection_loop.capability import Effect
from state_projection_loop.events import InMemoryLedger
from state_projection_loop.run import Run, RunStateError


def make_run() -> Run:
    ledger = InMemoryLedger()
    return Run("run_1", "ses_1", ledger)


class TestTransitions:
    def test_default_state_is_running(self):
        run = make_run()
        assert run.state == "RUNNING"

    def test_complete_is_terminal(self):
        run = make_run()
        run.complete("done")
        assert run.state == "COMPLETED"
        with pytest.raises(RunStateError):
            run.new_command("demo.thing", {}, "pure")

    def test_cannot_leave_terminal_state(self):
        run = make_run()
        run.fail("boom")
        with pytest.raises(RunStateError):
            run.transition("RUNNING")


class TestCommands:
    def test_new_command_records_started_event(self):
        run = make_run()
        cmd = run.new_command("demo.thing", {"x": 1}, "pure")
        events = list(run.ledger.iter_run(run.id))
        assert events[-1].type == "command_started"
        assert events[-1].data["command_id"] == cmd.id

    def test_record_outcome_ok(self):
        run = make_run()
        cmd = run.new_command("demo.thing", {}, "pure")
        run.record_outcome(cmd, "ok", result_ref="art_1")
        assert cmd.outcome == "ok"
        assert cmd.result_ref == "art_1"

    def test_record_outcome_unknown_never_auto_marked_failed(self):
        run = make_run()
        cmd = run.new_command("demo.thing", {}, "never_retry")
        run.record_outcome(cmd, "unknown", error="timed out")
        assert cmd.outcome == "unknown"  # distinct from "failed"


class TestApproval:
    def test_request_approval_transitions_and_records(self):
        run = make_run()
        cmd = run.new_command("fs.file.write", {"path": "a"}, "never_retry")
        req = run.request_approval(cmd, [Effect(kind="write", resource="workspace:*")],
                                    "needs review", policy_revision=1)
        assert run.state == "WAITING_FOR_APPROVAL"
        assert run.pending_approval is req

    def test_resolve_approval_returns_to_running(self):
        run = make_run()
        cmd = run.new_command("fs.file.write", {}, "never_retry")
        run.request_approval(cmd, [], "x", policy_revision=1)
        req = run.resolve_approval("approved", current_policy_revision=1)
        assert req.resolution == "approved"
        assert run.state == "RUNNING"
        assert run.pending_approval is None
        assert run.last_resolved_approval is req

    def test_resolve_with_stale_policy_revision_raises(self):
        run = make_run()
        cmd = run.new_command("fs.file.write", {}, "never_retry")
        run.request_approval(cmd, [], "x", policy_revision=1)
        with pytest.raises(RunStateError, match="Policy changed"):
            run.resolve_approval("approved", current_policy_revision=2)

    def test_resolve_without_pending_raises(self):
        run = make_run()
        with pytest.raises(RunStateError, match="no pending approval"):
            run.resolve_approval("approved", current_policy_revision=0)

    def test_expired_approval_raises_and_marks_expired(self):
        run = make_run()
        cmd = run.new_command("fs.file.write", {}, "never_retry")
        req = run.request_approval(cmd, [], "x", policy_revision=1, expires_in_s=-1)
        with pytest.raises(RunStateError, match="expired"):
            run.resolve_approval("approved", current_policy_revision=1)
        assert req.resolution == "expired"


class TestSnapshotRoundTrip:
    def test_round_trip_preserves_pending_state(self):
        from state_projection_loop.messages import ToolCall

        run = make_run()
        cmd = run.new_command("fs.file.write", {"path": "a"}, "never_retry")
        run.pending_calls = [ToolCall(name="fs.file.write", arguments={"path": "a"}, id=cmd.id)]
        run.request_approval(cmd, [Effect(kind="write", resource="workspace:*")], "x", policy_revision=2)

        state = run.to_snapshot_state()
        restored = Run.from_snapshot_state(run.id, run.ledger, state)

        assert restored.state == "WAITING_FOR_APPROVAL"
        assert restored.pending_approval.command_id == cmd.id
        assert restored.pending_approval.policy_revision == 2
        assert [c.name for c in restored.pending_calls] == ["fs.file.write"]
        assert restored.commands[cmd.id].capability_name == "fs.file.write"
