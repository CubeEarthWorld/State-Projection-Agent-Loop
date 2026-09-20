"""Run state machine: the unit of resumable execution.

A :class:`Run` is one job/conversation execution. Its state is not implicit
in "is the Python object still alive" — it is an explicit, ledger-recorded
value that a new process can read back after a restart. Waiting is a
first-class state (``WAITING_FOR_APPROVAL``, ``WAITING_FOR_USER``) with its
own persisted record (:class:`ApprovalRequest`, :class:`PendingQuestion`),
not just a hook that blocks a batch and forgets why.

A :class:`Command` is one planned invocation of a capability. Its
``command_id`` is stable across retries of the *same* logical attempt (never
regenerated on retry) so an external API can use it as an idempotency key,
and its ``outcome`` distinguishes three states that a timeout collapses
together in naive implementations: the call never started (``failed``
before execution), it demonstrably failed, or its result is unknown because
the timeout fired mid-flight (``unknown`` — never safe to blindly retry).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from .capability import Effect
from .events import EventLedger
from .ids import new_id
from .messages import ToolCall

RUN_STATES = ("RUNNING", "WAITING_FOR_APPROVAL", "WAITING_FOR_USER", "COMPLETED", "FAILED", "CANCELLED")
TERMINAL_STATES = ("COMPLETED", "FAILED", "CANCELLED")

COMMAND_OUTCOMES = ("pending", "ok", "failed", "unknown")


class RunStateError(Exception):
    """Raised on an illegal state transition (e.g. mutating a terminal run)."""


@dataclass
class Command:
    id: str
    capability_name: str
    arguments: dict[str, Any]
    retry_safety: str
    outcome: str = "pending"
    attempts: int = 0
    result_ref: Optional[str] = None  # artifact id, when ok
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Command":
        return cls(**d)


@dataclass
class ApprovalRequest:
    id: str
    command_id: str
    effects: list[Effect]
    reason: str
    policy_revision: int
    expires_at: Optional[float] = None
    resolution: Optional[str] = None  # "approved" | "denied" | "expired" | None (pending)

    def to_dict(self) -> dict[str, Any]:
        return {**vars(self), "effects": [e.to_dict() for e in self.effects]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ApprovalRequest":
        return cls(**{**d, "effects": [Effect.from_dict(e) for e in d["effects"]]})

    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() >= self.expires_at


@dataclass
class Question:
    """What a handler returns to pause the run until the user answers (see
    the ``ask`` pack). The runtime turns it into a :class:`PendingQuestion`."""

    text: str
    choices: Optional[list[str]] = None


@dataclass
class PendingQuestion:
    """A question the run is waiting on, persisted in the snapshot so the
    pause survives a restart. ``Session.answer`` fills ``answer`` and resumes."""

    id: str
    command_id: str
    call_id: str
    text: str
    choices: Optional[list[str]] = None
    answer: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PendingQuestion":
        return cls(**d)


class Run:
    """The state machine for one execution. Every transition and approval
    event is written to the ledger *before* ``self.state`` is updated, so a
    crash between the write and the in-memory update is self-healing on
    replay (the ledger, not the object, is the source of truth)."""

    def __init__(self, run_id: str, session_id: str, ledger: EventLedger) -> None:
        self.id = run_id
        self.session_id = session_id
        self.ledger = ledger
        self.state = "RUNNING"
        self.commands: dict[str, Command] = {}
        self.pending_approval: Optional[ApprovalRequest] = None
        # Kept around after resolve_approval() clears pending_approval, so
        # resume can find the exact command_id that was approved instead of
        # minting a fresh one (an approved command must keep its
        # idempotency key across the pause). The runtime clears it once consumed.
        self.last_resolved_approval: Optional[ApprovalRequest] = None
        self.pending_question: Optional[PendingQuestion] = None
        self.pending_calls: list[ToolCall] = []
        self.result: Any = None

    # -- state transitions ------------------------------------------------

    def _assert_not_terminal(self) -> None:
        if self.state in TERMINAL_STATES:
            raise RunStateError(f"Run {self.id} is terminal ({self.state}); no further commands may execute")

    def transition(self, new_state: str, *, reason: str = "") -> None:
        if new_state not in RUN_STATES:
            raise ValueError(f"Unknown run state {new_state!r}")
        if self.state in TERMINAL_STATES and new_state != self.state:
            raise RunStateError(f"Run {self.id} is terminal ({self.state}); cannot transition to {new_state}")
        self.ledger.append(self.id, "run_state_changed",
                            {"from": self.state, "to": new_state, "reason": reason})
        self.state = new_state

    def complete(self, result: Any) -> None:
        self.result = result
        self.transition("COMPLETED", reason="finish")

    def fail(self, reason: str) -> None:
        self.transition("FAILED", reason=reason)

    def cancel(self, reason: str = "cancelled") -> None:
        self.transition("CANCELLED", reason=reason)

    # -- commands -----------------------------------------------------------

    def new_command(self, capability_name: str, arguments: dict[str, Any], retry_safety: str) -> Command:
        self._assert_not_terminal()
        cmd = Command(id=new_id("command"), capability_name=capability_name, arguments=arguments,
                      retry_safety=retry_safety)
        self.commands[cmd.id] = cmd
        self.ledger.append(self.id, "command_started",
                            {"command_id": cmd.id, "capability": capability_name, "arguments": arguments})
        return cmd

    def record_outcome(self, command: Command, outcome: str, *, error: Optional[str] = None,
                        result_ref: Optional[str] = None, duration_ms: Optional[int] = None) -> None:
        if outcome not in COMMAND_OUTCOMES:
            raise ValueError(f"Unknown command outcome {outcome!r}")
        command.outcome = outcome
        command.error = error
        command.result_ref = result_ref
        event_type = {"ok": "command_completed", "failed": "command_failed",
                      "unknown": "command_outcome_unknown"}[outcome]
        self.ledger.append(self.id, event_type, {"command_id": command.id, "error": error,
                                                 "result_ref": result_ref, "duration_ms": duration_ms})

    # -- approval -------------------------------------------------------------

    def request_approval(
        self, command: Command, effects: list[Effect], reason: str, *,
        policy_revision: int, expires_in_s: Optional[float] = None,
    ) -> ApprovalRequest:
        expires_at = time.time() + expires_in_s if expires_in_s is not None else None
        request = ApprovalRequest(
            id=new_id("approval"), command_id=command.id, effects=effects, reason=reason,
            policy_revision=policy_revision, expires_at=expires_at,
        )
        self.pending_approval = request
        self.ledger.append(self.id, "approval_requested", {
            "approval_id": request.id, "command_id": command.id, "reason": reason,
            "effects": [e.to_dict() for e in effects],
            "policy_revision": policy_revision, "expires_at": expires_at,
        })
        self.transition("WAITING_FOR_APPROVAL", reason=reason)
        return request

    def resolve_approval(self, decision: str, *, current_policy_revision: int) -> ApprovalRequest:
        """Resolve the pending approval. ``decision`` is 'approved' or
        'denied'. If the policy revision has moved since the request was
        made, the approval is stale and must be re-requested — approving
        blind to a changed policy would defeat the whole point of layered
        deny ("premise changed" rule)."""
        request = self.pending_approval
        if request is None:
            raise RunStateError(f"Run {self.id} has no pending approval to resolve")
        if request.is_expired():
            request.resolution = "expired"
            self.ledger.append(self.id, "approval_resolved",
                                {"approval_id": request.id, "resolution": "expired"})
            raise RunStateError(f"Approval {request.id} expired at {request.expires_at}")
        if current_policy_revision != request.policy_revision:
            raise RunStateError(
                f"Policy changed (revision {request.policy_revision} -> {current_policy_revision}) "
                f"since approval {request.id} was requested; re-evaluate before resolving"
            )
        if decision not in ("approved", "denied"):
            raise ValueError("decision must be 'approved' or 'denied'")
        request.resolution = decision
        self.ledger.append(self.id, "approval_resolved", {"approval_id": request.id, "resolution": decision})
        self.pending_approval = None
        self.last_resolved_approval = request
        # pending_calls is intentionally left intact on denial: the runtime's
        # resume path consumes it to tell the model *why* nothing ran, then
        # clears it — dropping it here would silently swallow that context.
        self.transition("RUNNING", reason=f"approval {decision}")
        return request

    # -- questions ------------------------------------------------------------

    def ask_question(self, command: Command, call_id: str, question: Question) -> PendingQuestion:
        """Park the run on a question the model asked the user (the ``ask``
        pack). One at a time: concurrently executed read-only calls can both
        ask, and overwriting the first would lose it without a trace."""
        if self.pending_question is not None:
            raise RunStateError(
                f"Run {self.id} is already waiting on question {self.pending_question.id}; "
                "answer it before asking another"
            )
        pending = PendingQuestion(id=new_id("question"), command_id=command.id, call_id=call_id,
                                  text=question.text, choices=question.choices)
        self.pending_question = pending
        self.ledger.append(self.id, "question_asked", {
            "question_id": pending.id, "command_id": command.id, "call_id": call_id,
            "text": question.text, "choices": question.choices,
        })
        self.transition("WAITING_FOR_USER", reason="question")
        return pending

    def answer(self, text: str) -> PendingQuestion:
        """Answer the pending question; the asking command completes with the
        answer as its result and the run is ``RUNNING`` again."""
        pending = self.pending_question
        if pending is None:
            raise RunStateError(f"Run {self.id} has no pending question to answer")
        pending.answer = text
        self.ledger.append(self.id, "question_answered", {"question_id": pending.id, "answer": text})
        self.record_outcome(self.commands[pending.command_id], "ok")
        self.pending_question = None
        self.transition("RUNNING", reason="answered")
        return pending

    # -- persistence snapshot ------------------------------------------------

    def to_snapshot_state(self) -> dict[str, Any]:
        def optional(record: Any) -> Optional[dict[str, Any]]:
            return None if record is None else record.to_dict()

        return {
            "session_id": self.session_id,
            "state": self.state,
            "result": self.result,
            "commands": {cid: c.to_dict() for cid, c in self.commands.items()},
            "pending_approval": optional(self.pending_approval),
            "pending_question": optional(self.pending_question),
            "pending_calls": [c.to_dict() for c in self.pending_calls],
            "last_resolved_approval": optional(self.last_resolved_approval),
        }

    @classmethod
    def from_snapshot_state(cls, run_id: str, ledger: EventLedger, state: dict[str, Any]) -> "Run":
        run = cls(run_id, state["session_id"], ledger)
        run.state = state["state"]
        run.result = state.get("result")
        run.commands = {cid: Command.from_dict(c) for cid, c in (state.get("commands") or {}).items()}
        if state.get("pending_approval"):
            run.pending_approval = ApprovalRequest.from_dict(state["pending_approval"])
        if state.get("pending_question"):
            run.pending_question = PendingQuestion.from_dict(state["pending_question"])
        run.pending_calls = [ToolCall.from_dict(c) for c in (state.get("pending_calls") or [])]
        if state.get("last_resolved_approval"):
            run.last_resolved_approval = ApprovalRequest.from_dict(state["last_resolved_approval"])
        return run
