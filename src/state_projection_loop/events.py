"""Event Ledger: the single append-only source of truth for a run.

Every fact worth remembering — what the user said, what was sent to the
model, what it decided, what policy allowed, what a command did, what got
approved — is appended here as an :class:`Event`. Nothing else is
authoritative: conversation views, working state, and run status are all
*derived* by replaying (or partially replaying, via a :class:`Snapshot`)
this log. That is what makes a run resumable after a process restart and
makes "what actually happened" answerable after the fact.

Sensitive payloads are never embedded directly in an event: callers pass an
artifact reference (see :mod:`state_projection_loop.artifacts`) and only
that opaque id is written to the ledger, so a ledger file can be shipped or
deleted independently of the artifact store it references.
"""
from __future__ import annotations

import json
import threading
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Protocol, runtime_checkable

from .ids import new_id
from .messages import ASSISTANT, OBSERVATION, SYSTEM, USER, Message, ToolCall
from .serialization import dumps

EVENT_TYPES = (
    "user_input",
    "projection_compiled",
    "model_response",
    "decision_validated",
    "command_started",
    "command_completed",
    "command_failed",
    "command_outcome_unknown",
    "approval_requested",
    "approval_resolved",
    "run_state_changed",
    "branch_created",
    "notice",
    "observation",
    "checkpoint",
    "rewound",
    "checklists_changed",
    "question_asked",
    "question_answered",
    "state_folded",
    "model_call_failed",
    "hook_intervened",
)


@dataclass
class Event:
    id: str
    run_id: str
    sequence: int
    type: str
    ts: float
    data: dict[str, Any] = field(default_factory=dict)

    def to_line(self) -> str:
        return dumps({
            "id": self.id, "run_id": self.run_id, "sequence": self.sequence,
            "type": self.type, "ts": self.ts, "data": self.data,
        })

    @classmethod
    def from_line(cls, line: str) -> "Event":
        d = json.loads(line)
        return cls(id=d["id"], run_id=d["run_id"], sequence=d["sequence"],
                    type=d["type"], ts=d["ts"], data=d.get("data") or {})


@dataclass
class Snapshot:
    run_id: str
    sequence: int  # last event sequence folded into this snapshot
    ts: float
    state: dict[str, Any]


@dataclass
class RunSummary:
    """What a ledger knows about a run without reading its events: enough
    to list, pick and resume one."""

    run_id: str
    session_id: str
    state: str
    ts: float  # the last snapshot's time


@runtime_checkable
class EventLedger(Protocol):
    def append(self, run_id: str, type: str, data: dict[str, Any]) -> Event: ...

    def iter_run(self, run_id: str, *, after: int = 0) -> Iterator[Event]: ...

    def last_sequence(self, run_id: str) -> int: ...

    def save_snapshot(self, snapshot: Snapshot) -> None: ...

    def load_snapshot(self, run_id: str) -> Optional[Snapshot]: ...

    def list_runs(self) -> list[RunSummary]:
        """Every run with a snapshot, newest first."""
        ...


def _new_event(run_id: str, sequence: int, type: str, data: dict[str, Any]) -> Event:
    if type not in EVENT_TYPES:
        raise ValueError(f"Unknown event type {type!r}; expected one of {EVENT_TYPES}")
    return Event(id=new_id("event"), run_id=run_id, sequence=sequence, type=type, ts=time.time(), data=data)


class InMemoryLedger:
    """Process-local ledger: fast, exercised by every unit test, but does
    not survive a process restart. Use :class:`JsonlLedger` for that."""

    def __init__(self) -> None:
        self._events: dict[str, list[Event]] = {}
        self._snapshots: dict[str, Snapshot] = {}
        self._lock = threading.Lock()

    def append(self, run_id: str, type: str, data: dict[str, Any]) -> Event:
        with self._lock:
            event = _new_event(run_id, len(self._events.get(run_id, [])) + 1, type, data)
            self._events.setdefault(run_id, []).append(event)
            return event

    def iter_run(self, run_id: str, *, after: int = 0) -> Iterator[Event]:
        for event in self._events.get(run_id, []):
            if event.sequence > after:
                yield event

    def last_sequence(self, run_id: str) -> int:
        events = self._events.get(run_id)
        return events[-1].sequence if events else 0

    def save_snapshot(self, snapshot: Snapshot) -> None:
        self._snapshots[snapshot.run_id] = snapshot

    def load_snapshot(self, run_id: str) -> Optional[Snapshot]:
        return self._snapshots.get(run_id)

    def list_runs(self) -> list[RunSummary]:
        return _summaries(self._snapshots.values())


class JsonlLedger:
    """File-backed ledger: one append-only ``<run_id>.jsonl`` per run plus a
    ``<run_id>.snapshot.json`` sidecar. Surviving a process restart is the
    entire point — :meth:`state_projection_loop.session.Session.resume`
    reads this back to restore a ``WAITING_FOR_APPROVAL`` run."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_seq: dict[str, int] = {}

    def _path(self, run_id: str) -> Path:
        return self.directory / f"{run_id}.jsonl"

    def _snapshot_path(self, run_id: str) -> Path:
        return self.directory / f"{run_id}.snapshot.json"

    def _seq(self, run_id: str) -> int:
        if run_id in self._last_seq:
            return self._last_seq[run_id]
        path = self._path(run_id)
        n = 0
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
        self._last_seq[run_id] = n
        return n

    def append(self, run_id: str, type: str, data: dict[str, Any]) -> Event:
        with self._lock:
            event = _new_event(run_id, self._seq(run_id) + 1, type, data)
            with self._path(run_id).open("a", encoding="utf-8") as f:
                f.write(event.to_line() + "\n")
            self._last_seq[run_id] = event.sequence
            return event

    def iter_run(self, run_id: str, *, after: int = 0) -> Iterator[Event]:
        path = self._path(run_id)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = Event.from_line(line)
                except (ValueError, KeyError, TypeError):
                    # Appends are unbuffered, so a crash mid-append leaves a
                    # torn last line. One bad line must not make the run
                    # unresumable; every good line before it still replays.
                    continue
                if event.sequence > after:
                    yield event

    def last_sequence(self, run_id: str) -> int:
        return self._seq(run_id)

    def save_snapshot(self, snapshot: Snapshot) -> None:
        payload = {
            "run_id": snapshot.run_id, "sequence": snapshot.sequence,
            "ts": snapshot.ts, "state": snapshot.state,
        }
        tmp = self._snapshot_path(snapshot.run_id).with_suffix(".json.tmp")
        tmp.write_text(dumps(payload), encoding="utf-8")
        tmp.replace(self._snapshot_path(snapshot.run_id))

    def load_snapshot(self, run_id: str) -> Optional[Snapshot]:
        path = self._snapshot_path(run_id)
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        return Snapshot(run_id=d["run_id"], sequence=d["sequence"], ts=d["ts"], state=d["state"])

    def list_runs(self) -> list[RunSummary]:
        snapshots = (self.load_snapshot(p.name[:-len(".snapshot.json")])
                     for p in self.directory.glob("*.snapshot.json"))
        return _summaries(snap for snap in snapshots if snap is not None)


class ObservedLedger:
    """A ledger that also hands every appended :class:`Event` to an observer.

    Observers are read-only by contract: they see what happened, they cannot
    veto or rewrite it (that is the policy engine's job). An observer that
    raises is ignored so it can never take the loop down with it.
    """

    def __init__(self, inner: EventLedger, on_event: Callable[[Event], None]) -> None:
        self.inner = inner
        self.on_event = on_event

    def append(self, run_id: str, type: str, data: dict[str, Any]) -> Event:
        event = self.inner.append(run_id, type, data)
        try:
            self.on_event(event)
        except Exception:  # noqa: BLE001 — observers never break the loop
            pass
        return event

    def __getattr__(self, name: str) -> Any:
        # Everything but append is pass-through, so the protocol can grow
        # without another hand-written forwarder here.
        return getattr(self.inner, name)


def _summaries(snapshots: Iterable[Snapshot]) -> list[RunSummary]:
    runs = [RunSummary(run_id=s.run_id, session_id=s.state.get("session_id", ""),
                       state=s.state.get("state", ""), ts=s.ts) for s in snapshots]
    return sorted(runs, key=lambda r: r.ts, reverse=True)


_MESSAGE_BUILDERS: dict[str, Callable[[dict[str, Any]], Message]] = {
    "user_input": lambda d: Message(role=USER, content=d.get("text", "")),
    "model_response": lambda d: Message(
        role=ASSISTANT, content=d.get("text", ""),
        tool_calls=[ToolCall.from_dict(c) for c in (d.get("calls") or [])]),
    "observation": lambda d: Message(role=OBSERVATION, content=d.get("text", ""),
                                     tool_call_id=d.get("call_id"), name=d.get("name")),
    "notice": lambda d: Message(role=SYSTEM, content=d.get("text", "")),
}

# Derived, so a new renderable type is one entry above and not a second list.
RENDERABLE_TYPES = tuple(_MESSAGE_BUILDERS)


def event_to_message(event: Event) -> Optional[Message]:
    """The message a renderable event projects to; None for any other type."""
    build = _MESSAGE_BUILDERS.get(event.type)
    return build(event.data) if build is not None else None


_RENDERED: Any = weakref.WeakKeyDictionary()  # ledger -> (run_id, last sequence, rendering)


def renderable(ledger: EventLedger, run_id: str) -> list[tuple[Event, Message]]:
    """The run's conversation, oldest first: each renderable event with the
    message it projects to. The one scan every reader of the history shares —
    memoised until the next append. The messages are shared with every other
    reader: render from them, never mutate them."""
    key = (run_id, ledger.last_sequence(run_id))
    hit = _RENDERED.get(ledger)
    if hit is None or hit[:2] != key:
        hit = (*key, [(e, m) for e in ledger.iter_run(run_id) if (m := event_to_message(e)) is not None])
        _RENDERED[ledger] = hit
    return hit[2]
