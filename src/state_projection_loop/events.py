"""Event Ledger: the single append-only source of truth for a run.

Every fact worth remembering — what the user said, what was sent to the
model, what it decided, what policy allowed, what a command did, what got
approved — is appended here as an :class:`Event`. Nothing else is
authoritative: conversation views, working state, and run status are all
*derived* by replaying (or partially replaying, via a :class:`Snapshot`)
this log. That is what makes a run resumable after a process restart and
makes "what actually happened" answerable after the fact (P1-3).

Sensitive payloads are never embedded directly in an event: callers pass an
artifact reference (see :mod:`state_projection_loop.artifacts`) and only
that opaque id is written to the ledger, so a ledger file can be shipped or
deleted independently of the artifact store it references.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol, runtime_checkable

from .ids import new_id

EVENT_TYPES = (
    "user_input",
    "projection_compiled",
    "model_response",
    "decision_validated",
    "policy_decision",
    "command_started",
    "command_completed",
    "command_failed",
    "command_outcome_unknown",
    "artifact_stored",
    "approval_requested",
    "approval_resolved",
    "run_state_changed",
    "state_folded",
    "policy_changed",
    "branch_created",
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
        return json.dumps(
            {
                "id": self.id, "run_id": self.run_id, "sequence": self.sequence,
                "type": self.type, "ts": self.ts, "data": self.data,
            },
            ensure_ascii=False, default=str,
        )

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


@runtime_checkable
class EventLedger(Protocol):
    def append(self, run_id: str, type: str, data: dict[str, Any]) -> Event: ...

    def iter_run(self, run_id: str, *, after: int = 0) -> Iterator[Event]: ...

    def last_sequence(self, run_id: str) -> int: ...

    def save_snapshot(self, snapshot: Snapshot) -> None: ...

    def load_snapshot(self, run_id: str) -> Optional[Snapshot]: ...


class InMemoryLedger:
    """Process-local ledger: fast, exercised by every unit test, but does
    not survive a process restart. Use :class:`JsonlLedger` for that."""

    def __init__(self) -> None:
        self._events: dict[str, list[Event]] = {}
        self._snapshots: dict[str, Snapshot] = {}
        self._lock = threading.Lock()

    def append(self, run_id: str, type: str, data: dict[str, Any]) -> Event:
        if type not in EVENT_TYPES:
            raise ValueError(f"Unknown event type {type!r}; expected one of {EVENT_TYPES}")
        with self._lock:
            seq = len(self._events.get(run_id, [])) + 1
            import time as _time
            event = Event(id=new_id("event"), run_id=run_id, sequence=seq, type=type,
                           ts=_time.time(), data=data)
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
        n = 0
        path = self._path(run_id)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        n += 1
        self._last_seq[run_id] = n
        return n

    def append(self, run_id: str, type: str, data: dict[str, Any]) -> Event:
        if type not in EVENT_TYPES:
            raise ValueError(f"Unknown event type {type!r}; expected one of {EVENT_TYPES}")
        with self._lock:
            import time as _time
            seq = self._seq(run_id) + 1
            event = Event(id=new_id("event"), run_id=run_id, sequence=seq, type=type,
                           ts=_time.time(), data=data)
            with self._path(run_id).open("a", encoding="utf-8") as f:
                f.write(event.to_line() + "\n")
            self._last_seq[run_id] = seq
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
                event = Event.from_line(line)
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
        tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(self._snapshot_path(snapshot.run_id))

    def load_snapshot(self, run_id: str) -> Optional[Snapshot]:
        path = self._snapshot_path(run_id)
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        return Snapshot(run_id=d["run_id"], sequence=d["sequence"], ts=d["ts"], state=d["state"])
