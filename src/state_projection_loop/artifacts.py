"""Artifact store (replaces the old string-handle ``ValueStore``).

Large tool results, stored projections, and model responses never pass
through the model's context a second time: they are stored here and
projected as a preview card. A reference is a *structured* JSON object,
never a bare string — ``"$h1"`` used to be silently rewritten into a lookup
whenever it appeared as an argument, which meant a user could never pass
that literal string through a tool, and a mis-detected reference could leak
one tool's output into another tool's arguments. The fix is representational:
only ``{"$artifact": "<id>"}`` is ever resolved; every other string,
including one that happens to look like an id, passes through untouched.

Artifacts are namespaced by run so a sub-agent (or a resumed run) can never
address another run's data by guessing an id; a parent must explicitly
``move()`` a child artifact into its own namespace to receive it (I9).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .ids import new_id
from .tokens import estimate_tokens

REF_KEY = "$artifact"


def serialize_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=None, default=str)
    except (TypeError, ValueError):
        return str(value)


def is_ref(value: Any) -> bool:
    return isinstance(value, dict) and set(value.keys()) == {REF_KEY} and isinstance(value[REF_KEY], str)


def ref(artifact_id: str) -> dict[str, str]:
    return {REF_KEY: artifact_id}


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


@dataclass
class ArtifactRecord:
    id: str
    run_id: str
    value: Any
    text: str
    type_name: str
    tokens: int
    source: str = ""
    created: float = field(default_factory=time.time)

    def size_desc(self) -> str:
        v = self.value
        if isinstance(v, str):
            return f"{len(v)} chars, {v.count(chr(10)) + 1} lines"
        if isinstance(v, (list, tuple)):
            return f"len={len(v)}"
        if isinstance(v, dict):
            return f"{len(v)} keys"
        return f"{len(self.text)} chars"


class ArtifactStore:
    """Namespaced by ``run_id``: artifacts from one run are invisible to
    another unless explicitly moved. Optionally persists to
    ``directory/<run_id>/<artifact_id>.json`` so a resumed run can recover
    large payloads that never made it into the ledger body."""

    def __init__(self, run_id: str, *, directory: Optional[Path] = None) -> None:
        self.run_id = run_id
        self.directory = directory
        self._records: dict[str, ArtifactRecord] = {}

    def put(self, value: Any, *, source: str = "") -> ArtifactRecord:
        aid = new_id("artifact")
        text = serialize_value(value)
        record = ArtifactRecord(
            id=aid,
            run_id=self.run_id,
            value=value,
            text=text,
            type_name=type(value).__name__,
            tokens=estimate_tokens(text),
            source=source,
        )
        self._records[aid] = record
        self._persist(record)
        return record

    def _persist(self, record: ArtifactRecord) -> None:
        if self.directory is None:
            return
        run_dir = self.directory / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": record.id, "run_id": record.run_id, "type_name": record.type_name,
            "source": record.source, "created": record.created, "text": record.text,
        }
        (run_dir / f"{record.id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8"
        )

    def get(self, aid: str) -> Any:
        return self._records[aid].value

    def get_record(self, aid: str) -> ArtifactRecord:
        return self._records[aid]

    def exists(self, aid: str) -> bool:
        return aid in self._records

    def move(self, record: ArtifactRecord, *, source: str = "") -> ArtifactRecord:
        """Explicitly import a record from another store's namespace into
        this one (spawn child -> parent handoff, I9)."""
        return self.put(record.value, source=source or record.source)

    def ref_text(self, record: ArtifactRecord, *, preview: str = "head", preview_tokens: int = 120) -> str:
        """Projection form of an artifact: id + type + size + preview (I7)."""
        if preview == "tail":
            body = record.text[-preview_tokens * 6:]
            body = truncate_to_tokens(body[::-1], preview_tokens)[::-1]
            snippet = "…" + body
        else:
            snippet = truncate_to_tokens(record.text, preview_tokens)
            if len(snippet) < len(record.text):
                snippet += "…"
        return (
            f"[{record.id} {record.type_name} {record.size_desc()} ~{record.tokens}tk"
            + (f" from {record.source}" if record.source else "")
            + f"] preview: {snippet}"
        )

    # -- peek (resident meta tool) --------------------------------------

    def peek(
        self,
        aid: str,
        query: Optional[str] = None,
        range_: Optional[str] = None,
        *,
        max_tokens: int = 600,
    ) -> str:
        if not isinstance(aid, str) or not self.exists(aid):
            known = ", ".join(sorted(self._records)) or "(none)"
            shown = repr(aid)[:80]
            return f"Error: unknown artifact {shown}. Known artifacts: {known}"
        record = self._records[aid]
        if range_:
            result = self._peek_range(record, range_)
        elif query:
            result = self._peek_query(record, query)
        else:
            result = record.text
        out = truncate_to_tokens(result, max_tokens)
        if len(out) < len(result):
            out += f"\n…[truncated; {estimate_tokens(result) - max_tokens}tk more — narrow with query/range]"
        return out

    @staticmethod
    def _peek_range(record: ArtifactRecord, range_: str) -> str:
        m = re.match(r"^\s*(\d+)\s*(?:-\s*(\d+))?\s*$", range_)
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else start
            lines = record.text.split("\n")
            sel = lines[max(0, start - 1): end]
            return "\n".join(f"{i}: {line}" for i, line in enumerate(sel, start=max(1, start)))
        value = record.value
        try:
            for part in re.findall(r"[^.\[\]]+|\[\d+\]", range_):
                if part.startswith("["):
                    value = value[int(part[1:-1])]
                else:
                    value = value[part] if isinstance(value, dict) else getattr(value, part)
            return serialize_value(value)
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            return f"Error: cannot resolve range/path {range_!r}: {exc}"

    @staticmethod
    def _peek_query(record: ArtifactRecord, query: str) -> str:
        lines = record.text.split("\n")
        q = query.lower()
        hits = [i for i, line in enumerate(lines) if q in line.lower()]
        if not hits:
            return f"No lines matching {query!r} in {record.id}."
        out: list[str] = []
        shown: set[int] = set()
        for i in hits[:40]:
            for j in range(max(0, i - 1), min(len(lines), i + 2)):
                if j not in shown:
                    shown.add(j)
                    out.append(f"{j + 1}: {lines[j]}")
        return "\n".join(out)

    # -- reference resolution in tool arguments --------------------------

    def resolve_args(self, args: Any) -> Any:
        """Deep-replace ``{"$artifact": "..."}`` objects with stored values.

        Deliberately does NOT special-case bare strings: ``"$h1"`` (or any
        string) always passes through as literal data. Only the structured
        reference form is ever resolved.
        """
        if is_ref(args):
            aid = args[REF_KEY]
            if self.exists(aid):
                return self.get(aid)
            return args  # unknown ref: leave as-is, let schema validation surface it
        if isinstance(args, list):
            return [self.resolve_args(a) for a in args]
        if isinstance(args, dict):
            return {k: self.resolve_args(v) for k, v in args.items()}
        return args
