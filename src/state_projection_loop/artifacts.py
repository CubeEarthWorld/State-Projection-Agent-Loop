"""Artifact store.

Large tool results, stored projections, and model responses never pass
through the model's context a second time: they are stored here and
projected as a preview card. A reference is a *structured* JSON object,
never a bare string: if a string could be a reference, a user could never
pass that literal string through a tool, and a mis-detected reference could
leak one tool's output into another tool's arguments. Only
``{"$artifact": "<id>"}`` is ever resolved; every other string, including
one that happens to look like an id, passes through untouched.

Artifacts are namespaced by run so a sub-agent (or a resumed run) can never
address another run's data by guessing an id.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .ids import new_id
from .tokens import estimate_tokens, truncate_to_tokens
from .serialization import dumps

REF_KEY = "$artifact"


def serialize_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return dumps(value)
    except (TypeError, ValueError):
        return str(value)


def is_ref(value: Any) -> bool:
    return isinstance(value, dict) and set(value.keys()) == {REF_KEY} and isinstance(value[REF_KEY], str)


def ref(artifact_id: str) -> dict[str, str]:
    return {REF_KEY: artifact_id}


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

    def to_payload(self) -> dict[str, Any]:
        return {"id": self.id, "run_id": self.run_id, "type_name": self.type_name,
                "source": self.source, "created": self.created, "text": self.text}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ArtifactRecord":
        """Only the serialized text is persisted; anything that was not a
        string is parsed back, so a handler resolving the reference gets the
        dict it stored rather than its JSON."""
        text, type_name = payload["text"], payload.get("type_name", "str")
        value: Any = text
        if type_name != "str":
            try:
                value = json.loads(text)
            except ValueError:
                pass  # serialized with str(): the text is all there is
        return cls(id=payload["id"], run_id=payload["run_id"], value=value, text=text, type_name=type_name,
                   tokens=estimate_tokens(text), source=payload.get("source", ""),
                   created=payload.get("created", 0.0))


class ArtifactStore:
    """Namespaced by ``run_id``: artifacts from one run are invisible to
    another. Optionally persists to
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

    def _path(self, aid: str) -> Path:
        return self.directory / self.run_id / f"{aid}.json"

    def _persist(self, record: ArtifactRecord) -> None:
        if self.directory is None:
            return
        path = self._path(record.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(record.to_payload()), encoding="utf-8")

    def _find(self, aid: str) -> Optional[ArtifactRecord]:
        """The record, recovered from disk when an earlier process wrote it:
        a resumed run can still read a payload that was too large to keep
        in the ledger body."""
        record = self._records.get(aid)
        if record is None and self.directory is not None and self._path(aid).exists():
            record = ArtifactRecord.from_payload(json.loads(self._path(aid).read_text(encoding="utf-8")))
            self._records[aid] = record
        return record

    def get_record(self, aid: str) -> ArtifactRecord:
        record = self._find(aid)
        if record is None:
            raise KeyError(aid)
        return record

    def get(self, aid: str) -> Any:
        return self.get_record(aid).value

    def exists(self, aid: str) -> bool:
        return self._find(aid) is not None

    def ref_text(self, record: ArtifactRecord, *, preview: str = "head", preview_tokens: int = 120) -> str:
        """Projection form of an artifact: id + type + size + preview."""
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
        record = self.get_record(aid)
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
