"""Value store and handles (spec §8.3, invariant I7).

Large tool results never pass through the model's context: they are stored
here and projected as ``$hN`` references (type + size + preview). Tools may
receive handles as arguments — the runtime resolves them to real values.
The store lives for the session and is discarded with it (§15).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .tokens import estimate_tokens

HANDLE_RE = re.compile(r"^\$h\d+$")


def serialize_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=None, default=str)
    except (TypeError, ValueError):
        return str(value)


@dataclass
class HandleRecord:
    id: str
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


class ValueStore:
    def __init__(self) -> None:
        self._records: dict[str, HandleRecord] = {}
        self._counter = 0

    def put(self, value: Any, *, source: str = "") -> HandleRecord:
        self._counter += 1
        hid = f"$h{self._counter}"
        text = serialize_value(value)
        record = HandleRecord(
            id=hid,
            value=value,
            text=text,
            type_name=type(value).__name__,
            tokens=estimate_tokens(text),
            source=source,
        )
        self._records[hid] = record
        return record

    def get(self, hid: str) -> Any:
        return self._records[hid].value

    def get_record(self, hid: str) -> HandleRecord:
        return self._records[hid]

    def exists(self, hid: str) -> bool:
        return hid in self._records

    def ref_text(self, record: HandleRecord, *, preview: str = "head", preview_tokens: int = 120) -> str:
        """Projection form of a handle: id + type + size + preview (I7)."""
        if preview == "tail":
            body = record.text[-preview_tokens * 6 :]
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

    # -- peek (§12 resident meta tool) --------------------------------------

    def peek(
        self,
        hid: str,
        query: Optional[str] = None,
        range_: Optional[str] = None,
        *,
        max_tokens: int = 600,
    ) -> str:
        if not isinstance(hid, str) or not self.exists(hid):
            known = ", ".join(sorted(self._records)) or "(none)"
            shown = repr(hid)[:80]
            return f"Error: unknown handle {shown}. Known handles: {known}"
        record = self._records[hid]
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
    def _peek_range(record: HandleRecord, range_: str) -> str:
        m = re.match(r"^\s*(\d+)\s*(?:-\s*(\d+))?\s*$", range_)
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else start
            lines = record.text.split("\n")
            sel = lines[max(0, start - 1) : end]
            return "\n".join(f"{i}: {line}" for i, line in enumerate(sel, start=max(1, start)))
        # key path like a.b[2].c into JSON-like values
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
    def _peek_query(record: HandleRecord, query: str) -> str:
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

    # -- handle resolution in tool arguments (§8.3) --------------------------

    def resolve_args(self, args: Any) -> Any:
        """Deep-replace strings that are exactly ``$hN`` with stored values."""
        if isinstance(args, str):
            if HANDLE_RE.match(args) and self.exists(args):
                return self.get(args)
            return args
        if isinstance(args, list):
            return [self.resolve_args(a) for a in args]
        if isinstance(args, dict):
            return {k: self.resolve_args(v) for k, v in args.items()}
        return args
