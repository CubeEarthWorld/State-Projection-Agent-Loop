"""Cross-session memory: notes keyed outside any run namespace.

Notes outlive runs and sessions. Nothing is injected into the context
automatically — only what a search returns enters it, as an observation —
so a stale note can never masquerade as an instruction. The store is two
methods, so a database or vector index can replace the default without
touching the pack.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from .discovery import _tokenize
from .ids import new_id
from .serialization import dumps


@dataclass
class Note:
    id: str
    text: str
    tags: list[str] = field(default_factory=list)
    ts: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "Note":
        """Unknown fields are ignored, so a file written by a later version
        still loads."""
        return cls(id=d["id"], text=d["text"], tags=list(d.get("tags") or []),
                   ts=float(d.get("ts") or 0.0))


@runtime_checkable
class MemoryStore(Protocol):
    def save(self, text: str, tags: list[str]) -> Note: ...

    def search(self, query: str, k: int) -> list[Note]: ...


class JsonlMemoryStore:
    """One JSONL file of notes, or process memory when ``path`` is None.
    Search is lexical: the notes sharing the most query tokens (text and
    tags) come first, newest first among equals."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path is not None else None
        self._notes: list[Note] = []
        if self.path is not None and self.path.exists():
            with self.path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        self._notes.append(Note.from_dict(json.loads(line)))
                    except (ValueError, KeyError, TypeError):
                        # Appends are unbuffered: a crash mid-append leaves a
                        # torn last line. Skipping it beats failing every
                        # later Session() construction.
                        continue

    def save(self, text: str, tags: list[str]) -> Note:
        note = Note(id=new_id("note"), text=text, tags=list(tags), ts=time.time())
        self._notes.append(note)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(dumps(asdict(note)) + "\n")
        return note

    def search(self, query: str, k: int) -> list[Note]:
        terms = set(_tokenize(query))
        if not terms:
            return []
        scored = [(len(terms & set(_tokenize(n.text + " " + " ".join(n.tags)))), n.ts, n) for n in self._notes]
        hits = sorted((s for s in scored if s[0] > 0), key=lambda s: (s[0], s[1]), reverse=True)
        return [n for _, _, n in hits[:k]]
