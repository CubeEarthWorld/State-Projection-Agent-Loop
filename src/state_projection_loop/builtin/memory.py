"""Handlers of the ``memory`` pack over the session's :class:`MemoryStore`."""
from __future__ import annotations

import time
from typing import Any, Optional

from ..context import ToolContext


def _save(ctx: ToolContext, text: str, tags: Optional[list[str]] = None) -> str:
    note = ctx.session.memory.save(text, tags or [])
    return f"saved note {note.id}"


def _search(ctx: ToolContext, query: str, k: int = 5) -> Any:
    notes = ctx.session.memory.search(query, k)
    if not notes:
        return "No notes matched."
    return [{"id": n.id, "text": n.text, "tags": n.tags,
             "saved": time.strftime("%Y-%m-%d", time.gmtime(n.ts))} for n in notes]


MEMORY_HANDLERS = {"memory.note.save": _save, "memory.note.search": _search}
