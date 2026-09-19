"""Handler of the ``ask`` pack: ``meta.user.ask`` returns a :class:`Question`,
which the runtime turns into a ``WAITING_FOR_USER`` pause. ``Session.answer``
resumes."""
from __future__ import annotations

from typing import Optional

from ..context import ToolContext
from ..run import Question


def _ask(ctx: ToolContext, question: str, choices: Optional[list[str]] = None) -> Question:
    return Question(text=question, choices=choices)


ASK_HANDLERS = {"meta.user.ask": _ask}
