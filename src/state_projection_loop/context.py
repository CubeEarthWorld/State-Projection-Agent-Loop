"""What a tool handler and a projection section are handed.

Kept apart from the modules whose objects it carries, so none of them has to
import another just to name a field's type.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .artifacts import ArtifactStore
    from .config import Config
    from .discovery import ScoredTool, ToolSearch
    from .events import EventLedger
    from .registry import Registry
    from .run import Run
    from .session import Session
    from .working_state import WorkingState


@dataclass
class ToolContext:
    """What a tool handler receives.

    A handler opts in by declaring a first parameter named ``ctx`` (or
    annotated with ``ToolContext``); it is excluded from the JSON schema and
    injected by the runtime with ``command_id`` set. ``command_id`` is stable
    across retries of the *same* logical attempt and is the correct
    idempotency key to hand to an external API.

    Sections render from the superset :class:`TurnContext`; the runtime
    narrows it with :meth:`for_command` before a handler runs, so projection
    state never reaches a tool.
    """

    config: Optional["Config"] = None
    registry: Optional["Registry"] = None
    ledger: Optional["EventLedger"] = None
    run: Optional["Run"] = None
    working_state: Optional["WorkingState"] = None
    session: Optional["Session"] = None
    store: Optional["ArtifactStore"] = None
    search: Optional["ToolSearch"] = None
    command_id: str = ""

    @property
    def run_id(self) -> str:
        return self.run.id if self.run is not None else ""

    def for_command(self, command_id: str) -> "ToolContext":
        """The handler-facing view of this context for one command."""
        shared = {f.name: getattr(self, f.name) for f in fields(ToolContext)}
        return ToolContext(**{**shared, "command_id": command_id})


@dataclass
class TurnContext(ToolContext):
    """What a section renders from: the handler context plus this turn's
    projection state. ``api_tools`` is the list of native schemas that will
    be sent; sections may drop entries from it while shrinking, and the
    session sends whatever is left."""

    candidates: list["ScoredTool"] = field(default_factory=list)
    api_tools: list[dict[str, Any]] = field(default_factory=list)
