"""Structured working state: a finite, typed record of what the agent
knows and has decided, rather than an unbounded stack of free-text summaries.

Prose has no schema: a summary asked to "preserve reasons" keeps a
decision's reason exactly as reliably as any other sentence survives a
second fold, which is to say not reliably. ``WorkingState`` makes the shape
the promise: decisions are ``(text, reason)`` pairs in a list, not sentences
buried in a paragraph, so folding *appends* to a field instead of
re-summarizing a summary.

The original conversation text is never lost either way — it stays in the
Event Ledger (``user_input``/``model_response``/``command_*`` events) and is
reachable via ``meta.history.search`` even after being folded out
of the live projection.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from .tokens import truncate_to_tokens
from .checklists import ChecklistStore
from .serialization import dumps


@dataclass
class RecordedDecision:
    text: str
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "reason": self.reason}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RecordedDecision":
        return cls(text=str(d.get("text", "")), reason=str(d.get("reason", "")))


# The fields that are plain lists of text: copied and parsed alike.
_LIST_FIELDS = ("acceptance_criteria", "constraints", "confirmed_facts", "open_questions", "next_actions",
                "artifact_refs")


@dataclass
class WorkingState:
    goal: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    confirmed_facts: list[str] = field(default_factory=list)
    decisions: list[RecordedDecision] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    # Free-form escape hatch for application-specific state (game flags,
    # domain variables) that doesn't fit the fixed fields above. Editors of
    # `extra` are user code, the LLM (via the state.extra.* capabilities) and
    # the session seed.
    extra: dict[str, Any] = field(default_factory=dict)
    checklists: ChecklistStore = field(default_factory=ChecklistStore)
    # Ledger sequence up to which history has been folded into this state by
    # compaction; only the user's own words of those events render afterwards.
    folded_sequence: int = 0
    # Ledger sequence from which history renders verbatim. Everything older
    # is tiered by its distance from this point, and the point moves only in
    # steps (see Session._advance_tiers), so the rendered prefix stays
    # byte-identical between steps and a provider's prompt cache keeps hitting.
    verbatim_sequence: int = 0

    def is_empty(self) -> bool:
        return not any(getattr(self, f.name) for f in fields(self)
                       if f.name not in ("folded_sequence", "verbatim_sequence"))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {f.name: getattr(self, f.name) for f in fields(self)}  # declaration order
        out.update({name: list(out[name]) for name in _LIST_FIELDS})
        out.update(decisions=[d.to_dict() for d in self.decisions], extra=dict(self.extra),
                   checklists=self.checklists.to_dict())
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkingState":
        return cls(
            goal=str(d.get("goal", "")),
            decisions=[RecordedDecision.from_dict(x) for x in (d.get("decisions") or [])],
            extra=dict(d.get("extra") or {}),
            checklists=ChecklistStore.from_dict(d["checklists"]) if "checklists" in d else ChecklistStore(),
            folded_sequence=int(d.get("folded_sequence") or 0),
            verbatim_sequence=int(d.get("verbatim_sequence") or 0),
            **{name: list(d.get(name) or []) for name in _LIST_FIELDS},
        )

    def render(self, *, max_tokens: int = 800) -> str:
        parts: list[str] = [f"goal: {self.goal}"] if self.goal else []

        def bullets(name: str, lines: list[str]) -> None:
            if lines:
                parts.append(f"{name}:\n" + "\n".join(f"- {line}" for line in lines))

        bullets("acceptance_criteria", self.acceptance_criteria)
        bullets("constraints", self.constraints)
        bullets("confirmed_facts", self.confirmed_facts)
        bullets("decisions", [d.text + (f" (because: {d.reason})" if d.reason else "") for d in self.decisions])
        bullets("open_questions", self.open_questions)
        bullets("next_actions", self.next_actions)
        if self.artifact_refs:
            parts.append("artifact_refs: " + ", ".join(self.artifact_refs))
        if self.extra:
            parts.append("extra: " + dumps(self.extra))
        return truncate_to_tokens("\n".join(parts), max_tokens)


# The typed fields of WorkingState, i.e. the keys from_dict understands.
# Anything else a caller seeds is app-specific state and belongs in `extra`.
WORKING_STATE_FIELDS = frozenset(f.name for f in fields(WorkingState))
