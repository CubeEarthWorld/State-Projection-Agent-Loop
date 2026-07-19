"""Structured working state (P1-1): a finite, typed record of what the agent
knows and has decided, replacing an unbounded stack of free-text summaries.

The old summary contract asked an LLM to write prose that "preserves
reasons" and hoped later re-folds wouldn't lose them. Prose has no schema,
so nothing enforced that promise — a decision's reason was exactly as likely
to survive a second fold as any other sentence, which is to say: not
reliably. ``WorkingState`` makes the shape the promise: decisions are
``(text, reason)`` pairs in a list, not sentences buried in a paragraph, so
folding *appends* to a field instead of re-summarizing a summary.

The original conversation text is never lost either way — it stays in the
Event Ledger (``user_input``/``model_response``/``command_*`` events) and is
reachable via the ``search_history`` capability even after being folded out
of the live projection.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .messages import Message, SYSTEM
from .tokens import estimate_tokens


@dataclass
class RecordedDecision:
    text: str
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "reason": self.reason}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RecordedDecision":
        return cls(text=str(d.get("text", "")), reason=str(d.get("reason", "")))


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
    # `extra` are the same three as before: user code, the LLM (via the
    # state.extra.* capabilities), and the session seed.
    extra: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any([
            self.goal, self.acceptance_criteria, self.constraints, self.confirmed_facts,
            self.decisions, self.open_questions, self.next_actions, self.artifact_refs, self.extra,
        ])

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "acceptance_criteria": list(self.acceptance_criteria),
            "constraints": list(self.constraints),
            "confirmed_facts": list(self.confirmed_facts),
            "decisions": [d.to_dict() for d in self.decisions],
            "open_questions": list(self.open_questions),
            "next_actions": list(self.next_actions),
            "artifact_refs": list(self.artifact_refs),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkingState":
        return cls(
            goal=str(d.get("goal", "")),
            acceptance_criteria=list(d.get("acceptance_criteria") or []),
            constraints=list(d.get("constraints") or []),
            confirmed_facts=list(d.get("confirmed_facts") or []),
            decisions=[RecordedDecision.from_dict(x) for x in (d.get("decisions") or [])],
            open_questions=list(d.get("open_questions") or []),
            next_actions=list(d.get("next_actions") or []),
            artifact_refs=list(d.get("artifact_refs") or []),
            extra=dict(d.get("extra") or {}),
        )

    def merge_fold(self, delta: dict[str, Any]) -> None:
        """Apply a compaction-contract-v2 delta (see compaction.py): additive
        for facts/decisions/artifact_refs, replace-if-present for goal, and
        add/remove for open_questions."""
        if delta.get("goal"):
            self.goal = str(delta["goal"])
        for key in ("acceptance_criteria", "constraints"):
            for item in delta.get(key) or []:
                if item not in getattr(self, key):
                    getattr(self, key).append(item)
        for fact in delta.get("new_facts") or []:
            if fact not in self.confirmed_facts:
                self.confirmed_facts.append(fact)
        for d in delta.get("new_decisions") or []:
            if isinstance(d, dict):
                self.decisions.append(RecordedDecision.from_dict(d))
        for q in delta.get("new_open_questions") or []:
            if q not in self.open_questions:
                self.open_questions.append(q)
        for q in delta.get("resolved_open_questions") or []:
            self.open_questions = [x for x in self.open_questions if x != q]
        if delta.get("next_actions") is not None:
            self.next_actions = list(delta["next_actions"])
        for ref in delta.get("artifact_refs") or []:
            if ref not in self.artifact_refs:
                self.artifact_refs.append(ref)

    def render(self, *, max_tokens: int = 800) -> str:
        parts: list[str] = []
        if self.goal:
            parts.append(f"goal: {self.goal}")
        if self.acceptance_criteria:
            parts.append("acceptance_criteria:\n" + "\n".join(f"- {c}" for c in self.acceptance_criteria))
        if self.constraints:
            parts.append("constraints:\n" + "\n".join(f"- {c}" for c in self.constraints))
        if self.confirmed_facts:
            parts.append("confirmed_facts:\n" + "\n".join(f"- {c}" for c in self.confirmed_facts))
        if self.decisions:
            parts.append("decisions:\n" + "\n".join(
                f"- {d.text}" + (f" (because: {d.reason})" if d.reason else "") for d in self.decisions
            ))
        if self.open_questions:
            parts.append("open_questions:\n" + "\n".join(f"- {q}" for q in self.open_questions))
        if self.next_actions:
            parts.append("next_actions:\n" + "\n".join(f"- {a}" for a in self.next_actions))
        if self.artifact_refs:
            parts.append("artifact_refs: " + ", ".join(self.artifact_refs))
        if self.extra:
            parts.append("extra: " + json.dumps(self.extra, ensure_ascii=False, default=str))
        body = "\n".join(parts)
        if estimate_tokens(body) > max_tokens:
            # Truncate the least time-critical sections first: facts, then
            # decisions, keeping goal/constraints/open_questions/next_actions
            # (the parts most load-bearing for not losing the thread).
            body = body[: max_tokens * 4]
        return body


class WorkingStateSection:
    """Projects the working state each turn (volatile — always near the tail)."""

    name = "working_state"
    cache_class = "volatile"

    def __init__(self, *, max_tokens: int = 800) -> None:
        self.max_tokens = max_tokens

    def render(self, turn: Any) -> list[Message]:
        ws: Optional[WorkingState] = getattr(turn, "working_state", None)
        if ws is None or ws.is_empty():
            return []
        return [Message(role=SYSTEM, content="[Working state]\n" + ws.render(max_tokens=self.max_tokens))]
