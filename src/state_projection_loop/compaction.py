"""Compaction — folding overflowed conversation into the working state
under fold contract v2.

Contract v1 asked the summarizer to write free prose and hope decision
reasons survived a later re-fold. Contract v2 instead asks for a small,
schema-shaped delta that is merged straight into
:class:`~state_projection_loop.working_state.WorkingState`
(:meth:`WorkingState.merge_fold`): new facts and decisions are *appended*,
not re-summarized, so a decision's reason recorded three folds ago is still
there verbatim. The folded messages themselves are never discarded — they
remain in the Event Ledger and stay reachable via the ``search_history``
capability after being dropped from the live projection.
"""
from __future__ import annotations

import json
from typing import Optional

from .config import Config
from .messages import Message, SYSTEM, USER
from .tokens import estimate_tokens
from .working_state import WorkingState

CONTRACT_V2 = """You are the compaction summarizer of an agent loop. Fold the transcript below into a JSON delta that will be merged into a structured working-state record. Contract v2 — every rule is mandatory:
1. Output ONLY a single JSON object, no prose, no markdown fence.
2. Fields (all optional, omit what doesn't apply):
   "goal": string — only if the goal changed or was clarified,
   "new_facts": array of strings — confirmed facts/user constraints, kept verbatim where the user stated them,
   "new_decisions": array of {"text": string, "reason": string} — every decision the agent made and WHY, in first person,
   "new_open_questions": array of strings,
   "resolved_open_questions": array of strings — exact text of questions that are no longer open,
   "next_actions": array of strings — REPLACES the previous next_actions list; give the full current list,
   "artifact_refs": array of strings — any artifact ids mentioned that remain relevant.
3. Never copy large raw data bodies into a field; reference their artifact id instead.
4. Preserve chronological order within each array.
Output only the JSON object."""


def render_transcript(messages: list[Message]) -> str:
    lines: list[str] = []
    for m in messages:
        if m.role == USER:
            lines.append(f"[user] {m.text()}")
        elif m.role == "assistant":
            text = m.text()
            if text:
                lines.append(f"[assistant] {text}")
            for tc in m.tool_calls:
                args = json.dumps(tc.arguments, ensure_ascii=False, default=str)
                lines.append(f"[assistant→call] {tc.name}({_truncate(args, 60)})")
        elif m.role == "tool":
            lines.append(f"[observation:{m.name}] {_truncate(m.text(), 150)}")
        elif m.role == SYSTEM:
            lines.append(f"[runtime] {m.text()}")
    return "\n".join(lines)


def _truncate(text: str, max_tokens: int) -> str:
    from .artifacts import truncate_to_tokens

    return truncate_to_tokens(text, max_tokens)


def deterministic_fold(messages: list[Message]) -> dict:
    """LLM-free fallback (compaction.model="none"): mechanical, cannot
    reconstruct reasons, so it says so explicitly in a fact entry."""
    facts: list[str] = []
    for m in messages:
        if m.role == USER:
            facts.append(f'User said (verbatim): "{m.text()}"')
        elif m.role == "assistant":
            for tc in m.tool_calls:
                facts.append(f"(mechanical fold) called {tc.name}")
    return {"new_facts": facts[:50]}


class Compactor:
    def __init__(self, config: Config, summarizer=None) -> None:
        """``summarizer`` is an LLMAdapter or None (deterministic fallback)."""
        self.config = config
        self.summarizer = summarizer

    def should_compact(self, conversation: list[Message], window_tokens: int) -> bool:
        threshold = window_tokens * self.config.compaction.trigger_ratio
        return estimate_tokens(conversation) > threshold

    def split_point(self, conversation: list[Message]) -> int:
        """Index splitting messages to fold (older half by tokens) from the
        rest. Never orphans tool observations and always leaves at least the
        last exchange unfolded."""
        total = estimate_tokens(conversation)
        target = total // 2
        acc = 0
        i = 0
        while i < len(conversation) and acc < target:
            acc += estimate_tokens(conversation[i])
            i += 1
        while i < len(conversation) and conversation[i].role == "tool":
            i += 1
        if i >= len(conversation):
            i = max(0, len(conversation) - 1)
            while i > 0 and conversation[i].role == "tool":
                i -= 1
        return i

    def fold(self, conversation: list[Message], working_state: WorkingState) -> tuple[bool, list[Message]]:
        """Fold the older half of ``conversation`` into ``working_state`` in
        place. Returns (folded_anything, remaining_conversation)."""
        i = self.split_point(conversation)
        folded, remaining = conversation[:i], conversation[i:]
        if not folded:
            return False, conversation
        if self.summarizer is None:
            delta = deterministic_fold(folded)
        else:
            prompt = [
                Message(role=SYSTEM, content=CONTRACT_V2),
                Message(role=USER, content=render_transcript(folded)),
            ]
            decision = self.summarizer.complete(prompt, None)
            delta = _parse_delta(decision.text)
        working_state.merge_fold(delta)
        return True, remaining


def _parse_delta(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {"new_facts": [f"(fold parse failed) {text[:200]}"]}
