"""Compaction — folding overflowed conversation into the summary section
(spec §10) under the summary contract v1 (§10.2).

The agent's continuity is reconstructed every turn from the projection, so
the summary MUST preserve the *reasons* behind actions and unfinished
intentions, keep user constraints verbatim, and replace raw data with
handle references ($hN).
"""
from __future__ import annotations

import json
from typing import Optional

from .config import Config
from .handles import truncate_to_tokens
from .messages import Message, SYSTEM, USER
from .tokens import estimate_tokens

CONTRACT_V1 = """You are the compaction summarizer of an agent loop. Fold the transcript below into a summary that preserves the agent's continuity. Contract v1 — every rule is mandatory:
1. Write in first person as the agent ("I decided ... because ...").
2. Preserve chronological order.
3. For each item keep: the action taken, the gist of the observation, the decision and its reason, and any unfinished intentions.
4. Keep the user's explicit instructions, constraints and confirmed facts verbatim.
5. Replace raw data bodies with their handle references ($hN); never copy large data into the summary.
6. Keep the summary under {max_tokens} tokens (rough estimate is fine).
Output only the summary text, no preamble."""


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
                lines.append(f"[assistant→call] {tc.name}({truncate_to_tokens(args, 60)})")
        elif m.role == "tool":
            lines.append(f"[observation:{m.name}] {truncate_to_tokens(m.text(), 150)}")
        elif m.role == SYSTEM:
            lines.append(f"[runtime] {m.text()}")
    return "\n".join(lines)


def deterministic_fold(messages: list[Message], max_tokens: int) -> str:
    """LLM-free fallback (compaction.model="none").

    Keeps user messages verbatim (contract rule 4), actions and observation
    gists in order; cannot reconstruct reasons, so it notes that.
    """
    lines: list[str] = ["(mechanical fold — reasons unavailable)"]
    for m in messages:
        if m.role == USER:
            lines.append(f'User said (verbatim): "{m.text()}"')
        elif m.role == "assistant":
            for tc in m.tool_calls:
                lines.append(f"I called {tc.name}.")
            text = m.text().strip()
            if text:
                lines.append(f"I replied: {truncate_to_tokens(text, 60)}")
        elif m.role == "tool":
            gist = truncate_to_tokens(m.text(), 40)
            lines.append(f"→ {m.name}: {gist}")
    return truncate_to_tokens("\n".join(lines), max_tokens)


class Compactor:
    def __init__(self, config: Config, summarizer=None) -> None:
        """``summarizer`` is an LLMAdapter or None (deterministic fallback)."""
        self.config = config
        self.summarizer = summarizer

    def should_compact(self, conversation: list[Message], window_tokens: int) -> bool:
        threshold = window_tokens * self.config.compaction.trigger_ratio
        return estimate_tokens(conversation) > threshold

    def split_point(self, conversation: list[Message]) -> int:
        """Index splitting messages to fold (older half by tokens) from the rest.

        Never orphans tool observations and always leaves at least the last
        exchange unfolded.
        """
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

    def fold(self, conversation: list[Message]) -> tuple[str, list[Message]]:
        i = self.split_point(conversation)
        folded, remaining = conversation[:i], conversation[i:]
        if not folded:
            return "", conversation
        folded_tokens = estimate_tokens(folded)
        max_summary = max(150, int(folded_tokens * self.config.compaction.max_summary_ratio))
        if self.summarizer is None:
            summary = deterministic_fold(folded, max_summary)
        else:
            prompt = [
                Message(role=SYSTEM, content=CONTRACT_V1.format(max_tokens=max_summary)),
                Message(role=USER, content=render_transcript(folded)),
            ]
            decision = self.summarizer.complete(prompt, None)
            summary = truncate_to_tokens(decision.text.strip(), int(max_summary * 1.5))
        return summary, remaining
