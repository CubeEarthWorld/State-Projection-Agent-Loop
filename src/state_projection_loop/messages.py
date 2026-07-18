"""Internal message and decision representation (spec §17 — implementation
discretion; §8.2 — observations carry a structurally distinct role).

``content`` may be a plain string or a list of part dicts
(e.g. ``[{"type": "text", "text": ...}, {"type": "image_url", ...}]``) so
multimodal input can pass through without core changes.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Optional

# Role constants. Tool results MUST use OBSERVATION so untrusted data stays
# structurally distinct from instructions (invariant I6; mitigation, not a
# full defense — see spec §16).
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"
OBSERVATION = "tool"

_call_counter = itertools.count(1)


def new_call_id() -> str:
    return f"call_{next(_call_counter)}"


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=new_call_id)
    raw_arguments: Optional[str] = None
    """Original argument string when the provider returned unparseable JSON;
    validation will fail and route through the self-repair path (§6)."""


@dataclass
class Message:
    role: str
    content: Any = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return "\n".join(
                p.get("text", "") for p in self.content if isinstance(p, dict) and p.get("type") == "text"
            )
        return str(self.content)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Decision:
    """One model output: plain text and/or a batch of tool calls (§2.3)."""

    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    thought: str = ""
    usage: Optional[Usage] = None
    raw: Any = None

    @property
    def is_text_only(self) -> bool:
        return not self.calls
