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

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role, "content": self.content,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments, "id": tc.id, "raw_arguments": tc.raw_arguments}
                for tc in self.tool_calls
            ],
            "tool_call_id": self.tool_call_id, "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(
            role=d["role"], content=d.get("content", ""),
            tool_calls=[
                ToolCall(name=tc["name"], arguments=tc.get("arguments") or {}, id=tc.get("id") or new_call_id(),
                          raw_arguments=tc.get("raw_arguments"))
                for tc in (d.get("tool_calls") or [])
            ],
            tool_call_id=d.get("tool_call_id"), name=d.get("name"),
        )


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Decision:
    """One model output: plain text and/or a batch of tool calls.

    ``finish`` is the formal completion signal (P0-3): it is a property of
    the *decision itself*, not a tool call routed through the runtime like
    any other. A decision that sets ``finish`` together with a non-empty
    ``calls`` is invalid and MUST be rejected by validation before anything
    executes — declaring the job done and still queuing side effects in the
    same breath is exactly the bug this separation prevents.
    """

    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    thought: str = ""
    usage: Optional[Usage] = None
    raw: Any = None
    finish: bool = False
    result: Any = None

    @property
    def is_text_only(self) -> bool:
        return not self.calls and not self.finish
