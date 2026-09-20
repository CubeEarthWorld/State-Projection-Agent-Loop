"""Internal message and decision representation; observations carry a
structurally distinct role.

``content`` may be a plain string or a list of part dicts
(e.g. ``[{"type": "text", "text": ...}, {"type": "image_url", ...}]``) so
multimodal input can pass through without core changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .ids import new_id

# Role constants. Tool results MUST use OBSERVATION so untrusted data stays
# structurally distinct from instructions (a mitigation, not a full defense).
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"
OBSERVATION = "tool"

def new_call_id() -> str:
    # A ULID, not a counter: a counter restarts with the process and would
    # reuse ids already in a resumed ledger, where calls pair with results by id.
    return new_id("call")


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=new_call_id)
    raw_arguments: Optional[str] = None
    """Original argument string when the provider returned unparseable JSON;
    validation will fail and route through the self-repair path."""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "id": self.id, "raw_arguments": self.raw_arguments}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolCall":
        return cls(name=d["name"], arguments=d.get("arguments") or {}, id=d.get("id") or new_call_id(),
                   raw_arguments=d.get("raw_arguments"))


@dataclass
class Message:
    role: str
    content: Any = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return "\n".join(
                p.get("text", "") for p in self.content if isinstance(p, dict) and p.get("type") == "text"
            )
        return str(self.content)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(
            role=d["role"], content=d.get("content", ""),
            tool_calls=[ToolCall.from_dict(tc) for tc in (d.get("tool_calls") or [])],
            tool_call_id=d.get("tool_call_id"), name=d.get("name"),
        )


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Of prompt_tokens, how many the provider served from its prompt cache
    # (0 when it does not say): the number that tells whether the
    # projection's prefix stayed byte-stable between turns.
    cached_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "cached_tokens": self.cached_tokens}


@dataclass
class Decision:
    """One model output: plain text and/or a batch of tool calls.

    ``finish`` is the formal completion signal: it is a property of
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

