"""Constrained middleware hooks (defect-1 fix).

Exactly two interception points are exposed, chosen so approval gates and
guardrails fit without forking the loop, while the loop's four verbs stay
fixed:

* ``after_decide``  — inspect the model's decision *before* execution;
  may block it (human-in-the-loop approval, policy engines).
* ``after_execute`` — transform an observation *after* execution
  (redaction, enrichment).

Hooks receive only the decision/result and the turn context. The API
deliberately offers no way to load specs wholesale or inject ahead of the
prefix cache, so hooks cannot violate the invariants (I1–I4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .messages import Decision, ToolCall


@dataclass
class HookBlock:
    """Returned by an ``after_decide`` hook to stop the pending tool calls.

    ``observation`` (or ``reason``) is projected back to the model as the
    result of each blocked call, so it can self-correct next turn.
    """

    reason: str
    observation: Optional[str] = None

    def text(self) -> str:
        return self.observation or f"[blocked by policy] {self.reason}"


# after_decide(decision, turn) -> HookBlock | None (None = proceed)
AfterDecideHook = Callable[[Decision, Any], Optional[HookBlock]]
# after_execute(call, result, turn) -> replacement result | None (None = keep)
AfterExecuteHook = Callable[[ToolCall, Any, Any], Optional[Any]]


@dataclass
class Hooks:
    after_decide: list[AfterDecideHook] = field(default_factory=list)
    after_execute: list[AfterExecuteHook] = field(default_factory=list)
