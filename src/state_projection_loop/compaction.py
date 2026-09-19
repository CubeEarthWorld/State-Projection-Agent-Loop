"""Compaction: fold old history into the structured working state with one
model call, instead of re-summarising prose.

The model returns a JSON delta. Its *shape* is validated with the same
schema validator as tool arguments, and each entry is checked for
grounding: an identifier, path or number the transcript never mentions is
the mark of an invented fact, and such an entry is dropped (the delta
records it under ``ungrounded``). The pre-fold working state is written to
the ledger with the delta. Folded events keep living in the ledger; the
projection renders only the user's own words of them afterwards.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from .compression import ungrounded
from .json_schema import validate_value
from .working_state import RecordedDecision, WorkingState

_ITEM = {"type": "string", "maxLength": 500}

FOLD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts_add": {"type": "array", "items": _ITEM, "maxItems": 20},
        "decisions_add": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": _ITEM, "reason": _ITEM},
                "required": ["text"],
                "additionalProperties": False,
            },
            "maxItems": 20,
        },
        "questions_add": {"type": "array", "items": _ITEM, "maxItems": 20},
        "questions_resolve": {"type": "array", "items": _ITEM, "maxItems": 20},
        "next_actions": {"type": "array", "items": _ITEM, "maxItems": 20},
    },
    "additionalProperties": False,
}

FOLD_INSTRUCTIONS = (
    "You compact an agent transcript into structured working state. Read the transcript "
    "and answer with ONE JSON object and nothing else, with these optional keys: "
    '"facts_add" (confirmed facts worth keeping), "decisions_add" (objects with "text" and '
    '"reason"), "questions_add" (still-open questions), "questions_resolve" (open questions '
    'now answered, verbatim), "next_actions" (the full remaining plan, replacing the old '
    "one). Add only what the transcript states; never invent. Keep each entry under 500 "
    "characters and each list under 20 entries."
)

_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def parse_fold_reply(text: str) -> Optional[dict[str, Any]]:
    """Parse the model's fold reply: a JSON object, optionally fenced."""
    body = text.strip()
    fence = _FENCE.search(body)
    if fence:
        body = fence.group(1).strip()
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def apply_fold_delta(ws: WorkingState, delta: dict[str, Any], *, transcript: str = "") -> Optional[str]:
    """Validate and merge a fold delta. Returns an error message, or None.

    With a ``transcript``, entries carrying identifiers it never mentions
    are dropped from the delta (and listed under ``delta["ungrounded"]``)
    before the merge."""
    error = validate_value(FOLD_SCHEMA, delta)
    if error is not None:
        return error
    if transcript:
        dropped: list[str] = []

        def grounded(entry: Any) -> bool:
            text = entry if isinstance(entry, str) else f"{entry.get('text', '')} {entry.get('reason', '')}"
            missing = ungrounded(text, transcript)
            if missing:
                dropped.append(f"{text} (unknown: {', '.join(missing)})")
            return not missing

        for key in ("facts_add", "decisions_add", "questions_add", "next_actions"):
            if key in delta:
                delta[key] = [entry for entry in delta[key] if grounded(entry)]
        if dropped:
            delta["ungrounded"] = dropped
    resolve = list(delta.get("questions_resolve") or [])
    for q in resolve:
        if q not in ws.open_questions:
            return f"questions_resolve names an unknown question: {q}"
    for f in delta.get("facts_add") or []:
        if f not in ws.confirmed_facts:
            ws.confirmed_facts.append(f)
    for d in delta.get("decisions_add") or []:
        ws.decisions.append(RecordedDecision(text=d["text"], reason=d.get("reason", "")))
    for q in delta.get("questions_add") or []:
        if q not in ws.open_questions:
            ws.open_questions.append(q)
    ws.open_questions = [q for q in ws.open_questions if q not in resolve]
    if "next_actions" in delta:
        ws.next_actions = list(delta.get("next_actions") or [])
    return None
