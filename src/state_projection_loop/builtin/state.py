"""Working-state-as-tools: the LLM edits the structured working state
through a small set of typed capabilities instead of an arbitrary dict.

Editors of the working state are exactly two: user code
(``session.working_state`` / seed) and the LLM via these capabilities.
A game master registers all of this; a simple support bot registers none
— the core projection is identical either way.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from ..capability import ToolContext
from .defs import load


def _walk_extra(extra: dict, path: str, *, create: bool = False) -> tuple[Any, str]:
    parts = [p for p in path.split(".") if p]
    if not parts:
        raise ValueError("empty path")
    node = extra
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            if not create:
                raise KeyError(path)
            node[part] = {}
        node = node[part]
    return node, parts[-1]


def _set_goal(ctx: ToolContext, text: str) -> str:
    ctx.working_state.goal = text
    return f"goal set: {text}"


def _add_fact(ctx: ToolContext, text: str) -> str:
    if text not in ctx.working_state.confirmed_facts:
        ctx.working_state.confirmed_facts.append(text)
    return f"fact recorded: {text}"


def _add_constraint(ctx: ToolContext, text: str) -> str:
    if text not in ctx.working_state.constraints:
        ctx.working_state.constraints.append(text)
    return f"constraint recorded: {text}"


def _record_decision(ctx: ToolContext, text: str, reason: str = "") -> str:
    from ..working_state import RecordedDecision

    ctx.working_state.decisions.append(RecordedDecision(text=text, reason=reason))
    return f"decision recorded: {text}" + (f" (because: {reason})" if reason else "")


def _add_open_question(ctx: ToolContext, text: str) -> str:
    if text not in ctx.working_state.open_questions:
        ctx.working_state.open_questions.append(text)
    return f"open question added: {text}"


def _resolve_open_question(ctx: ToolContext, text: str) -> str:
    ctx.working_state.open_questions = [q for q in ctx.working_state.open_questions if q != text]
    return f"open question resolved: {text}"


def _set_next_actions(ctx: ToolContext, actions: list[str]) -> str:
    ctx.working_state.next_actions = list(actions)
    return f"next_actions set: {actions}"


def _extra_set(ctx: ToolContext, path: str, value: Any = None) -> str:
    node, leaf = _walk_extra(ctx.working_state.extra, path, create=True)
    node[leaf] = value
    return f"extra.{path} = {json.dumps(value, ensure_ascii=False, default=str)}"


def _extra_get(ctx: ToolContext, path: str) -> Any:
    try:
        node, leaf = _walk_extra(ctx.working_state.extra, path)
        return node[leaf]
    except KeyError:
        return f"(not set: {path})"



STATE_HANDLERS = {
    "state.goal.set": _set_goal,
    "state.fact.add": _add_fact,
    "state.constraint.add": _add_constraint,
    "state.decision.record": _record_decision,
    "state.question.add": _add_open_question,
    "state.question.resolve": _resolve_open_question,
    "state.next_actions.set": _set_next_actions,
    "state.extra.set": _extra_set,
    "state.extra.get": _extra_get,
}


def install_state(registry) -> None:
    """Register the bundled working-state capabilities.

    Takes a Registry, not a Session: nothing here needs the session, and the
    Dart port takes a Registry for the same reason.

    The working state is projected automatically by ``WorkingStateSection``
    whenever it is part of ``config.projection.sections`` (the default).
    """
    for definition in load("state"):
        if definition["name"] not in registry:
            registry.register(definition, handler=STATE_HANDLERS[definition["name"]])
