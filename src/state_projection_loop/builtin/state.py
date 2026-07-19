"""Working-state-as-tools: the LLM edits the structured working state
through a small set of typed capabilities instead of an arbitrary dict.

Editors of the working state are exactly three: user code
(``session.run.working_state`` / seed), the LLM via these capabilities, and
compaction folds (`compaction.py`). A game master registers all of this;
a simple support bot registers none — the core projection is identical
either way.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from ..capability import ToolContext


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


STATE_CAPABILITY_DEFS: list[dict[str, Any]] = [
    {
        "name": "state.goal.set",
        "category": "state",
        "spec": {
            "description": "現在の目標(ゴール)を設定する。working_stateとcandidate検索クエリに反映される。",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
        "discovery": {"embedding_text": "目標 ゴール クリア条件 目的 goal objective"},
        "execution": {"timeout_s": 5, "retry_safety": "idempotent"},
        "effects": [{"kind": "none"}],
    },
    {
        "name": "state.fact.add",
        "category": "state",
        "spec": {
            "description": "確認済みの事実・ユーザーの制約を working_state に追記する。",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
        "discovery": {"embedding_text": "事実 記録 確認 remember fact constraint"},
        "execution": {"timeout_s": 5, "retry_safety": "idempotent"},
        "effects": [{"kind": "none"}],
    },
    {
        "name": "state.constraint.add",
        "category": "state",
        "spec": {
            "description": "制約を working_state に追記する。",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
        "execution": {"timeout_s": 5, "retry_safety": "idempotent"},
        "effects": [{"kind": "none"}],
    },
    {
        "name": "state.decision.record",
        "category": "state",
        "spec": {
            "description": "判断とその理由を working_state.decisions に記録する。理由は必ず埋めること。",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "reason": {"type": "string", "default": ""}},
                "required": ["text"],
            },
        },
        "discovery": {"embedding_text": "判断 決定 理由 decision reason record"},
        "execution": {"timeout_s": 5, "retry_safety": "idempotent"},
        "effects": [{"kind": "none"}],
    },
    {
        "name": "state.question.add",
        "category": "state",
        "spec": {
            "description": "未解決の疑問を working_state.open_questions に追加する。",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
        "execution": {"timeout_s": 5, "retry_safety": "idempotent"},
        "effects": [{"kind": "none"}],
    },
    {
        "name": "state.question.resolve",
        "category": "state",
        "spec": {
            "description": "working_state.open_questions から該当項目を削除する(完全一致)。",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
        "execution": {"timeout_s": 5, "retry_safety": "idempotent"},
        "effects": [{"kind": "none"}],
    },
    {
        "name": "state.next_actions.set",
        "category": "state",
        "spec": {
            "description": "working_state.next_actions を丸ごと置き換える。",
            "parameters": {
                "type": "object",
                "properties": {"actions": {"type": "array", "items": {"type": "string"}}},
                "required": ["actions"],
            },
        },
        "execution": {"timeout_s": 5, "retry_safety": "idempotent"},
        "effects": [{"kind": "none"}],
    },
    {
        "name": "state.extra.set",
        "category": "state",
        "spec": {
            "description": "working_state.extra にパス指定で任意のアプリ固有状態(フラグ・変数)を書き込む。",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "value": {"description": "任意のJSON値"}},
                "required": ["path", "value"],
            },
        },
        "discovery": {"embedding_text": "状態 変数 フラグ 保存 記録 セット flag variable"},
        "execution": {"timeout_s": 5, "retry_safety": "idempotent"},
        "effects": [{"kind": "none"}],
    },
    {
        "name": "state.extra.get",
        "category": "state",
        "spec": {
            "description": "working_state.extra からパス指定で値を読む。",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
        "execution": {"timeout_s": 5, "retry_safety": "pure"},
        "effects": [{"kind": "none"}],
    },
]

_HANDLERS = {
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


def install_state(session) -> None:
    """Register the bundled working-state capabilities. The working state
    is projected automatically by ``WorkingStateSection`` whenever it is
    part of ``config.projection.sections`` (the default)."""
    for definition in STATE_CAPABILITY_DEFS:
        if definition["name"] not in session.registry:
            session.registry.register(definition, handler=_HANDLERS[definition["name"]])
