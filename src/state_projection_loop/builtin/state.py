"""State-as-tools (spec §7): goal / flags / variables live *outside* the
core, provided here as ordinary bundled tools plus a bundled section.

Editors of state are exactly three: user code (``session.state``), the LLM
(via these tools), and the session seed. A game master registers all of
this; a simple support bot registers none — the core is identical.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from ..handles import truncate_to_tokens
from ..messages import Message, SYSTEM
from ..projection import TurnContext
from ..tooldef import ToolContext


def _walk(state: dict, path: str, *, create: bool = False) -> tuple[Any, str]:
    parts = [p for p in path.split(".") if p]
    if not parts:
        raise ValueError("empty state path")
    node = state
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            if not create:
                raise KeyError(path)
            node[part] = {}
        node = node[part]
    return node, parts[-1]


def _state_set(ctx: ToolContext, path: str, value: Any = None) -> str:
    node, leaf = _walk(ctx.state, path, create=True)
    node[leaf] = value
    return f"state.{path} = {json.dumps(value, ensure_ascii=False, default=str)}"


def _state_get(ctx: ToolContext, path: str) -> Any:
    try:
        node, leaf = _walk(ctx.state, path)
        return node[leaf]
    except KeyError:
        return f"(not set: {path})"


def _state_delete(ctx: ToolContext, path: str) -> str:
    try:
        node, leaf = _walk(ctx.state, path)
        del node[leaf]
        return f"deleted state.{path}"
    except KeyError:
        return f"(not set: {path})"


def _set_goal(ctx: ToolContext, text: str) -> str:
    ctx.state["goal"] = text
    return f"goal set: {text}"


def _set_flag(ctx: ToolContext, name: str, value: Any = True) -> str:
    ctx.state.setdefault("flags", {})[name] = value
    return f"flag {name} = {value!r}"


STATE_TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "state_set",
        "category": "state",
        "spec": {
            "description": "構造化状態にパス指定で値を書き込む(例 path='party.hp', value=12)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "ドット区切りパス"},
                    "value": {"description": "任意のJSON値"},
                },
                "required": ["path", "value"],
            },
        },
        "discovery": {"embedding_text": "状態 変数 保存 記録 セット 更新 remember store variable"},
        "execution": {"timeout_s": 5},
    },
    {
        "name": "state_get",
        "category": "state",
        "spec": {
            "description": "構造化状態からパス指定で値を読む。",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        "discovery": {"embedding_text": "状態 変数 参照 読む 確認 get read variable"},
        "execution": {"timeout_s": 5, "parallel_safe": True},
    },
    {
        "name": "state_delete",
        "category": "state",
        "spec": {
            "description": "構造化状態からパス指定で値を削除する。",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        "execution": {"timeout_s": 5},
    },
    {
        "name": "set_goal",
        "category": "state",
        "spec": {
            "description": "現在の目標(ゴール)を設定する。state_viewと候補検索クエリに反映される。",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
        "discovery": {"embedding_text": "目標 ゴール クリア条件 目的 goal objective"},
        "execution": {"timeout_s": 5},
    },
    {
        "name": "set_flag",
        "category": "state",
        "spec": {
            "description": "フラグ(名前付き真偽値・任意値)を設定する。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"description": "省略時 true", "default": True},
                },
                "required": ["name"],
            },
        },
        "discovery": {"embedding_text": "フラグ 立てる イベント 進行 flag event trigger"},
        "execution": {"timeout_s": 5},
    },
]

_HANDLERS = {
    "state_set": _state_set,
    "state_get": _state_get,
    "state_delete": _state_delete,
    "set_goal": _set_goal,
    "set_flag": _set_flag,
}


class StateViewSection:
    """Projects the state essentials each turn (volatile), structurally
    preventing goal drift (§7). A custom ``template(state) -> str`` may
    replace the default rendering."""

    name = "state_view"
    cache_class = "volatile"

    def __init__(self, template=None, *, max_tokens: int = 600) -> None:
        self.template = template
        self.max_tokens = max_tokens

    def render(self, turn: TurnContext) -> list[Message]:
        state = turn.state
        if not state:
            return []
        if self.template is not None:
            body = self.template(state)
        else:
            parts = []
            if "goal" in state:
                parts.append(f"goal: {state['goal']}")
            if state.get("flags"):
                parts.append("flags: " + json.dumps(state["flags"], ensure_ascii=False, default=str))
            rest = {k: v for k, v in state.items() if k not in ("goal", "flags")}
            if rest:
                parts.append(json.dumps(rest, ensure_ascii=False, default=str))
            body = "\n".join(parts)
        if not body:
            return []
        return [Message(role=SYSTEM, content="[State]\n" + truncate_to_tokens(body, self.max_tokens))]


def install_state(session, *, view: bool = True, template=None) -> None:
    """Register the bundled state tools (and optionally the state_view
    section, inserted just before candidates)."""
    for definition in STATE_TOOL_DEFS:
        if definition["name"] not in session.registry:
            session.registry.register(definition, handler=_HANDLERS[definition["name"]])
    if view and session.projection.get("state_view") is None:
        session.projection.insert_before("candidates", StateViewSection(template))
