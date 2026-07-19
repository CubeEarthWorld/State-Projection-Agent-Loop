"""Resident meta capabilities: ``find_tools``, ``peek``, and
``search_history`` are always present; ``spawn`` is opt-in via
``install_spawn``.

There is no ``done`` capability anymore (P0-3): completion is
``Decision.finish``, a property of the model's response handled directly by
the session loop, not something routed through the runtime like any other
call. See :func:`state_projection_loop.llm.extract_finish`.
"""
from __future__ import annotations

import copy
from typing import Any, Optional

from ..artifacts import ref as artifact_ref
from ..capability import ToolContext
from ..registry import Registry

FIND_TOOLS_DEF: dict[str, Any] = {
    "name": "meta.tool.find",
    "category": "meta",
    "card": {
        "summary": "ツール台帳を自然文で検索し、該当ツールのカード一覧を返す",
        "signature": "find_tools(query: str, category: str | None = None, k: int = 8) -> list[ToolCard]",
        "tags": ["meta", "検索"],
    },
    "spec": {
        "description": "Search the capability registry with a natural-language query and return matching cards.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "やりたいことを自然文で"},
                "category": {"type": ["string", "null"], "description": "目次のカテゴリで絞り込み"},
                "k": {"type": "integer", "default": 8, "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
        },
        "usage_notes": "自動候補に必要なツールが見当たらない時に使う。目次のカテゴリ名で絞れる。",
    },
    "discovery": {"pinned": True, "no_embed": True},
    "execution": {"timeout_s": 10, "retry_safety": "pure"},
    "effects": [{"kind": "none"}],
}


def _find_tools(ctx: ToolContext, query: str, category: Optional[str] = None, k: int = 8) -> Any:
    results = ctx.search.search(query, category=category, k=k, layer=3)
    if not results:
        toc = ctx.registry.toc_text()
        return f"No tools matched {query!r}. Categories: {toc or '(none)'}"
    if ctx.session is not None:
        ctx.session._activate_tools([s.tool.name for s in results])
    return [
        {"name": s.tool.name, "category": s.tool.category, "card": s.tool.card_text(), "score": round(s.score, 3)}
        for s in results
    ]


PEEK_DEF: dict[str, Any] = {
    "name": "meta.artifact.peek",
    "category": "meta",
    "card": {
        "summary": "アーティファクト参照の中身を部分閲覧する",
        "signature": 'peek(artifact: {"$artifact": str}, query: str | null = None, range: str | null = None) -> str',
        "tags": ["meta", "参照"],
    },
    "spec": {
        "description": "Partially inspect the value stored behind an artifact reference ({\"$artifact\": \"...\"}).",
        "parameters": {
            "type": "object",
            "properties": {
                "artifact": {"type": "object", "description": "構造化参照 {\"$artifact\": \"art_...\"}"},
                "query": {"type": ["string", "null"], "description": "中身から探したい内容"},
                "range": {"type": ["string", "null"], "description": "行範囲(例 '10-40')やキーパス(例 'items[0].name')"},
            },
            "required": ["artifact"],
        },
        "usage_notes": "プレビューで足りない時のみ使う。全量展開は避け、queryかrangeで絞る。",
    },
    "discovery": {"pinned": True, "no_embed": True},
    "execution": {"timeout_s": 10, "retry_safety": "pure", "resolve_handles": False},
    "effects": [{"kind": "none"}],
}


def _peek(ctx: ToolContext, artifact: dict, query: Optional[str] = None, range: Optional[str] = None) -> str:  # noqa: A002
    from ..artifacts import is_ref

    if not is_ref(artifact):
        return f"Error: {artifact!r} is not a valid artifact reference; expected {{'$artifact': '<id>'}}"
    return ctx.store.peek(artifact["$artifact"], query=query, range_=range)


SEARCH_HISTORY_DEF: dict[str, Any] = {
    "name": "meta.history.search",
    "category": "meta",
    "card": {
        "summary": "折り畳まれた過去の会話をイベント台帳から検索する",
        "signature": "search_history(query: str, k: int = 10) -> list[str]",
        "tags": ["meta", "検索", "履歴"],
    },
    "spec": {
        "description": (
            "Search the append-only event ledger for this run, including messages folded out of the "
            "live conversation by compaction. Use when working_state doesn't have enough detail."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
        },
    },
    "discovery": {"pinned": True, "no_embed": True},
    "execution": {"timeout_s": 10, "retry_safety": "pure"},
    "effects": [{"kind": "none"}],
}


def _search_history(ctx: ToolContext, query: str, k: int = 10) -> Any:
    if ctx.ledger is None or ctx.run is None:
        return "History search is unavailable (no ledger configured for this session)."
    q = query.lower()
    hits: list[str] = []
    for event in ctx.ledger.iter_run(ctx.run.id):
        blob = str(event.data)
        if q in blob.lower():
            hits.append(f"[{event.sequence}] {event.type}: {blob[:300]}")
            if len(hits) >= k:
                break
    return hits or [f"No ledger events matched {query!r}."]


SPAWN_DEF: dict[str, Any] = {
    "name": "meta.agent.spawn",
    "category": "meta",
    "card": {
        "summary": "サブエージェントを起動しタスクを委任、結果アーティファクトを受け取る",
        "signature": "spawn(task: str, kernel: str | None = None, tool_scope: list | None = None, model: str | None = None, max_steps: int = 15) -> Any",
        "tags": ["meta", "swarm", "サブエージェント"],
    },
    "spec": {
        "description": (
            "Run a sub-agent with its own independent context on the given task. Parent and child share "
            "ONLY the task string (input) and the result (output); artifacts must be explicitly moved."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "委任するタスクの完全な記述(子は親の文脈を一切見られない)"},
                "kernel": {"type": ["string", "null"], "description": "子のシステムプロンプト(省略時は汎用ジョブカーネル)"},
                "tool_scope": {
                    "type": ["array", "null"], "items": {"type": "string"},
                    "description": "子に許可するツール名/カテゴリ(例 ['web/*','file'])。省略時は親と同じ台帳",
                },
                "model": {"type": ["string", "null"], "description": "子で使うモデル名(spawn_llm_factory が必要)"},
                "max_steps": {"type": "integer", "default": 15, "minimum": 1, "maximum": 100},
            },
            "required": ["task"],
        },
        "usage_notes": "自己完結したタスクの記述を渡すこと。親の会話内容は共有されない。",
    },
    "discovery": {"pinned": True, "no_embed": True},
    "execution": {"timeout_s": 600, "retry_safety": "never_retry"},
    "effects": [{"kind": "external", "resource": "subagent:*"}],
}


def _spawn(
    ctx: ToolContext, task: str, kernel: Optional[str] = None, tool_scope: Optional[list[str]] = None,
    model: Optional[str] = None, max_steps: int = 15,
) -> Any:
    from ..session import Session

    parent = ctx.session
    if parent is None:
        raise RuntimeError("spawn requires a session context")
    if model is not None and parent.spawn_llm_factory is None:
        raise RuntimeError("spawn(model=...) requires Session(spawn_llm_factory=...)")
    llm = parent.spawn_llm_factory(model) if parent.spawn_llm_factory else parent.llm

    child_registry = parent.registry.subset(tool_scope) if tool_scope else Registry()
    if not tool_scope:
        for cap in parent.registry:
            if cap.name != "meta.agent.spawn":  # no recursive swarm by default
                child_registry.register(cap, replace=True)

    child_config = copy.deepcopy(parent.config)
    child_config.mode = "job"
    child_config.budget.max_steps = max_steps
    child_config.persistence.ledger_directory = None  # child ledger is not persisted independently

    child = Session(
        llm=llm,
        kernel=kernel or "You are a focused sub-agent. Complete the task, then call finish(result) with the outcome.",
        config=child_config,
        registry=child_registry,
        embedder=getattr(parent.search, "embedder", None),
        summarizer=parent.summarizer,
        policy=parent.policy,
    )
    return child.run_job(task)


def ensure_meta_tools(registry: Registry) -> None:
    """Register the resident meta capabilities if absent."""
    if "meta.tool.find" not in registry:
        registry.register(FIND_TOOLS_DEF, handler=_find_tools)
    if "meta.artifact.peek" not in registry:
        registry.register(PEEK_DEF, handler=_peek)
    if "meta.history.search" not in registry:
        registry.register(SEARCH_HISTORY_DEF, handler=_search_history)


def install_spawn(registry: Registry) -> None:
    """Opt-in sub-agent capability for swarm-style setups."""
    if "meta.agent.spawn" not in registry:
        registry.register(SPAWN_DEF, handler=_spawn)
