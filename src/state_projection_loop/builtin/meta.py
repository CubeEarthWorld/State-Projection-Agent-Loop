"""Resident meta tools (spec §12): find_tools and peek are always present;
done is added in job mode; spawn (§11) is opt-in via ``install_spawn``.
"""
from __future__ import annotations

import copy
from typing import Any, Optional

from ..registry import Registry
from ..tooldef import ToolContext, ToolDef

FIND_TOOLS_DEF: dict[str, Any] = {
    "name": "find_tools",
    "category": "meta",
    "card": {
        "summary": "ツール台帳を自然文で検索し、該当ツールのカード一覧を返す",
        "signature": "find_tools(query: str, category: str | None = None, k: int = 8) -> list[ToolCard]",
        "tags": ["meta", "検索"],
    },
    "spec": {
        "description": "Search the tool registry with a natural-language query and return matching tool cards.",
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
    "execution": {"timeout_s": 10, "parallel_safe": True},
}


def _find_tools(ctx: ToolContext, query: str, category: Optional[str] = None, k: int = 8) -> Any:
    results = ctx.search.search(query, category=category, k=k, layer=3)
    if not results:
        toc = ctx.registry.toc_text()
        return f"No tools matched {query!r}. Categories: {toc or '(none)'}"
    if ctx.session is not None:
        ctx.session._activate_tools([s.tool.name for s in results])
    return [
        {
            "name": s.tool.name,
            "category": s.tool.category,
            "card": s.tool.card_text(),
            "score": round(s.score, 3),
        }
        for s in results
    ]


PEEK_DEF: dict[str, Any] = {
    "name": "peek",
    "category": "meta",
    "card": {
        "summary": "ハンドル($hN)の中身を部分閲覧する",
        "signature": "peek(handle: str, query: str | null = None, range: str | null = None) -> str",
        "tags": ["meta", "参照"],
    },
    "spec": {
        "description": "Partially inspect the value stored behind a handle ($hN).",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "query": {"type": ["string", "null"], "description": "中身から探したい内容"},
                "range": {"type": ["string", "null"], "description": "行範囲(例 '10-40')やキーパス(例 'items[0].name')"},
            },
            "required": ["handle"],
        },
        "usage_notes": "要約プレビューで足りない時のみ使う。全量展開は避け、queryかrangeで絞る。",
    },
    "discovery": {"pinned": True, "no_embed": True},
    "execution": {"timeout_s": 10, "parallel_safe": True, "resolve_handles": False},
}


def _peek(ctx: ToolContext, handle: str, query: Optional[str] = None, range: Optional[str] = None) -> str:  # noqa: A002
    return ctx.store.peek(handle, query=query, range_=range)


DONE_DEF: dict[str, Any] = {
    "name": "done",
    "category": "meta",
    "card": {
        "summary": "ジョブを完了し最終結果を返す",
        "signature": "done(result: Any) -> None",
        "tags": ["meta", "終了"],
    },
    "spec": {
        "description": "Finish the current job and hand back the final result. Ends the loop.",
        "parameters": {
            "type": "object",
            "properties": {"result": {"description": "最終成果物(文字列・オブジェクト・ハンドル参照など)"}},
            "required": ["result"],
        },
        "usage_notes": "jobモードの終了条件。作業が完了したら必ず呼ぶ。",
    },
    "discovery": {"pinned": True, "no_embed": True},
    "execution": {"timeout_s": 5},
}


def _done(ctx: ToolContext, result: Any = None) -> str:
    if ctx.session is not None:
        ctx.session._finish(result)
    return "Job finished."


SPAWN_DEF: dict[str, Any] = {
    "name": "spawn",
    "category": "meta",
    "card": {
        "summary": "サブエージェントを起動しタスクを委任、結果ハンドルを受け取る",
        "signature": "spawn(task: str, kernel: str | None = None, tool_scope: list | None = None, model: str | None = None, max_steps: int = 15) -> Any",
        "tags": ["meta", "swarm", "サブエージェント"],
    },
    "spec": {
        "description": (
            "Run a sub-agent with its own independent context on the given task. "
            "Parent and child share ONLY the task string (input) and the result (output) — invariant I9."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "委任するタスクの完全な記述(子は親の文脈を一切見られない)"},
                "kernel": {"type": ["string", "null"], "description": "子のシステムプロンプト(省略時は汎用ジョブカーネル)"},
                "tool_scope": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "子に許可するツール名/カテゴリ(例 ['web/*','file'])。省略時は親と同じ台帳",
                },
                "model": {"type": ["string", "null"], "description": "子で使うモデル名(spawn_llm_factory が必要)"},
                "max_steps": {"type": "integer", "default": 15, "minimum": 1, "maximum": 100},
            },
            "required": ["task"],
        },
        "usage_notes": "自己完結したタスクの記述を渡すこと。親の会話内容は共有されない。結果が大きい場合はハンドルとして返る。",
    },
    "discovery": {"pinned": True, "no_embed": True},
    "execution": {"timeout_s": 600},
}


def _spawn(
    ctx: ToolContext,
    task: str,
    kernel: Optional[str] = None,
    tool_scope: Optional[list[str]] = None,
    model: Optional[str] = None,
    max_steps: int = 15,
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
        for td in parent.registry:
            if td.name != "spawn":  # no recursive swarm by default
                child_registry.register(td, replace=True)

    child_config = copy.deepcopy(parent.config)
    child_config.mode = "job"
    child_config.budget.max_steps = max_steps
    child_config.log_path = None

    child = Session(
        llm=llm,
        kernel=kernel or "You are a focused sub-agent. Complete the task, then call done(result) with the outcome.",
        config=child_config,
        registry=child_registry,
        embedder=getattr(parent.search, "embedder", None),
        summarizer=parent.summarizer,
    )
    parent.logger.log("spawn", task=task[:200], tool_scope=tool_scope, model=model)
    return child.run_job(task)


def ensure_meta_tools(registry: Registry, *, mode: str = "chat") -> None:
    """Register the resident meta tools if absent (find_tools, peek; done in job mode)."""
    if "find_tools" not in registry:
        registry.register(FIND_TOOLS_DEF, handler=_find_tools)
    if "peek" not in registry:
        registry.register(PEEK_DEF, handler=_peek)
    if mode == "job" and "done" not in registry:
        registry.register(DONE_DEF, handler=_done)


def install_spawn(registry: Registry) -> None:
    """Opt-in sub-agent tool (§11) for swarm-style setups."""
    if "spawn" not in registry:
        registry.register(SPAWN_DEF, handler=_spawn)
