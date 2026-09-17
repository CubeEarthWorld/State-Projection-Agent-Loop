"""Handlers of the ``meta`` pack (``meta.tool.find``, ``meta.artifact.peek``,
``meta.history.search``) and the opt-in ``spawn`` pack (``meta.agent.spawn``).

There is no ``done`` capability: completion is ``Decision.finish``, a
property of the model's response handled directly by the session loop, not
something routed through the runtime like any other call. See
:func:`state_projection_loop.llm.extract_finish`.
"""
from __future__ import annotations

import copy
from typing import Any, Optional

from ..artifacts import is_ref
from ..capability import ToolContext
from ..registry import Registry


def _find_tools(ctx: ToolContext, query: str, category: Optional[str] = None, k: int = 8) -> Any:
    results = ctx.search.search(query, category=category, k=k, layer=3)
    if not results:
        toc = ctx.registry.toc_text()
        return f"No tools matched {query!r}. Categories: {toc or '(none)'}"
    if ctx.session is not None:
        ctx.session.activate([s.tool.name for s in results])
    return [
        {"name": s.tool.name, "category": s.tool.category, "card": s.tool.card_text(), "score": round(s.score, 3)}
        for s in results
    ]


def _peek(ctx: ToolContext, artifact: dict, query: Optional[str] = None, range: Optional[str] = None) -> str:  # noqa: A002
    if not is_ref(artifact):
        return f"Error: {artifact!r} is not a valid artifact reference; expected {{'$artifact': '<id>'}}"
    return ctx.store.peek(artifact["$artifact"], query=query, range_=range)


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


async def _spawn(
    ctx: ToolContext, task: str, kernel: Optional[str] = None, tool_scope: Optional[list[str]] = None,
    model: Optional[str] = None, max_steps: int = 15, checklist_ids: Optional[list[str]] = None,
) -> Any:
    from ..session import Session

    parent = ctx.session
    if parent is None:
        raise RuntimeError("spawn requires a session context")
    if model is not None and parent.spawn_llm_factory is None:
        raise RuntimeError("spawn(model=...) requires Session(spawn_llm_factory=...)")
    documents = [parent.checklists.execute("export", id=i)["checklists"][0] for i in (checklist_ids or [])]
    if len(set(checklist_ids or [])) != len(checklist_ids or []):
        raise ValueError("Duplicate checklist_ids")
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
        seed={"checklists": {"version": 1, "checklists": documents}},
        policy=parent.policy,
    )
    result = await child.arun_job(task)
    if checklist_ids is not None:
        return {"result": result, "checklists": child.checklists.to_dict()}
    return result


META_HANDLERS = {
    "meta.tool.find": _find_tools,
    "meta.artifact.peek": _peek,
    "meta.history.search": _search_history,
}

SPAWN_HANDLERS = {"meta.agent.spawn": _spawn}
