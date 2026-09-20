"""Handlers of the ``meta`` pack (``meta.tool.find``, ``meta.artifact.peek``,
``meta.history.search``) and the opt-in ``spawn`` pack (``meta.agent.spawn``).

There is no ``done`` capability: completion is ``Decision.finish``, a
property of the model's response handled directly by the session loop, not
something routed through the runtime like any other call. See
:func:`state_projection_loop.llm.extract_finish`.
"""
from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import replace
from typing import Any, Optional

from ..artifacts import is_ref
from ..context import ToolContext
from ..serialization import dumps


def _find_tools(ctx: ToolContext, query: str, category: Optional[str] = None, k: int = 8) -> Any:
    results = ctx.search.search(query, category=category, k=k, layer=3)
    if not results:
        toc = ctx.registry.toc_text()
        return f"No tools matched \"{query}\". Categories: {toc or '(none)'}"
    if ctx.session is not None:
        ctx.session.activate([s.tool.name for s in results])
    return [
        {"name": s.tool.name, "category": s.tool.category, "card": s.tool.card_text(), "score": round(s.score, 3)}
        for s in results
    ]


def _peek(ctx: ToolContext, artifact: dict, query: Optional[str] = None, range: Optional[str] = None) -> str:  # noqa: A002
    if not is_ref(artifact):
        return f'Error: {dumps(artifact)} is not a valid artifact reference; expected {{"$artifact": "<id>"}}'
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
    return hits or [f"No ledger events matched \"{query}\"."]


SPAWN_NAME = "meta.agent.spawn"
ASK_NAME = "meta.user.ask"
# Wide enough for any real fan-out; narrow enough that a confused model
# cannot open fifty model streams in one call.
MAX_FANOUT = 8
DEFAULT_KERNEL = ("You are a focused sub-agent. Complete the task, then call finish(result) "
                  "with the outcome. You cannot ask the user anything.")


def _share(limit: Any, used: float, n: int) -> Any:
    """Each child's slice of what the parent has left of one limit."""
    return None if limit is None else type(limit)(max(0, limit - used) / n)


def _child_session(parent: Any, spec: dict[str, Any], n: int, restored: Any = None) -> Any:
    """One sub-agent: an ordinary Run in the parent's ledger, so it is
    auditable, resumable and recoverable by the machinery that already
    exists. ``restored`` reattaches a child spawned by an earlier invocation
    of this same command."""
    from ..session import Session

    # No scope means everything but spawn itself (no recursive swarm by
    # default). Always a subset(), so the parent's deny-list carries over.
    scope = spec.get("tool_scope") or [c.name for c in parent.registry if c.name != SPAWN_NAME]
    registry = parent.registry.subset(scope)
    registry.disable(ASK_NAME)  # a sub-agent has no user to ask

    cfg = copy.deepcopy(parent.config)
    cfg.mode = "job"
    cfg.result_schema = None  # the parent's finish schema is not the child's contract
    cfg.budget.max_steps = spec.get("max_steps") or 15
    used = parent.budget
    cfg.budget.max_tokens = _share(cfg.budget.max_tokens, used.prompt_tokens + used.completion_tokens, n)
    cfg.budget.max_cost = _share(cfg.budget.max_cost, used.cost, n)
    cfg.budget.max_seconds = _share(cfg.budget.max_seconds, time.time() - used.started, 1)

    documents = [parent.checklists.execute("export", id=i)["checklists"][0]
                 for i in (spec.get("checklist_ids") or [])]
    model = spec.get("model")
    return Session(
        llm=parent.spawn_llm_factory(model) if parent.spawn_llm_factory else parent.llm,
        kernel=spec.get("kernel") or DEFAULT_KERNEL,
        config=cfg,
        registry=registry,
        builtins=(),  # whatever the scope carried, nothing re-installed behind it
        embedder=parent.search.embedder,
        seed=None if restored else {"checklists": {"version": 1, "checklists": documents}},
        policy=parent.policy,
        ledger=parent.ledger,
        memory=parent.memory,
        hooks=parent.runtime.hooks,
        spawn_llm_factory=parent.spawn_llm_factory,
        _restored=restored,
    )


async def _drive(child: Any, spec: dict[str, Any], resolution: Optional[str], fresh: bool) -> None:
    """Run one child to a terminal state or to its next pause."""
    from ..run import TERMINAL_STATES

    try:
        if resolution is not None and child.run.state == "WAITING_FOR_APPROVAL":
            child.resolve_approval(resolution)
        if child.run.state in TERMINAL_STATES or child.run.state.startswith("WAITING"):
            return
        await (child.arun_job(spec["task"]) if fresh else child.aresume())
    except Exception as exc:  # noqa: BLE001 — a child's failure is the parent's observation
        if child.run.state not in TERMINAL_STATES:
            child.cancel(f"{type(exc).__name__}: {exc}")


def _entry(child: Any, spec: dict[str, Any]) -> dict[str, Any]:
    reason = next((e.data.get("reason") for e in reversed(list(child.ledger.iter_run(child.run.id)))
                   if e.type == "run_state_changed"), "")
    entry: dict[str, Any] = {"run_id": child.run.id, "state": child.run.state,
                             "result": child.run.result if child.run.state == "COMPLETED" else None}
    if child.run.state != "COMPLETED":
        entry["error"] = reason
    if spec.get("checklist_ids"):
        entry["checklists"] = child.checklists.to_dict()
    return entry


async def _spawn(ctx: ToolContext, tasks: list[dict[str, Any]]) -> Any:
    from ..run import TERMINAL_STATES

    parent = ctx.session
    if parent is None:
        raise RuntimeError("spawn requires a session context")
    if not 1 <= len(tasks) <= MAX_FANOUT:
        raise ValueError(f"spawn takes 1 to {MAX_FANOUT} tasks, got {len(tasks)}")
    for spec in tasks:
        ids = spec.get("checklist_ids") or []
        # Check before exporting: a duplicated id made the export raise
        # IndexError on the second lookup instead of saying what was wrong.
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate checklist_ids")
        if spec.get("model") is not None and parent.spawn_llm_factory is None:
            raise RuntimeError("spawn(model=...) requires Session(spawn_llm_factory=...)")

    # Re-invoked after forwarding a child's approval? Pick the same children
    # back up out of the ledger instead of starting the work again.
    known = next((e.data["child_run_ids"] for e in ctx.ledger.iter_run(ctx.run.id)
                  if e.type == "run_spawned" and e.data["command_id"] == ctx.command_id), None)
    fresh = known is None
    if fresh:
        children = [_child_session(parent, spec, len(tasks)) for spec in tasks]
        ctx.ledger.append(ctx.run.id, "run_spawned", {
            "command_id": ctx.command_id, "child_run_ids": [c.run.id for c in children]})
    else:
        children = [_child_session(parent, spec, len(tasks), restored=ctx.ledger.load_snapshot(rid))
                    for spec, rid in zip(tasks, known)]

    # The forwarded approval belongs to the first child still waiting: the
    # ones before it are terminal, or they would have been forwarded first.
    forwarded = (next((c for c in children if c.run.state == "WAITING_FOR_APPROVAL"), None)
                 if ctx.resolution else None)
    spent = [(c.budget.prompt_tokens, c.budget.completion_tokens) for c in children]

    parent._children.extend(children)
    try:
        await asyncio.gather(*(
            _drive(child, spec, ctx.resolution if child is forwarded else None, fresh)
            for child, spec in zip(children, tasks)))
    finally:
        for child in children:
            parent._children.remove(child)
    for child, (prompt, completion) in zip(children, spent):
        parent.budget.note_usage(child.budget.prompt_tokens - prompt,
                                 child.budget.completion_tokens - completion, parent.config)

    for child in children:
        if child.run.state == "WAITING_FOR_USER":
            child.cancel("a sub-agent has no user to ask")
        elif child.run.state not in TERMINAL_STATES and child.run.state != "WAITING_FOR_APPROVAL":
            child.cancel("interrupted")
    waiting = next((c for c in children if c.run.state == "WAITING_FOR_APPROVAL"), None)
    if waiting is not None:
        # Park this command on the child's approval; the host resolves it on
        # the root session and the runtime re-invokes us with the decision.
        request = waiting.run.pending_approval
        return replace(request, reason=f"sub-agent {waiting.run.id}: {request.reason}")
    return [_entry(child, spec) for child, spec in zip(children, tasks)]


META_HANDLERS = {
    "meta.tool.find": _find_tools,
    "meta.artifact.peek": _peek,
    "meta.history.search": _search_history,
}

SPAWN_HANDLERS = {"meta.agent.spawn": _spawn}
