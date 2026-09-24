"""Handlers of the ``meta`` pack (``meta.tool.find``, ``meta.artifact.peek``,
``meta.history.search``) and the opt-in ``spawn`` pack (``meta.agent.spawn``,
``meta.agent.join``).

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
            hits.append(f"[{event.sequence}] {event.type}: {blob}")
            if len(hits) >= k:
                break
    return hits or [f"No ledger events matched \"{query}\"."]


SPAWN_NAME = "meta.agent.spawn"
JOIN_NAME = "meta.agent.join"
ASK_NAME = "meta.user.ask"
# Wide enough for any real fan-out; narrow enough that a confused model
# cannot open fifty model streams in one call.
MAX_FANOUT = 8
DEFAULT_KERNEL = ("You are a focused sub-agent. Complete the task, then call finish(result) "
                  "with the outcome. You cannot ask the user anything.")


def _share(limit: Any, used: float, n: int) -> Any:
    """Each child's slice of what the parent has left of one limit."""
    return None if limit is None else type(limit)(max(0, limit - used) / n)


def child_session(parent: Any, spec: dict[str, Any], n: int, restored: Any = None) -> Any:
    """One sub-agent: an ordinary Run in the parent's ledger, so it is
    auditable, resumable and recoverable by the machinery that already
    exists. ``restored`` reattaches a child spawned by an earlier invocation
    of this same command."""
    from ..session import Session

    # No scope means everything but spawn itself (no recursive swarm by
    # default). Always a subset(), so the parent's deny-list carries over.
    scope = (spec.get("tool_scope")
             or [c.name for c in parent.registry if c.name not in (SPAWN_NAME, JOIN_NAME)])
    registry = parent.registry.subset(scope)
    registry.disable(ASK_NAME)  # a sub-agent has no user to ask

    cfg = copy.deepcopy(parent.config)
    cfg.mode = "job"
    cfg.result_schema = None  # the parent's finish schema is not the child's contract
    cfg.budget.max_steps = spec.get("max_steps") or 15
    used = parent.budget
    # Sub-agents still outstanding have spent from the same allowance but
    # have not been charged back yet; a new child must not be handed it twice.
    live = [c.budget for c in parent.children]
    spent = used.prompt_tokens + used.completion_tokens + sum(
        b.prompt_tokens + b.completion_tokens for b in live)
    cfg.budget.max_tokens = _share(cfg.budget.max_tokens, spent, n)
    cfg.budget.max_cost = _share(cfg.budget.max_cost, used.cost + sum(b.cost for b in live), n)
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


def _spawned(ctx: ToolContext, command_id: Optional[str] = None) -> Any:
    """This run's ``run_spawned`` events, newest last."""
    return [e for e in ctx.ledger.iter_run(ctx.run.id)
            if e.type == "run_spawned" and (command_id is None or e.data["command_id"] == command_id)]


def _lookup(ctx: ToolContext, parent: Any, run_ids: Optional[list[str]]) -> list[tuple[Any, dict]]:
    """The (child, spec) pairs a join should act on. Outstanding children are
    the live sessions; anything already collected is rebuilt from its
    snapshot, so a join can still read a result the nudge announced."""
    outstanding = {c.run.id: c for c in parent.children}
    specs: dict[str, dict[str, Any]] = {}
    for event in _spawned(ctx):
        command = ctx.run.commands.get(event.data["command_id"])
        for spec, run_id in zip((command.arguments.get("tasks") or []) if command else [],
                                event.data["child_run_ids"]):
            specs[run_id] = spec
    if run_ids is None:
        return [(child, specs.get(rid, {})) for rid, child in outstanding.items()]
    pairs = []
    for run_id in run_ids:
        if run_id not in specs:
            raise ValueError(f"{run_id} is not a sub-agent of this run")
        spec = specs[run_id]
        child = outstanding.get(run_id)
        if child is None:
            child = child_session(parent, spec, 1, restored=ctx.ledger.load_snapshot(run_id))
        pairs.append((child, spec))
    return pairs


async def _join_children(parent: Any, pairs: list[tuple[Any, dict]], resolution: Optional[str],
                         *, cancel: bool) -> Any:
    """Drive every child to a terminal state (or stop it), then report.

    Returns an ``ApprovalRequest`` instead when a child is waiting on one:
    the runtime parks this command and re-invokes it with the decision.
    """
    from ..run import TERMINAL_STATES

    children = [child for child, _ in pairs]
    # The forwarded approval belongs to the first child still waiting: the
    # ones before it are terminal, or they would have been forwarded first.
    if resolution is not None:
        waiting = next((c for c in children if c.run.state == "WAITING_FOR_APPROVAL"), None)
        if waiting is not None:
            waiting.resolve_approval(resolution)
    spent = [(c.budget.prompt_tokens, c.budget.completion_tokens) for c in children]

    for child in children:
        if cancel:
            child.interrupt()
        else:
            child._drive()
    await asyncio.gather(*(c._driver for c in children if c._driver is not None),
                         return_exceptions=True)

    for child in children:
        if child.run.state == "WAITING_FOR_USER":
            child.cancel("a sub-agent has no user to ask")
        elif cancel and child.run.state not in TERMINAL_STATES:
            child.cancel("cancelled by the parent")
    for child, (prompt, completion) in zip(children, spent):
        parent.budget.note_usage(child.budget.prompt_tokens - prompt,
                                 child.budget.completion_tokens - completion, parent.config)

    waiting = next((c for c in children if c.run.state == "WAITING_FOR_APPROVAL"), None)
    if waiting is not None:
        # Park this command on the child's approval; the host resolves it on
        # the root session and the runtime re-invokes us with the decision.
        request = waiting.run.pending_approval
        return replace(request, reason=f"sub-agent {waiting.run.id}: {request.reason}")
    for child in children:
        if child.run.state in TERMINAL_STATES and child in parent.children:
            parent.children.remove(child)
    return [_entry(child, spec) for child, spec in pairs]


async def _spawn(ctx: ToolContext, tasks: list[dict[str, Any]], background: bool = False) -> Any:
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
    known = next((e.data["child_run_ids"] for e in _spawned(ctx, ctx.command_id)), None)
    if known is None:
        children = [child_session(parent, spec, len(tasks)) for spec in tasks]
        parent.children.extend(children)
        ctx.ledger.append(ctx.run.id, "run_spawned", {
            "command_id": ctx.command_id, "background": background,
            "child_run_ids": [c.run.id for c in children]})
        for child, spec in zip(children, tasks):
            child._drive(spec["task"])
    else:
        outstanding = {c.run.id: c for c in parent.children}
        children = [outstanding.get(rid) or child_session(parent, spec, len(tasks),
                                                          restored=ctx.ledger.load_snapshot(rid))
                    for spec, rid in zip(tasks, known)]

    if background:
        # The loop head tends them from here: it nudges the parent when one
        # finishes, and meta.agent.join collects the results.
        return [{"run_id": c.run.id, "state": c.run.state} for c in children]
    return await _join_children(parent, list(zip(children, tasks)), ctx.resolution, cancel=False)


async def _join(ctx: ToolContext, run_ids: Optional[list[str]] = None, cancel: bool = False) -> Any:
    parent = ctx.session
    if parent is None:
        raise RuntimeError("join requires a session context")
    pairs = _lookup(ctx, parent, run_ids)
    if not pairs:
        return "No sub-agents are outstanding."
    return await _join_children(parent, pairs, ctx.resolution, cancel=cancel)


META_HANDLERS = {
    "meta.tool.find": _find_tools,
    "meta.artifact.peek": _peek,
    "meta.history.search": _search_history,
}

SPAWN_HANDLERS = {"meta.agent.spawn": _spawn, "meta.agent.join": _join}
