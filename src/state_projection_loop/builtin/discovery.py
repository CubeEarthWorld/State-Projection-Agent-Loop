"""Category browsing with ledger-backed, request-wide exploration progress."""
from __future__ import annotations

from typing import Any, Optional

from ..context import ToolContext


DISCOVERY_EVENT = "tool_discovery"


def exploration(ctx: ToolContext) -> tuple[set[str], set[str], dict[str, dict]]:
    """Qualified names keep an updated capability discoverable again."""
    if ctx.ledger is None or ctx.run is None:
        raise RuntimeError("Tool discovery requires an event ledger and run")
    seen: set[str] = set()
    rejected: set[str] = set()
    responses: dict[str, dict] = {}
    # ponytail: replay the run on each browse; add a sequence cache if profiling warrants it.
    for event in ctx.ledger.iter_run(ctx.run.id):
        if event.type == "user_input":
            seen.clear()
            rejected.clear()
            responses.clear()
        elif event.type == DISCOVERY_EVENT:
            data = event.data
            if data["action"] == "reset":
                seen.clear()
                rejected.clear()
            seen.update(data.get("shown", []))
            rejected.update(data.get("rejected", []))
            responses[data["command_id"]] = data
    return seen, rejected, responses


def _categories(ctx: ToolContext, seen: set[str], rejected: set[str]) -> list[dict]:
    categories: dict[str, dict] = {}
    for tool in ctx.registry:
        category = tool.category or "misc"
        item = categories.setdefault(category, {"category": category, "total": 0,
                                                "pinned": 0, "remaining": 0, "rejected": 0})
        item["total"] += 1
        item["pinned"] += int(tool.discovery.pinned)
        item["remaining"] += int(not tool.discovery.pinned and tool.qualified_name not in seen
                                 and tool.qualified_name not in rejected)
        item["rejected"] += int(tool.qualified_name in rejected)
    return [categories[name] for name in sorted(categories)]


async def find_tools(ctx: ToolContext, query: Optional[str] = None, category: Optional[str] = None,
                     k: int = 8, action: str = "categories", name: Optional[str] = None,
                     reason: str = "") -> Any:
    """Listing never activates tools; only an explicit describe does."""
    if ctx.session is None:
        raise RuntimeError("Tool discovery requires a session")
    if not 1 <= k <= 50:
        raise ValueError("k must be between 1 and 50")
    seen, rejected, responses = exploration(ctx)
    previous = responses.get(ctx.command_id) if ctx.command_id else None
    if previous is not None:
        if previous["action"] == "describe":
            ctx.session.activate([previous["name"]])
            ctx.session.runtime.seen_specs.add(previous["name"])
        elif previous["action"] == "reject":
            ctx.session.deactivate([previous["name"]])
        return previous["response"]

    shown: list[str] = []
    declined: list[str] = []
    if action == "reset":
        seen.clear()
        rejected.clear()
        response = {"categories": _categories(ctx, seen, rejected), "reset": True}
    elif action == "categories":
        response = {"categories": _categories(ctx, seen, rejected)}
    elif action in ("describe", "reject"):
        tool = ctx.registry.get(name or "")
        if tool is None:
            raise ValueError(f"Unknown or disabled tool: {name}")
        if action == "reject":
            if tool.discovery.pinned:
                raise ValueError("Pinned tools cannot be rejected from discovery")
            declined = [tool.qualified_name]
            rejected.update(declined)
            response = {"rejected": tool.name, "reason": reason,
                        "categories": _categories(ctx, seen, rejected)}
        else:
            if tool.qualified_name in rejected:
                raise ValueError("This tool was rejected; reset exploration to reconsider it")
            shown = [tool.qualified_name]
            response = {"name": tool.name, "category": tool.category or "misc",
                        "spec": tool.spec_text()}
    elif action in ("list", "search"):
        available = [t for t in ctx.registry if not t.discovery.pinned
                     and t.qualified_name not in seen and t.qualified_name not in rejected]
        if action == "list":
            if not category:
                raise ValueError("list requires category; use categories to see the index")
            prefix = category.rstrip("/*").rstrip("/")
            matching = [t for t in available if category == "*" or (t.category or "misc") == prefix
                        or (t.category or "misc").startswith(prefix + "/")]
            matching.sort(key=lambda t: t.name)
            tools = matching[:k]
            remaining_matches = len(matching) - len(tools)
        else:
            if not query or not query.strip():
                raise ValueError("search requires a non-empty query")
            allowed = {t.name for t in available}
            excluded = {t.name for t in ctx.registry if t.name not in allowed}
            results = ctx.search.search(query, category=category, k=k + 1, layer=3, exclude=excluded)
            tools = [s.tool for s in results[:k]]
            remaining_matches = None
        shown = [t.qualified_name for t in tools]
        seen.update(shown)
        response = {"tools": [{"name": t.name, "category": t.category or "misc",
                               "summary": t.card.summary, "tags": t.card.tags} for t in tools],
                    "has_more": remaining_matches > 0 if action == "list" else len(results) > k,
                    "remaining_matches": remaining_matches,
                    "categories": _categories(ctx, seen, rejected),
                    "next": ("Repeat to see unseen tools, or describe a tool by name." if tools else
                             "No unseen matches. Choose another category/query; reset only to revisit tools.")}
    else:
        raise ValueError(f"Unknown discovery action: {action}")

    ctx.ledger.append(ctx.run.id, DISCOVERY_EVENT, {"command_id": ctx.command_id, "action": action,
                      "query": query, "category": category, "name": name, "reason": reason,
                      "shown": shown, "rejected": declined, "response": response})
    if action == "describe":
        ctx.session.activate([tool.name])
        ctx.session.runtime.seen_specs.add(tool.name)
    elif action == "reject":
        ctx.session.deactivate([tool.name])
    return response
