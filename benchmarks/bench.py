"""SPAL vs conventional preloading - same runtime, one variable.

Both arms run the *same* Session, the same tasks, the same model and the
same tools. The only difference is how capabilities reach the model:

  preload : every capability pinned, full spec resident every turn,
            discovery off. This is the conventional agent loop.
  spal    : nothing pinned; TOC + BM25/tag candidate cards (layer 1-2),
            meta.tool.find as the fallback (layer 3).

Holding the runtime fixed is deliberate: any difference is the context
strategy, not a different harness being a different harness.

    python -m benchmarks.bench --sizes 25,200 --repeat 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from state_projection_loop import Config, Session
from state_projection_loop.capability import build_capability_from_function

from benchmarks.adapter import PROVIDERS

KERNEL = (
    "You are a back-office operations assistant. Use the available tools to answer; "
    "never invent data. When you have the answer, call finish(result) with it. "
    "If no tool you can see fits, use meta.tool.find to search the registry before giving up."
)

# --------------------------------------------------------------------------
# Real capabilities - the ones the tasks actually need.
# --------------------------------------------------------------------------

_STOCK = {"tokyo": 42, "osaka": 17, "fukuoka": 8}
_ORDERS = {"A-1001": {"customer_id": "C-7", "total": 12800, "status": "shipped"}}
_CUSTOMERS = {"C-7": {"name": "Hoshino Kikaku", "email": "ops@hoshino-kikaku.example"}}
_REVENUE = [{"site": "tokyo", "revenue": 4200}, {"site": "osaka", "revenue": 3100},
            {"site": "fukuoka", "revenue": 1750}]


def stock_get(warehouse: str) -> dict:
    """Current stock on hand for one warehouse.

    Args:
        warehouse: Warehouse key: tokyo, osaka or fukuoka.
    """
    return {"warehouse": warehouse, "stock": _STOCK.get(warehouse.lower(), 0)}


def order_get(order_id: str) -> dict:
    """Look up one order by its id.

    Args:
        order_id: The order id, e.g. A-1001.
    """
    return {"order_id": order_id, **_ORDERS.get(order_id, {})}


def customer_get(customer_id: str) -> dict:
    """Fetch a customer record.

    Args:
        customer_id: The customer id, e.g. C-7.
    """
    return {"customer_id": customer_id, **_CUSTOMERS.get(customer_id, {})}


def revenue_list(period: str = "last_month") -> dict:
    """Revenue per site for a period.

    Args:
        period: Reporting period key, e.g. last_month.
    """
    return {"period": period, "rows": _REVENUE}


def delayed_count(region: str = "all") -> dict:
    """How many shipments are currently behind their promised date.

    Args:
        region: Region filter, or 'all'.
    """
    return {"region": region, "delayed": 23}


REAL: list[tuple[Callable[..., Any], dict]] = [
    (stock_get, dict(name="inventory.stock.get", category="inventory", retry_safety="pure",
                     effects=[("none", "*")],
                     embedding_text="warehouse stock level on hand inventory count")),
    (order_get, dict(name="orders.order.get", category="orders", retry_safety="pure",
                     effects=[("none", "*")],
                     embedding_text="order lookup by id status total customer")),
    (customer_get, dict(name="crm.customer.get", category="crm", retry_safety="pure",
                        effects=[("none", "*")],
                        embedding_text="customer record name email contact details")),
    (revenue_list, dict(name="sales.revenue.list", category="sales", retry_safety="pure",
                        effects=[("none", "*")],
                        embedding_text="revenue sales figures per site last month")),
    (delayed_count, dict(name="logistics.shipment.delayed_count", category="logistics",
                         retry_safety="pure", effects=[("none", "*")],
                         embedding_text="shipments running behind schedule late overdue delivery backlog")),
]

# --------------------------------------------------------------------------
# Filler capabilities - registry scale. Three parameter shapes are cycled so
# the specs are realistically sized rather than all identical.
# --------------------------------------------------------------------------


def _shape_a(target: str, limit: int = 20) -> dict:
    """Operate on a resource in this service.

    Args:
        target: The resource identifier to act on.
        limit: Maximum number of rows to return.
    """
    return {"ok": True}


def _shape_b(query: str, since: str = "", include_archived: bool = False) -> dict:
    """Search or filter records in this service.

    Args:
        query: Free-text query.
        since: ISO date lower bound.
        include_archived: Whether to include archived records.
    """
    return {"ok": True}


def _shape_c(record_id: str, fields: str = "", dry_run: bool = True) -> dict:
    """Read or modify one record in this service.

    Args:
        record_id: The record identifier.
        fields: Comma-separated field list.
        dry_run: Validate without committing.
    """
    return {"ok": True}


_SHAPES = [_shape_a, _shape_b, _shape_c]
_SERVICES = ["billing", "hr", "procurement", "support", "analytics", "fleet", "catalog",
             "payroll", "compliance", "marketing", "facilities", "legal", "security",
             "training", "vendor", "quality", "research", "travel", "tax", "asset"]
_RESOURCES = ["record", "report", "ticket", "invoice", "contract", "schedule", "policy",
              "batch", "audit", "profile"]
_OPS = ["get", "list", "search", "create", "update", "cancel", "export", "summarize"]


def filler_capabilities(n: int) -> list[tuple[Callable[..., Any], dict]]:
    out: list[tuple[Callable[..., Any], dict]] = []
    i = 0
    for svc in _SERVICES:
        for res in _RESOURCES:
            for op in _OPS:
                if len(out) >= n:
                    return out
                fn = _SHAPES[i % 3]
                out.append((fn, dict(
                    name=f"{svc}.{res}.{op}",
                    category=svc,
                    summary=f"{op.capitalize()} a {svc} {res}.",
                    usage_notes=(f"Applies only to {svc} {res} objects; not for inventory, "
                                 f"orders, customers, revenue or shipments."),
                    embedding_text=f"{svc} {res} {op}",
                    retry_safety="pure",
                    effects=[("none", "*")],
                )))
                i += 1
    return out


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------


@dataclass
class Task:
    key: str
    prompt: str
    expect_tools: set
    check: Callable[[str], bool]


TASKS = [
    Task("stock", "How many units are on hand in the Tokyo warehouse?",
         {"inventory.stock.get"}, lambda s: "42" in s),
    Task("two_step", "What is the email address of the customer who placed order A-1001?",
         {"orders.order.get", "crm.customer.get"},
         lambda s: "ops@hoshino-kikaku.example" in s.lower()),
    Task("arith", "What was total revenue across all sites last month?",
         {"sales.revenue.list"}, lambda s: "9050" in s.replace(",", "")),
    Task("obscure", "I need to know how many deliveries are currently running behind schedule.",
         {"logistics.shipment.delayed_count"}, lambda s: "23" in s),
]

# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------

ARMS = {
    # Conventional: every spec resident every turn, no discovery layer.
    "preload": dict(pinned=True,
                    discovery={"vector": "off", "k": 0, "toc": False, "active_tools": 0}),
    # SPAL as designed: TOC + candidate cards + meta.tool.find.
    "spal": dict(pinned=False,
                 discovery={"vector": "off", "k": 8, "toc": True, "active_tools": 48}),
}


@dataclass
class Result:
    arm: str
    size: int
    task: str
    run: int
    ok: bool = False
    steps: int = 0
    api_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: float = 0.0
    seconds: float = 0.0
    recall: bool = False
    error: str = ""
    calls: list = field(default_factory=list)


def run_one(arm: str, size: int, task: Task, run: int, model: str,
            provider: str = "anthropic") -> Result:
    spec = ARMS[arm]
    adapter = PROVIDERS[provider](model=model)
    config = Config.from_dict({
        # Job mode is what offers finish(result) and gives the run a terminal
        # state - without it there is nothing deterministic to grade.
        "mode": "job",
        "discovery": spec["discovery"],
        "compaction": {"trigger_ratio": 0.0},
        "budget": {"max_steps": 12},
        # Big enough that the preload arm is never truncated by the window
        # budget - the comparison is about strategy, not clipping.
        "projection": {"window_tokens": 180_000, "reserved_output_tokens": 2048},
    })
    session = Session(adapter, kernel=KERNEL, config=config)
    for fn, meta in REAL + filler_capabilities(size):
        session.registry.register(
            build_capability_from_function(fn, pinned=spec["pinned"], **meta), fn)

    r = Result(arm=arm, size=size, task=task.key, run=run)
    t0 = time.time()
    try:
        answer = session.run_job(task.prompt)
        text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False, default=str)
        r.ok = bool(task.check(text))
    except Exception as exc:  # an arm that cannot run IS the finding; record it
        r.error = f"{type(exc).__name__}: {exc}"
    r.seconds = time.time() - t0
    r.steps = session.budget.steps
    r.api_calls = adapter.api_calls
    r.tokens_in = adapter.prompt_total
    r.tokens_out = adapter.usage["out"]
    r.cache_read = adapter.usage["cache_read"]
    r.cache_write = adapter.usage["cache_write"]
    r.cost = adapter.cost
    r.calls = adapter.seen_calls
    dotted = {c.replace("__", ".") for c in adapter.seen_calls}
    r.recall = task.expect_tools <= dotted
    return r


def report(results: list, sizes: list, arms: list) -> None:
    print("\n" + "=" * 80)
    print(f"{'N':>6} {'arm':<9} {'success':>8} {'recall':>8} {'turns':>6} "
          f"{'in/turn':>9} {'out':>7} {'$/task':>9} {'sec':>6}")
    print("-" * 80)
    for size in sizes:
        for arm in arms:
            rows = [r for r in results if r.size == size and r.arm == arm]
            if not rows:
                continue
            ok = sum(r.ok for r in rows)
            rec = sum(r.recall for r in rows)
            turns = [r.api_calls for r in rows if r.api_calls]
            per_turn = [r.tokens_in / r.api_calls for r in rows if r.api_calls]
            print(f"{size:>6} {arm:<9} {ok:>3}/{len(rows):<4} {rec:>3}/{len(rows):<4} "
                  f"{statistics.median(turns) if turns else 0:>6.1f} "
                  f"{statistics.median(per_turn) if per_turn else 0:>9.0f} "
                  f"{statistics.median([r.tokens_out for r in rows]):>7.0f} "
                  f"{statistics.median([r.cost for r in rows]):>9.4f} "
                  f"{statistics.median([r.seconds for r in rows]):>6.1f}")
    print("=" * 80)
    print(f"total spend this run: ${sum(r.cost for r in results):.4f}")
    errs = [r for r in results if r.error]
    if errs:
        print(f"\n{len(errs)} run(s) errored:")
        for r in errs[:10]:
            print(f"  N={r.size} {r.arm} {r.task} run{r.run}: {r.error}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="25,200")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--provider", default="deepseek", choices=sorted(PROVIDERS))
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--arms", default="preload,spal")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--out", default="benchmarks/results.json")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    arms = args.arms.split(",")
    tasks = [t for t in TASKS if not args.tasks or t.key in args.tasks.split(",")]

    results: list[Result] = []
    total = len(sizes) * len(arms) * len(tasks) * args.repeat
    n = 0
    for size in sizes:
        for arm in arms:
            for task in tasks:
                for run in range(args.repeat):
                    n += 1
                    r = run_one(arm, size, task, run, args.model, args.provider)
                    results.append(r)
                    flag = "ok  " if r.ok else ("ERR " if r.error else "MISS")
                    print(f"[{n}/{total}] N={size:<5} {arm:<8} {r.task:<9} run{run} "
                          f"{flag} turns={r.api_calls} in={r.tokens_in:<7} "
                          f"out={r.tokens_out:<5} ${r.cost:.4f} {r.seconds:.1f}s"
                          + (f"  {r.error}" if r.error else ""), flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump([vars(r) for r in results], f, indent=2, ensure_ascii=False)
    report(results, sizes, arms)
    print(f"\nRaw results -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
