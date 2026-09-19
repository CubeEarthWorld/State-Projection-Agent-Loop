"""Structural half of the benchmark - no API key, no spend, deterministic.

A real model is needed to measure success rate, turns and whether the model
*uses* what it is shown. It is NOT needed to measure what the runtime would
have sent, which is where the package makes its one published numeric claim
(under 3k tokens of per-turn overhead at 1,000 capabilities, against
~150k for full-spec preloading).

ScriptedLLM replays a fixed decision sequence, so both arms take the exact
same path through the runtime and the only thing that varies is what the
projection put in front of the model. Two things are measured:

  context  - tokens the arm would have sent, per turn and per task
  visible  - whether the capability the task needs was actually in front of
             the model on turn 1 (tool-recall, measured without a model:
             discovery either surfaced it or it did not)

Token counts use the package's own ``estimate_tokens`` - the same estimator
the acceptance test and the window budget use, so this checks the claim on
its own terms. It is an estimate, not a provider tokenizer.

    python -m benchmarks.bench_offline --sizes 25,200,1000
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field

from state_projection_loop import Config, Session
from state_projection_loop.capability import build_capability_from_function, to_api_name
from state_projection_loop.llm import ScriptedLLM
from state_projection_loop.tokens import estimate_tokens

from benchmarks.bench import ARMS, KERNEL, REAL, TASKS, filler_capabilities

# The tool calls each task would make, in order, if the model were perfect.
SCRIPTS = {
    "stock": [("inventory.stock.get", {"warehouse": "tokyo"})],
    "two_step": [("orders.order.get", {"order_id": "A-1001"}),
                 ("crm.customer.get", {"customer_id": "C-7"})],
    "arith": [("sales.revenue.list", {"period": "last_month"})],
    "obscure": [("logistics.shipment.delayed_count", {"region": "all"})],
}
ANSWERS = {"stock": "42", "two_step": "ops@hoshino-kikaku.example",
           "arith": "9050", "obscure": "23"}


@dataclass
class Row:
    arm: str
    size: int
    task: str
    tools_sent: int = 0
    schema_tokens: int = 0
    message_tokens: int = 0
    turn1_tokens: int = 0
    total_prompt_tokens: int = 0
    turns: int = 0
    visible: bool = False
    visible_where: str = ""
    completed: bool = False
    notes: list = field(default_factory=list)


def measure(arm: str, size: int, task_key: str) -> Row:
    spec = ARMS[arm]
    steps = [ScriptedLLM.call(name, **args) for name, args in SCRIPTS[task_key]]
    steps.append(ScriptedLLM.finish(ANSWERS[task_key]))
    llm = ScriptedLLM(steps, strict=False)

    config = Config.from_dict({
        "mode": "job",
        "discovery": spec["discovery"],
        "compaction": {"trigger_ratio": 0.0},
        "budget": {"max_steps": 12},
        "projection": {"window_tokens": 180_000, "reserved_output_tokens": 2048},
    })
    session = Session(llm, kernel=KERNEL, config=config)
    for fn, meta in REAL + filler_capabilities(size):
        session.registry.register(
            build_capability_from_function(fn, pinned=spec["pinned"], **meta), fn)

    row = Row(arm=arm, size=size, task=task_key)
    try:
        session.run_job(next(t.prompt for t in TASKS if t.key == task_key))
        row.completed = True
    except Exception as exc:
        row.notes.append(f"{type(exc).__name__}: {exc}"[:200])

    if not llm.requests:
        return row
    first = llm.requests[0]
    row.turns = len(llm.requests)
    row.tools_sent = len(first["tools"])
    row.schema_tokens = estimate_tokens(first["tools"])
    row.message_tokens = estimate_tokens(first["messages"])
    row.turn1_tokens = row.schema_tokens + row.message_tokens
    row.total_prompt_tokens = sum(
        estimate_tokens(r["tools"]) + estimate_tokens(r["messages"]) for r in llm.requests)

    # Was the needed capability in front of the model on turn 1?
    needed = [n for n, _ in SCRIPTS[task_key]]
    schema_names = {t.get("function", t)["name"] for t in first["tools"]}
    blob = "\n".join(
        m.content if isinstance(m.content, str) else json.dumps(m.content, default=str)
        for m in first["messages"])
    in_schemas = all(to_api_name(n) in schema_names for n in needed)
    in_text = all(n in blob for n in needed)
    row.visible = in_schemas or in_text
    row.visible_where = ("native schema" if in_schemas else
                         "candidate card" if in_text else "NOT SHOWN")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="25,200,1000")
    ap.add_argument("--out", default="benchmarks/results-offline.json")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    rows = [measure(arm, size, t.key)
            for size in sizes for arm in ARMS for t in TASKS]

    print("=" * 92)
    print(f"{'N':>6} {'arm':<9} {'tools':>7} {'schema tk':>10} {'msg tk':>8} "
          f"{'turn-1 tk':>10} {'task tk':>9} {'shown':>6} {'via':<15}")
    print("-" * 92)
    for size in sizes:
        for arm in ARMS:
            sel = [r for r in rows if r.size == size and r.arm == arm]
            shown = sum(r.visible for r in sel)
            wheres = {r.visible_where for r in sel}
            print(f"{size:>6} {arm:<9} "
                  f"{statistics.median([r.tools_sent for r in sel]):>7.0f} "
                  f"{statistics.median([r.schema_tokens for r in sel]):>10.0f} "
                  f"{statistics.median([r.message_tokens for r in sel]):>8.0f} "
                  f"{statistics.median([r.turn1_tokens for r in sel]):>10.0f} "
                  f"{statistics.median([r.total_prompt_tokens for r in sel]):>9.0f} "
                  f"{shown:>3}/{len(sel):<2} {'/'.join(sorted(wheres)):<15}")
    print("=" * 92)

    misses = [r for r in rows if not r.visible]
    if misses:
        print("\ncapability NOT in front of the model on turn 1:")
        for r in misses:
            print(f"  N={r.size:<5} {r.arm:<8} {r.task}")
    bad = [r for r in rows if not r.completed]
    if bad:
        print(f"\n{len(bad)} run(s) did not complete:")
        for r in bad[:10]:
            print(f"  N={r.size} {r.arm} {r.task}: {r.notes}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump([vars(r) for r in rows], f, indent=2, ensure_ascii=False)
    print(f"\nRaw -> {args.out}")


if __name__ == "__main__":
    main()
