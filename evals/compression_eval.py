"""Does history compression lose what the agent needs? A long-horizon recall
eval, runnable against any OpenAI-compatible model.

    python -m evals.compression_eval --turns 30 --window 12000 --repeat 2
    python -m evals.compression_eval --fold 0.75          # with LLM compaction

The agent drives a fake record-lookup tool for ``--turns`` turns. Each turn's
tool output is ~30 lines of noise around one fact (an id, an amount, a
status), some facts are later *updated*, and the user drops a few facts of
their own. Then the same session is asked questions whose answers sit at
known depths, graded by exact substring — no judge model. Alongside
accuracy it reports what compression is for: prompt tokens per turn, and
the provider's prompt-cache hit ratio, which is only high when the rendered
prefix stays byte-identical between turns.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from state_projection_loop import Config, Registry, Session  # noqa: E402
from state_projection_loop.policy import PolicyEngine  # noqa: E402

from examples.llm_adapters import OpenAICompatAdapter  # noqa: E402

KERNEL = (
    "You are an operations assistant for a small logistics company. When the user asks you to look up a "
    "record, call ops.record.lookup with its id, then reply in ONE short sentence that repeats the key figure "
    "or status the record shows. When the user asks a question, answer from what you have seen in this "
    "conversation only, in one short sentence; if it was never stated, answer exactly: not stated."
)

NOISE = [
    "region: EU-WEST", "carrier: Nordfreight", "priority: normal", "sla: 48h", "checked_by: system",
    "warehouse: Rotterdam-3", "pallets: 12", "temperature: ambient", "customs: cleared", "insurance: standard",
    "route: RTM-HAM-CPH", "docs: complete", "handler: crew-B", "seal: intact", "weight_class: C",
]


def scenario(seed: int, turns: int) -> tuple[list[dict], list[dict]]:
    """``turns`` lookups plus the questions to ask afterwards, with answers."""
    rng = random.Random(seed)
    records: dict[str, dict] = {}
    steps: list[dict] = []
    questions: list[dict] = []
    for i in range(turns):
        rid = f"R{seed * 1000 + i:05d}"
        amount = rng.randint(1000, 9999)
        status = rng.choice(["delivered", "in transit", "held at customs", "returned"])
        invoice = f"INV-{rng.randint(10000, 99999)}"
        records[rid] = {"amount": amount, "status": status, "invoice": invoice}
        steps.append({"say": f"Look up record {rid}.", "rid": rid})
    # updates: three records change amount later in the conversation
    updated = rng.sample(list(records), 3)
    for n, rid in enumerate(updated):
        new_amount = records[rid]["amount"] + 500
        steps.insert(turns // 2 + n * 2, {"say": f"Correction on {rid}: the amount is now {new_amount} EUR, "
                                                 "please note it.", "user_fact": (rid, new_amount)})
        records[rid]["updated"] = new_amount
    # user-stated facts
    steps.insert(2, {"say": "By the way, our account manager is Petra Lindqvist and the site code is ZX-41."})
    early, late = list(records)[1], list(records)[-2]
    questions += [
        {"ask": f"What was the invoice number for {early}?", "answer": records[early]["invoice"], "kind": "early"},
        {"ask": f"What status did {late} have?", "answer": records[late]["status"], "kind": "late"},
        {"ask": f"What is the current amount for {updated[0]}?", "answer": str(records[updated[0]]["updated"]),
         "kind": "update", "stale": str(records[updated[0]]["amount"])},
        {"ask": "Who is our account manager?", "answer": "Lindqvist", "kind": "user_fact"},
        {"ask": "What is the site code?", "answer": "ZX-41", "kind": "user_fact"},
        {"ask": f"What was the delivery driver's name for {early}?", "answer": "not stated", "kind": "abstain"},
    ]
    return steps, questions, records


def build_registry(records: dict[str, dict], seed: int) -> Registry:
    rng = random.Random(seed + 7)
    registry = Registry()

    def lookup(id: str) -> str:
        r = records.get(id)
        if r is None:
            return f"record {id}: not found"
        lines = [f"record {id}", f"status: {r['status']}"]
        lines += [rng.choice(NOISE) for _ in range(rng.randint(24, 34))]
        lines.insert(rng.randint(3, len(lines)), f"amount: {r['amount']} EUR")
        lines.insert(rng.randint(3, len(lines)), f"invoice: {r['invoice']}")
        return "\n".join(lines)

    registry.register({
        "name": "ops.record.lookup", "category": "ops",
        "spec": {"description": "Look up a shipment record by id; returns its status, amount and invoice.",
                 "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
        "discovery": {"pinned": True},
        "execution": {"retry_safety": "pure"},
        "effects": [{"kind": "read", "resource": "ops:records"}],
    }, handler=lookup)
    return registry


def run_one(seed: int, args: argparse.Namespace) -> dict:
    steps, questions, records = scenario(seed, args.turns)
    llm = OpenAICompatAdapter.from_env(temperature=0.0)
    config = Config.from_dict({
        "projection": {"window_tokens": args.window},
        "compaction": {"trigger_ratio": args.fold},
        "budget": {"max_steps": 400},
    })
    session = Session(llm, kernel=KERNEL, registry=build_registry(records, seed), config=config,
                      policy=PolicyEngine(default_decision="allow"), builtins=())
    started = time.time()
    for step in steps:
        session.send(step["say"])
    usages = [e.data["usage"] for e in session.ledger.iter_run(session.run.id)
              if e.type == "model_response" and e.data.get("usage")]
    results = []
    for q in questions:
        reply = str(session.send(q["ask"]))
        ok = q["answer"].lower() in reply.lower()
        if q["kind"] == "update" and q["stale"] in reply and reply.find(q["stale"]) < reply.find(q["answer"]):
            ok = False  # the stale figure presented as the current one
        results.append({**q, "reply": reply, "ok": ok})
    prompt = [u["prompt_tokens"] for u in usages]
    cached = [u["cached_tokens"] for u in usages]
    return {
        "seed": seed, "accuracy": sum(r["ok"] for r in results) / len(results),
        "by_kind": {r["kind"]: r["ok"] for r in results},
        "prompt_tokens_per_turn": mean(prompt) if prompt else 0,
        "peak_prompt_tokens": max(prompt) if prompt else 0,
        "cache_hit_ratio": (sum(cached) / sum(prompt)) if sum(prompt) else 0.0,
        "model_calls": len(usages), "seconds": time.time() - started,
        "folds": sum(1 for e in session.ledger.iter_run(session.run.id) if e.type == "state_folded"),
        "answers": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--window", type=int, default=12000)
    parser.add_argument("--fold", type=float, default=0.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    runs = [run_one(seed, args) for seed in range(1, args.repeat + 1)]
    summary = {
        "label": args.label, "turns": args.turns, "window": args.window, "fold": args.fold,
        "accuracy": mean(r["accuracy"] for r in runs),
        "by_kind": {k: mean(r["by_kind"][k] for r in runs) for k in runs[0]["by_kind"]},
        "prompt_tokens_per_turn": mean(r["prompt_tokens_per_turn"] for r in runs),
        "peak_prompt_tokens": max(r["peak_prompt_tokens"] for r in runs),
        "cache_hit_ratio": mean(r["cache_hit_ratio"] for r in runs),
        "folds": sum(r["folds"] for r in runs),
        "runs": runs,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
