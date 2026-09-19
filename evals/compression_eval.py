"""Does history compression lose what the agent needs? Long-horizon recall
evals of several shapes, in two languages, runnable against any
OpenAI-compatible model.

    python -m evals.compression_eval --task records --lang en --turns 30 --window 12000 --repeat 2
    python -m evals.compression_eval --task all --lang all --fold 0.75   # the whole matrix, with folds

The agent drives a fake tool for ``--turns`` turns; the tool output is
noisy, some facts are later corrected, and the user drops facts of their
own. Then the same session is asked questions whose answers sit at known
depths, graded by exact substring — no judge model. Alongside accuracy it
reports what compression is for: prompt tokens per turn, and the
provider's prompt-cache hit ratio, which is only high when the rendered
prefix stays byte-identical between turns. See ``scenarios.py`` for the
shapes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from state_projection_loop import Config, Session  # noqa: E402
from state_projection_loop.policy import PolicyEngine  # noqa: E402

from evals.scenarios import SCENARIOS  # noqa: E402
from examples.llm_adapters import OpenAICompatAdapter  # noqa: E402


def run_one(task: str, lang: str, seed: int, args: argparse.Namespace) -> dict:
    scenario = SCENARIOS[task](seed, args.turns, lang)
    llm = OpenAICompatAdapter.from_env(temperature=0.0)
    config = Config.from_dict({
        "projection": {"window_tokens": args.window},
        "compaction": {"trigger_ratio": args.fold},
        "budget": {"max_steps": 600},
    })
    session = Session(llm, kernel=scenario.kernel, registry=scenario.registry, config=config,
                      policy=PolicyEngine(default_decision="allow"), builtins=scenario.builtins)
    started = time.time()
    for step in scenario.steps:
        session.send(step)
    usages = [e.data["usage"] for e in session.ledger.iter_run(session.run.id)
              if e.type == "model_response" and e.data.get("usage")]
    results = []
    for q in scenario.questions:
        if not q["answer"]:
            continue
        reply = str(session.send(q["ask"]))
        ok = q["answer"].lower() in reply.lower()
        stale = q.get("stale")
        if ok and stale and stale in reply and reply.find(stale) < reply.find(q["answer"]):
            ok = False  # the stale figure presented as the current one
        results.append({**q, "reply": reply, "ok": ok})
    prompt = [u["prompt_tokens"] for u in usages]
    cached = [u["cached_tokens"] for u in usages]
    return {
        "task": task, "lang": lang, "seed": seed,
        "accuracy": sum(r["ok"] for r in results) / len(results),
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
    parser.add_argument("--task", default="records", help="one of %s or all" % ", ".join(SCENARIOS))
    parser.add_argument("--lang", default="en", help="en, ja or all")
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--window", type=int, default=12000)
    parser.add_argument("--fold", type=float, default=0.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    tasks = list(SCENARIOS) if args.task == "all" else [args.task]
    langs = ["en", "ja"] if args.lang == "all" else [args.lang]
    runs = [run_one(task, lang, seed, args) for task in tasks for lang in langs for seed in range(1, args.repeat + 1)]
    cells = {}
    for run in runs:
        cell = cells.setdefault((run["task"], run["lang"]), [])
        cell.append(run)
    summary = {
        "label": args.label, "turns": args.turns, "window": args.window, "fold": args.fold,
        "cells": [{"task": task, "lang": lang, "accuracy": mean(r["accuracy"] for r in rs),
                   "prompt_tokens_per_turn": mean(r["prompt_tokens_per_turn"] for r in rs),
                   "cache_hit_ratio": mean(r["cache_hit_ratio"] for r in rs), "folds": sum(r["folds"] for r in rs),
                   "misses": [f'{a["kind"]}: expected {a["answer"]!r}, got {a["reply"][:60]!r}'
                              for r in rs for a in r["answers"] if not a["ok"]]}
                  for (task, lang), rs in cells.items()],
        "accuracy": mean(r["accuracy"] for r in runs),
        "runs": runs,
    }
    for cell in summary["cells"]:
        print(f'{cell["task"]:9} {cell["lang"]}  acc {cell["accuracy"]:.2f}  tok/turn {cell["prompt_tokens_per_turn"]:.0f}  '
              f'cache {cell["cache_hit_ratio"]:.2f}  folds {cell["folds"]}')
        for miss in cell["misses"]:
            print("           miss", miss)
    print(f"overall accuracy {summary['accuracy']:.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
