"""Does history compression lose what the agent needs? Long-horizon recall
evals of several shapes, in two languages, runnable against any
OpenAI-compatible model — or offline, against no model at all.

    python -m evals.compression_eval --task records --lang en --turns 30 --window 12000 --repeat 2
    python -m evals.compression_eval --task all --lang all --fold 0.75   # the whole matrix, with folds
    python -m evals.compression_eval --task all --lang all --offline     # projection only, no API

The agent drives a fake tool for ``--turns`` turns; the tool output is
noisy, some facts are later corrected, and the user drops facts of their
own. Then the same session is asked questions whose answers sit at known
depths, graded by exact substring — no judge model. Alongside accuracy it
reports what compression is for: prompt tokens per turn, and the
provider's prompt-cache hit ratio, which is only high when the rendered
prefix stays byte-identical between turns. See ``scenarios.py`` for the
shapes.

``--offline`` replaces the model with a script that calls the scenario's
tool for every step that names one and answers everything else in a line.
It cannot grade recall, but it measures the projection itself for free:
tokens per turn by the estimator, and a cache proxy — the share of each
prompt's characters that are a byte-identical prefix of the previous
prompt — so a change to the tiers can be checked for size and prefix
stability before spending on the live matrix.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from state_projection_loop import FOLD_INSTRUCTIONS, Config, Decision, Message, Session, ToolCall  # noqa: E402
from state_projection_loop.messages import OBSERVATION, SYSTEM, USER  # noqa: E402
from state_projection_loop.policy import PolicyEngine  # noqa: E402
from state_projection_loop.tokens import estimate_tokens  # noqa: E402

from evals.scenarios import SCENARIOS  # noqa: E402


class OfflineLLM:
    """Stands in for the model: one tool call for a step that names one, a
    one-line answer to everything else, an empty fold. Records every prompt
    so the projection can be measured without a provider."""

    def __init__(self, calls: dict[str, tuple[str, dict[str, str]]]) -> None:
        self.calls = calls
        self.requests: list[list[Message]] = []

    async def complete(self, messages: list[Message], tools=None, *, on_delta=None) -> Decision:
        self.requests.append(list(messages))
        if messages[0].content == FOLD_INSTRUCTIONS:
            return Decision(text='{"facts_add": []}')
        last = next(m for m in reversed(messages) if m.role != SYSTEM)
        if last.role == OBSERVATION:
            return Decision(text=str(last.content).splitlines()[0][:80])
        call = self.calls.get(str(last.content)) if last.role == USER else None
        if call:
            return Decision(text="", calls=[ToolCall(name=call[0], arguments=call[1])])
        return Decision(text="noted")

    def measured(self) -> tuple[list[int], float]:
        """Prompt tokens per call, and the cache proxy over consecutive calls."""
        texts = ["\n".join(f"{m.role}:{m.content}" for m in req) for req in self.requests]
        shared = [len(os.path.commonprefix([a, b])) for a, b in zip(texts, texts[1:])]
        total = sum(len(t) for t in texts[1:])
        return [estimate_tokens(req) for req in self.requests], (sum(shared) / total if total else 0.0)


def run_one(task: str, lang: str, seed: int, args: argparse.Namespace) -> dict:
    scenario = SCENARIOS[task](seed, args.turns, lang)
    if args.offline:
        llm = OfflineLLM(scenario.calls)
    else:
        from examples.llm_adapters import OpenAICompatAdapter
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
    results = []
    if args.offline:
        prompt, cache = llm.measured()
    else:
        usages = [e.data["usage"] for e in session.ledger.iter_run(session.run.id)
                  if e.type == "model_response" and e.data.get("usage")]
        prompt = [u["prompt_tokens"] for u in usages]
        cache = (sum(u["cached_tokens"] for u in usages) / sum(prompt)) if sum(prompt) else 0.0
        for q in scenario.questions:
            if not q["answer"]:
                continue
            reply = str(session.send(q["ask"]))
            ok = q["answer"].lower() in reply.lower()
            stale = q.get("stale")
            if ok and stale and stale in reply and reply.find(stale) < reply.find(q["answer"]):
                ok = False  # the stale figure presented as the current one
            results.append({**q, "reply": reply, "ok": ok})
    return {
        "task": task, "lang": lang, "seed": seed,
        "accuracy": sum(r["ok"] for r in results) / len(results) if results else None,
        "by_kind": {r["kind"]: r["ok"] for r in results},
        "prompt_tokens_per_turn": mean(prompt) if prompt else 0,
        "peak_prompt_tokens": max(prompt) if prompt else 0,
        "cache_hit_ratio": cache,
        "model_calls": len(prompt), "seconds": time.time() - started,
        "folds": sum(1 for e in session.ledger.iter_run(session.run.id) if e.type == "state_folded"),
        "answers": results,
    }


def _mean(values: list[Optional[float]]) -> Optional[float]:
    known = [v for v in values if v is not None]
    return mean(known) if known else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="records", help="one of %s or all" % ", ".join(SCENARIOS))
    parser.add_argument("--lang", default="en", help="en, ja or all")
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--window", type=int, default=12000)
    parser.add_argument("--fold", type=float, default=0.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--offline", action="store_true", help="no model: measure the projection only")
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
        "label": args.label, "turns": args.turns, "window": args.window, "fold": args.fold, "offline": args.offline,
        "cells": [{"task": task, "lang": lang, "accuracy": _mean([r["accuracy"] for r in rs]),
                   "prompt_tokens_per_turn": mean(r["prompt_tokens_per_turn"] for r in rs),
                   "cache_hit_ratio": mean(r["cache_hit_ratio"] for r in rs), "folds": sum(r["folds"] for r in rs),
                   "misses": [f'{a["kind"]}: expected {a["answer"]!r}, got {a["reply"][:60]!r}'
                              for r in rs for a in r["answers"] if not a["ok"]]}
                  for (task, lang), rs in cells.items()],
        "accuracy": _mean([r["accuracy"] for r in runs]),
        "runs": runs,
    }
    for cell in summary["cells"]:
        acc = "  --" if cell["accuracy"] is None else f'{cell["accuracy"]:.2f}'
        print(f'{cell["task"]:9} {cell["lang"]}  acc {acc}  tok/turn {cell["prompt_tokens_per_turn"]:.0f}  '
              f'cache {cell["cache_hit_ratio"]:.2f}  folds {cell["folds"]}')
        for miss in cell["misses"]:
            print("           miss", miss)
    if summary["accuracy"] is not None:
        print(f"overall accuracy {summary['accuracy']:.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
