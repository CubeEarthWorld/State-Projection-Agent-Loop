"""Small agent CLI. Install the existing [examples] extra for the OpenAI adapter."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time

try:
    from state_projection_loop import Config, PolicyEngine, Session
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from state_projection_loop import Config, PolicyEngine, Session

from llm_adapters import OpenAICompatAdapter


class MeasuredAdapter(OpenAICompatAdapter):
    async def complete(self, messages, tools=None, *, on_delta=None):
        decision = await super().complete(messages, tools, on_delta=on_delta)
        if decision.raw is not None:
            usage = getattr(decision.raw, "usage", None)
            cost = getattr(usage, "cost", None)
            self.api_cost += float(cost or 0)
            self.cost_reported |= cost is not None
        return decision


def main():
    parser = argparse.ArgumentParser(description="Run one coding task through State Projection Loop")
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek/deepseek-v4-flash-0731"))
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--json", action="store_true", help="Return result and usage metrics as JSON")
    parser.add_argument("--trace", type=Path, help="Save the event ledger, excluding credentials")
    args = parser.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY")
    if not key:
        parser.error("Set OPENROUTER_API_KEY or LLM_API_KEY")
    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    if not prompt:
        parser.error("Provide a prompt or --prompt-file")
    root = args.cwd.resolve()
    if not root.is_dir():
        parser.error("--cwd must be an existing directory")
    config = Config.from_dict({"mode": "job", "budget": {"max_steps": args.steps, "max_seconds": args.seconds}})
    adapter = MeasuredAdapter(args.model, api_key=key, base_url=args.base_url, temperature=0,
                              max_tokens=8192, timeout=min(args.seconds, 120),
                              extra_body={"reasoning": {"enabled": False}})
    adapter.api_cost = 0.0
    adapter.cost_reported = False
    session = Session(adapter, config=config, policy=PolicyEngine(default_decision="allow"), workspace_root=root,
                      kernel="You are a coding agent. Implement the requested change in the working directory. "
                             "Use read, write, edit, grep, find, ls and bash. "
                             "Call finish(result) when the task is complete.")
    started = time.perf_counter()
    error = None
    try:
        result = asyncio.run(session.arun_job(prompt))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result = None
    events = list(session.ledger.iter_run(session.run.id))
    summary = {"model": args.model, "result": result, "error": error, "state": session.run.state,
               "seconds": round(time.perf_counter() - started, 3),
               "model_calls": sum(e.type == "model_response" for e in events),
               "tool_calls": sum(e.type == "command_started" for e in events),
               "input_tokens": session.budget.prompt_tokens, "output_tokens": session.budget.completion_tokens,
               "cached_tokens": sum(e.data.get("usage", {}).get("cached_tokens", 0)
                                    for e in events if e.type == "model_response" and e.data.get("usage")),
               "api_cost_usd": adapter.api_cost if adapter.cost_reported else None}
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text("\n".join(e.to_line() for e in events) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False) if args.json else result or error)
    return 1 if error or session.run.state != "COMPLETED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
