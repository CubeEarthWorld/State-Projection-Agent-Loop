"""Interactive customer-support agent over any OpenAI-compatible API.

    python -m examples.customer_support.run_live

Type questions about AquaPure AP-100 / SmartBrew SB-2; 'quit' to exit.
Escalations and chart cards are printed at the end (in-memory backend).
"""
from __future__ import annotations

import json
import os

from state_projection_loop import Session
from state_projection_loop.policy import Rule

from ..llm_adapters import OpenAICompatAdapter
from .tools import SUPPORT_KERNEL, SupportBackend, build_support_registry

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def main() -> None:
    llm = OpenAICompatAdapter(
        model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
    )
    backend = SupportBackend()
    # Interactive multi-turn chat: the run stays RUNNING across many send()
    # calls (job mode's finish() would otherwise terminate it on turn one).
    session = Session(llm, kernel=SUPPORT_KERNEL, registry=build_support_registry(backend))
    # Escalation and chart rendering are the two effectful actions this
    # demo exposes; grant the whole support.* namespace so the console
    # demo doesn't stop to ask for approval on every reply.
    session.policy.add_rule("workspace", Rule(decision="allow", capability_pattern="support.*"))

    print("カスタマーサポートAIです。ご質問をどうぞ(quit で終了)")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in ("quit", "exit"):
            break
        print("ai >", session.send(text))

    if backend.tickets:
        print("\n--- escalated tickets ---")
        print(json.dumps(backend.tickets, ensure_ascii=False, indent=2))
    if backend.charts:
        print("\n--- rendered chart cards ---")
        print(json.dumps(backend.charts, ensure_ascii=False, indent=2))
    print("\n--- usage metrics ---")
    print(json.dumps(backend.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
