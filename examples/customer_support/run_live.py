"""Interactive customer-support agent over DeepSeek.

    python -m examples.customer_support.run_live

Type questions about AquaPure AP-100 / SmartBrew SB-2; 'quit' to exit.
Escalations and chart cards are printed at the end (in-memory backend).
"""
from __future__ import annotations

import json

from state_projection_loop import Session
from state_projection_loop.adapters import DeepSeekAdapter

from .tools import SUPPORT_KERNEL, SupportBackend, build_support_registry

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def main() -> None:
    backend = SupportBackend()
    session = Session(DeepSeekAdapter(), kernel=SUPPORT_KERNEL,
                      registry=build_support_registry(backend))
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
