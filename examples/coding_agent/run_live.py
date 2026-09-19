"""One-shot coding agent over any OpenAI-compatible API: fixes a failing
test in a temp workspace.

    python -m examples.coding_agent.run_live
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from state_projection_loop import Config

from ..llm_adapters import OpenAICompatAdapter
from .tools import make_session, seed_workspace


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        seed_workspace(root)
        session = make_session(OpenAICompatAdapter.from_env(), root,
                               config=Config.from_dict({"mode": "job", "budget": {"max_steps": 12}}))
        reply = session.run_job(
            "dev.tests.run を実行し、失敗しているテストを修正してください。修正後は必ず dev.tests.run で確認し、finish(result) で報告してください。"
        )
        print("assistant:", reply)
        print("\n--- fixed calculator.py ---")
        print((root / "calculator.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
