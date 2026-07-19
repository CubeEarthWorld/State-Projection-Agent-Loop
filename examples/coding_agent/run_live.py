"""One-shot coding agent over any OpenAI-compatible API: fixes a failing
test in a temp workspace.

    python -m examples.coding_agent.run_live
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from state_projection_loop import Config, Session

from ..llm_adapters import OpenAICompatAdapter
from .tools import CODING_KERNEL, build_coding_registry, seed_workspace

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
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        seed_workspace(root)
        session = Session(
            llm,
            kernel=CODING_KERNEL,
            registry=build_coding_registry(root),
            config=Config.from_dict({"mode": "job", "budget": {"max_steps": 12}}),
        )
        # This example's own file tools live entirely inside the temp
        # workspace, so we grant them automatically instead of pausing on
        # every write for approval.
        session.policy.set_scope("workspace_write", "allow")
        session.policy.set_scope("sandbox_command", "allow")
        reply = session.run_job(
            "dev.tests.run を実行し、失敗しているテストを修正してください。修正後は必ず dev.tests.run で確認し、finish(result) で報告してください。"
        )
        print("assistant:", reply)
        print("\n--- fixed calculator.py ---")
        print((root / "calculator.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
