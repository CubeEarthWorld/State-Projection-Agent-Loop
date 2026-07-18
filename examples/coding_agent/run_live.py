"""One-shot coding agent over DeepSeek: fixes a failing test in a temp workspace.

    python -m examples.coding_agent.run_live
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from state_projection_loop import Config, Session
from state_projection_loop.adapters import DeepSeekAdapter

from .tools import CODING_KERNEL, build_coding_registry, seed_workspace

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        seed_workspace(root)
        session = Session(
            DeepSeekAdapter(),
            kernel=CODING_KERNEL,
            registry=build_coding_registry(root),
            config=Config.from_dict({"budget": {"max_steps": 12}}),
        )
        reply = session.send(
            "run_tests を実行し、失敗しているテストを修正してください。修正後は必ず run_tests で確認を。"
        )
        print("assistant:", reply)
        print("\n--- fixed calculator.py ---")
        print((root / "calculator.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
