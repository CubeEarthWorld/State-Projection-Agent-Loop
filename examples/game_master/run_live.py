"""Interactive TRPG game master over DeepSeek, with full state management.

    python -m examples.game_master.run_live
"""
from __future__ import annotations

import json

from state_projection_loop import Session, install_state
from state_projection_loop.adapters import DeepSeekAdapter

from .tools import GM_KERNEL, MediaLog, build_game_registry, initial_seed

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def main() -> None:
    log = MediaLog()
    session = Session(DeepSeekAdapter(temperature=0.8), kernel=GM_KERNEL,
                      registry=build_game_registry(log), seed=initial_seed())
    install_state(session)
    print("=== 地下迷宮からの脱出 ===")
    print("GM>", session.send("ゲームを開始してください。オープニングの場面を描写して。"))
    while not session.state.get("flags", {}).get("cleared"):
        try:
            action = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not action or action.lower() in ("quit", "exit"):
            break
        print("GM>", session.send(action))
        if log.bgm:
            print(f"   [bgm: {log.bgm[-1]}]")
        if log.images:
            print(f"   [scene: {log.images[-1]}]")

    print("\n--- final state ---")
    print(json.dumps(session.state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
