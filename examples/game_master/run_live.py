"""Interactive TRPG game master over any OpenAI-compatible API, with full
working-state management.

    python -m examples.game_master.run_live
"""
from __future__ import annotations

import json

from ..llm_adapters import OpenAICompatAdapter
from .tools import MediaLog, make_session


def main() -> None:
    log = MediaLog()
    # Interactive multi-turn narration: chat mode, so the run stays RUNNING
    # across many send() calls instead of terminating on the first finish().
    session = make_session(OpenAICompatAdapter.from_env(temperature=0.8), log)

    print("=== 地下迷宮からの脱出 ===")
    print("GM>", session.send("ゲームを開始してください。オープニングの場面を描写して。"))
    while not session.working_state.extra.get("flags", {}).get("cleared"):
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
    print(json.dumps(session.working_state.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
