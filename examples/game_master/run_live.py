"""Interactive TRPG game master over any OpenAI-compatible API, with full
working-state management.

    python -m examples.game_master.run_live
"""
from __future__ import annotations

import json
import os

from state_projection_loop import Session, install_state
from state_projection_loop.policy import Rule

from ..llm_adapters import OpenAICompatAdapter
from .tools import GM_KERNEL, MediaLog, build_game_registry, initial_seed

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
        temperature=0.8,
    )
    log = MediaLog()
    # Interactive multi-turn narration: chat mode, so the run stays RUNNING
    # across many send() calls instead of terminating on the first finish().
    session = Session(llm, kernel=GM_KERNEL, registry=build_game_registry(log), seed=initial_seed())
    install_state(session)
    # A single-player narrative game: media cues and dice rolls are the
    # only effects, and they're the whole point of the game master — grant
    # them instead of pausing the story to ask for approval.
    session.policy.add_rule("workspace", Rule(decision="allow", capability_pattern="game.*"))
    session.policy.add_rule("workspace", Rule(decision="allow", capability_pattern="state.*"))

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
