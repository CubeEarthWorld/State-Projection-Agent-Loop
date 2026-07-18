"""Scenario: game master — narration + presentation (BGM/image/expression)
+ dice + full state management (goal / flags / variables) per spec §7."""
from __future__ import annotations

import pytest

from state_projection_loop import ScriptedLLM, Session, install_state

from examples.game_master.tools import GM_KERNEL, MediaLog, build_game_registry, initial_seed


@pytest.fixture()
def game():
    log = MediaLog()
    session = Session(
        ScriptedLLM([], strict=False),  # replaced per-test
        kernel=GM_KERNEL,
        registry=build_game_registry(log, dice_seed=42),
        seed=initial_seed(),
    )
    install_state(session)
    return log, session


def make_session(log, steps, seed=None):
    session = Session(
        ScriptedLLM(steps),
        kernel=GM_KERNEL,
        registry=build_game_registry(log, dice_seed=42),
        seed=seed or initial_seed(),
    )
    install_state(session)
    return session


class TestPresentation:
    def test_scene_change_drives_media_tools_in_parallel(self):
        log = MediaLog()
        session = make_session(log, [
            ScriptedLLM.calls(
                ("show_image", {"scene": "dungeon_door"}),
                ("play_bgm", {"track": "tension"}),
                ("set_expression", {"character": "ナビィ", "expression": "surprised"}),
                ("set_flag", {"name": "door_examined", "value": True}),
            ),
            "重厚な扉だ。表面には古代文字が刻まれている……。",
        ])
        reply = session.send("扉を調べる")
        assert "扉" in reply
        assert log.images == ["dungeon_door"]
        assert log.bgm == ["tension"]
        assert log.expressions == [{"character": "ナビィ", "expression": "surprised"}]
        assert session.state["flags"]["door_examined"] is True

    def test_invalid_expression_self_repairs(self):
        log = MediaLog()
        session = make_session(log, [
            ScriptedLLM.call("set_expression", character="ナビィ", expression="grinning"),
            ScriptedLLM.call("set_expression", character="ナビィ", expression="smile"),
            "ナビィはにっこり笑った。",
        ])
        session.send("ナビィを笑わせて")
        obs = [m.content for m in session.conversation if m.role == "tool"]
        assert "Validation error" in obs[0] and "### set_expression" in obs[0]
        assert log.expressions == [{"character": "ナビィ", "expression": "smile"}]


class TestDice:
    def test_deterministic_rolls_with_seed(self):
        log = MediaLog()
        session = make_session(log, [
            ScriptedLLM.call("roll_dice", sides=6, count=2),
            "ダイスの結果で判定した。",
        ])
        session.send("鍵開けに挑戦する")
        assert len(log.dice) == 1
        assert log.dice[0]["total"] == sum(log.dice[0]["rolls"])
        assert all(1 <= r <= 6 for r in log.dice[0]["rolls"])


class TestStateManagement:
    def test_state_view_prevents_goal_drift(self):
        """The goal is re-projected every turn — structural drift prevention."""
        seen = []

        def capture(messages, tools):
            seen.append("\n".join(str(m.content) for m in messages))
            return "……(様子を見ている)"

        log = MediaLog()
        session = make_session(log, [capture, capture])
        session.send("あたりを見回す")
        session.send("先に進む")
        for joined in seen:
            assert "宝物庫の鍵を見つけて地下迷宮から脱出する" in joined  # goal always visible
            assert "[State]" in joined

    def test_hp_and_inventory_updates(self):
        log = MediaLog()
        session = make_session(log, [
            ScriptedLLM.calls(
                ("state_set", {"path": "party.hero.hp", "value": 14}),
                ("state_set", {"path": "party.hero.items", "value": ["たいまつ", "宝物庫の鍵"]}),
                ("set_flag", {"name": "key_found", "value": True}),
            ),
            "罠でダメージを受けたが、鍵を手に入れた!",
        ])
        session.send("宝箱を開ける")
        hero = session.state["party"]["hero"]
        assert hero["hp"] == 14
        assert "宝物庫の鍵" in hero["items"]
        assert session.state["flags"]["key_found"] is True

    def test_goal_completion_flow(self):
        log = MediaLog()
        session = make_session(log, [
            ScriptedLLM.calls(
                ("show_image", {"scene": "exit_gate"}),
                ("play_bgm", {"track": "victory"}),
                ("set_flag", {"name": "cleared", "value": True}),
            ),
            "扉が開いた!まばゆい光の中、君たちは地上へ帰還した。──完──",
        ])
        reply = session.send("鍵を使って脱出する")
        assert session.state["flags"]["cleared"] is True
        assert log.bgm[-1] == "victory"
        assert "完" in reply

    def test_seed_is_projected_from_session_start(self):
        captured = {}

        def check(messages, tools):
            captured["joined"] = "\n".join(str(m.content) for m in messages)
            return "ようこそ、地下迷宮へ。"

        log = MediaLog()
        session = make_session(log, [check])
        session.send("ゲームを始めよう")
        assert '"hp": 20' in captured["joined"]        # party seeded
        assert "dungeon_entrance" in captured["joined"]  # scene seeded
