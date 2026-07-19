"""Game-master scenario toolkit (§7 full-equipment usage).

The GM manages narration (plain text), presentation (BGM / images /
character expressions), dice, and — via the bundled state tools — the
scenario's goal, flags and variables. ``MediaLog`` records every
presentation call so a front end (or a test) can observe them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

from state_projection_loop import Registry

EXPRESSIONS = ["neutral", "smile", "angry", "sad", "surprised", "fear"]


@dataclass
class MediaLog:
    bgm: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    expressions: list[dict[str, str]] = field(default_factory=list)
    dice: list[dict[str, Any]] = field(default_factory=list)


def build_game_registry(log: MediaLog, *, dice_seed: Optional[int] = None) -> Registry:
    registry = Registry()
    rng = random.Random(dice_seed)

    def play_bgm(track: str, loop: bool = True) -> str:
        log.bgm.append(track)
        return f"♪ BGM再生: {track} (loop={loop})"

    def show_image(scene: str) -> str:
        log.images.append(scene)
        return f"🖼 シーン画像表示: {scene}"

    def set_expression(character: str, expression: str) -> str:
        log.expressions.append({"character": character, "expression": expression})
        return f"{character} の表情を {expression} に変更"

    def roll_dice(sides: int = 6, count: int = 1) -> dict[str, Any]:
        rolls = [rng.randint(1, sides) for _ in range(count)]
        result = {"rolls": rolls, "total": sum(rolls)}
        log.dice.append(result)
        return result

    registry.register({
        "name": "game.media.play_bgm",
        "category": "game.media",
        "spec": {
            "description": "BGMトラックを再生する。場面の空気が変わったら切り替える。",
            "parameters": {
                "type": "object",
                "properties": {
                    "track": {"type": "string", "description": "例: tension, battle, calm, mystery"},
                    "loop": {"type": "boolean", "default": True},
                },
                "required": ["track"],
            },
        },
        "discovery": {"embedding_text": "音楽 BGM 曲 雰囲気 緊張 戦闘 静か 場面転換"},
        "execution": {"timeout_s": 5, "retry_safety": "idempotent"},
        "effects": [{"kind": "write", "resource": "ui:media"}],
    }, handler=play_bgm)

    registry.register({
        "name": "game.media.show_image",
        "category": "game.media",
        "spec": {
            "description": "シーン画像(背景・イベント絵)を表示する。",
            "parameters": {
                "type": "object",
                "properties": {"scene": {"type": "string", "description": "例: dungeon_door, treasure_room"}},
                "required": ["scene"],
            },
        },
        "discovery": {"embedding_text": "画像 背景 シーン 場面 表示 見せる イベント絵"},
        "execution": {"timeout_s": 5, "retry_safety": "idempotent"},
        "effects": [{"kind": "write", "resource": "ui:media"}],
    }, handler=show_image)

    registry.register({
        "name": "game.media.set_expression",
        "category": "game.media",
        "spec": {
            "description": "キャラクターの立ち絵の表情を変更する。",
            "parameters": {
                "type": "object",
                "properties": {
                    "character": {"type": "string"},
                    "expression": {"enum": EXPRESSIONS},
                },
                "required": ["character", "expression"],
            },
        },
        "discovery": {"embedding_text": "表情 感情 笑顔 怒り 悲しみ 驚き 立ち絵"},
        "execution": {"timeout_s": 5, "retry_safety": "idempotent"},
        "effects": [{"kind": "write", "resource": "ui:media"}],
    }, handler=set_expression)

    registry.register({
        "name": "game.dice.roll",
        "category": "game.dice",
        "spec": {
            "description": "ダイスを振る。判定・ダメージ計算に使う。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sides": {"type": "integer", "default": 6, "minimum": 2, "maximum": 100},
                    "count": {"type": "integer", "default": 1, "minimum": 1, "maximum": 10},
                },
            },
        },
        "discovery": {"embedding_text": "ダイス サイコロ 判定 ロール 運試し 1d6 2d6"},
        "execution": {"timeout_s": 5, "retry_safety": "never_retry"},
        "effects": [{"kind": "write", "resource": "game:dice_log"}],
    }, handler=roll_dice)

    return registry


GM_KERNEL = """あなたはTRPGのゲームマスターです。ルール:
- 地の文・NPCのセリフはテキストで語る。場面が変わったら game.media.show_image と game.media.play_bgm、
  感情が動いたら game.media.set_expression で演出する。
- 進行状況は必ず working state で管理する: クリア条件は state.goal.set、イベント達成は state.extra.set、
  HP・所持品などは state.extra.set。[Working state] に常に現在の状態が表示される。
- 判定が必要な行動は game.dice.roll で解決する。
- goal の達成条件を満たしたら祝福の演出をして state.extra.set("flags.cleared", true) を立てる。"""


def initial_seed() -> dict[str, Any]:
    return {
        "goal": "宝物庫の鍵を見つけて地下迷宮から脱出する",
        "extra": {
            "flags": {"cleared": False},
            "party": {"hero": {"hp": 20, "items": ["たいまつ"]}},
            "scene": "dungeon_entrance",
        },
    }
