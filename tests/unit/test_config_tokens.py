"""Config defaults (§13) and token estimation."""
from __future__ import annotations

import pytest

from state_projection_loop import Config, Message
from state_projection_loop.tokens import estimate_text_tokens, estimate_tokens


class TestConfigDefaults:
    def test_spec_defaults(self):
        cfg = Config()
        assert cfg.mode == "chat"
        assert cfg.projection.window_tokens == 30000
        assert cfg.projection.sections == ["kernel", "toc", "history", "working_state", "candidates"]
        assert cfg.discovery.vector == "auto"
        assert cfg.discovery.k == 8
        assert cfg.discovery.toc is True
        assert cfg.discovery.query_sources == [
            "last_user_message", "last_model_thought", "goal_if_exists",
        ]
        assert cfg.compression.full_window == 6
        assert cfg.compression.compressed_window == 24
        assert cfg.compression.summary_window == 60
        assert cfg.budget.max_steps == 50
        assert cfg.budget.max_tokens is None
        assert cfg.artifacts.inline_threshold_tokens == 800
        assert cfg.limits.max_validation_retries == 2
        assert cfg.limits.approval_expires_s == 3600.0
        assert cfg.persistence.ledger_directory is None

    def test_from_dict_nested_override(self):
        cfg = Config.from_dict({
            "mode": "job",
            "projection": {"window_tokens": 8000},
            "discovery": {"vector": "off", "k": 4},
            "budget": {"max_steps": 10},
        })
        assert cfg.mode == "job"
        assert cfg.projection.window_tokens == 8000
        assert cfg.projection.sections[0] == "kernel"  # untouched defaults survive
        assert cfg.discovery.vector == "off"
        assert cfg.discovery.k == 4
        assert cfg.budget.max_steps == 10

    def test_from_dict_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config key"):
            Config.from_dict({"projektion": {}})
        with pytest.raises(ValueError, match="Unknown config key"):
            Config.from_dict({"discovery": {"vektor": "on"}})


class TestTokenEstimation:
    def test_empty(self):
        assert estimate_text_tokens("") == 0
        assert estimate_tokens(None) == 0

    def test_ascii_roughly_quarter(self):
        text = "a" * 400
        assert estimate_text_tokens(text) == 100

    def test_cjk_counts_per_char(self):
        assert estimate_text_tokens("こんにちは") == 5
        assert estimate_text_tokens("宝物庫の鍵") == 5

    def test_mixed(self):
        # 4 CJK chars + 8 ascii chars -> 4 + 2
        assert estimate_text_tokens("日本語だabcdefgh") == 6

    def test_message_overhead_and_calls(self):
        m = Message(role="user", content="hello world!")
        base = estimate_tokens(m)
        assert base >= 4 + 3
        from state_projection_loop import ToolCall

        m2 = Message(role="assistant", content="", tool_calls=[ToolCall(name="t", arguments={"a": 1})])
        assert estimate_tokens(m2) > estimate_tokens(Message(role="assistant", content=""))

    def test_list_of_messages(self):
        msgs = [Message(role="user", content="abcd"), Message(role="assistant", content="efgh")]
        assert estimate_tokens(msgs) == estimate_tokens(msgs[0]) + estimate_tokens(msgs[1])
