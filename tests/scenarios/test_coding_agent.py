"""Scenario: coding agent (Claude Code 相当の縮小版).

A deterministic ScriptedLLM drives the *real* tools: real files on disk,
real subprocess test runs — only the model is scripted.
"""
from __future__ import annotations

import pytest

from state_projection_loop import ScriptedLLM, Session

from examples.coding_agent.tools import (
    CODING_KERNEL,
    FIXED_CALCULATOR,
    build_coding_registry,
    seed_workspace,
)


@pytest.fixture()
def workspace(tmp_path):
    seed_workspace(tmp_path)
    return tmp_path


class TestBugFixWorkflow:
    def test_full_red_green_cycle(self, workspace):
        llm = ScriptedLLM([
            ScriptedLLM.call("run_tests"),
            ScriptedLLM.call("read_file", path="calculator.py"),
            ScriptedLLM.call("write_file", path="calculator.py", content=FIXED_CALCULATOR),
            ScriptedLLM.call("run_tests"),
            "divide() にゼロ除算ガードを追加し、全テストが通ることを確認しました。",
        ])
        session = Session(llm, kernel=CODING_KERNEL, registry=build_coding_registry(workspace))
        reply = session.send("テストが落ちているので calculator.py を直してください")

        assert "全テスト" in reply
        observations = [m.content for m in session.conversation if m.role == "tool"]
        assert "FAILED" in observations[0]          # red
        assert "def divide" in observations[1]      # read the source
        assert "wrote" in observations[2]
        assert "ALL TESTS PASSED" in observations[3]  # green
        assert "raise ValueError" in (workspace / "calculator.py").read_text(encoding="utf-8")

    def test_path_traversal_is_blocked(self, workspace):
        secret = workspace.parent / "secret.txt"
        secret.write_text("do not read", encoding="utf-8")
        llm = ScriptedLLM([
            ScriptedLLM.call("read_file", path="../secret.txt"),
            "読めませんでした。",
        ])
        session = Session(llm, registry=build_coding_registry(workspace))
        session.send("親ディレクトリのファイルを読んで")
        obs = next(m.content for m in session.conversation if m.role == "tool")
        assert "escapes the workspace" in obs
        assert "do not read" not in obs

    def test_large_file_read_becomes_handle_and_peek_works(self, workspace):
        big = "\n".join(f"line {i}: {'x' * 60}" for i in range(400))
        (workspace / "big.txt").write_text(big, encoding="utf-8")
        llm = ScriptedLLM([
            ScriptedLLM.call("read_file", path="big.txt"),
            ScriptedLLM.call("peek", handle="$h1", range="399-400"),
            "確認しました。",
        ])
        session = Session(llm, registry=build_coding_registry(workspace))
        session.send("big.txt の最後の方を見せて")
        obs = [m.content for m in session.conversation if m.role == "tool"]
        assert "$h1" in obs[0] and "peek" in obs[0]   # handled, not inlined (I7)
        assert "line 398" in obs[1] or "line 399" in obs[1]
