"""Scenario: coding agent (a reduced Claude-Code-like workflow).

A deterministic ScriptedLLM drives the *real* tools: real files on disk,
real subprocess test runs — only the model is scripted.
"""
from __future__ import annotations

import pytest

from state_projection_loop import Config, ScriptedLLM, Session
from state_projection_loop.artifacts import ref
from state_projection_loop.policy import PolicyEngine

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


def allow_all() -> PolicyEngine:
    return PolicyEngine(default_decision="allow")


class TestBugFixWorkflow:
    def test_full_red_green_cycle(self, workspace):
        llm = ScriptedLLM([
            ScriptedLLM.call("dev.tests.run"),
            ScriptedLLM.call("filesystem.file.read", path="calculator.py"),
            ScriptedLLM.call("filesystem.file.write", path="calculator.py", content=FIXED_CALCULATOR),
            ScriptedLLM.call("dev.tests.run"),
            ScriptedLLM.finish(result="divide() にゼロ除算ガードを追加し、全テストが通ることを確認しました。"),
        ])
        session = Session(llm, kernel=CODING_KERNEL, registry=build_coding_registry(workspace),
                          config=Config.from_dict({"mode": "job"}), policy=allow_all())
        reply = session.run_job("テストが落ちているので calculator.py を直してください")

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
            ScriptedLLM.call("filesystem.file.read", path="../secret.txt"),
            "読めませんでした。",
        ])
        session = Session(llm, registry=build_coding_registry(workspace), policy=allow_all())
        session.send("親ディレクトリのファイルを読んで")
        obs = next(m.content for m in session.conversation if m.role == "tool")
        assert "escapes the workspace" in obs
        assert "do not read" not in obs

    def test_large_file_read_becomes_artifact_and_peek_works(self, workspace):
        big = "\n".join(f"line {i}: {'x' * 60}" for i in range(400))
        (workspace / "big.txt").write_text(big, encoding="utf-8")

        def peek_step(messages, tools):
            obs = next(m for m in reversed(messages) if m.role == "tool")
            artifact_id = obs.content.split("[", 1)[1].split(" ", 1)[0]
            return ScriptedLLM.call("meta.artifact.peek", artifact=ref(artifact_id), range="399-400")

        llm = ScriptedLLM([
            ScriptedLLM.call("filesystem.file.read", path="big.txt"),
            peek_step,
            "確認しました。",
        ])
        session = Session(llm, registry=build_coding_registry(workspace), policy=allow_all())
        session.send("big.txt の最後の方を見せて")
        obs = [m.content for m in session.conversation if m.role == "tool"]
        assert "art_" in obs[0] and "peek" in obs[0]   # became an artifact, not inlined
        assert "line 398" in obs[1] or "line 399" in obs[1]
