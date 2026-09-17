"""Coding-agent scenario toolkit: sandboxed file tools + a test runner.

All file access is confined to the workspace root (path-traversal safe);
``dev.tests.run`` executes the workspace's test scripts in a subprocess and
reports pass/fail output as an observation.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from state_projection_loop import Registry, Session, install_toolkits


def build_coding_registry(root: Path) -> Registry:
    root = Path(root).resolve()
    registry = Registry()

    def run_tests() -> str:
        outputs = []
        for test in sorted(root.glob("test_*.py")):
            proc = subprocess.run(
                [sys.executable, str(test)],
                capture_output=True, text=True, cwd=root, timeout=30,
            )
            status = "PASSED" if proc.returncode == 0 else "FAILED"
            detail = (proc.stdout + proc.stderr).strip()
            outputs.append(f"{test.name}: {status}\n{detail}")
        return "\n\n".join(outputs) or "no test files found"

    install_toolkits(registry, root, shell=False)  # filesystem.file.* confined to the workspace

    registry.register({
        "name": "dev.tests.run",
        "category": "dev",
        "spec": {
            "description": "ワークスペースの test_*.py を実行し結果を返す。",
            "parameters": {"type": "object", "properties": {}},
        },
        "discovery": {"embedding_text": "テスト実行 テストを走らせる 検証 pytest test run"},
        "execution": {"timeout_s": 60, "retry_safety": "never_retry",
                      "output_policy": {"max_inline_tokens": 1200}},
        "effects": [{"kind": "external", "resource": "sandbox:subprocess"}],
    }, handler=run_tests)

    return registry


CODING_KERNEL = """あなたはコーディングエージェントです。手順:
1. dev.tests.run でまず現状を確認する。
2. 失敗があれば filesystem.file.read で該当コードを読み、原因を特定する。
3. filesystem.file.write で修正し、必ず dev.tests.run で修正を検証する。
4. テストが全て通ったら、行った修正を簡潔に報告してから finish(result) を呼ぶ。"""

BUGGY_CALCULATOR = '''\
def divide(a, b):
    """Divide a by b. Must raise ValueError on b == 0."""
    return a / b


def add(a, b):
    return a + b
'''

CALCULATOR_TESTS = '''\
from calculator import add, divide

assert add(2, 3) == 5
assert divide(6, 2) == 3

try:
    divide(1, 0)
except ValueError:
    pass
else:
    raise AssertionError("divide(1, 0) must raise ValueError")

print("ALL TESTS PASSED")
'''

FIXED_CALCULATOR = '''\
def divide(a, b):
    """Divide a by b. Must raise ValueError on b == 0."""
    if b == 0:
        raise ValueError("division by zero is not allowed")
    return a / b


def add(a, b):
    return a + b
'''


def seed_workspace(root: Path) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "calculator.py").write_text(BUGGY_CALCULATOR, encoding="utf-8")
    (root / "test_calculator.py").write_text(CALCULATOR_TESTS, encoding="utf-8")


def make_session(llm, root: Path, **session_args) -> Session:
    """The coding agent over the workspace at ``root``. Its file tools live
    entirely inside that workspace, so they are granted up front instead of
    pausing on every write for approval."""
    session = Session(llm, kernel=CODING_KERNEL, registry=build_coding_registry(root), **session_args)
    session.policy.set_scope("workspace_write", "allow")
    session.policy.set_scope("sandbox_command", "allow")
    return session
