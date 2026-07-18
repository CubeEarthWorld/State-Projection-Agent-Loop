"""Coding-agent scenario toolkit: sandboxed file tools + a test runner.

All file access is confined to the workspace root (path-traversal safe);
``run_tests`` executes the workspace's test scripts in a subprocess and
reports pass/fail output as an observation.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from state_projection_loop import Registry


def build_coding_registry(root: Path) -> Registry:
    root = Path(root).resolve()
    registry = Registry()

    def _resolve(path: str) -> Path:
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root):
            raise PermissionError(f"path escapes the workspace: {path}")
        return candidate

    def list_files(pattern: str = "**/*") -> list[str]:
        return sorted(
            str(p.relative_to(root)).replace("\\", "/")
            for p in root.glob(pattern) if p.is_file()
        )

    def read_file(path: str) -> str:
        return _resolve(path).read_text(encoding="utf-8")

    def write_file(path: str, content: str) -> str:
        target = _resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"

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

    registry.register({
        "name": "list_files",
        "category": "file",
        "spec": {
            "description": "ワークスペース内のファイル一覧を返す。",
            "parameters": {"type": "object", "properties": {
                "pattern": {"type": "string", "default": "**/*"}}},
        },
        "discovery": {"embedding_text": "ファイル一覧 構成 どんなファイル ls list"},
        "execution": {"timeout_s": 10, "parallel_safe": True},
    }, handler=list_files)

    registry.register({
        "name": "read_file",
        "category": "file",
        "spec": {
            "description": "ワークスペース内のファイルを読む。",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}}, "required": ["path"]},
        },
        "discovery": {"embedding_text": "ファイルを読む 中身 コード 確認 read cat"},
        "execution": {"timeout_s": 10, "parallel_safe": True,
                      "output_policy": {"max_inline_tokens": 1200}},
    }, handler=read_file)

    registry.register({
        "name": "write_file",
        "category": "file/edit",
        "spec": {
            "description": "ワークスペース内のファイルへ全文を書き込む(上書き)。",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}}, "required": ["path", "content"]},
            "usage_notes": "部分編集ではなくファイル全文を渡すこと。",
        },
        "discovery": {"embedding_text": "ファイルを書く 保存 修正 編集 write save fix"},
        "execution": {"timeout_s": 10},
    }, handler=write_file)

    registry.register({
        "name": "run_tests",
        "category": "dev",
        "spec": {
            "description": "ワークスペースの test_*.py を実行し結果を返す。",
            "parameters": {"type": "object", "properties": {}},
        },
        "discovery": {"embedding_text": "テスト実行 テストを走らせる 検証 pytest test run"},
        "execution": {"timeout_s": 60,
                      "output_policy": {"max_inline_tokens": 1200}},
    }, handler=run_tests)

    return registry


CODING_KERNEL = """あなたはコーディングエージェントです。手順:
1. run_tests でまず現状を確認する。
2. 失敗があれば read_file で該当コードを読み、原因を特定する。
3. write_file で修正し、必ず run_tests で修正を検証する。
4. テストが全て通ったら、行った修正を簡潔に報告する。"""

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
