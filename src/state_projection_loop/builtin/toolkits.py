"""Toolkits: root-confined filesystem and shell capabilities.

Off unless installed: a generic runtime must not assume a filesystem.
Definitions are the shared ``toolkits.json``; every path is resolved under
``root`` and rejected if it escapes it.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..registry import Registry
from . import _install
from .defs import load


def install_toolkits(registry: Registry, root: str | Path, *, shell: bool = True) -> None:
    """Install ``filesystem.file.*`` (and, with ``shell``, ``shell.command.run``)
    confined to ``root``."""
    root_path = Path(root).resolve()

    def resolve(relative: str) -> Path:
        candidate = (root_path / relative).resolve()
        if candidate != root_path and root_path not in candidate.parents:
            raise ValueError(f"path escapes the workspace: {relative}")
        return candidate

    def list_files(path: str = "") -> list[str]:
        directory = resolve(path)
        if not directory.exists():
            return []
        return sorted(
            str(p.relative_to(root_path)).replace("\\", "/") for p in directory.rglob("*") if p.is_file()
        )

    def read(path: str) -> str:
        return resolve(path).read_text(encoding="utf-8")

    def write(path: str, content: str) -> str:
        target = resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"

    def run(command: str) -> str:
        proc = subprocess.run(command, shell=True, cwd=root_path, capture_output=True, text=True)
        return f"exit={proc.returncode}\n{proc.stdout}{proc.stderr}".rstrip()

    handlers = {
        "filesystem.file.list": list_files,
        "filesystem.file.read": read,
        "filesystem.file.write": write,
        **({"shell.command.run": run} if shell else {}),
    }
    _install(registry, [d for d in load("toolkits") if d["name"] in handlers], handlers)
