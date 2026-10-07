"""Pinned agent tools. File operations stay under root; bash uses root as cwd."""
from __future__ import annotations

import subprocess
import fnmatch
import os
import re
import shutil
from pathlib import Path

from ..registry import Registry
from . import _install
from .defs import load


def install_toolkits(registry: Registry, root: str | Path, *, shell: bool = True) -> None:
    """Install read/write/edit/grep/find/ls and optionally bash, all pinned."""
    root_path = Path(root).resolve()

    def resolve(relative: str) -> Path:
        candidate = (root_path / relative).resolve()
        if candidate != root_path and root_path not in candidate.parents:
            raise ValueError(f"path escapes the workspace: {relative}")
        return candidate

    def list_files(path: str = "", pattern: str = "*", max_results: int = 1000) -> list[str]:
        directory = resolve(path)
        if not directory.exists():
            return []
        matches = []
        for p in directory.rglob("*"):
            if not p.is_file() or not p.resolve().is_relative_to(root_path):
                continue
            relative = p.relative_to(root_path).as_posix()
            if fnmatch.fnmatchcase(p.name, pattern) or fnmatch.fnmatchcase(relative, pattern):
                matches.append(relative)
                if len(matches) >= max_results:
                    break
        return sorted(matches)

    def ls(path: str = "") -> list[str]:
        return sorted(p.relative_to(root_path).as_posix() + ("/" if p.is_dir() else "")
                      for p in resolve(path).iterdir() if p.resolve().is_relative_to(root_path))

    def read(path: str) -> str:
        return resolve(path).read_text(encoding="utf-8")

    def write(path: str, content: str) -> str:
        target = resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"

    def edit(path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        if not old_text:
            raise ValueError("old_text must not be empty")
        content = read(path)
        count = content.count(old_text)
        if count == 0 or (count > 1 and not replace_all):
            raise ValueError(f"old_text matched {count} times; require one match or replace_all=True")
        return write(path, content.replace(old_text, new_text, -1 if replace_all else 1))

    def grep(pattern: str, path: str = "", include: str = "*", regex: bool = False,
             max_results: int = 100) -> list[dict]:
        matcher = re.compile(pattern if regex else re.escape(pattern))
        target = resolve(path)
        candidates = [target] if target.is_file() else target.rglob("*")
        matches = []
        for candidate in candidates:
            if not candidate.is_file() or not candidate.resolve().is_relative_to(root_path):
                continue
            relative = candidate.relative_to(root_path).as_posix()
            if not (fnmatch.fnmatchcase(candidate.name, include) or fnmatch.fnmatchcase(relative, include)):
                continue
            try:
                with candidate.open(encoding="utf-8") as source:
                    for line, text in enumerate(source, 1):
                        if matcher.search(text):
                            matches.append({"path": relative, "line": line, "text": text.rstrip("\r\n")})
                            if len(matches) >= max_results:
                                return matches
            except UnicodeDecodeError:
                continue  # Skip non-UTF-8 files.
        return matches

    def run(command: str, timeout: int = 60) -> str:
        executable = shutil.which("bash")
        if os.name == "nt":
            git = shutil.which("git")
            git_bash = Path(git).parent.parent / "bin/bash.exe" if git else None
            if git_bash is not None and git_bash.is_file():
                executable = str(git_bash)
        if executable is None:
            raise RuntimeError("bash is not installed; install Bash or supply your own bash tool")
        proc = subprocess.run([executable, "-lc", command], cwd=root_path, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return f"exit={proc.returncode}\n{proc.stdout}{proc.stderr}".rstrip()

    handlers = {
        "find": list_files,
        "read": read,
        "write": write,
        "edit": edit,
        "grep": grep,
        "ls": ls,
        **({"bash": run} if shell else {}),
    }
    _install(registry, [d for d in load("toolkits") if d["name"] in handlers], handlers)
