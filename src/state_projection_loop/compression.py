"""Deterministic content compression — pure functions, no LLM, no I/O.

Applied by the projection pipeline when rendering older events at reduced
fidelity. Every function here is total (never raises on any string input)
and idempotent where noted. The guarantee: compression never fabricates
content that was not in the original; it only removes or abbreviates.

Fidelity levels (applied by projection based on event age):

* ``full``       — verbatim, no compression
* ``compressed`` — pattern noise removed, long outputs head+tail truncated
* ``summary``    — first meaningful line + token/line count
* ``handle``     — artifact reference only (projection decides externally)
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional

_NOISE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^diff --git .+\n", re.M), ""),
    (re.compile(r"^index [0-9a-f]+\.\.[0-9a-f]+.*\n", re.M), ""),
    (re.compile(r"^--- a/.+\n", re.M), ""),
    (re.compile(r"^\+\+\+ b/.+\n", re.M), ""),
    (re.compile(r"^@@ [^@]+ @@[^\n]*\n", re.M), ""),
    (re.compile(r"^(node_modules|\.venv|__pycache__|\.git/|\.dart_tool/)[^\n]*\n", re.M), ""),
    (re.compile(r"^\s*$\n(\s*$\n)+", re.M), "\n"),
    (re.compile(r"\x1b\[[0-9;]*[a-zA-Z]"), ""),
    (re.compile(r"^Progress:.*\r?", re.M), ""),
    (re.compile(r"^\[?\d+/\d+\]?\s*(Downloading|Installing|Collecting|Using cached)[^\n]*\n", re.M), ""),
]

_HEAD_RATIO = 0.6
_TAIL_RATIO = 0.25


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def strip_noise(text: str) -> str:
    for pattern, repl in _NOISE_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def head_tail_truncate(text: str, max_lines: int) -> str:
    lines = text.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return text
    head_n = max(1, int(max_lines * _HEAD_RATIO))
    tail_n = max(1, int(max_lines * _TAIL_RATIO))
    omitted = len(lines) - head_n - tail_n
    if omitted <= 0:
        return text
    head = lines[:head_n]
    tail = lines[-tail_n:] if tail_n > 0 else []
    marker = f"  [... {omitted} lines omitted ...]\n"
    return "".join(head) + marker + "".join(tail)


def compress_text(text: str, *, max_lines: int = 80) -> str:
    """Full compression pipeline: strip noise then truncate.

    Idempotent for already-short texts. Never returns empty for non-empty
    input — at minimum the first line survives.
    """
    if not text:
        return text
    result = strip_noise(text)
    result = head_tail_truncate(result, max_lines)
    if not result.strip() and text.strip():
        result = text.splitlines(keepends=True)[0]
    return result


def first_meaningful_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "//", "/*", "*", "---")):
            return stripped
    lines = text.splitlines()
    return lines[0].strip() if lines else ""


def summarize_text(text: str) -> str:
    """Reduce to a single descriptive line preserving the most salient content."""
    if not text:
        return text
    first = first_meaningful_line(text)
    line_count = text.count("\n") + 1
    char_count = len(text)
    if line_count <= 1 and char_count <= 120:
        return text.strip()
    suffix = f"  [{line_count} lines, {char_count} chars]"
    max_first = 120
    if len(first) > max_first:
        first = first[:max_first] + "…"
    return first + suffix


def compress_observation(text: str, *, max_lines: int = 40) -> str:
    """Aggressive compression for tool observations at 'compressed' fidelity.

    Tool outputs tend to be noisier than user/assistant text (build logs,
    diffs, stack traces), so we use a tighter line budget and the same
    noise-stripping pipeline.
    """
    return compress_text(text, max_lines=max_lines)


def dedupe_key(content: str) -> str:
    """Content-addressed key for detecting repeated observations."""
    return content_hash(content)
