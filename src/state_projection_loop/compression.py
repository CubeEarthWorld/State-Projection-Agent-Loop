"""Deterministic content compression — pure functions, no LLM, no I/O.

Applied by the projection pipeline when rendering older events at reduced
fidelity. Every function here is total (never raises on any string input)
and idempotent where noted. The guarantee: compression never fabricates
content that was not in the original; it only removes or abbreviates.

Fidelity levels (the projection picks one per message; see
:class:`~state_projection_loop.projection.HistorySection`):

* ``full``       — verbatim, no compression
* ``compressed`` — pattern noise removed, long outputs head+tail truncated;
                   for tool results, :func:`mask_observation`
* ``summary``    — first meaningful line + token/line count
"""
from __future__ import annotations

import re

from .hashing import fnv1a_64_hex

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
    return fnv1a_64_hex(text)


def strip_noise(text: str) -> str:
    for pattern, repl in _NOISE_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def head_tail_truncate(text: str, max_lines: int) -> str:
    lines = text.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return text
    head_n = max(1, int(max_lines * _HEAD_RATIO))
    # The marker is itself a line, so head + tail + marker must still fit
    # `max_lines` — with the shipped defaults this clamp never binds.
    tail_n = max(0, min(max(1, int(max_lines * _TAIL_RATIO)), max_lines - head_n - 1))
    omitted = len(lines) - head_n - tail_n
    if omitted <= 0:
        return text
    head = lines[:head_n]
    tail = lines[-tail_n:] if tail_n > 0 else []
    marker = f"  [... {omitted} lines omitted ...]\n"
    result = "".join(head) + marker + "".join(tail)
    # Truncating many short lines costs more characters than it saves; the
    # budget is in characters, so hand back the original when that happens.
    return result if len(result) < len(text) else text


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


# What an error looks like in a tool result, whatever the language it is
# reported in: the structural traces first (exit codes, stack frames), then
# the word for it in the languages agents commonly work in. A result whose
# call *failed* is treated as an error without consulting this at all.
_ERROR_MARKER = re.compile(
    r"(?i)\b(error|traceback|exception|failed|denied|fatal|panic|fehler|erreur|errore)\b"
    # A process that exited non-zero: the whole code, not its first digit, and
    # never a bare "status" (an HTTP "status: 200" is not a failure).
    r"|\b(?:exit(?: code| status)?|returncode)[=: ]+(?!0+\b)\d{1,3}\b"
    # A stack frame's file:line. The token before the colon must contain a
    # non-digit, so a clock time ("at 14:30") is not mistaken for a frame.
    r"|\bline \d+, in \b|\bat [^\n]*[^\s:\d][^\s:]*:\d+"
    r"|エラー|失敗|例外|错误|失败|异常|오류|실패|ошибка|исключение"
)


def mask_observation(text: str, *, max_lines: int = 40, failed: bool = False) -> str:
    """The compressed form of an old tool result: cleared down to its first
    line and size — the model already acted on it — unless the call failed
    or the text reports an error, which stays readable (head and tail)
    because errors are what a later step most often needs to look back at."""
    if failed or _ERROR_MARKER.search(text):
        return compress_text(text, max_lines=max_lines)
    return summarize_text(text)


_IDENTIFIER = re.compile(r"[A-Za-z0-9_][\w./:-]*[\w/]")


def _grounded(haystack: str, token: str) -> bool:
    """Whether ``token`` occurs in ``haystack`` on its own, rather than
    buried inside a longer run of letters and digits.

    A plain substring test grounds an invented "order 942" on an unrelated
    "commit 8942", which is exactly the fabrication the caller is trying to
    catch. Punctuation still counts as a boundary, so "src/main.py" is
    grounded by "a/src/main.py:42".
    """
    start = haystack.find(token)
    while start >= 0:
        end = start + len(token)
        before = haystack[start - 1] if start else ""
        after = haystack[end] if end < len(haystack) else ""
        if not before.isalnum() and not after.isalnum():
            return True
        start = haystack.find(token, start + 1)
    return False


def ungrounded(entry: str, transcript: str) -> list[str]:
    """Identifier-like tokens of ``entry`` (paths, ids, numbers, names with
    digits or punctuation) that never occur in ``transcript``: the parts a
    summary could only have invented."""
    haystack = transcript.lower()
    return [
        token for token in _IDENTIFIER.findall(entry)
        if len(token) >= 3 and (any(ch.isdigit() for ch in token) or any(ch in "./:_-" for ch in token))
        and not _grounded(haystack, token.lower())
    ]


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
