"""Token estimation utilities (spec §3.3, §13).

Budgets are enforced against a conservative estimate, never an exact
tokenizer count. The estimator is pluggable via :func:`set_estimator` so a
real tokenizer can be swapped in when precision matters.

Heuristic: CJK characters count as ~1 token each, everything else as ~1
token per 4 characters. This overestimates slightly for English and is close
for Japanese, which keeps budget enforcement on the safe side.
"""
from __future__ import annotations

import json
import math
from typing import Any, Callable

_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x2E80, 0x2FDF),  # CJK radicals
    (0x3000, 0x303F),  # CJK punctuation
    (0x3040, 0x30FF),  # Hiragana / Katakana
    (0x3130, 0x318F),  # Hangul compatibility Jamo
    (0x3400, 0x4DBF),  # CJK ext A
    (0x4E00, 0x9FFF),  # CJK unified
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK compat ideographs
    (0xFF00, 0xFFEF),  # fullwidth forms
)


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _CJK_RANGES)


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for c in text if _is_cjk(c))
    other = len(text) - cjk
    return cjk + math.ceil(other / 4)


_estimator: Callable[[str], int] = estimate_text_tokens


def set_estimator(fn: Callable[[str], int]) -> None:
    """Replace the global token estimator (e.g. with a real tokenizer)."""
    global _estimator
    _estimator = fn


def estimate_tokens(obj: Any) -> int:
    """Estimate tokens for text, Message-like objects, or containers."""
    if obj is None:
        return 0
    if isinstance(obj, str):
        return _estimator(obj)
    if isinstance(obj, (list, tuple)):
        return sum(estimate_tokens(x) for x in obj)
    if hasattr(obj, "role") and hasattr(obj, "content"):  # Message-like
        total = 4 + estimate_tokens(obj.content)
        for tc in getattr(obj, "tool_calls", None) or []:
            args = getattr(tc, "arguments", {})
            total += 6 + _estimator(getattr(tc, "name", "")) + _estimator(
                json.dumps(args, ensure_ascii=False, default=str)
            )
        return total
    if isinstance(obj, dict):
        return _estimator(json.dumps(obj, ensure_ascii=False, default=str))
    return _estimator(str(obj))
