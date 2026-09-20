"""Token estimation utilities.

Budgets are enforced against a conservative estimate, never an exact
tokenizer count. The estimator is pluggable via ``set_estimator`` so a real
tokenizer can be swapped in when precision matters.

Heuristic: CJK characters count as ~1 token each, everything else as ~1
token per 4 characters. This overestimates slightly for English and is close
for Japanese, which keeps budget enforcement on the safe side.
"""
from __future__ import annotations

import math
from typing import Any, Callable

from .serialization import dumps

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
    # 0x1100 is the lowest range start, so this rejects ASCII — the common
    # case, run per character of every message — in one comparison.
    return o >= 0x1100 and any(lo <= o <= hi for lo, hi in _CJK_RANGES)


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for c in text if _is_cjk(c))
    other = len(text) - cjk
    return cjk + math.ceil(other / 4)


_estimator: Callable[[str], int] = estimate_text_tokens


def set_estimator(fn: Callable[[str], int]) -> None:
    """Replace the global token estimator (e.g. with a real tokenizer).

    The counterpart of the Dart port's ``setEstimator``. Without it the
    ``_estimator`` indirection below would be flexibility nothing can reach.
    """
    global _estimator
    _estimator = fn

# What one image part costs: a provider's typical per-image charge. Counting
# the base64 text instead would overshoot by two orders of magnitude.
IMAGE_TOKENS = 1000


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
            total += 6 + _estimator(getattr(tc, "name", "")) + _estimator(dumps(getattr(tc, "arguments", {})))
        return total
    if isinstance(obj, dict):
        if obj.get("type") in ("image_url", "image"):
            return IMAGE_TOKENS
        return _estimator(dumps(obj))
    return _estimator(str(obj))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """The longest prefix of ``text`` that fits ``max_tokens``."""
    if estimate_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]
