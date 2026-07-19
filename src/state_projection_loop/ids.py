"""Identifiers: dependency-free ULIDs with typed, human-readable prefixes.

Every entity that appears in the Event Ledger carries a prefixed ULID
(``ses_``, ``run_``, ``evt_``, ``cmd_``, ``apr_``, ``art_``). ULIDs are
lexicographically sortable by creation time, which keeps ledger files and
directory listings naturally ordered without a separate index.

Event *order within a run* is never inferred from the ID's timestamp —
that's what :class:`~state_projection_loop.events.EventLedger` sequence
numbers are for. IDs only need to be unique and roughly time-ordered.
"""
from __future__ import annotations

import os
import threading
import time

_CROCKFORD32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_LOCK = threading.Lock()
_LAST_MS = 0
_LAST_RANDOM = 0


def _encode(value: int, length: int) -> str:
    chars = ["0"] * length
    for i in range(length - 1, -1, -1):
        chars[i] = _CROCKFORD32[value & 0x1F]
        value >>= 5
    return "".join(chars)


def new_ulid() -> str:
    """A 26-char Crockford-base32 ULID: 48-bit ms timestamp + 80-bit random.

    Monotonic within a process: if called twice in the same millisecond the
    random part is incremented instead of redrawn, so IDs generated back to
    back still sort in call order.
    """
    global _LAST_MS, _LAST_RANDOM
    with _LOCK:
        ms = int(time.time() * 1000)
        if ms <= _LAST_MS:
            ms = _LAST_MS
            _LAST_RANDOM += 1
        else:
            _LAST_RANDOM = int.from_bytes(os.urandom(10), "big")
        _LAST_MS = ms
        random_part = _LAST_RANDOM & ((1 << 80) - 1)
    return _encode(ms, 10) + _encode(random_part, 16)


_PREFIXES = {
    "session": "ses",
    "run": "run",
    "event": "evt",
    "command": "cmd",
    "approval": "apr",
    "artifact": "art",
    "branch": "brn",
}


def new_id(kind: str) -> str:
    """A prefixed ULID for the given entity kind, e.g. ``new_id("run")``."""
    prefix = _PREFIXES.get(kind)
    if prefix is None:
        raise ValueError(f"Unknown id kind {kind!r}; expected one of {sorted(_PREFIXES)}")
    return f"{prefix}_{new_ulid()}"


def kind_of(entity_id: str) -> str:
    """Reverse-lookup the entity kind from a prefixed id (for assertions/logging)."""
    prefix, _, _ = entity_id.partition("_")
    for kind, p in _PREFIXES.items():
        if p == prefix:
            return kind
    return "unknown"
