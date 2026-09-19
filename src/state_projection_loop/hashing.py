"""FNV-1a, the one hash both ports share.

Two call sites need a stable hash: the compression fingerprint and the
bag-of-ngrams embedding. Neither is a security boundary - the fingerprint
dedupes identical text and the embedding buckets features - so neither
needs a cryptographic primitive, and the previous SHA-256 was truncated to
64 bits anyway, which leaves roughly 32 bits of collision resistance under
the birthday bound.

FNV-1a is chosen because it is the same handful of lines in every
language. Dart has no standard-library digest, so any cryptographic choice
would mean either a third-party package or a hand-written SHA-256; this
keeps both ports dependency-free and byte-identical instead.

``fnv1a_64`` and ``fnv1a_32`` must stay bit-identical to
``lib/src/hashing.dart``; ``spec/fixtures/compression.json`` pins that.
"""
from __future__ import annotations

_FNV64_OFFSET = 0xCBF29CE484222325
_FNV64_PRIME = 0x100000001B3
_FNV32_OFFSET = 0x811C9DC5
_FNV32_PRIME = 0x01000193
_MASK64 = 0xFFFFFFFFFFFFFFFF
_MASK32 = 0xFFFFFFFF


def fnv1a_64(data: bytes) -> int:
    h = _FNV64_OFFSET
    for byte in data:
        h = ((h ^ byte) * _FNV64_PRIME) & _MASK64
    return h


def fnv1a_32(data: bytes) -> int:
    h = _FNV32_OFFSET
    for byte in data:
        h = ((h ^ byte) * _FNV32_PRIME) & _MASK32
    return h


def fnv1a_64_hex(text: str) -> str:
    """16 lowercase hex digits - the full 64-bit digest, not a truncation."""
    return format(fnv1a_64(text.encode("utf-8", errors="replace")), "016x")
