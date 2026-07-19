"""Embedding backend protocol (swappable abstraction).

The core never requires vectors: with no backend, discovery degrades to
lexical matching (BM25) and every capability stays reachable. The only
concrete implementation shipped here is :class:`HashingEmbedding` —
dependency-free and deterministic, calls no external model or API. Any real
embedding model (an OpenAI-compatible ``/embeddings`` endpoint, a local
GGUF model via llama-cpp-python, etc.) is the integrator's own adapter,
implementing this same two-method Protocol; see ``examples/llm_adapters.py``
for reference implementations. The package intentionally does not bundle
or depend on any LLM/embedding provider SDK.
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence, runtime_checkable

Vector = list[float]


@runtime_checkable
class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class HashingEmbedding:
    """Deterministic char-trigram hashing embedding. No dependencies, no
    network calls, no model weights.

    Not semantically smart, but stable, fast, and shares surface-form
    overlap, which is enough for tests and lightweight deployments.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> Vector:
        vec = [0.0] * self.dim
        text = (text or "").lower()
        for n in (2, 3):
            for i in range(max(0, len(text) - n + 1)):
                gram = text[i: i + n]
                h = int.from_bytes(hashlib.md5(gram.encode("utf-8")).digest()[:4], "little")
                vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)
