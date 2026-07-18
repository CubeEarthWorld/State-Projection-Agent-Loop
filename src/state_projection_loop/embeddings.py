"""Embedding backends (spec §9 — swappable abstraction).

The core never requires vectors: with no backend, discovery degrades to
lexical matching (BM25) and every tool stays reachable (invariant I10).

Backends:

* :class:`HashingEmbedding` — dependency-free deterministic character
  n-gram hashing. Useful for tests and as a cheap semantic-ish fallback.
* :class:`LlamaCppEmbedding` — GGUF models via ``llama-cpp-python``
  (optional extra ``embeddings``). Defaults target
  ``google/embeddinggemma-300m`` community GGUF builds and apply the
  EmbeddingGemma prompt prefixes for queries vs. documents.
"""
from __future__ import annotations

import hashlib
import math
import os
from typing import Optional, Protocol, Sequence, runtime_checkable

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
    """Deterministic char-trigram hashing embedding. No dependencies.

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
                gram = text[i : i + n]
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


# EmbeddingGemma prompt prefixes (model card recommendation).
GEMMA_QUERY_PREFIX = "task: search result | query: "
GEMMA_DOC_PREFIX = "title: none | text: "

DEFAULT_GGUF_REPO = "ggml-org/embeddinggemma-300M-GGUF"
DEFAULT_GGUF_FILE = "embeddinggemma-300M-Q8_0.gguf"


class LlamaCppEmbedding:
    """GGUF embedding via llama-cpp-python (optional).

    Resolution order for the model file: explicit ``model_path`` argument →
    ``SPAL_EMBED_GGUF`` env var → download ``repo_id``/``filename`` from
    Hugging Face into the local HF cache (requires ``huggingface-hub``).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        repo_id: str = DEFAULT_GGUF_REPO,
        filename: str = DEFAULT_GGUF_FILE,
        n_ctx: int = 2048,
        query_prefix: str = GEMMA_QUERY_PREFIX,
        doc_prefix: str = GEMMA_DOC_PREFIX,
        verbose: bool = False,
    ) -> None:
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        path = model_path or os.environ.get("SPAL_EMBED_GGUF") or ""
        if not path:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "No GGUF path given and huggingface-hub is not installed. "
                    "Install the 'embeddings' extra or set SPAL_EMBED_GGUF."
                ) from exc
            path = hf_hub_download(repo_id=repo_id, filename=filename)
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "llama-cpp-python is required for GGUF embeddings. "
                "Install the 'embeddings' extra: pip install state-projection-loop[embeddings]"
            ) from exc
        self._llama = Llama(model_path=path, embedding=True, n_ctx=n_ctx, verbose=verbose)

    def _embed_one(self, text: str) -> Vector:
        out = self._llama.embed(text)
        # llama-cpp may return a single vector or a list of per-token vectors.
        if out and isinstance(out[0], (list, tuple)):
            dim = len(out[0])
            pooled = [sum(tok[i] for tok in out) / len(out) for i in range(dim)]
            return [float(x) for x in pooled]
        return [float(x) for x in out]

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed_one(self.doc_prefix + t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self._embed_one(self.query_prefix + text)
