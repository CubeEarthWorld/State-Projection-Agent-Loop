"""Tool discovery search engine, shared by layer 2 (auto candidates) and
layer 3 (find_tools) — spec §5, §9.

Pure computation: no LLM is ever involved. Scoring mixes vector similarity,
BM25 lexical match, and tag/name match. With vectors disabled or
unavailable, weights renormalize over the remaining components (§9).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from .capability import Capability
from .embeddings import EmbeddingBackend, cosine
from .registry import Registry

_ASCII_WORD = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """ASCII words + CJK bigrams (unigram only for isolated single chars).

    Single-char CJK grams are deliberately excluded from longer runs:
    particles like 「の」「ー」 match almost everything and turn BM25 into
    noise (found by the embeddinggemma integration test).
    """
    text = (text or "").lower()
    tokens = _ASCII_WORD.findall(text)
    cjk_run: list[str] = []

    def flush() -> None:
        if not cjk_run:
            return
        if len(cjk_run) == 1:
            tokens.append(cjk_run[0])
        else:
            tokens.extend(a + b for a, b in zip(cjk_run, cjk_run[1:]))
        cjk_run.clear()

    for ch in text:
        o = ord(ch)
        if 0x3000 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0xAC00 <= o <= 0xD7AF:
            cjk_run.append(ch)
        else:
            flush()
    flush()
    return tokens


@dataclass
class ScoredTool:
    tool: Capability
    score: float
    components: dict[str, float] = field(default_factory=dict)


class ToolSearch:
    """Mixed vector + lexical + tag scoring over the registry.

    The index rebuilds lazily whenever the registry epoch changes, so
    provider-driven tool changes are picked up automatically.
    """

    DEFAULT_WEIGHTS = {"vector": 0.55, "lexical": 0.30, "tags": 0.15}

    def __init__(
        self,
        registry: Registry,
        *,
        embedder: Optional[EmbeddingBackend] = None,
        vector: str = "auto",
        weights: Optional[dict[str, float]] = None,
    ) -> None:
        if vector not in ("auto", "on", "off"):
            raise ValueError(f"vector must be auto|on|off, got {vector!r}")
        if vector == "on" and embedder is None:
            raise ValueError("discovery.vector='on' requires an embedding backend")
        self.registry = registry
        self.embedder = embedder if vector != "off" else None
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)
        self._epoch = -1
        self._doc_tokens: dict[str, list[str]] = {}
        self._df: dict[str, int] = {}
        self._avgdl = 1.0
        self._vectors: dict[str, list[float]] = {}

    # -- index --------------------------------------------------------------

    def _doc_text(self, tool: Capability) -> str:
        parts = [
            tool.name.replace("_", " "),
            tool.name,
            tool.category,
            tool.card.summary,
            " ".join(tool.card.tags),
            tool.discovery.embedding_text,
            tool.spec.description,
        ]
        return " ".join(p for p in parts if p)

    def _ensure_index(self) -> None:
        if self._epoch == self.registry.epoch:
            return
        self._doc_tokens = {t.name: _tokenize(self._doc_text(t)) for t in self.registry}
        self._df = {}
        for toks in self._doc_tokens.values():
            for tok in set(toks):
                self._df[tok] = self._df.get(tok, 0) + 1
        lengths = [len(t) for t in self._doc_tokens.values()]
        self._avgdl = (sum(lengths) / len(lengths)) if lengths else 1.0
        if self.embedder is not None:
            to_embed = [t for t in self.registry if not t.discovery.no_embed]
            texts = [t.embedding_source() for t in to_embed]
            vecs = self.embedder.embed_documents(texts) if texts else []
            self._vectors = {t.name: v for t, v in zip(to_embed, vecs)}
        else:
            self._vectors = {}
        self._epoch = self.registry.epoch

    # -- scoring ------------------------------------------------------------

    def _bm25(self, query_tokens: list[str], name: str, k1: float = 1.5, b: float = 0.75) -> float:
        toks = self._doc_tokens.get(name, [])
        if not toks:
            return 0.0
        n_docs = max(1, len(self._doc_tokens))
        counts: dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        score = 0.0
        for qt in query_tokens:
            tf = counts.get(qt, 0)
            if tf == 0:
                continue
            df = self._df.get(qt, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * len(toks) / self._avgdl)
            score += idf * (tf * (k1 + 1)) / denom
        return score

    def _tag_score(self, query: str, query_tokens: list[str], tool: Capability) -> float:
        q = query.lower()
        score = 0.0
        if tool.name.lower() in q or q.strip() == tool.name.lower():
            score += 1.0
        qset = set(query_tokens)
        tags = {t.lower() for t in tool.card.tags}
        if tags:
            overlap = sum(1 for t in tags if t in qset or t in q)
            score += min(1.0, overlap / max(1, len(tags)) * 1.5)
        return min(score, 1.0)

    def search(
        self,
        query: str,
        *,
        category: Optional[str] = None,
        k: int = 8,
        layer: int = 3,
        exclude: Optional[set[str]] = None,
    ) -> list[ScoredTool]:
        """Rank tools for a natural-language query.

        ``layer=2`` (auto candidates) excludes ``no_embed`` tools (§4.2);
        ``layer=3`` (find_tools) searches everything.
        """
        self._ensure_index()
        exclude = exclude or set()
        tools = []
        for t in self.registry:
            if t.name in exclude:
                continue
            if layer == 2 and t.discovery.no_embed:
                continue
            if category:
                cat = t.category or "misc"
                norm = category.rstrip("/*").rstrip("/")
                if not (cat == norm or cat.startswith(norm + "/")):
                    continue
            tools.append(t)
        if not tools or not (query or "").strip():
            return []

        query_tokens = _tokenize(query)
        # absolute squash, not max-normalization: a tiny incidental match
        # must stay tiny instead of being amplified to full scale
        lexical = {t.name: self._bm25(query_tokens, t.name) for t in tools}

        qvec: Optional[list[float]] = None
        if self.embedder is not None and self._vectors:
            qvec = self.embedder.embed_query(query)

        results: list[ScoredTool] = []
        for t in tools:
            components: dict[str, float] = {}
            weights: dict[str, float] = {}
            if qvec is not None and t.name in self._vectors:
                components["vector"] = (cosine(qvec, self._vectors[t.name]) + 1.0) / 2.0
                weights["vector"] = self.weights["vector"]
            components["lexical"] = lexical[t.name] / (lexical[t.name] + 1.0)
            weights["lexical"] = self.weights["lexical"]
            components["tags"] = self._tag_score(query, query_tokens, t)
            weights["tags"] = self.weights["tags"]
            wsum = sum(weights.values()) or 1.0
            score = sum(components[k_] * weights[k_] for k_ in components) / wsum
            if score > 0:
                results.append(ScoredTool(tool=t, score=score, components=components))
        results.sort(key=lambda s: s.score, reverse=True)
        return results[:k]
