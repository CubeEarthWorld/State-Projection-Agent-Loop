"""Live integration: embeddinggemma-300m GGUF via llama-cpp-python.

``LlamaCppEmbedding`` lives in ``examples/llm_adapters.py`` (a local model,
but still a concrete backend implementation, so it is kept out of the
LLM-agnostic package — see that module's docstring).

Skipped unless SPAL_RUN_LIVE=1 and llama-cpp-python is importable. The
model file resolves from SPAL_EMBED_GGUF or is downloaded from Hugging
Face on first run (cached afterwards).
"""
from __future__ import annotations

import os

import pytest

from state_projection_loop import Registry, ToolSearch
from state_projection_loop.embeddings import cosine

from examples.llm_adapters import LlamaCppEmbedding

from _util import capability_dict

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("SPAL_RUN_LIVE") != "1", reason="SPAL_RUN_LIVE != 1"),
]

llama_cpp = pytest.importorskip("llama_cpp", reason="llama-cpp-python not installed")


@pytest.fixture(scope="module")
def embedder() -> LlamaCppEmbedding:
    return LlamaCppEmbedding()


class TestEmbeddingBasics:
    def test_vectors_have_consistent_dimension(self, embedder):
        vecs = embedder.embed_documents(["水を浄化するフィルター", "コーヒーを淹れる機械"])
        assert len(vecs) == 2
        assert len(vecs[0]) == len(vecs[1]) > 100

    def test_semantic_similarity_beats_unrelated(self, embedder):
        q = embedder.embed_query("ウェブで最新ニュースを調べたい")
        related = embedder.embed_documents(["調べて 検索して 最新情報 ニュース ウェブ検索"])[0]
        unrelated = embedder.embed_documents(["BGM 音楽 曲を流す 雰囲気"])[0]
        assert cosine(q, related) > cosine(q, unrelated)


class TestVectorDiscovery:
    def test_layer2_semantic_tool_selection(self, embedder):
        reg = Registry()
        reg.register(capability_dict("web.search.query", category="web/search",
                                      summary="ウェブを検索し上位結果を返す",
                                      embedding_text="調べて 検索して 最新情報 ニュース 現在の 価格"))
        reg.register(capability_dict("game.media.play_bgm", category="game/media",
                                      summary="BGMを再生する",
                                      embedding_text="音楽 BGM 曲を流す 雰囲気"))
        reg.register(capability_dict("mail.send", category="mail",
                                      summary="メールを送信する",
                                      embedding_text="メール 送信 連絡 通知 mail send"))
        search = ToolSearch(reg, embedder=embedder, vector="on")

        top = search.search("今日の為替レートがいくらか知りたい", k=2, layer=2)
        assert top[0].tool.name == "web.search.query"

        top = search.search("しっとりした曲をかけて雰囲気を出して", k=2, layer=2)
        assert top[0].tool.name == "game.media.play_bgm"
