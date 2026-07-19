"""Discovery search engine: BM25, vector mixing, no_embed exclusion,
category filters, epoch-driven reindexing."""
from __future__ import annotations

import pytest

from state_projection_loop import HashingEmbedding, Registry, ToolSearch

from _util import capability_dict


def make_registry() -> Registry:
    reg = Registry()
    reg.register(capability_dict(
        "web.search", category="web/search",
        summary="ウェブを検索し上位結果を返す",
        tags=["web", "検索"],
        embedding_text="調べて 検索して 最新情報 ニュース 現在の 価格",
    ))
    reg.register(capability_dict(
        "file.read", category="file",
        summary="ファイルを読み込む",
        tags=["file"],
        embedding_text="ファイルを開く 読む 中身を見る open read file",
    ))
    reg.register(capability_dict(
        "game.media.play_bgm", category="game/media",
        summary="BGMを再生する",
        tags=["bgm", "音楽"],
        embedding_text="音楽 BGM 曲を流す 雰囲気",
    ))
    reg.register(capability_dict(
        "admin.secret_tool", category="admin",
        summary="hidden admin tool",
        no_embed=True,
    ))
    return reg


class TestLexical:
    def test_japanese_query_hits_embedding_text(self):
        search = ToolSearch(make_registry(), vector="off")
        results = search.search("最新情報を検索して", k=3)
        assert results[0].tool.name == "web.search"

    def test_english_query(self):
        search = ToolSearch(make_registry(), vector="off")
        results = search.search("read a file", k=3)
        assert results[0].tool.name == "file.read"

    def test_exact_name_query_ranks_first(self):
        search = ToolSearch(make_registry(), vector="off")
        results = search.search("play_bgm", k=3)
        assert results[0].tool.name == "game.media.play_bgm"

    def test_empty_query_returns_nothing(self):
        search = ToolSearch(make_registry(), vector="off")
        assert search.search("") == []
        assert search.search("   ") == []


class TestLayers:
    def test_layer2_excludes_no_embed(self):
        search = ToolSearch(make_registry(), vector="off")
        names = [s.tool.name for s in search.search("hidden admin tool", layer=2, k=8)]
        assert "admin.secret_tool" not in names

    def test_layer3_reaches_no_embed(self):
        search = ToolSearch(make_registry(), vector="off")
        names = [s.tool.name for s in search.search("hidden admin tool", layer=3, k=8)]
        assert "admin.secret_tool" in names

    def test_exclude_set(self):
        search = ToolSearch(make_registry(), vector="off")
        names = [s.tool.name for s in search.search("検索して", exclude={"web.search"}, k=8)]
        assert "web.search" not in names


class TestCategoryFilter:
    def test_prefix_match(self):
        search = ToolSearch(make_registry(), vector="off")
        results = search.search("検索", category="web", k=8)
        assert {s.tool.name for s in results} == {"web.search"}

    def test_exact_match(self):
        search = ToolSearch(make_registry(), vector="off")
        results = search.search("BGM 音楽", category="game/media", k=8)
        assert [s.tool.name for s in results] == ["game.media.play_bgm"]


class TestVectorMixing:
    def test_vector_on_requires_backend(self):
        with pytest.raises(ValueError, match="requires an embedding backend"):
            ToolSearch(make_registry(), vector="on")

    def test_vector_component_present_with_backend(self):
        search = ToolSearch(make_registry(), embedder=HashingEmbedding(), vector="on")
        results = search.search("最新情報を検索して", k=3)
        assert results[0].tool.name == "web.search"
        assert "vector" in results[0].components
        assert "lexical" in results[0].components

    def test_vector_off_ignores_backend(self):
        search = ToolSearch(make_registry(), embedder=HashingEmbedding(), vector="off")
        results = search.search("検索して", k=3)
        assert all("vector" not in s.components for s in results)

    def test_invalid_vector_mode(self):
        with pytest.raises(ValueError, match="auto|on|off"):
            ToolSearch(make_registry(), vector="maybe")


class TestReindexing:
    def test_new_tool_found_after_epoch_bump(self):
        reg = make_registry()
        search = ToolSearch(reg, vector="off")
        assert not any(s.tool.name == "text.translate" for s in search.search("翻訳 translate", k=8))
        reg.register(capability_dict("text.translate", summary="テキストを翻訳する",
                                      embedding_text="翻訳して 英語に 日本語に translate"))
        results = search.search("翻訳して translate", k=8)
        assert results[0].tool.name == "text.translate"
