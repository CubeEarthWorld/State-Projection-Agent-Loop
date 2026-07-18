"""Tool definition parsing, card derivation (§4), and the @tool decorator."""
from __future__ import annotations

from typing import Optional

import pytest

from state_projection_loop import ToolContext, ToolDef, tool
from state_projection_loop.tooldef import synthesize_signature

WEB_SEARCH = {
    "name": "web_search",
    "category": "web/search",
    "card": {
        "summary": "ウェブを検索し上位結果(タイトル・URL・抜粋)を返す",
        "signature": "web_search(query: str, max_results: int = 5) -> list[SearchResult]",
        "tags": ["web", "検索", "調査"],
    },
    "spec": {
        "description": "検索エンジン経由でウェブ全体を検索する。結果は関連度順。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "検索クエリ。1〜6語が最適"},
                "max_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
        "usage_notes": "変化しうる事実・最新情報の確認に使う。",
        "examples": [{"call": {"query": "USD JPY 為替 今日"}, "note": "単純な事実確認は1回で足りる"}],
    },
    "discovery": {"embedding_text": "調べて 検索して 最新情報 ニュース 現在の 価格 いつ 誰が"},
    "execution": {"timeout_s": 20, "retries": 2, "parallel_safe": True,
                  "output_policy": {"max_inline_tokens": 800, "overflow": "handle", "preview": "head"}},
}


class TestFromDict:
    def test_full_spec_json(self):
        td = ToolDef.from_dict(WEB_SEARCH)
        assert td.name == "web_search"
        assert td.category == "web/search"
        assert td.card.summary.startswith("ウェブを検索")
        assert td.card.tags == ["web", "検索", "調査"]
        assert td.discovery.embedding_text.startswith("調べて")
        assert td.execution.timeout_s == 20
        assert td.execution.retries == 2
        assert td.execution.parallel_safe is True
        assert td.execution.output_policy.max_inline_tokens == 800

    def test_name_required(self):
        with pytest.raises(ValueError, match="name"):
            ToolDef.from_dict({"category": "x"})

    def test_card_auto_derivation(self):
        td = ToolDef.from_dict({
            "name": "web_search",
            "spec": WEB_SEARCH["spec"],
        })
        assert td.card.summary == "検索エンジン経由でウェブ全体を検索する。"
        assert "query: str" in td.card.signature
        assert "max_results: int = 5" in td.card.signature

    def test_card_text_is_compact(self):
        from state_projection_loop.tokens import estimate_tokens

        td = ToolDef.from_dict(WEB_SEARCH)
        assert estimate_tokens(td.card_text()) <= 60  # card ≈ 30tk target (§4.2)
        assert td.card_text().startswith("- web_search(")

    def test_spec_text_contains_schema_and_examples(self):
        td = ToolDef.from_dict(WEB_SEARCH)
        text = td.spec_text()
        assert "### web_search" in text
        assert '"required": ["query"]' in text
        assert "Usage notes:" in text
        assert "USD JPY" in text

    def test_api_schema_shape(self):
        td = ToolDef.from_dict(WEB_SEARCH)
        schema = td.api_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "web_search"
        assert schema["function"]["parameters"]["required"] == ["query"]
        assert "Usage:" in schema["function"]["description"]

    def test_embedding_source_prefers_embedding_text(self):
        td = ToolDef.from_dict(WEB_SEARCH)
        assert td.embedding_source() == WEB_SEARCH["discovery"]["embedding_text"]
        td2 = ToolDef.from_dict({"name": "x", "card": {"summary": "does x", "tags": ["a"]},
                                 "spec": {"parameters": {"type": "object", "properties": {}}}})
        assert td2.embedding_source() == "does x a"

    def test_missing_parameters_defaults_to_empty_object(self):
        td = ToolDef.from_dict({"name": "noop"})
        assert td.spec.parameters == {"type": "object", "properties": {}}


class TestSignatureSynthesis:
    def test_types_defaults_and_nullables(self):
        sig = synthesize_signature("f", {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer", "default": 3},
                "c": {"type": ["string", "null"]},
            },
            "required": ["a"],
        }, returns={"type": "array"})
        assert sig == "f(a: str, b: int = 3, c: str | None = None) -> list"


class TestDecorator:
    def test_metadata_from_hints_and_docstring(self):
        @tool(category="math", tags=["calc"])
        def add(a: int, b: int = 2) -> int:
            """Add two numbers.

            Args:
                a: first operand
                b: second operand
            """
            return a + b

        td: ToolDef = add.__spal_tool__
        assert td.name == "add"
        assert td.category == "math"
        assert td.spec.description == "Add two numbers."
        props = td.spec.parameters["properties"]
        assert props["a"] == {"type": "integer", "description": "first operand"}
        assert props["b"]["type"] == "integer"
        assert props["b"]["default"] == 2
        assert td.spec.parameters["required"] == ["a"]
        assert td.execution.handler is add
        assert td.wants_ctx is False

    def test_ctx_param_excluded_and_detected(self):
        @tool
        def stateful(ctx: ToolContext, key: str) -> str:
            """Read a state key."""
            return str(ctx.state.get(key))

        td: ToolDef = stateful.__spal_tool__
        assert "ctx" not in td.spec.parameters["properties"]
        assert td.spec.parameters["required"] == ["key"]
        assert td.wants_ctx is True

    def test_optional_and_list_hints(self):
        @tool
        def f(names: list[str], limit: Optional[int] = None) -> str:
            """Do f."""
            return ""

        props = f.__spal_tool__.spec.parameters["properties"]
        assert props["names"] == {"type": "array", "items": {"type": "string"}}
        assert props["limit"]["type"] == ["integer", "null"]

    def test_bare_decorator(self):
        @tool
        def g(x: str) -> str:
            """G tool."""
            return x

        assert g.__spal_tool__.name == "g"
        assert g("a") == "a"  # function unchanged
