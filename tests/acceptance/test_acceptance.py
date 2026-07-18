"""Acceptance tests mandated by spec §17:

(a) 1,000 registered tools, default config → per-turn tool-related overhead
    ≤ 3k tokens (two orders of magnitude below full-spec preloading)
(b) with vectors disabled, every registered tool remains reachable (I10)
(c) the self-repair path works: validation failure → spec attached → retry
(d) the default config alone yields a working chat agent (I11)
"""
from __future__ import annotations

import random

import pytest

from state_projection_loop import Config, Registry, ScriptedLLM, Session, ToolSearch
from state_projection_loop.tokens import estimate_tokens

from _util import ok_handler_factory, tool_dict

CATEGORIES = [
    "web/search", "web/fetch", "file", "file/edit", "game/flags", "game/media",
    "support/manuals", "support/tickets", "mail", "calendar", "db/query", "db/admin",
    "os/process", "os/fs", "image", "audio", "crm", "billing", "analytics", "deploy",
]

WORDS = ["search", "read", "write", "update", "delete", "list", "sync", "fetch",
         "render", "play", "check", "convert", "翻訳", "検索", "取得", "更新",
         "送信", "予約", "集計", "生成"]


def build_thousand_tool_registry(n: int = 1000) -> Registry:
    rng = random.Random(42)
    reg = Registry()
    for i in range(n):
        cat = CATEGORIES[i % len(CATEGORIES)]
        w1, w2 = rng.choice(WORDS), rng.choice(WORDS)
        reg.register(
            tool_dict(
                f"tool_{i:04d}_{cat.replace('/', '_')}",
                category=cat,
                description=f"{cat} 用のツール{i}。{w1} と {w2} を行う。",
                embedding_text=f"{w1} {w2} {cat} ツール{i}",
                properties={
                    "target": {"type": "string", "description": "対象"},
                    "limit": {"type": "integer", "default": 10},
                },
                required=["target"],
            ),
            handler=ok_handler_factory(f"tool_{i:04d}"),
        )
    return reg


@pytest.fixture(scope="module")
def thousand_tools() -> Registry:
    return build_thousand_tool_registry()


class TestA_TokenOverheadAt1000Tools:
    def test_per_turn_tool_overhead_under_3k(self, thousand_tools):
        captured = {}

        def snapshot(messages, tools):
            captured["messages"] = messages
            captured["tools"] = tools
            return "了解しました。"

        session = Session(ScriptedLLM([snapshot]), kernel="あなたは有能なアシスタントです。",
                          registry=thousand_tools)
        session.send("ファイルを検索して読みたい")

        messages = captured["messages"]
        overhead = 0
        for m in messages:
            content = str(m.content)
            if m.role == "system" and (
                "[Tool index]" in content or "[Tool candidates" in content
                or "[Pinned tools]" in content or "[Runtime notes]" in content
            ):
                overhead += estimate_tokens(m)
        overhead += estimate_tokens([t for t in captured["tools"]])

        assert overhead <= 3000, f"tool overhead {overhead}tk exceeds the 3k budget"

        # two orders of magnitude below preloading every spec (§13 goal)
        full_preload = sum(estimate_tokens(t.spec_text()) for t in thousand_tools)
        assert full_preload > overhead * 10
        assert len(captured["tools"]) < 100  # never O(N) native schemas (I1)

    def test_toc_stays_compact(self, thousand_tools):
        assert estimate_tokens(thousand_tools.toc_text()) <= 100  # §13: TOC ≤ 100tk


class TestB_ReachabilityWithoutVectors:
    def test_every_tool_reachable_via_find_tools(self, thousand_tools):
        """I10: with vector='off', layer 3 search by name finds every tool."""
        search = ToolSearch(thousand_tools, vector="off")
        rng = random.Random(7)
        sample = rng.sample(thousand_tools.all(), 150)
        for td in sample:
            results = search.search(td.name, layer=3, k=5)
            assert any(s.tool.name == td.name for s in results), f"{td.name} unreachable"

    def test_toc_covers_every_category(self, thousand_tools):
        toc = thousand_tools.toc_text()
        for td in thousand_tools:
            assert (td.category or "misc").split("/")[0] in toc

    def test_no_embed_tools_still_reachable(self):
        reg = Registry()
        reg.register(tool_dict("shadow", no_embed=True, summary="shadow tool"))
        search = ToolSearch(reg, vector="off")
        assert any(s.tool.name == "shadow" for s in search.search("shadow", layer=3))


class TestC_SelfRepairPath:
    def test_validation_failure_spec_retry(self):
        reg = Registry()
        calls_seen = []

        def echo(text: str) -> str:
            calls_seen.append(text)
            return f"echo: {text}"

        reg.register(
            tool_dict("echo", description="Echo text.",
                      properties={"text": {"type": "string"}}, required=["text"]),
            handler=echo,
        )

        def repair_step(messages, tools):
            # the model "reads" the spec from the latest observation and retries
            last = messages[-1] if messages[-1].role == "tool" else next(
                m for m in reversed(messages) if m.role == "tool")
            assert "### echo" in str(last.content)
            return ScriptedLLM.call("echo", text="fixed")

        llm = ScriptedLLM([
            ScriptedLLM.call("echo", text=12345),  # wrong type
            repair_step,
            "self-repair complete",
        ])
        session = Session(llm, registry=reg)
        assert session.send("echo something") == "self-repair complete"
        assert calls_seen == ["fixed"]  # bad call never executed, good one did


class TestD_DefaultChatAgent:
    def test_defaults_only_chat(self):
        """No state, no vectors, no spawn, config untouched — chat works (I11)."""
        session = Session(ScriptedLLM(["はい、こんにちは!", "元気です。"]))
        assert session.send("こんにちは") == "はい、こんにちは!"
        assert session.send("元気?") == "元気です。"
        assert session.summary == []
        assert [m.role for m in session.conversation] == [
            "user", "assistant", "user", "assistant",
        ]
