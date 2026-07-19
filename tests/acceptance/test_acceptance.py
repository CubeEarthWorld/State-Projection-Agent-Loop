"""General acceptance tests (carried over from the original design spec):

(a) 1,000 registered capabilities, default config -> per-turn tool-related
    overhead stays small (two orders of magnitude below full-spec preloading)
(b) with vectors disabled, every registered capability remains reachable
(c) the self-repair path works: validation failure -> spec attached -> retry
(d) the default config alone yields a working chat agent

The Codex-review-driven P0/P1 fixes have their own focused suite in
``test_p0_p1_acceptance.py``.
"""
from __future__ import annotations

import random

import pytest

from state_projection_loop import Config, Registry, ScriptedLLM, Session, ToolSearch
from state_projection_loop.tokens import estimate_tokens

from _util import capability_dict, ok_handler_factory

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
            capability_dict(
                f"demo.tool_{i:04d}",
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

        # two orders of magnitude below preloading every spec
        full_preload = sum(estimate_tokens(t.spec_text()) for t in thousand_tools)
        assert full_preload > overhead * 10
        assert len(captured["tools"]) < 100  # never O(N) native schemas

    def test_toc_stays_compact(self, thousand_tools):
        assert estimate_tokens(thousand_tools.toc_text()) <= 100


class TestB_ReachabilityWithoutVectors:
    def test_every_tool_reachable_via_find_tools(self, thousand_tools):
        """With vector='off', layer 3 search by name finds every capability."""
        search = ToolSearch(thousand_tools, vector="off")
        rng = random.Random(7)
        sample = rng.sample(thousand_tools.all(), 150)
        for cap in sample:
            results = search.search(cap.name, layer=3, k=5)
            assert any(s.tool.name == cap.name for s in results), f"{cap.name} unreachable"

    def test_toc_covers_every_category(self, thousand_tools):
        toc = thousand_tools.toc_text()
        for cap in thousand_tools:
            assert (cap.category or "misc").split("/")[0] in toc

    def test_no_embed_tools_still_reachable(self):
        reg = Registry()
        reg.register(capability_dict("demo.shadow", no_embed=True, summary="shadow tool"))
        search = ToolSearch(reg, vector="off")
        assert any(s.tool.name == "demo.shadow" for s in search.search("shadow", layer=3))


class TestC_SelfRepairPath:
    def test_validation_failure_spec_retry(self):
        reg = Registry()
        calls_seen = []

        def echo(text: str) -> str:
            calls_seen.append(text)
            return f"echo: {text}"

        reg.register(
            capability_dict("demo.echo", description="Echo text.",
                             properties={"text": {"type": "string"}}, required=["text"]),
            handler=echo,
        )

        def repair_step(messages, tools):
            last = messages[-1] if messages[-1].role == "tool" else next(
                m for m in reversed(messages) if m.role == "tool")
            assert "### demo.echo" in str(last.content)
            return ScriptedLLM.call("demo.echo", text="fixed")

        llm = ScriptedLLM([
            ScriptedLLM.call("demo.echo", text=12345),  # wrong type
            repair_step,
            "self-repair complete",
        ])
        session = Session(llm, registry=reg)
        assert session.send("echo something") == "self-repair complete"
        assert calls_seen == ["fixed"]  # bad call never executed, good one did


class TestD_DefaultChatAgent:
    def test_defaults_only_chat(self):
        """No policy config, no vectors, no spawn — chat works out of the box."""
        session = Session(ScriptedLLM(["はい、こんにちは!", "元気です。"]))
        assert session.send("こんにちは") == "はい、こんにちは!"
        assert session.send("元気?") == "元気です。"
        assert session.working_state.is_empty()
        assert [m.role for m in session.conversation] == [
            "user", "assistant", "user", "assistant",
        ]
