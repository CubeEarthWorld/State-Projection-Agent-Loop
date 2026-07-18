"""Bundled state tools & state_view (§7), spawn sub-agents (§11, I9),
and compaction contract details (§10.2)."""
from __future__ import annotations

from state_projection_loop import (
    Config,
    Message,
    Registry,
    ScriptedLLM,
    Session,
    install_spawn,
    install_state,
)
from state_projection_loop.builtin.state import StateViewSection
from state_projection_loop.compaction import Compactor, deterministic_fold
from state_projection_loop.messages import ToolCall
from state_projection_loop.projection import TurnContext

from _util import tool_dict


class TestStateTools:
    def test_install_registers_tools_and_section(self):
        session = Session(ScriptedLLM(["ok"]))
        install_state(session)
        for name in ("state_set", "state_get", "state_delete", "set_goal", "set_flag"):
            assert name in session.registry
        names = [s.name for s in session.projection.sections]
        assert names.index("state_view") == names.index("candidates") - 1

    def test_llm_edits_state_via_tools(self):
        llm = ScriptedLLM([
            ScriptedLLM.calls(
                ("set_goal", {"text": "宝物庫の鍵を見つける"}),
                ("set_flag", {"name": "door_open", "value": True}),
                ("state_set", {"path": "party.hero.hp", "value": 12}),
            ),
            "state updated",
        ])
        session = Session(llm)
        install_state(session)
        session.send("start")
        assert session.state["goal"] == "宝物庫の鍵を見つける"
        assert session.state["flags"]["door_open"] is True
        assert session.state["party"]["hero"]["hp"] == 12

    def test_state_view_projected(self):
        seen = {}

        def capture(messages, tools):
            seen["joined"] = "\n".join(str(m.content) for m in messages)
            return "ok"

        session = Session(ScriptedLLM([capture]),
                          seed={"goal": "escape", "flags": {"lit": True}})
        install_state(session)
        session.send("hi")
        assert "[State]" in seen["joined"]
        assert "goal: escape" in seen["joined"]
        assert '"lit": true' in seen["joined"]

    def test_state_view_custom_template_and_empty_state(self):
        section = StateViewSection(template=lambda s: f"HP={s['hp']}")
        turn = TurnContext(config=Config(), registry=Registry(),
                          conversation=[], summary=[], state={"hp": 3})
        assert section.render(turn)[0].content == "[State]\nHP=3"
        turn.state = {}
        assert section.render(turn) == []

    def test_seed_and_get_delete(self):
        llm = ScriptedLLM([
            ScriptedLLM.call("state_get", path="inventory.torch"),
            ScriptedLLM.call("state_delete", path="inventory.torch"),
            ScriptedLLM.call("state_get", path="inventory.torch"),
            "done",
        ])
        session = Session(llm, seed={"inventory": {"torch": 1}})
        install_state(session, view=False)
        session.send("check")
        obs = [m.content for m in session.conversation if m.role == "tool"]
        assert obs[0] == "1"
        assert "deleted" in obs[1]
        assert "(not set" in obs[2]


class TestSpawn:
    def _parent(self, child_llm, parent_steps, registry=None):
        reg = registry or Registry()
        install_spawn(reg)
        return Session(
            ScriptedLLM(parent_steps),
            registry=reg,
            spawn_llm_factory=lambda model: child_llm,
        )

    def test_child_runs_job_and_returns_result(self):
        child_llm = ScriptedLLM([ScriptedLLM.call("done", result="child-result-42")])
        session = self._parent(child_llm, [
            ScriptedLLM.call("spawn", task="compute the answer"),
            "parent finished",
        ])
        assert session.send("delegate this") == "parent finished"
        obs = next(m for m in session.conversation if m.role == "tool" and m.name == "spawn")
        assert "child-result-42" in str(obs.content)

    def test_isolation_child_sees_only_task(self):
        """Invariant I9: parent/child share only the task string and result."""
        child_llm = ScriptedLLM([ScriptedLLM.call("done", result="ok")])
        session = self._parent(child_llm, [
            ScriptedLLM.call("spawn", task="isolated task description"),
            "done",
        ])
        session.send("SECRET-PARENT-CONTEXT do not leak")
        child_messages = child_llm.requests[0]["messages"]
        joined = "\n".join(str(m.content) for m in child_messages)
        assert "isolated task description" in joined
        assert "SECRET-PARENT-CONTEXT" not in joined

    def test_tool_scope_limits_child_registry(self):
        reg = Registry()
        reg.register(tool_dict("game_flag", category="game/flags"))
        reg.register(tool_dict("web_search", category="web/search"))

        def child_step(messages, tools):
            names = {t["function"]["name"] for t in tools}
            assert "done" in names and "find_tools" in names
            return ScriptedLLM.call("find_tools", query="web_search search the web")

        def child_step2(messages, tools):
            return ScriptedLLM.call("done", result="scoped")

        child_llm = ScriptedLLM([child_step, child_step2])
        session = self._parent(child_llm, [
            ScriptedLLM.call("spawn", task="try tools", tool_scope=["game/*"]),
            "done",
        ], registry=reg)
        session.send("go")
        find_obs = next(m for m in child_llm.requests[-1]["messages"]
                        if m.role == "tool" and m.name == "find_tools")
        assert "web_search" not in str(find_obs.content)  # out of scope
        assert session.send  # parent still alive

    def test_spawn_without_factory_uses_parent_llm(self):
        reg = Registry()
        install_spawn(reg)
        llm = ScriptedLLM([
            ScriptedLLM.call("spawn", task="sub"),
            ScriptedLLM.call("done", result="from shared llm"),
            "parent done",
        ])
        session = Session(llm, registry=reg)
        assert session.send("go") == "parent done"


class TestCompactionContract:
    def _long_conversation(self):
        conv = []
        for i in range(10):
            conv.append(Message(role="user", content=f"指示{i}: 必ず敬語で話すこと " + "詳細 " * 30))
            conv.append(Message(role="assistant", content=f"了解しました {i} " + "説明 " * 30,
                                tool_calls=[ToolCall(name="echo", arguments={"text": str(i)})]))
            conv.append(Message(role="tool", content=f"echo: {i}", tool_call_id=f"c{i}", name="echo"))
        return conv

    def test_fold_reduces_and_respects_split(self):
        cfg = Config()
        compactor = Compactor(cfg, summarizer=ScriptedLLM(
            ["私はユーザーの指示に従い、echoを呼び出した。未完了の意図はない。"]))
        conv = self._long_conversation()
        summary, remaining = compactor.fold(conv)
        assert summary.startswith("私は")
        assert 0 < len(remaining) < len(conv)
        assert remaining[0].role != "tool"  # never orphan observations

    def test_contract_prompt_carries_all_musts(self):
        summarizer = ScriptedLLM(["summary"])
        compactor = Compactor(Config(), summarizer=summarizer)
        compactor.fold(self._long_conversation())
        prompt = summarizer.requests[0]["messages"][0].content
        for fragment in ("first person", "chronological", "unfinished intentions",
                         "verbatim", "$hN", "tokens"):
            assert fragment in prompt

    def test_deterministic_fold_keeps_user_verbatim(self):
        msgs = [Message(role="user", content="絶対に丁寧語を使うこと"),
                Message(role="assistant", content="わかりました",
                        tool_calls=[ToolCall(name="t", arguments={})]),
                Message(role="tool", content="ok", name="t")]
        out = deterministic_fold(msgs, 500)
        assert "絶対に丁寧語を使うこと" in out
        assert "I called t." in out

    def test_summary_length_cap(self):
        cfg = Config()  # max_summary_ratio 0.1
        long_summary = "とても長い要約 " * 400
        compactor = Compactor(cfg, summarizer=ScriptedLLM([long_summary]))
        conv = self._long_conversation()
        summary, _ = compactor.fold(conv)
        from state_projection_loop.tokens import estimate_tokens

        folded_tokens = estimate_tokens(conv) // 2
        assert estimate_tokens(summary) <= int(max(150, folded_tokens * 0.1) * 1.5) + 5
