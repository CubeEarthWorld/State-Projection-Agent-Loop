"""Session loop: chat & job modes, candidates injection, find_tools flow,
budget grace, interruption, hooks (defect-1 fix), compaction wiring."""
from __future__ import annotations

import pytest

from state_projection_loop import (
    Config,
    Decision,
    HookBlock,
    Hooks,
    Registry,
    ScriptedLLM,
    Session,
)

from _util import echo_handler, tool_dict


def echo_registry() -> Registry:
    reg = Registry()
    reg.register(
        tool_dict("echo", description="Echo the text back.",
                  properties={"text": {"type": "string"}}, required=["text"],
                  embedding_text="echo repeat say オウム返し"),
        handler=echo_handler,
    )
    return reg


class TestChatMode:
    def test_default_config_plain_chat(self):
        """Invariant I11: defaults alone give a working chat agent."""
        session = Session(ScriptedLLM(["こんにちは!ご用件をどうぞ。"]))
        reply = session.send("こんにちは")
        assert reply == "こんにちは!ご用件をどうぞ。"
        roles = [m.role for m in session.conversation]
        assert roles == ["user", "assistant"]

    def test_multi_turn(self):
        session = Session(ScriptedLLM(["reply 1", "reply 2"]))
        assert session.send("one") == "reply 1"
        assert session.send("two") == "reply 2"
        assert len(session.conversation) == 4

    def test_tool_call_then_answer(self):
        llm = ScriptedLLM([
            ScriptedLLM.call("echo", text="hello"),
            "The tool said: echo: hello",
        ])
        session = Session(llm, registry=echo_registry())
        reply = session.send("please echo hello")
        assert reply == "The tool said: echo: hello"
        obs = [m for m in session.conversation if m.role == "tool"]
        assert len(obs) == 1 and obs[0].content == "echo: hello"
        assert obs[0].name == "echo" and obs[0].tool_call_id

    def test_meta_tools_always_present(self):
        session = Session(ScriptedLLM(["ok"]))
        assert "find_tools" in session.registry
        assert "peek" in session.registry
        assert "done" not in session.registry  # chat mode

    def test_kernel_carries_pinned_meta_specs(self):
        llm = ScriptedLLM([lambda messages, tools: "ok"])
        session = Session(llm, kernel="You are a helper.")
        session.send("hi")
        kernel = llm.requests[0]["messages"][0]
        assert kernel.role == "system"
        assert "You are a helper." in kernel.content
        assert "### find_tools" in kernel.content and "### peek" in kernel.content

    def test_candidates_injected_from_user_message(self):
        def check(messages, tools):
            joined = "\n".join(str(m.content) for m in messages)
            assert "[Tool candidates" in joined and "- echo(" in joined
            tool_names = [t["function"]["name"] for t in tools]
            assert "echo" in tool_names and "find_tools" in tool_names
            return "saw candidates"

        session = Session(ScriptedLLM([check]), registry=echo_registry())
        assert session.send("echo repeat this") == "saw candidates"

    def test_find_tools_activates_results(self):
        reg = echo_registry()

        def step2(messages, tools):
            names = [t["function"]["name"] for t in tools]
            assert "echo" in names  # activated by find_tools even without candidates
            return ScriptedLLM.call("echo", text="via find_tools")

        llm = ScriptedLLM([
            ScriptedLLM.call("find_tools", query="オウム返し echo"),
            step2,
            "done",
        ])
        cfg = Config.from_dict({"discovery": {"query_sources": []}})  # kill layer 2
        session = Session(llm, registry=reg, config=cfg)
        assert session.send("noise") == "done"
        find_obs = next(m for m in session.conversation if m.role == "tool" and m.name == "find_tools")
        assert "echo" in str(find_obs.content)


class TestJobMode:
    def job_config(self, **budget):
        return Config.from_dict({"mode": "job", "budget": {"max_steps": budget.get("max_steps", 50)}})

    def test_done_ends_job_with_result(self):
        llm = ScriptedLLM([
            ScriptedLLM.call("echo", text="working"),
            ScriptedLLM.call("done", result={"status": "ok", "count": 3}),
        ])
        session = Session(llm, registry=echo_registry(), config=self.job_config())
        result = session.run_job("do the thing")
        assert result == {"status": "ok", "count": 3}
        assert "done" in session.registry

    def test_text_only_turn_gets_nudged(self):
        llm = ScriptedLLM([
            "just thinking out loud",
            ScriptedLLM.call("done", result="finished"),
        ])
        session = Session(llm, config=self.job_config())
        assert session.run_job("task") == "finished"
        notices = [m for m in session.conversation
                   if m.role == "system" and "done(result)" in str(m.content)]
        assert notices

    def test_budget_grace_turn_then_stop(self):
        llm = ScriptedLLM([
            ScriptedLLM.call("echo", text="a"),
            ScriptedLLM.call("echo", text="b"),
            "final wrap-up summary",
        ])
        session = Session(llm, registry=echo_registry(), config=self.job_config(max_steps=2))
        result = session.run_job("loop forever")
        assert result == "final wrap-up summary"
        assert any("Budget exceeded" in str(m.content) for m in session.conversation
                   if m.role == "system")

    def test_idle_limit_returns_text(self):
        cfg = Config.from_dict({"mode": "job", "limits": {"max_idle_turns": 1}})
        llm = ScriptedLLM(["thinking...", "still thinking, giving my answer"])
        session = Session(llm, config=cfg)
        assert session.run_job("task") == "still thinking, giving my answer"


class TestInterruption:
    def test_interrupt_stops_loop(self):
        llm = ScriptedLLM(["never reached"])
        session = Session(llm)
        session.interrupt()
        assert session.send("hi") == "[interrupted]"
        assert llm.requests == []  # stopped before calling the model


class TestHooks:
    def test_after_decide_blocks_execution(self):
        executed = []

        def dangerous() -> str:
            executed.append(True)
            return "boom"

        reg = Registry()
        reg.register(tool_dict("rm_rf"), handler=dangerous)

        def approval_gate(decision: Decision, turn) -> HookBlock | None:
            if any(c.name == "rm_rf" for c in decision.calls):
                return HookBlock(reason="human approval required")
            return None

        llm = ScriptedLLM([
            ScriptedLLM.call("rm_rf"),
            "I could not run it.",
        ])
        session = Session(llm, registry=reg, hooks=Hooks(after_decide=[approval_gate]))
        reply = session.send("delete everything")
        assert reply == "I could not run it."
        assert executed == []  # never ran
        blocked = [m for m in session.conversation if m.role == "tool"]
        assert blocked and "[blocked by policy]" in blocked[0].content

    def test_after_execute_redacts_observation(self):
        from state_projection_loop.runtime import ToolResult

        reg = Registry()
        reg.register(tool_dict("fetch_secret"), handler=lambda: "password=hunter2")

        def redact(call, result: ToolResult, turn):
            if "password=" in result.observation:
                return ToolResult(call=result.call, ok=result.ok,
                                  observation="password=[REDACTED]")
            return None

        llm = ScriptedLLM([ScriptedLLM.call("fetch_secret"), "done"])
        session = Session(llm, registry=reg, hooks=Hooks(after_execute=[redact]))
        session.send("get it")
        obs = next(m for m in session.conversation if m.role == "tool")
        assert obs.content == "password=[REDACTED]"


class TestCompactionWiring:
    def test_session_folds_overflowing_conversation(self):
        summarizer = ScriptedLLM(
            ["I asked about topic A and answered; nothing is pending."] * 10, strict=False
        )
        cfg = Config.from_dict({"projection": {"window_tokens": 700}})
        llm = ScriptedLLM([f"reply {i}: " + "filler words here " * 40 for i in range(6)])
        session = Session(llm, config=cfg, summarizer=summarizer)
        for i in range(6):
            session.send(f"question {i}")
        assert session.summary, "overflow should have been folded into the summary"
        assert summarizer.requests, "the summarizer LLM should have been called"
        # contract v1 prompt reached the summarizer
        prompt = summarizer.requests[0]["messages"][0].content
        assert "first person" in prompt and "$hN" in prompt

    def test_compaction_model_none_uses_deterministic_fold(self):
        cfg = Config.from_dict({"projection": {"window_tokens": 700},
                                "compaction": {"model": "none"}})
        llm = ScriptedLLM([f"reply {i}: " + "filler words here " * 40 for i in range(6)])
        session = Session(llm, config=cfg)
        for i in range(6):
            session.send(f"question {i}")
        assert session.summary
        assert any("verbatim" in e or "question 0" in e for e in session.summary)


class TestBudgetTokens:
    def test_estimated_usage_accumulates_without_provider_usage(self):
        session = Session(ScriptedLLM(["short reply"]))
        session.send("hello")
        assert session.budget.steps == 1
        assert session.budget.prompt_tokens > 0
        assert session.budget.completion_tokens > 0


class TestAsyncGuard:
    def test_sync_api_inside_event_loop_raises(self):
        import asyncio

        async def inner():
            session = Session(ScriptedLLM(["x"]))
            with pytest.raises(RuntimeError, match="asend"):
                session.send("hi")

        asyncio.run(inner())

    async def test_async_api(self):
        session = Session(ScriptedLLM(["async reply"]))
        assert await session.asend("hi") == "async reply"
