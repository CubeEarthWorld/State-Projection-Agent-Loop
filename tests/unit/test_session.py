"""Session loop: chat & job modes, candidates injection, meta capabilities,
finish validation (P0-3), concurrency guard (P0-4), policy gating, budget
grace, interruption, compaction wiring."""
from __future__ import annotations

import pytest

from state_projection_loop import Config, Registry, ScriptedLLM, Session
from state_projection_loop.policy import PolicyEngine, Rule
from state_projection_loop.session import ConcurrencyError

from _util import echo_handler, capability_dict


def echo_registry() -> Registry:
    reg = Registry()
    reg.register(
        capability_dict("demo.echo", description="Echo the text back.",
                         properties={"text": {"type": "string"}}, required=["text"],
                         embedding_text="echo repeat say オウム返し"),
        handler=echo_handler,
    )
    return reg


def allow_all_policy() -> PolicyEngine:
    return PolicyEngine(default_decision="allow")


class TestChatMode:
    def test_default_config_plain_chat(self):
        """Defaults alone give a working chat agent."""
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
        assert session.run.state == "RUNNING"  # chat mode never auto-completes the run

    def test_tool_call_then_answer(self):
        llm = ScriptedLLM([
            ScriptedLLM.call("demo.echo", text="hello"),
            "The tool said: echo: hello",
        ])
        session = Session(llm, registry=echo_registry(), policy=allow_all_policy())
        reply = session.send("please echo hello")
        assert reply == "The tool said: echo: hello"
        obs = [m for m in session.conversation if m.role == "tool"]
        assert len(obs) == 1 and obs[0].content == "echo: hello"
        assert obs[0].name == "demo.echo" and obs[0].tool_call_id

    def test_meta_capabilities_always_present(self):
        session = Session(ScriptedLLM(["ok"]))
        assert "meta.tool.find" in session.registry
        assert "meta.artifact.peek" in session.registry
        assert "meta.history.search" in session.registry

    def test_kernel_carries_pinned_meta_specs(self):
        llm = ScriptedLLM([lambda messages, tools: "ok"])
        session = Session(llm, kernel="You are a helper.")
        session.send("hi")
        kernel = llm.requests[0]["messages"][0]
        assert kernel.role == "system"
        assert "You are a helper." in kernel.content
        assert "### meta.tool.find@1" in kernel.content and "### meta.artifact.peek@1" in kernel.content

    def test_candidates_injected_from_user_message(self):
        def check(messages, tools):
            joined = "\n".join(str(m.content) for m in messages)
            # Native schemas are sent, so the candidate card dedupes down to
            # just the signature (P0-5) instead of repeating the full card.
            assert "[Tool candidates" in joined and "demo.echo(" in joined
            tool_names = [t["function"]["name"] for t in tools]
            # native schema names are provider-safe encoded (dots -> "__")
            assert "demo__echo" in tool_names and "meta__tool__find" in tool_names
            return "saw candidates"

        session = Session(ScriptedLLM([check]), registry=echo_registry())
        assert session.send("echo repeat this") == "saw candidates"

    def test_find_tools_activates_results(self):
        reg = echo_registry()

        def step2(messages, tools):
            names = [t["function"]["name"] for t in tools]
            assert "demo__echo" in names  # activated by find even without candidates
            return ScriptedLLM.call("demo.echo", text="via find_tools")

        llm = ScriptedLLM([
            ScriptedLLM.call("meta.tool.find", query="オウム返し echo"),
            step2,
            "done",
        ])
        cfg = Config.from_dict({"discovery": {"query_sources": []}})  # kill layer 2
        session = Session(llm, registry=reg, config=cfg, policy=allow_all_policy())
        assert session.send("noise") == "done"
        find_obs = next(m for m in session.conversation if m.role == "tool" and m.name == "meta.tool.find")
        assert "demo.echo" in str(find_obs.content)


class TestJobMode:
    def job_config(self, **budget):
        return Config.from_dict({"mode": "job", "budget": {"max_steps": budget.get("max_steps", 50)}})

    def test_finish_ends_job_with_result(self):
        llm = ScriptedLLM([
            ScriptedLLM.call("demo.echo", text="working"),
            ScriptedLLM.finish(result={"status": "ok", "count": 3}),
        ])
        session = Session(llm, registry=echo_registry(), config=self.job_config(), policy=allow_all_policy())
        result = session.run_job("do the thing")
        assert result == {"status": "ok", "count": 3}
        assert session.run.state == "COMPLETED"

    def test_finish_combined_with_calls_is_rejected(self):
        from state_projection_loop.messages import Decision, ToolCall

        mixed = Decision(text="", calls=[ToolCall(name="demo.echo", arguments={"text": "x"})],
                          finish=True, result="premature")
        llm = ScriptedLLM([mixed, ScriptedLLM.finish(result="actually done")])
        session = Session(llm, registry=echo_registry(), config=self.job_config(), policy=allow_all_policy())
        result = session.run_job("do the thing")
        assert result == "actually done"
        rejected = [m for m in session.conversation if m.role == "tool" and "Rejected" in str(m.content)]
        assert rejected  # the mixed decision produced a rejection observation, not an execution

    def test_text_only_turn_gets_nudged(self):
        llm = ScriptedLLM([
            "just thinking out loud",
            ScriptedLLM.finish(result="finished"),
        ])
        session = Session(llm, config=self.job_config())
        assert session.run_job("task") == "finished"
        notices = [m for m in session.conversation
                   if m.role == "system" and "finish(result)" in str(m.content)]
        assert notices

    def test_budget_grace_turn_then_stop(self):
        llm = ScriptedLLM([
            ScriptedLLM.call("demo.echo", text="a"),
            ScriptedLLM.call("demo.echo", text="b"),
            "final wrap-up summary",
        ])
        session = Session(llm, registry=echo_registry(), config=self.job_config(max_steps=2),
                          policy=allow_all_policy())
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


class TestPolicyGating:
    def test_deny_blocks_execution_without_running_handler(self):
        executed = []

        def dangerous() -> str:
            executed.append(True)
            return "boom"

        reg = Registry()
        reg.register(capability_dict("demo.rm_rf", effects=[("external", "*")]), handler=dangerous)

        policy = PolicyEngine(default_decision="deny")
        llm = ScriptedLLM([ScriptedLLM.call("demo.rm_rf"), "I could not run it."])
        session = Session(llm, registry=reg, policy=policy)
        reply = session.send("delete everything")
        assert reply == "I could not run it."
        assert executed == []
        blocked = [m for m in session.conversation if m.role == "tool"]
        assert blocked and "Denied by policy" in blocked[0].content

    def test_require_approval_pauses_the_run(self):
        reg = Registry()
        reg.register(capability_dict("demo.rm_rf", effects=[("external", "*")]), handler=lambda: "boom")
        policy = PolicyEngine(default_decision="require_approval")
        llm = ScriptedLLM([ScriptedLLM.call("demo.rm_rf")])
        session = Session(llm, registry=reg, policy=policy)
        result = session.send("delete everything")
        assert session.run.state == "WAITING_FOR_APPROVAL"
        assert result.reason  # ApprovalRequest


class TestConcurrencyGuard:
    async def test_second_concurrent_call_raises(self):
        # The model decision step is synchronous, so the only way a second
        # asend() can race the first is while a tool call is genuinely
        # in flight (an async handler awaiting something). Use that as the
        # yield point.
        import asyncio

        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_tool() -> str:
            started.set()
            await release.wait()
            return "done"

        reg = Registry()
        reg.register(capability_dict("demo.slow"), handler=slow_tool)
        llm = ScriptedLLM([ScriptedLLM.call("demo.slow"), "finished"])
        session = Session(llm, registry=reg, policy=allow_all_policy())

        task = asyncio.create_task(session.asend("go"))
        await started.wait()
        with pytest.raises(ConcurrencyError):
            await session.asend("again")
        release.set()
        assert await task == "finished"


class TestCompactionWiring:
    def test_session_folds_overflow_into_working_state(self):
        summarizer = ScriptedLLM(
            ['{"new_facts": ["topic A was discussed"], "new_decisions": [], "next_actions": []}'] * 10,
            strict=False,
        )
        cfg = Config.from_dict({"projection": {"window_tokens": 700}})
        llm = ScriptedLLM([f"reply {i}: " + "filler words here " * 40 for i in range(6)])
        session = Session(llm, config=cfg, summarizer=summarizer)
        for i in range(6):
            session.send(f"question {i}")
        assert session.working_state.confirmed_facts, "overflow should have been folded into working_state"
        assert summarizer.requests, "the summarizer LLM should have been called"
        prompt = summarizer.requests[0]["messages"][0].content
        assert "JSON" in prompt

    def test_compaction_model_none_uses_deterministic_fold(self):
        cfg = Config.from_dict({"projection": {"window_tokens": 700},
                                "compaction": {"model": "none"}})
        llm = ScriptedLLM([f"reply {i}: " + "filler words here " * 40 for i in range(6)])
        session = Session(llm, config=cfg)
        for i in range(6):
            session.send(f"question {i}")
        assert session.working_state.confirmed_facts
        assert any("verbatim" in f or "question 0" in f for f in session.working_state.confirmed_facts)


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
