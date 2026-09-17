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


class TestFidelityCompression:
    def test_old_messages_are_compressed_in_projection(self):
        """With a small window, older messages get fidelity-compressed rather
        than compacted by an LLM — the projection handles it deterministically."""
        cfg = Config.from_dict({"projection": {"window_tokens": 2000}})
        llm = ScriptedLLM([f"reply {i}: " + "filler words here " * 40 for i in range(8)])
        session = Session(llm, config=cfg)
        for i in range(8):
            session.send(f"question {i}")
        assert session.budget.steps == 8
        assert len(session.conversation) == 16


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


class TestRewind:
    def test_rewind_cancels_old_run_and_restores_state(self):
        llm = ScriptedLLM(["reply 0", "reply 1", "reply 2", "after rewind"])
        session = Session(llm)
        session.send("msg 0")
        session.send("msg 1")
        session.send("msg 2")
        assert len(session.conversation) == 6
        old_run_id = session.run.id

        irreversible = session.rewind(to_turn=1)

        assert session.run.id != old_run_id
        assert session.run.state == "RUNNING"
        assert len(session.conversation) == 2
        assert session.conversation[0].content == "msg 0"
        assert session.conversation[1].content == "reply 0"
        assert irreversible == []

    def test_rewind_reports_external_effects(self):
        reg = Registry()
        reg.register(capability_dict("mail.send", effects=[("external", "smtp:*")]),
                     handler=lambda: "sent")
        llm = ScriptedLLM([
            ScriptedLLM.call("mail.send"),
            "sent the email",
            "reply 1",
        ])
        session = Session(llm, registry=reg, policy=allow_all_policy())
        session.send("send the email")
        session.send("do something else")

        irreversible = session.rewind(to_turn=1)
        assert any("mail.send" in note for note in irreversible)

    def test_rewind_restores_working_state(self):
        from state_projection_loop import install_builtins

        llm = ScriptedLLM([
            ScriptedLLM.call("state.goal.set", text="find the key"),
            "goal set",
            ScriptedLLM.call("state.goal.set", text="escape the room"),
            "goal changed",
            "after rewind",
        ])
        session = Session(llm, policy=allow_all_policy())
        install_builtins(session.registry, ["state"])
        session.send("set goal")
        session.send("change goal")
        assert session.working_state.goal == "escape the room"

        session.rewind(to_turn=1)
        assert session.working_state.goal == "find the key"

    def test_conversation_after_rewind_continues_normally(self):
        llm = ScriptedLLM(["reply 0", "reply 1", "new reply after rewind"])
        session = Session(llm)
        session.send("msg 0")
        session.send("msg 1")
        session.rewind(to_turn=1)
        reply = session.send("msg after rewind")
        assert reply == "new reply after rewind"
        assert len(session.conversation) == 4


class TestDisabledCapabilitiesAreInvisible:
    """The point of disabling: the model can neither see nor call the tool.

    Asserted against what actually reaches the adapter — the rendered
    messages and the native tool schemas — because that is the only view
    the model has, and every surface (schemas, pinned specs, runtime notes,
    tool index, candidates) lands in exactly one of those two.
    """

    def _session(self, *disabled: str, steps=None) -> Session:
        registry = Registry(disabled=disabled)
        registry.register(capability_dict("demo.echo", description="Echo the text back.",
                                          properties={"text": {"type": "string"}},
                                          required=["text"],
                                          embedding_text="echo repeat say"), handler=echo_handler)
        return Session(ScriptedLLM(steps if steps is not None else ["hi"]), kernel="K",
                       registry=registry, policy=allow_all_policy())

    @staticmethod
    def _sent(session: Session) -> tuple[str, list[str]]:
        request = session.llm.requests[-1]
        prompt = "\n".join(m.content for m in request["messages"] if isinstance(m.content, str))
        return prompt, [t["function"]["name"] for t in request["tools"]]

    def test_bundled_checklist_tool_can_be_disabled(self):
        session = self._session("planning.checklist.manage")
        session.send("hello")
        prompt, tools = self._sent(session)
        assert "planning__checklist__manage" not in tools
        assert "planning.checklist.manage" not in prompt  # no pinned spec, no runtime note
        assert "planning" not in prompt                    # and no tool-index entry

    def test_disabled_tool_is_not_discoverable(self):
        session = self._session("demo.echo")
        assert session.search.search("echo repeat", k=5, layer=3) == []
        assert session.registry.get("demo.echo") is None

    def test_disabled_tool_cannot_be_executed(self):
        session = self._session("demo.echo", steps=[ScriptedLLM.call("demo.echo", text="x"), "done"])
        session.send("use echo")
        observations = [e.data for e in session.ledger.iter_run(session.run.id) if e.type == "observation"]
        assert any("not registered" in str(o) for o in observations)

    def test_disabling_mid_session_takes_effect_on_the_next_turn(self):
        session = self._session(steps=["one", "two"])
        session.send("hello")
        prompt, tools = self._sent(session)
        assert "planning__checklist__manage" in tools and "planning.checklist.manage" in prompt

        session.registry.disable("planning.checklist.manage")
        session.send("hello again")
        prompt, tools = self._sent(session)
        assert "planning__checklist__manage" not in tools
        assert "planning.checklist.manage" not in prompt


class TestResumedRunArtifacts:
    """A resumed run installs a fresh ArtifactStore for its new run id.

    The runtime must write into that one, not into a copy captured when it
    was constructed, or every artifact produced after the resume becomes
    unreachable to meta.artifact.peek.
    """

    def test_artifacts_produced_after_resume_are_readable(self, tmp_path):
        import re

        from state_projection_loop import Config
        from state_projection_loop.artifacts import is_ref

        registry = Registry()
        registry.register(capability_dict("demo.big", properties={}, max_inline_tokens=1),
                          handler=lambda: "x" * 4000)
        config = Config.from_dict({
            "mode": "job",
            "persistence": {"ledger_directory": str(tmp_path)},
            "artifacts": {"directory": str(tmp_path / "artifacts")},
        })
        first = Session(ScriptedLLM([ScriptedLLM.finish(result="ok")]), registry=registry,
                        config=config, policy=allow_all_policy())
        first.run_job("nothing")

        resumed = Session.resume_from_ledger(
            ScriptedLLM([ScriptedLLM.call("demo.big"), ScriptedLLM.finish(result="done")]),
            first.run.id, config=config, registry=registry, policy=allow_all_policy(),
        )
        assert resumed.store is not None
        resumed.run.state = "RUNNING"
        resumed.run_job("make a big result")

        observations = "\n".join(e.data["text"] for e in resumed.ledger.iter_run(resumed.run.id)
                                 if e.type == "observation")
        ids = re.findall(r"art_[0-9A-Z]+", observations)
        assert ids, f"expected the oversized result to become an artifact, got {observations!r}"
        assert is_ref({"$artifact": ids[0]})
        # The store the session hands to meta.artifact.peek must be the one
        # the runtime just wrote to.
        assert "xxx" in resumed.store.peek(ids[0])


class TestStateToolsDeclareTheirWrites:
    """state.* mutates the working state, so it must not be declared as
    effect-free: the runtime uses that declaration to decide what may run
    concurrently, and a mislabelled write loses the model's stated order."""

    def test_mutating_state_tools_are_not_read_only(self):
        from state_projection_loop import install_builtins
        from state_projection_loop.runtime import Runtime

        session = Session(ScriptedLLM([]), registry=Registry(), policy=allow_all_policy())
        install_builtins(session.registry, ["state"])
        mutating = [c for c in session.registry if c.name.startswith("state.") and not c.name.endswith(".get")]
        assert mutating, "expected the bundled state tools to be installed"
        for capability in mutating:
            assert not Runtime.is_read_only(capability), f"{capability.name} claims to be read-only"

    def test_state_writes_are_auto_allowed_by_the_default_policy(self):
        from state_projection_loop import install_builtins

        session = Session(ScriptedLLM([]), registry=Registry())  # default (auto_safe) policy
        install_builtins(session.registry, ["state"])
        capability = session.registry.get("state.goal.set")
        assert session.policy.evaluate(capability, {"text": "x"}).decision == "allow"


class TestApprovalKeepsTheDecisionIntact:
    """A decision parked on an approval is either shown with all of its
    results or not shown at all. Anything in between is a 400 from a native
    tool-calling provider."""

    def _session(self, steps):
        registry = Registry()
        registry.register(capability_dict("demo.write", effects=[("external", "*")]),
                          handler=lambda: "written")
        registry.register(capability_dict("demo.second", effects=[("external", "*")]),
                          handler=lambda: "second")
        return Session(ScriptedLLM(steps), registry=registry,
                       policy=PolicyEngine(default_decision="require_approval"))

    @staticmethod
    def _observations(session):
        return [(e.data["call_id"], e.data["text"]) for e in session.ledger.iter_run(session.run.id)
                if e.type == "observation"]

    def test_a_parked_decision_is_not_projected_at_all(self):
        session = self._session([ScriptedLLM.call("demo.write")])
        session.send("do it")
        assert session.run.state == "WAITING_FOR_APPROVAL"
        assert self._observations(session) == []
        assert not any(m.role == "assistant" and m.tool_calls
                       for m in session.projection.get("history").render(session._context()))

    def test_approval_produces_exactly_one_result_per_call(self):
        session = self._session([ScriptedLLM.call("demo.write"), "done"])
        session.send("do it")
        session.resolve_approval("approved")
        session.resume()
        call_ids = [cid for cid, _ in self._observations(session)]
        assert len(call_ids) == len(set(call_ids)) == 1
        assert "written" in self._observations(session)[0][1]

    def test_denial_answers_every_parked_call(self):
        session = self._session([
            ScriptedLLM.calls(("demo.write", {}), ("demo.second", {})), "done",
        ])
        session.send("do both")
        session.resolve_approval("denied")
        session.resume()
        observations = self._observations(session)
        assert len(observations) == 2, f"both parked calls need a result, got {observations}"
        assert all("denied" in text.lower() or "not executed" in text.lower()
                   for _, text in observations)
        history = session.projection.get("history").render(session._context())
        assert any(m.role == "assistant" and m.tool_calls for m in history)


class TestTheLoopDoesNotBlock:
    """The provider round-trip is the longest wait in a turn. A synchronous
    adapter call would hold the event loop for its whole duration, freezing
    every other task in the host application."""

    def test_other_tasks_run_while_the_model_is_thinking(self):
        import asyncio

        async def scenario():
            other_ran = asyncio.Event()

            class WaitingLLM:
                # Returns only once another task has run: a handshake, not a
                # wall-clock tick count, so a coarse timer cannot fail it.
                async def complete(self, messages, tools=None):
                    from state_projection_loop.llm import Decision

                    await other_ran.wait()
                    return Decision(text="done")

            async def other():
                other_ran.set()

            task = asyncio.create_task(other())
            assert await asyncio.wait_for(Session(WaitingLLM()).asend("hello"), timeout=5) == "done"
            await task

        asyncio.run(scenario())
