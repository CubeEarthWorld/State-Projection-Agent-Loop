"""Bundled packs: installation rules and handler contracts.

Regression coverage for the builtin-pack bugs; the Dart port mirrors this
file as ``test/unit/builtins_test.dart``.
"""
from __future__ import annotations

import pytest

from state_projection_loop import Registry, ScriptedLLM, Session, install_builtins
from state_projection_loop.builtin.state import STATE_HANDLERS
from state_projection_loop.context import ToolContext
from state_projection_loop.working_state import WorkingState

from _util import allow_all, capability_dict


class TestInstall:
    def test_a_disabled_developer_definition_is_not_overwritten(self):
        """``get()`` returns None for a disabled capability, so testing
        membership with it made a switched-off name look unregistered and
        the pack replaced it — losing the developer's definition."""
        reg = Registry()
        reg.register(capability_dict("meta.tool.find", category="mine", description="Mine."))
        reg.disable("meta.tool.find")
        install_builtins(reg, ["meta"])
        reg.enable("meta.tool.find")
        assert reg.get("meta.tool.find").category == "mine"
        assert reg.get("meta.tool.find").spec.description == "Mine."

    def test_an_enabled_developer_definition_still_wins(self):
        reg = Registry()
        reg.register(capability_dict("meta.tool.find", category="mine"))
        install_builtins(reg, ["meta"])
        assert reg.get("meta.tool.find").category == "mine"

    def test_a_definition_without_a_handler_fails_at_install(self):
        """Adding a tool to a shared <pack>.json with no handler must not
        register a capability that only fails when the model calls it."""
        from state_projection_loop.builtin import _install

        reg = Registry()
        with pytest.raises(KeyError):
            _install(reg, [capability_dict("demo.no.handler")], {})
        assert not reg.has_definition("demo.no.handler")


def _ctx() -> ToolContext:
    return ToolContext(working_state=WorkingState())


class TestStateHandlers:
    def test_extra_get_reports_a_missing_key_but_raises_on_an_empty_path(self):
        """The schema allows ``path: ""``; only the missing-key case is an
        answer, an unusable path is an error."""
        ctx = _ctx()
        assert STATE_HANDLERS["state.extra.get"](ctx, path="nope") == "(not set: nope)"
        with pytest.raises(ValueError, match="empty path"):
            STATE_HANDLERS["state.extra.get"](ctx, path="")

    def test_next_actions_echoes_a_quoted_list(self):
        assert (STATE_HANDLERS["state.next_actions.set"](_ctx(), actions=["a", "b"])
                == "next_actions set: ['a', 'b']")

    def test_append_handlers_stay_unique(self):
        ctx = _ctx()
        for handler, field in (("state.fact.add", "confirmed_facts"),
                               ("state.constraint.add", "constraints"),
                               ("state.question.add", "open_questions")):
            STATE_HANDLERS[handler](ctx, text="x")
            STATE_HANDLERS[handler](ctx, text="x")
            assert getattr(ctx.working_state, field) == ["x"]


class TestSpawn:
    def test_duplicate_checklist_ids_are_rejected_before_any_work(self):
        """The check ran after the export, so a duplicate surfaced as an
        IndexError from the export's result instead of saying what was
        wrong."""
        session = Session(ScriptedLLM(["done"]), policy=allow_all(), builtins=["meta", "spawn"])
        session.checklists.execute("create", name="a", items=[{"text": "one"}])
        cid = session.checklists.to_dict()["checklists"][0]["id"]
        handler = session.registry.get("meta.agent.spawn").execution.handler
        with pytest.raises(ValueError, match="Duplicate checklist_ids"):
            import asyncio

            asyncio.run(handler(ToolContext(session=session),
                                tasks=[{"task": "t", "checklist_ids": [cid, cid, "missing"]}]))

    def test_a_child_is_an_ordinary_run_in_the_parents_ledger(self):
        """The child used to get a throwaway InMemoryLedger, so sub-agent
        work was absent from the one thing the package calls the truth."""
        session = _parent(ScriptedLLM([ScriptedLLM.finish(result="from the child")]))
        entry, = session.invoke("meta.agent.spawn", tasks=[{"task": "work"}])

        assert entry["state"] == "COMPLETED" and entry["result"] == "from the child"
        spawned = [e for e in session.ledger.iter_run(session.run.id) if e.type == "run_spawned"]
        assert [d for e in spawned for d in e.data["child_run_ids"]] == [entry["run_id"]]
        child_events = [e.type for e in session.ledger.iter_run(entry["run_id"])]
        assert "model_response" in child_events
        assert entry["run_id"] in [r.run_id for r in session.ledger.list_runs()]

    def test_tasks_in_one_call_run_concurrently(self):
        """Each child's first step waits for the other's; serial execution
        would deadlock, so only real concurrency gets past the timeout."""
        import asyncio

        arrived = [asyncio.Event(), asyncio.Event()]

        class Rendezvous:
            """A model adapter that will not answer until its sibling has
            also been asked. Serial execution can never get past this."""

            def __init__(self, i):
                self.i = i

            async def complete(self, messages, tools=None, *, on_delta=None):
                arrived[self.i].set()
                await asyncio.wait_for(arrived[1 - self.i].wait(), timeout=2)
                return ScriptedLLM.finish(result=f"child {self.i}")

        async def go():
            session = _parent(None)
            children = iter([Rendezvous(0), Rendezvous(1)])
            session.spawn_llm_factory = lambda model: next(children)
            return await session.ainvoke("meta.agent.spawn", tasks=[{"task": "a"}, {"task": "b"}])

        entries = asyncio.run(go())
        assert [e["result"] for e in entries] == ["child 0", "child 1"]

    @pytest.mark.parametrize("decision,ran", [("approved", True), ("denied", False)])
    def test_a_childs_approval_is_resolved_on_the_root_session(self, decision, ran):
        """A child that stopped for approval used to be returned as the
        parent's tool result and then dropped on the floor: unapprovable,
        unresumable, invisible."""
        session, done = _parent_with_guarded_tool()
        pending = session.invoke("meta.agent.spawn", tasks=[{"task": "work"}])

        assert session.run.state == "WAITING_FOR_APPROVAL"
        assert pending.reason.startswith("sub-agent run_")
        assert [e.resource for e in pending.effects] == ["guarded"]

        session.resolve_approval(decision)
        assert session.resume() == "parent done"
        assert done == ([{"text": "x"}] if ran else [])
        assert _child_state(session) == ("COMPLETED", "did it" if ran else "blocked")

    def test_a_parked_child_survives_a_restart(self, tmp_path):
        from state_projection_loop import Config

        config = Config.from_dict({"persistence": {"ledger_directory": str(tmp_path)}})
        session, done = _parent_with_guarded_tool(config=config)
        session.invoke("meta.agent.spawn", tasks=[{"task": "work"}])
        run_id = session.run.id
        del session

        resumed = _parent_with_guarded_tool(config=config, resume=run_id, done=done)[0]
        assert resumed.run.state == "WAITING_FOR_APPROVAL"
        resumed.resolve_approval("approved")
        assert resumed.resume() == "parent done"
        assert done == [{"text": "x"}]
        assert _child_state(resumed) == ("COMPLETED", "did it")

    def test_interrupt_reaches_a_running_child(self):
        session = _parent(None)

        def step(messages, tools):
            session.interrupt()
            return ScriptedLLM.call("meta.tool.find", query="anything")

        session.spawn_llm_factory = lambda model: ScriptedLLM([step, ScriptedLLM.finish(result="never")])
        entry, = session.invoke("meta.agent.spawn", tasks=[{"task": "work"}])
        assert entry["state"] == "CANCELLED" and entry["result"] is None

    def test_budget_is_split_between_children_and_charged_back(self):
        from state_projection_loop import Config

        from state_projection_loop.builtin.meta import _child_session

        session = _parent(None, config=Config.from_dict({"budget": {"max_tokens": 1000}}))
        session.budget.prompt_tokens = 200
        # Each child gets a slice of what is LEFT, not a fresh full limit.
        assert _child_session(session, {"task": "a"}, 2).config.budget.max_tokens == 400

        scripts = iter([ScriptedLLM([ScriptedLLM.finish(result="a")]),
                        ScriptedLLM([ScriptedLLM.finish(result="b")])])
        session.spawn_llm_factory = lambda model: next(scripts)
        session.invoke("meta.agent.spawn", tasks=[{"task": "a"}, {"task": "b"}])
        assert session.budget.prompt_tokens > 200  # the children's usage came home

    def test_a_child_cannot_ask_the_user(self):
        session = _parent(None, builtins=["meta", "spawn", "ask"])
        tools: list[list[str]] = []

        def step(messages, api_tools):
            tools.append([t["name"] for t in api_tools])
            return ScriptedLLM.finish(result="done")

        session.spawn_llm_factory = lambda model: ScriptedLLM([step])
        session.invoke("meta.agent.spawn", tasks=[{"task": "work", "tool_scope": ["*"]}])
        assert "meta__user__ask" not in tools[0]


def _parent(child_llm, *, config=None, builtins=("meta", "spawn"), **kwargs):
    return Session(ScriptedLLM([]), policy=allow_all(), builtins=list(builtins), config=config,
                   spawn_llm_factory=(lambda model: child_llm) if child_llm else None, **kwargs)


def _child_state(session):
    """The sub-agent's own run, read back out of the ledger the way an
    audit would: terminal state and result, from its snapshot."""
    run_id = next(e.data["child_run_ids"][0] for e in session.ledger.iter_run(session.run.id)
                  if e.type == "run_spawned")
    snapshot = session.ledger.load_snapshot(run_id)
    return snapshot.state["state"], snapshot.state["result"]


def _parent_with_guarded_tool(*, config=None, resume=None, done=None):
    """A parent whose child calls one capability the policy holds for
    approval, then finishes with whatever it was told."""
    from state_projection_loop import Config, PolicyEngine
    from state_projection_loop.capability import capability
    from state_projection_loop.policy import Rule

    done = [] if done is None else done

    def guarded(text: str) -> str:
        done.append({"text": text})
        return "written"

    registry = Registry()
    registry.register(capability(guarded, name="demo.guarded.write", category="demo",
                                 effects=[("write", "guarded")]))
    policy = PolicyEngine(default_decision="allow")
    policy.add_rule("developer", Rule(decision="require_approval", capability_pattern="demo.guarded.write",
                                      reason="needs a human"))

    def child_step(messages, tools):
        # Content-driven, not positional: a reattached child is handed a
        # fresh adapter, exactly as a real one would be.
        seen = " ".join(str(m.content) for m in messages)
        if "written" in seen:
            return ScriptedLLM.finish(result="did it")
        if "denied" in seen:
            return ScriptedLLM.finish(result="blocked")
        return ScriptedLLM.call("demo.guarded.write", text="x")

    def child_llm(model=None):
        return ScriptedLLM([child_step] * 4)

    args = dict(registry=registry, policy=policy, builtins=["meta", "spawn"],
                spawn_llm_factory=child_llm, config=config or Config())
    if resume is not None:
        args.pop("config")
        session = Session.resume_from_ledger(ScriptedLLM(["parent done"]), resume, config=config, **args)
    else:
        session = Session(ScriptedLLM(["parent done"]), **args)
    return session, done
