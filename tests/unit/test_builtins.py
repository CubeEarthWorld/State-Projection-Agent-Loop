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

            asyncio.run(handler(ToolContext(session=session), task="t",
                                checklist_ids=[cid, cid, "missing"]))
