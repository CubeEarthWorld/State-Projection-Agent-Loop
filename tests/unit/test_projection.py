"""Projection pipeline: section composition, window enforcement including
native tool-schema + reserved-output budgeting, kernel immutability,
epoch-cached TOC, candidate-card dedup against native schemas, and
fidelity-graded history rendering from the Event Ledger."""
from __future__ import annotations

import pytest

from state_projection_loop import (
    CandidatesSection,
    Config,
    HistorySection,
    InMemoryLedger,
    KernelSection,
    Message,
    Projection,
    Registry,
    TocSection,
    TurnContext,
)
from state_projection_loop.projection import build_default_sections
from state_projection_loop.run import Run
from state_projection_loop.tokens import estimate_tokens
from state_projection_loop.working_state import WorkingState

from _util import capability_dict


def make_ledger_with_events(n_user=3, n_obs=0):
    ledger = InMemoryLedger()
    run_id = "run_test"
    for i in range(n_user):
        ledger.append(run_id, "user_input", {"text": f"message {i} " + "pad " * 20})
        ledger.append(run_id, "model_response", {"text": f"reply {i}", "calls": []})
    if n_obs:
        # Observations only ever follow the decision that asked for them; the
        # projection drops a result whose call is not there (and vice versa).
        ledger.append(run_id, "model_response", {
            "text": "",
            "calls": [{"name": "tool", "arguments": {}, "id": f"c{i}"} for i in range(n_obs)],
        })
    for i in range(n_obs):
        ledger.append(run_id, "observation", {"call_id": f"c{i}", "name": "tool", "text": f"result {i} " + "data " * 30})
    return ledger, run_id


def make_turn(registry=None, ledger=None, run_id="run_test", working_state=None, candidates=None, window=30000):
    cfg = Config()
    cfg.projection.window_tokens = window
    ledger = ledger or InMemoryLedger()
    return TurnContext(
        config=cfg,
        registry=registry or Registry(),
        ledger=ledger,
        run=Run(run_id, "ses_test", ledger),
        working_state=working_state or WorkingState(),
        candidates=candidates or [],
    )


def default_projection(registry, kernel="You are helpful.", window=30000):
    sections = build_default_sections(
        ["kernel", "toc", "history", "working_state", "candidates"],
        kernel_text=kernel,
    )
    return Projection(sections, window_tokens=window)


class TestRenderComposition:
    def test_kernel_first_and_contains_pinned_spec(self):
        reg = Registry()
        reg.register(capability_dict("demo.pinned_tool", pinned=True, description="Pinned helper."))
        projection = default_projection(reg)
        msgs = projection.render(make_turn(registry=reg))
        assert msgs[0].role == "system"
        assert "You are helpful." in msgs[0].content
        assert "### demo.pinned_tool@1" in msgs[0].content

    def test_toc_present_and_working_state_absent_when_empty(self):
        reg = Registry()
        reg.register(capability_dict("web.t", category="web"))
        projection = default_projection(reg)
        msgs = projection.render(make_turn(registry=reg))
        contents = [str(m.content) for m in msgs]
        assert any("[Tool index] web(1)" in c for c in contents)
        assert not any("[Working state]" in c for c in contents)

    def test_toc_disabled_by_config(self):
        reg = Registry()
        reg.register(capability_dict("web.t", category="web"))
        projection = default_projection(reg)
        turn = make_turn(registry=reg)
        turn.config.discovery.toc = False
        msgs = projection.render(turn)
        assert not any("[Tool index]" in str(m.content) for m in msgs)

    def test_candidates_render_last(self):
        reg = Registry()
        cap = reg.register(capability_dict("demo.cand", summary="candidate tool"))
        from state_projection_loop import ScoredTool

        projection = default_projection(reg)
        ledger, run_id = make_ledger_with_events(n_user=1)
        turn = make_turn(registry=reg, ledger=ledger, run_id=run_id,
                         candidates=[ScoredTool(tool=cap, score=1.0)])
        msgs = projection.render(turn)
        assert "[Tool candidates" in str(msgs[-1].content)
        assert "- demo.cand(" in str(msgs[-1].content)

    def test_candidate_cards_deduped_against_native_schemas(self):
        reg = Registry()
        cap = reg.register(capability_dict("demo.cand", summary="a somewhat long description of the tool"))
        from state_projection_loop import ScoredTool

        projection = default_projection(reg)
        turn = make_turn(registry=reg, candidates=[ScoredTool(tool=cap, score=1.0)])
        msgs = projection.render(turn, api_tools=[cap.tool_spec()])
        last = str(msgs[-1].content)
        assert "schemas sent natively" in last
        assert "a somewhat long description" not in last

    def test_render_does_not_mutate_the_caller_s_tool_list(self):
        # `render` drops native schemas to fit the window; it must do that to
        # its own copy, not to the list the caller still holds.
        reg = Registry()
        cap = reg.register(capability_dict("demo.cand", summary="a somewhat long description of the tool"))
        projection = default_projection(reg, window=1)
        api_tools = [cap.tool_spec()]
        projection.render(make_turn(registry=reg), api_tools=api_tools)
        assert api_tools == [cap.tool_spec()]

    def test_working_state_rendered_when_present(self):
        projection = default_projection(Registry())
        ws = WorkingState(goal="ship the feature")
        msgs = projection.render(make_turn(working_state=ws))
        assert any("[Working state]" in str(m.content) and "ship the feature" in str(m.content) for m in msgs)


class TestHistorySection:
    def test_renders_user_and_assistant_from_ledger(self):
        ledger, run_id = make_ledger_with_events(n_user=2)
        section = HistorySection()
        turn = make_turn(ledger=ledger, run_id=run_id)
        msgs = section.render(turn)
        roles = [m.role for m in msgs]
        assert "user" in roles
        assert "assistant" in roles

    def test_renders_observations(self):
        ledger, run_id = make_ledger_with_events(n_user=1, n_obs=2)
        section = HistorySection()
        turn = make_turn(ledger=ledger, run_id=run_id)
        msgs = section.render(turn)
        obs = [m for m in msgs if m.role == "tool"]
        assert len(obs) == 2

    def test_renders_notices(self):
        ledger = InMemoryLedger()
        run_id = "run_test"
        ledger.append(run_id, "user_input", {"text": "hi"})
        ledger.append(run_id, "notice", {"text": "[runtime] Budget exceeded"})
        section = HistorySection()
        turn = make_turn(ledger=ledger, run_id=run_id)
        msgs = section.render(turn)
        assert any("Budget exceeded" in str(m.content) for m in msgs)

    def test_fidelity_full_for_recent(self):
        ledger = InMemoryLedger()
        run_id = "run_test"
        long_text = "x " * 200
        ledger.append(run_id, "user_input", {"text": long_text})
        section = HistorySection()
        turn = make_turn(ledger=ledger, run_id=run_id)
        msgs = section.render(turn)
        assert msgs[0].content == long_text

    @staticmethod
    def _long_conversation(turns: int):
        ledger, run_id = InMemoryLedger(), "run_test"
        for i in range(turns):
            ledger.append(run_id, "user_input", {"text": f"msg {i} " + "pad " * 50})
            ledger.append(run_id, "model_response", {"text": "\n".join(f"reply {i} line {j}" for j in range(120))})
        return ledger, run_id

    def test_before_the_verbatim_point_assistant_text_is_compressed_then_summarized(self):
        ledger, run_id = self._long_conversation(70)
        turn = make_turn(ledger=ledger, run_id=run_id)
        turn.working_state.verbatim_sequence = 2 * 69 + 1  # the last turn is the tail
        msgs = HistorySection().render(turn)
        replies = [str(m.content) for m in msgs if m.role == "assistant"]
        assert replies[-1].count("\n") == 119, "the tail is verbatim"
        assert "omitted" in replies[-2], "just before the point: head and tail"
        assert replies[0].startswith("reply ") and "  [120 lines, " in replies[0], "far before the point: one line"

    def test_the_user_is_never_compressed_and_the_oldest_are_dropped(self):
        ledger, run_id = self._long_conversation(100)
        turn = make_turn(ledger=ledger, run_id=run_id)
        turn.working_state.verbatim_sequence = 2 * 99 + 1
        msgs = HistorySection().render(turn)
        users = [str(m.content) for m in msgs if m.role == "user"]
        assert len(users) == 100 and all(u.endswith("pad ") for u in users)
        # windows count messages of every role: 24 compressed and 60 summarized
        # messages hold 12 and 30 assistant replies, plus the verbatim tail
        assert len([m for m in msgs if m.role == "assistant"]) == 12 + 30 + 1

    def test_without_a_verbatim_point_everything_is_verbatim(self):
        ledger, run_id = self._long_conversation(30)
        msgs = HistorySection().render(make_turn(ledger=ledger, run_id=run_id))
        assert all(str(m.content).count("\n") == 119 for m in msgs if m.role == "assistant")

    def test_empty_ledger_returns_empty(self):
        section = HistorySection()
        turn = make_turn()
        assert section.render(turn) == []

    def test_non_renderable_events_skipped(self):
        ledger = InMemoryLedger()
        run_id = "run_test"
        ledger.append(run_id, "user_input", {"text": "hi"})
        ledger.append(run_id, "projection_compiled", {"tokens": 100})
        ledger.append(run_id, "decision_validated", {"ok": True})
        section = HistorySection()
        turn = make_turn(ledger=ledger, run_id=run_id)
        msgs = section.render(turn)
        assert len(msgs) == 1
        assert msgs[0].content == "hi"


class TestTocEpochCaching:
    def test_toc_updates_after_registry_change(self):
        reg = Registry()
        reg.register(capability_dict("web.a", category="web"))
        section = TocSection()
        turn = make_turn(registry=reg)
        first = section.render(turn)
        assert "web(1)" in first[0].content
        assert section.render(turn)[0] is first[0]
        reg.register(capability_dict("web.b", category="web"))
        assert "web(2)" in section.render(turn)[0].content

    def test_kernel_is_stable_while_the_registry_is_unchanged(self):
        reg = Registry()
        reg.register(capability_dict("demo.p", pinned=True))
        section = KernelSection("kernel")
        first = section.render(make_turn(registry=reg))[0].content
        assert section.render(make_turn(registry=reg))[0].content == first

    def test_kernel_picks_up_a_capability_pinned_later(self):
        reg = Registry()
        reg.register(capability_dict("demo.p", pinned=True))
        section = KernelSection("kernel")
        before = section.render(make_turn(registry=reg))[0].content
        assert "demo.late_pin" not in before
        reg.register(capability_dict("demo.late_pin", pinned=True))
        assert "demo.late_pin" in section.render(make_turn(registry=reg))[0].content


class TestWindowEnforcement:
    def test_candidates_shrink_first(self):
        reg = Registry()
        caps = [reg.register(capability_dict(f"demo.tool_{i}", summary="x" * 120)) for i in range(10)]
        from state_projection_loop import ScoredTool

        projection = default_projection(reg, window=260)
        ledger, run_id = make_ledger_with_events(n_user=1)
        turn = make_turn(registry=reg, ledger=ledger, run_id=run_id, window=260,
                         candidates=[ScoredTool(tool=c, score=1.0) for c in caps])
        msgs = projection.render(turn)
        assert estimate_tokens(msgs) <= 260
        assert len(turn.candidates) < 10

    def test_history_emergency_trim(self):
        reg = Registry()
        projection = default_projection(reg, window=500)
        ledger = InMemoryLedger()
        run_id = "run_test"
        for i in range(20):
            ledger.append(run_id, "user_input", {"text": f"message {i} " + "long text " * 30})
            ledger.append(run_id, "model_response", {"text": f"reply {i} " + "long text " * 30})
        turn = make_turn(registry=reg, ledger=ledger, run_id=run_id, window=500)
        msgs = projection.render(turn)
        assert estimate_tokens(msgs) <= 500

    def test_native_tool_schemas_count_against_the_budget(self):
        reg = Registry()
        reg.register(capability_dict("demo.tool", properties={
            f"p{i}": {"type": "string", "description": "x" * 40} for i in range(6)
        }))
        cap = reg.get("demo.tool")
        projection = default_projection(reg, window=400)
        ledger = InMemoryLedger()
        run_id = "run_test"
        for i in range(10):
            ledger.append(run_id, "user_input", {"text": f"message {i} " + "pad " * 20})

        without_schema = projection.render(make_turn(registry=reg, ledger=ledger, run_id=run_id, window=400))
        with_schema = projection.render(
            make_turn(registry=reg, ledger=ledger, run_id=run_id, window=400),
            api_tools=[cap.tool_spec()],
        )
        assert estimate_tokens(with_schema) <= estimate_tokens(without_schema)

    def test_reserved_output_tokens_counted(self):
        reg = Registry()
        projection = default_projection(reg, window=400)
        ledger = InMemoryLedger()
        run_id = "run_test"
        for i in range(10):
            ledger.append(run_id, "user_input", {"text": f"message {i} " + "pad " * 20})

        unreserved = projection.render(make_turn(registry=reg, ledger=ledger, run_id=run_id, window=400))
        reserved = projection.render(
            make_turn(registry=reg, ledger=ledger, run_id=run_id, window=400), reserved_tokens=150,
        )
        assert estimate_tokens(reserved) <= estimate_tokens(unreserved)


class TestBuildDefaultSections:
    def test_unknown_section_name_rejected(self):
        with pytest.raises(ValueError, match="Unknown section"):
            build_default_sections(["kernel", "mystery"], kernel_text="")

    def test_default_section_order(self):
        sections = build_default_sections(
            ["kernel", "toc", "history", "working_state", "candidates"],
            kernel_text="k",
        )
        names = [s.name for s in sections]
        assert names == ["kernel", "toc", "history", "working_state", "candidates"]


class TestToolCallPairing:
    """A native tool-calling provider rejects an assistant message whose
    tool_calls have no matching results, and a result with no call. The
    projection must never emit either, whatever produced the gap."""

    @staticmethod
    def _decision(ledger, run_id, *call_ids, text=""):
        ledger.append(run_id, "model_response", {
            "text": text,
            "calls": [{"name": "demo.tool", "arguments": {}, "id": cid} for cid in call_ids],
        })

    @staticmethod
    def _result(ledger, run_id, call_id, text="ok"):
        ledger.append(run_id, "observation", {"call_id": call_id, "name": "demo.tool", "text": text})

    def _render(self, ledger, run_id, **cfg):
        turn = make_turn(ledger=ledger, run_id=run_id)
        for key, value in cfg.items():
            if key == "verbatim_sequence":
                turn.working_state.verbatim_sequence = value
            else:
                setattr(turn.config.compression, key, value)
        return HistorySection().render(turn)

    def test_a_decision_still_awaiting_its_results_is_hidden(self):
        ledger, run_id = InMemoryLedger(), "run_test"
        ledger.append(run_id, "user_input", {"text": "hi"})
        self._decision(ledger, run_id, "c0")  # parked on an approval: no result yet
        msgs = self._render(ledger, run_id)
        assert [m.role for m in msgs] == ["user"]

    def test_a_partly_answered_decision_is_hidden_whole(self):
        ledger, run_id = InMemoryLedger(), "run_test"
        ledger.append(run_id, "user_input", {"text": "hi"})
        self._decision(ledger, run_id, "c0", "c1")
        self._result(ledger, run_id, "c0")
        msgs = self._render(ledger, run_id)
        assert [m.role for m in msgs] == ["user"]

    def test_a_complete_decision_is_kept(self):
        ledger, run_id = InMemoryLedger(), "run_test"
        ledger.append(run_id, "user_input", {"text": "hi"})
        self._decision(ledger, run_id, "c0", "c1")
        self._result(ledger, run_id, "c0")
        self._result(ledger, run_id, "c1")
        msgs = self._render(ledger, run_id)
        assert [m.role for m in msgs] == ["user", "assistant", "tool", "tool"]

    def test_tier_exclusion_never_orphans_a_result(self):
        """The oldest events fall out of the window one at a time; the cut
        must not land between a decision and its results."""
        ledger, run_id = InMemoryLedger(), "run_test"
        self._decision(ledger, run_id, "c0")
        self._result(ledger, run_id, "c0")
        for i in range(4):
            ledger.append(run_id, "user_input", {"text": f"later {i}"})
        # With the point after everything, the decision is 6 messages before
        # it and its result 5: compressed_window=2 + summary_window=3 keeps
        # the result but not the decision.
        msgs = self._render(ledger, run_id, compressed_window=2, summary_window=3, verbatim_sequence=7)
        assert not any(m.role == "assistant" and m.tool_calls for m in msgs)
        assert not any(m.role == "tool" for m in msgs)
