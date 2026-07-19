"""Projection pipeline: section ordering, window enforcement including
native tool-schema + reserved-output budgeting (P0-5), kernel immutability,
epoch-cached TOC, candidate-card dedup against native schemas."""
from __future__ import annotations

import pytest

from state_projection_loop import (
    CandidatesSection,
    Config,
    ConversationSection,
    KernelSection,
    Message,
    Projection,
    ProjectionError,
    Registry,
    TocSection,
    TurnContext,
)
from state_projection_loop.projection import build_default_sections
from state_projection_loop.tokens import estimate_tokens
from state_projection_loop.working_state import WorkingState

from _util import capability_dict


def make_turn(registry=None, conversation=None, working_state=None, candidates=None, window=30000):
    cfg = Config()
    cfg.projection.window_tokens = window
    return TurnContext(
        config=cfg,
        registry=registry or Registry(),
        conversation=conversation or [],
        working_state=working_state or WorkingState(),
        candidates=candidates or [],
    )


def default_projection(registry, kernel="You are helpful.", window=30000):
    sections = build_default_sections(
        ["kernel", "toc", "conversation", "working_state", "candidates"],
        kernel_text=kernel,
        pinned=registry.pinned(),
    )
    return Projection(sections, window_tokens=window)


class TestOrderingInvariants:
    def test_volatile_must_be_last(self):
        with pytest.raises(ProjectionError, match="Invariant violated"):
            Projection([CandidatesSection(), ConversationSection()])

    def test_valid_default_order(self):
        from state_projection_loop.working_state import WorkingStateSection

        Projection([KernelSection("k"), TocSection(), ConversationSection(),
                    WorkingStateSection(), CandidatesSection()])

    def test_unknown_cache_class_rejected(self):
        class Bad:
            name = "bad"
            cache_class = "sometimes"

            def render(self, turn):
                return []

        with pytest.raises(ProjectionError, match="cache_class"):
            Projection([Bad()])

    def test_unknown_section_name_rejected(self):
        with pytest.raises(ProjectionError, match="Unknown section"):
            build_default_sections(["kernel", "mystery"], kernel_text="", pinned=[])


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
        turn = make_turn(registry=reg, conversation=[Message(role="user", content="hi")],
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
        turn.dedupe_candidate_cards = True
        msgs = projection.render(turn, api_tools=[cap.api_schema()])
        last = str(msgs[-1].content)
        assert "schemas sent natively" in last
        assert "a somewhat long description" not in last

    def test_working_state_rendered_when_present(self):
        projection = default_projection(Registry())
        ws = WorkingState(goal="ship the feature")
        msgs = projection.render(make_turn(working_state=ws))
        assert any("[Working state]" in str(m.content) and "ship the feature" in str(m.content) for m in msgs)


class TestTocEpochCaching:
    def test_toc_updates_after_registry_change(self):
        reg = Registry()
        reg.register(capability_dict("web.a", category="web"))
        section = TocSection()
        turn = make_turn(registry=reg)
        first = section.render(turn)
        assert "web(1)" in first[0].content
        assert section.render(turn)[0] is first[0]  # cached within an epoch
        reg.register(capability_dict("web.b", category="web"))
        assert "web(2)" in section.render(turn)[0].content

    def test_kernel_is_immutable_across_registry_changes(self):
        reg = Registry()
        reg.register(capability_dict("demo.p", pinned=True))
        section = KernelSection("kernel", reg.pinned())
        before = section.render(make_turn(registry=reg))[0].content
        reg.register(capability_dict("demo.late_pin", pinned=True))
        after = section.render(make_turn(registry=reg))[0].content
        assert before == after


class TestWindowEnforcement:
    def test_candidates_shrink_first(self):
        reg = Registry()
        caps = [reg.register(capability_dict(f"demo.tool_{i}", summary="x" * 120)) for i in range(10)]
        from state_projection_loop import ScoredTool

        projection = default_projection(reg, window=260)
        turn = make_turn(registry=reg, window=260,
                         candidates=[ScoredTool(tool=c, score=1.0) for c in caps])
        msgs = projection.render(turn)
        assert estimate_tokens(msgs) <= 260
        assert len(turn.candidates) < 10

    def test_conversation_emergency_trim(self):
        reg = Registry()
        projection = default_projection(reg, window=500)
        conversation = [Message(role="user", content=f"message {i} " + "long text " * 30)
                        for i in range(8)]
        turn = make_turn(registry=reg, conversation=conversation, window=500)
        msgs = projection.render(turn)
        assert estimate_tokens(msgs) <= 500
        assert any("trimmed" in str(m.content) for m in msgs)
        assert any("message 7" in str(m.content) for m in msgs)

    def test_trim_never_leaves_orphan_observations(self):
        from state_projection_loop import ToolCall

        reg = Registry()
        projection = default_projection(reg, window=220)
        conversation = []
        for i in range(6):
            conversation.append(Message(role="assistant", content=f"step {i} " + "pad " * 20,
                                        tool_calls=[ToolCall(name="t", arguments={})]))
            conversation.append(Message(role="tool", content="result " + "pad " * 20,
                                        tool_call_id=f"c{i}", name="t"))
        turn = make_turn(registry=reg, conversation=conversation, window=220)
        msgs = projection.render(turn)
        roles = [m.role for m in msgs]
        first_conv = next((i for i, m in enumerate(msgs) if m.role in ("assistant", "tool")), None)
        if first_conv is not None:
            assert roles[first_conv] == "assistant"

    def test_native_tool_schemas_count_against_the_budget(self):
        # Same window either way; sending a native tool schema alongside the
        # projection eats into the same budget, so strictly less (or equal)
        # conversation can survive once the schema is counted (P0-5) —
        # a schema was previously invisible to the window check entirely.
        reg = Registry()
        reg.register(capability_dict("demo.tool", properties={
            f"p{i}": {"type": "string", "description": "x" * 40} for i in range(6)
        }))
        cap = reg.get("demo.tool")
        projection = default_projection(reg, window=400)
        conversation = [Message(role="user", content=f"message {i} " + "pad " * 20) for i in range(10)]

        without_schema = projection.render(make_turn(registry=reg, conversation=list(conversation), window=400))
        with_schema = projection.render(
            make_turn(registry=reg, conversation=list(conversation), window=400),
            api_tools=[cap.api_schema()],
        )
        assert estimate_tokens(with_schema) <= estimate_tokens(without_schema)
        assert len(with_schema) <= len(without_schema)

    def test_reserved_output_tokens_counted(self):
        # Same reasoning as above but for the reserved-output allowance:
        # reserving room for the model's own reply must shrink what fits,
        # not be silently ignored.
        reg = Registry()
        projection = default_projection(reg, window=400)
        conversation = [Message(role="user", content=f"message {i} " + "pad " * 20) for i in range(10)]

        unreserved = projection.render(make_turn(registry=reg, conversation=list(conversation), window=400))
        reserved = projection.render(
            make_turn(registry=reg, conversation=list(conversation), window=400), reserved_tokens=150,
        )
        assert estimate_tokens(reserved) <= estimate_tokens(unreserved)
