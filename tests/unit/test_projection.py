"""Projection pipeline: section ordering (I3), window enforcement (I2),
kernel immutability (I4), epoch-cached TOC (defect-2 fix)."""
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
    SummarySection,
    TocSection,
    ToolSearch,
    TurnContext,
)
from state_projection_loop.projection import build_default_sections
from state_projection_loop.tokens import estimate_tokens

from _util import tool_dict


def make_turn(registry=None, conversation=None, summary=None, candidates=None, window=30000):
    cfg = Config()
    cfg.projection.window_tokens = window
    return TurnContext(
        config=cfg,
        registry=registry or Registry(),
        conversation=conversation or [],
        summary=summary or [],
        candidates=candidates or [],
    )


def default_projection(registry, kernel="You are helpful.", window=30000):
    sections = build_default_sections(
        ["kernel", "toc", "summary", "conversation", "candidates"],
        kernel_text=kernel,
        pinned=registry.pinned(),
    )
    return Projection(sections, window_tokens=window)


class TestOrderingInvariants:
    def test_volatile_must_be_last(self):
        with pytest.raises(ProjectionError, match="I3"):
            Projection([CandidatesSection(), ConversationSection()])

    def test_valid_default_order(self):
        Projection([KernelSection("k"), TocSection(), SummarySection(),
                    ConversationSection(), CandidatesSection()])

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
        reg.register(tool_dict("pinned_tool", pinned=True, description="Pinned helper."))
        projection = default_projection(reg)
        msgs = projection.render(make_turn(registry=reg))
        assert msgs[0].role == "system"
        assert "You are helpful." in msgs[0].content
        assert "### pinned_tool" in msgs[0].content

    def test_toc_present_and_summary_absent_when_empty(self):
        reg = Registry()
        reg.register(tool_dict("t", category="web"))
        projection = default_projection(reg)
        msgs = projection.render(make_turn(registry=reg))
        contents = [str(m.content) for m in msgs]
        assert any("[Tool index] web(1)" in c for c in contents)
        assert not any("[Summary" in c for c in contents)

    def test_toc_disabled_by_config(self):
        reg = Registry()
        reg.register(tool_dict("t", category="web"))
        projection = default_projection(reg)
        turn = make_turn(registry=reg)
        turn.config.discovery.toc = False
        msgs = projection.render(turn)
        assert not any("[Tool index]" in str(m.content) for m in msgs)

    def test_candidates_render_last(self):
        reg = Registry()
        td = reg.register(tool_dict("cand", summary="candidate tool"))
        from state_projection_loop import ScoredTool

        projection = default_projection(reg)
        turn = make_turn(registry=reg, conversation=[Message(role="user", content="hi")],
                         candidates=[ScoredTool(tool=td, score=1.0)])
        msgs = projection.render(turn)
        assert "[Tool candidates" in str(msgs[-1].content)
        assert "- cand(" in str(msgs[-1].content)

    def test_summary_rendered_when_present(self):
        projection = default_projection(Registry())
        msgs = projection.render(make_turn(summary=["I searched the web because the user asked."]))
        assert any("[Summary of earlier conversation]" in str(m.content) for m in msgs)


class TestTocEpochCaching:
    def test_toc_updates_after_registry_change(self):
        reg = Registry()
        reg.register(tool_dict("a", category="web"))
        section = TocSection()
        turn = make_turn(registry=reg)
        first = section.render(turn)
        assert "web(1)" in first[0].content
        assert section.render(turn)[0] is first[0]  # cached within an epoch
        reg.register(tool_dict("b", category="web"))
        assert "web(2)" in section.render(turn)[0].content

    def test_kernel_is_immutable_across_registry_changes(self):
        reg = Registry()
        reg.register(tool_dict("p", pinned=True))
        section = KernelSection("kernel", reg.pinned())
        before = section.render(make_turn(registry=reg))[0].content
        reg.register(tool_dict("late_pin", pinned=True))
        after = section.render(make_turn(registry=reg))[0].content
        assert before == after  # I4


class TestWindowEnforcement:
    def test_candidates_shrink_first(self):
        reg = Registry()
        tds = [reg.register(tool_dict(f"tool_{i}", summary="x" * 120)) for i in range(10)]
        from state_projection_loop import ScoredTool

        projection = default_projection(reg, window=260)
        turn = make_turn(registry=reg, window=260,
                         candidates=[ScoredTool(tool=t, score=1.0) for t in tds])
        msgs = projection.render(turn)
        assert estimate_tokens(msgs) <= 260
        assert len(turn.candidates) < 10  # dropped from the low-score end

    def test_conversation_emergency_trim(self):
        reg = Registry()
        projection = default_projection(reg, window=500)
        conversation = [Message(role="user", content=f"message {i} " + "long text " * 30)
                        for i in range(8)]
        turn = make_turn(registry=reg, conversation=conversation, window=500)
        msgs = projection.render(turn)
        assert estimate_tokens(msgs) <= 500
        assert any("trimmed" in str(m.content) for m in msgs)
        # newest message survives
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
