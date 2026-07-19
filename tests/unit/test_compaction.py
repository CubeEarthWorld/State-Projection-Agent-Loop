"""Compactor: split point safety (never orphans observations), fold
contract v2 JSON delta parsing, deterministic fallback."""
from __future__ import annotations

from state_projection_loop.compaction import Compactor, _parse_delta, deterministic_fold
from state_projection_loop.config import Config
from state_projection_loop.messages import ASSISTANT, Message, OBSERVATION, SYSTEM, USER, ToolCall
from state_projection_loop.working_state import WorkingState


class TestParseDelta:
    def test_parses_plain_json(self):
        d = _parse_delta('{"new_facts": ["a"]}')
        assert d == {"new_facts": ["a"]}

    def test_parses_fenced_json(self):
        d = _parse_delta('```json\n{"new_facts": ["a"]}\n```')
        assert d == {"new_facts": ["a"]}

    def test_malformed_json_becomes_a_fact_not_a_crash(self):
        d = _parse_delta("not json at all")
        assert "new_facts" in d and "not json" in d["new_facts"][0]


class TestSplitPoint:
    def test_never_splits_inside_a_tool_call_pair(self):
        compactor = Compactor(Config())
        conversation = []
        for i in range(6):
            conversation.append(Message(role=ASSISTANT, content=f"step {i}",
                                        tool_calls=[ToolCall(name="t", arguments={})]))
            conversation.append(Message(role=OBSERVATION, content="result", tool_call_id=f"c{i}", name="t"))
        i = compactor.split_point(conversation)
        assert conversation[i].role != OBSERVATION  # never orphans an observation at the head


class TestFold:
    def test_deterministic_fold_used_when_no_summarizer(self):
        # split_point always leaves at least the last exchange unfolded, so
        # a lone message never folds — use enough turns that a genuine
        # older half exists to fold away.
        compactor = Compactor(Config())
        ws = WorkingState()
        conversation = [Message(role=USER, content="please remember X" + " pad" * 20)] + [
            Message(role=ASSISTANT, content=f"reply {i}" + " pad" * 20) for i in range(5)
        ]
        folded, remaining = compactor.fold(conversation, ws)
        assert folded is True
        assert any("please remember X" in f for f in ws.confirmed_facts)
        assert len(remaining) < len(conversation)

    def test_fold_merges_summarizer_json_delta(self):
        from state_projection_loop.llm import ScriptedLLM

        summarizer = ScriptedLLM(['{"new_facts": ["the user prefers dark mode"]}'])
        compactor = Compactor(Config(), summarizer)
        ws = WorkingState()
        conversation = [Message(role=USER, content="I like dark mode")] * 2
        folded, remaining = compactor.fold(conversation, ws)
        assert folded is True
        assert "the user prefers dark mode" in ws.confirmed_facts

    def test_empty_conversation_does_not_fold(self):
        compactor = Compactor(Config())
        ws = WorkingState()
        folded, remaining = compactor.fold([], ws)
        assert folded is False
        assert ws.is_empty()

    def test_deterministic_fold_helper_direct(self):
        conversation = [Message(role=USER, content="hi"), Message(role=ASSISTANT, content="", tool_calls=[
            ToolCall(name="demo.tool", arguments={})])]
        delta = deterministic_fold(conversation)
        assert any("hi" in f for f in delta["new_facts"])
