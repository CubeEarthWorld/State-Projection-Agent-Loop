"""WorkingState: structured merge semantics (decisions keep their reason
through repeated folds — P1-1), rendering, round-trip serialization."""
from __future__ import annotations

from state_projection_loop.working_state import RecordedDecision, WorkingState


class TestMergeFold:
    def test_facts_are_additive_and_deduped(self):
        ws = WorkingState()
        ws.merge_fold({"new_facts": ["user wants JSON output"]})
        ws.merge_fold({"new_facts": ["user wants JSON output", "user is on Windows"]})
        assert ws.confirmed_facts == ["user wants JSON output", "user is on Windows"]

    def test_decisions_keep_reason_across_repeated_folds(self):
        ws = WorkingState()
        ws.merge_fold({"new_decisions": [{"text": "used SQLite", "reason": "no server needed for this scale"}]})
        ws.merge_fold({"new_decisions": [{"text": "added an index on user_id", "reason": "query was slow"}]})
        assert len(ws.decisions) == 2
        assert ws.decisions[0].reason == "no server needed for this scale"
        assert ws.decisions[1].reason == "query was slow"

    def test_open_questions_add_and_resolve(self):
        ws = WorkingState()
        ws.merge_fold({"new_open_questions": ["which timezone?"]})
        assert ws.open_questions == ["which timezone?"]
        ws.merge_fold({"resolved_open_questions": ["which timezone?"]})
        assert ws.open_questions == []

    def test_next_actions_replaced_not_appended(self):
        ws = WorkingState()
        ws.merge_fold({"next_actions": ["step 1", "step 2"]})
        ws.merge_fold({"next_actions": ["step 3"]})
        assert ws.next_actions == ["step 3"]

    def test_goal_only_updated_when_present(self):
        ws = WorkingState(goal="original goal")
        ws.merge_fold({"new_facts": ["irrelevant"]})
        assert ws.goal == "original goal"
        ws.merge_fold({"goal": "revised goal"})
        assert ws.goal == "revised goal"


class TestRenderAndSerialize:
    def test_render_includes_decision_reason(self):
        ws = WorkingState(decisions=[RecordedDecision(text="chose X", reason="Y was slower")])
        text = ws.render()
        assert "chose X" in text and "Y was slower" in text

    def test_is_empty(self):
        assert WorkingState().is_empty() is True
        assert WorkingState(goal="x").is_empty() is False
        assert WorkingState(extra={"flag": True}).is_empty() is False

    def test_round_trip(self):
        ws = WorkingState(
            goal="g", acceptance_criteria=["a"], constraints=["c"], confirmed_facts=["f"],
            decisions=[RecordedDecision(text="d", reason="r")], open_questions=["q"],
            next_actions=["n"], artifact_refs=["art_1"], extra={"k": "v"},
        )
        restored = WorkingState.from_dict(ws.to_dict())
        assert restored.to_dict() == ws.to_dict()
