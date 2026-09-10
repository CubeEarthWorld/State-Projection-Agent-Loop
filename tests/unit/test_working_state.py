"""WorkingState: rendering, round-trip serialization."""
from __future__ import annotations

from state_projection_loop.working_state import RecordedDecision, WorkingState


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


class TestSeeding:
    """Seeded typed fields go through the same parser snapshots use."""

    def test_seeded_decisions_survive_a_round_trip(self):
        from state_projection_loop import ScriptedLLM, Session

        session = Session(ScriptedLLM([]), seed={
            "goal": "g",
            "decisions": [{"text": "use sqlite", "reason": "single writer"}],
        })
        assert session.working_state.to_dict()["decisions"] == [
            {"text": "use sqlite", "reason": "single writer"}
        ]

    def test_seeded_extra_is_not_nested_under_itself(self):
        from state_projection_loop import ScriptedLLM, Session

        session = Session(ScriptedLLM([]), seed={"extra": {"flags": {}}})
        assert session.working_state.extra == {"flags": {}}

    def test_unknown_keys_fall_back_to_extra(self):
        from state_projection_loop import ScriptedLLM, Session

        session = Session(ScriptedLLM([]), seed={"campaign": {"chapter": 2}})
        assert session.working_state.extra == {"campaign": {"chapter": 2}}
