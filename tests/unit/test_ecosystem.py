"""What connects a session to the world around it: cross-session memory,
listing runs, SKILL.md directories, workspace instruction files."""
from __future__ import annotations

import pytest

from state_projection_loop import (
    Config, InMemoryLedger, InstructionsSection, JsonlLedger, JsonlMemoryStore, Registry, ScriptedLLM,
    Session, load_skills,
)

from _util import allow_all


class TestMemory:
    def test_notes_saved_in_one_session_are_found_by_the_next(self, tmp_path):
        config = Config.from_dict({"persistence": {"ledger_directory": str(tmp_path)}})
        first = Session(ScriptedLLM([ScriptedLLM.call("memory.note.save", text="The user prefers tabs over spaces",
                                                      tags=["style"]), "noted"]),
                        builtins=["memory"], config=config)
        assert first.send("remember my style") == "noted"
        assert (tmp_path / "memory.jsonl").exists()

        second = Session(ScriptedLLM([ScriptedLLM.call("memory.note.search", query="indentation style tabs"),
                                      "found"]), builtins=["memory"], config=config)
        second.send("how do I indent?")
        observation = [e.data["text"] for e in second.ledger.iter_run(second.run.id) if e.type == "observation"][0]
        assert "tabs over spaces" in observation and '"style"' in observation

    def test_the_default_policy_lets_the_model_use_memory(self):
        session = Session(ScriptedLLM([]), builtins=["memory"])
        save = session.registry.get("memory.note.save")
        assert session.policy.evaluate(save, {"text": "x"}).decision == "allow"

    def test_search_ranks_by_shared_terms_then_recency(self):
        store = JsonlMemoryStore()
        store.save("deploy with make release", ["ops"])
        store.save("the release branch is called stable", ["git"])
        store.save("unrelated note", [])
        assert [n.text for n in store.search("release branch", 5)] == [
            "the release branch is called stable", "deploy with make release"]
        assert store.search("zzz", 5) == []


class TestListRuns:
    def test_a_ledger_lists_its_runs_newest_first(self, tmp_path):
        config = Config.from_dict({"persistence": {"ledger_directory": str(tmp_path)}})
        a = Session(ScriptedLLM(["one"]), config=config)
        a.send("hi")
        b = Session(ScriptedLLM([ScriptedLLM.finish(result="done")]),
                    config=Config.from_dict({"mode": "job", "persistence": {"ledger_directory": str(tmp_path)}}))
        b.run_job("task")
        runs = JsonlLedger(tmp_path).list_runs()
        assert [(r.run_id, r.state) for r in runs] == [(b.run.id, "COMPLETED"), (a.run.id, "RUNNING")]
        assert runs[0].session_id == b.session_id
        assert InMemoryLedger().list_runs() == []


class TestSkillDirectories:
    def test_skill_md_files_become_skill_capabilities(self, tmp_path):
        (tmp_path / "deploy-service").mkdir()
        (tmp_path / "deploy-service" / "SKILL.md").write_text(
            '---\nname: deploy-service\ndescription: "How to deploy the service"\n---\n1. run make release\n',
            encoding="utf-8")
        (tmp_path / "bare").mkdir()
        (tmp_path / "bare" / "SKILL.md").write_text("Just the steps.", encoding="utf-8")
        skills = {c.name: c for c in load_skills(tmp_path)}
        assert set(skills) == {"skill.bare.load", "skill.deploy_service.load"}
        assert skills["skill.deploy_service.load"].card.summary == "How to deploy the service"
        assert skills["skill.deploy_service.load"].execution.handler() == "1. run make release\n"
        assert skills["skill.bare.load"].execution.handler() == "Just the steps."


class TestInstructions:
    def test_instruction_files_up_the_tree_are_projected_outermost_first(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Repo rule: run the tests.", encoding="utf-8")
        nested = tmp_path / "pkg" / "sub"
        nested.mkdir(parents=True)
        (nested / "CLAUDE.md").write_text("Package rule: no prints.", encoding="utf-8")
        text = InstructionsSection.load(nested)
        assert text.index("Repo rule") < text.index("Package rule")
        assert f"[Instructions from {tmp_path / 'AGENTS.md'}]" in text

        llm = ScriptedLLM(["ok"])
        session = Session(llm, sections=[InstructionsSection(nested)])
        session.send("hi")
        assert "Package rule: no prints." in llm.requests[0]["messages"][0].content

    def test_no_files_means_no_message(self, tmp_path):
        assert InstructionsSection(tmp_path).render(None) == []
