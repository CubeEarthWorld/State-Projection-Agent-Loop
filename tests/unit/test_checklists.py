"""Plans survive compaction/restarts, and edits/handoffs remain isolated."""
import copy
import json
from pathlib import Path

import pytest

from state_projection_loop import (
    ChecklistStore, ChecklistSection, Config, Decision, PolicyEngine, Registry,
    ScriptedLLM, Session, ToolCall, TurnContext, WorkingState, install_builtins,
)


def plan(store):
    return store.execute("create", name="出荷計画", items=[{"text": "実装"}, {"text": "検証"}])


def edit(store, value, action="update", **args):
    return store.execute(action, id=value["id"], expected_revision=value["revision"], **args)


def test_crud_progress_and_revision_conflicts():
    store = ChecklistStore()
    value = plan(store)
    assert len(value["id"]) == 26
    assert value["progress"]["fraction"] == 0
    first = value["items"][0]["id"]
    value = edit(store, value, "update_item", item_id=first, item={"status": "completed", "notes": "tests passed"})
    assert value["progress"]["fraction"] == .5
    before = store.to_dict()
    with pytest.raises(ValueError, match="Revision conflict"):
        store.execute("delete", id=value["id"], expected_revision=1)
    assert before == store.to_dict()
    value = edit(store, value, "add_item", item={"text": "publish"})
    value = edit(store, value, "delete_item", item_id=value["items"][-1]["id"])
    value = edit(store, value, items=[{**x, "status": "completed"} for x in value["items"]])
    assert value["status"] == "completed"
    assert value["progress"]["fraction"] == 1
    assert store.execute("list")[0]["id"] == value["id"]
    assert "items" not in store.execute("list")[0]
    edit(store, value, "delete")
    assert store.execute("list") == []


@pytest.mark.parametrize("patch", [
    {"name": " "}, {"include_in_context": 1}, {"context_mode": "bogus"},
    {"items": [{"text": "a", "status": "in_progress"}, {"text": "b", "status": "in_progress"}]},
    {"items": [{"text": "a", "status": "unknown"}]}, {"items": None},
    {"items": [{"text": "a"}] * 201}, {"unexpected": True},
])
def test_invalid_updates_are_atomic(patch):
    store = ChecklistStore()
    value = plan(store)
    before = store.to_dict()
    with pytest.raises(ValueError):
        edit(store, value, **patch)
    assert store.to_dict() == before


def test_item_ids_cannot_be_changed_or_duplicated():
    store = ChecklistStore()
    value = plan(store)
    before = store.to_dict()
    with pytest.raises(ValueError):
        edit(store, value, "update_item", item_id=value["items"][0]["id"], item={"id": value["items"][1]["id"]})
    with pytest.raises(ValueError):
        edit(store, value, items=[value["items"][0]] * 2)
    assert store.to_dict() == before


def test_derived_status_empty_cancelled_blocked_and_atomic_switch():
    store = ChecklistStore()
    value = store.execute("create", name="empty")
    assert value["status"] == "pending" and value["progress"]["fraction"] == 0
    value = edit(store, value, items=[{"text": "a", "status": "blocked", "notes": "waiting"}])
    assert value["status"] == "blocked"
    value = edit(store, value, items=[{"text": "a", "status": "cancelled"}])
    assert value["status"] == "cancelled" and value["progress"]["fraction"] == 0
    value = edit(store, value, items=[{"text": "a", "status": "completed"}, {"text": "b", "status": "cancelled"}])
    assert value["status"] == "completed" and value["progress"]["fraction"] == 1
    value = edit(store, value, items=[{"text": "a", "status": "in_progress"}, {"text": "b"}])
    ids = [x["id"] for x in value["items"]]
    value = edit(store, value, items=[{**value["items"][0], "status": "completed"}, {**value["items"][1], "status": "in_progress"}])
    assert [x["id"] for x in value["items"]] == ids


def test_export_import_is_portable_atomic_and_isolated():
    store = ChecklistStore()
    value = plan(store)
    document = json.loads(json.dumps(store.execute("export", id=value["id"])))
    child = ChecklistStore()
    child.execute("import", document=document)
    copied = child.execute("get", id=value["id"])
    edit(child, copied, name="child")
    assert store.execute("get", id=value["id"])["name"] == "出荷計画"
    document["checklists"][0]["name"] = "mutated"
    assert child.execute("get", id=value["id"])["name"] == "child"
    before = child.to_dict()
    with pytest.raises(ValueError):
        child.execute("import", document=document)
    malformed = copy.deepcopy(document)
    malformed["checklists"].append({})
    with pytest.raises(ValueError):
        child.execute("import", document=malformed)
    assert child.to_dict() == before
    assert WorkingState.from_dict({}).checklists.execute("list") == []


def test_projection_modes_visibility_and_budget():
    store = ChecklistStore()
    value = plan(store)
    value = edit(store, value, context_mode="name")
    assert json.loads(store.render()) == {"id": value["id"], "name": value["name"]}
    value = edit(store, value, context_mode="summary")
    assert "progress" in json.loads(store.render()) and "items" not in json.loads(store.render())
    value = edit(store, value, context_mode="full")
    assert len(json.loads(store.render())["items"]) == 2
    value = edit(store, value, include_in_context=False)
    assert store.render() == "" and len(store.execute("list")) == 1
    edit(store, value, include_in_context=True, items=[{"text": "x" * 500, "notes": "y" * 2000}] * 10)
    assert len(store.render(max_chars=1000)) <= 1000
    assert "progress" in store.render(max_chars=1000)


def test_default_tool_plans_execute_in_order_and_survive_compression():
    session = Session(ScriptedLLM(["ok"] * 3))
    assert session.registry.get("planning.checklist.manage").discovery.pinned
    value = session.invoke("planning.checklist.manage", action="create", name="durable", items=[{"text": "verify"}])
    session.llm = ScriptedLLM([
        Decision(calls=[
            ToolCall(name="planning.checklist.manage", arguments={"action": "update", "id": value["id"], "expected_revision": 1, "name": "first"}),
            ToolCall(name="planning.checklist.manage", arguments={"action": "update", "id": value["id"], "expected_revision": 2, "name": "second"}),
        ]), "ok", "ok",
    ])
    session.send("work")
    session.config.compression.summary_window = 0
    session.config.compression.full_window = 0
    session.config.compression.compressed_window = 0
    session.send("continue")
    messages = session.llm.requests[-1]["messages"]
    assert any("second" in str(m.content) and "[Checklists" in str(m.content) for m in messages)
    assert session.checklists.execute("get", id=value["id"])["revision"] == 3
    assert len([e for e in session.ledger.iter_run(session.run.id) if e.type == "checklists_changed"]) == 3


def test_policy_can_deny_changes():
    session = Session(ScriptedLLM([]), policy=PolicyEngine(default_decision="deny"))
    with pytest.raises(RuntimeError):
        session.invoke("planning.checklist.manage", action="create", name="no")
    assert session.checklists.execute("list") == []


def test_restart_recovers_after_snapshot_gap_and_keeps_deletion(tmp_path):
    config = Config.from_dict({"persistence": {"ledger_directory": str(tmp_path)}})
    session = Session(ScriptedLLM([]), config=config)
    initial = session.ledger.load_snapshot(session.run.id)
    value = session.invoke("planning.checklist.manage", action="create", name="persist", include_in_context=False)
    # Simulate a crash after the mutation event but before the next snapshot.
    session.ledger.save_snapshot(initial)
    restored = Session.resume_from_ledger(ScriptedLLM([]), session.run.id, config=config)
    assert restored.checklists.execute("get", id=value["id"])["include_in_context"] is False
    restored.invoke("planning.checklist.manage", action="delete", id=value["id"], expected_revision=1)
    restored.ledger.save_snapshot(initial)
    again = Session.resume_from_ledger(ScriptedLLM([]), session.run.id, config=config)
    assert again.checklists.execute("list") == []


def test_completed_run_retains_plans_on_restart(tmp_path):
    cfg = Config.from_dict({"mode": "job", "persistence": {"ledger_directory": str(tmp_path)}})
    session = Session(ScriptedLLM([ScriptedLLM.finish(result="done")]), config=cfg)
    value = session.invoke("planning.checklist.manage", action="create", name="retained")
    session.run_job("finish")
    restored = Session.resume_from_ledger(ScriptedLLM([]), session.run.id, config=cfg)
    assert restored.run.state == "COMPLETED"
    assert restored.checklists.execute("get", id=value["id"])["name"] == "retained"


def test_branch_rewind_and_spawn_do_not_share_plans():
    session = Session(ScriptedLLM(["one", "two"]), policy=PolicyEngine(default_decision="allow"))
    value = session.invoke("planning.checklist.manage", action="create", name="original")
    session.send("first")
    child, _ = session.branch()
    child.invoke("planning.checklist.manage", action="update", id=value["id"], expected_revision=1, name="branch")
    assert session.checklists.execute("get", id=value["id"])["name"] == "original"
    session.send("second")
    session.invoke("planning.checklist.manage", action="delete", id=value["id"], expected_revision=1)
    session.rewind(to_turn=1)
    assert session.checklists.execute("get", id=value["id"])["name"] == "original"
    install_builtins(session.registry, ["spawn"])
    session.spawn_llm_factory = lambda model: ScriptedLLM([
        ScriptedLLM.call("planning.checklist.manage", action="update", id=value["id"], expected_revision=1, name="delegated"),
        ScriptedLLM.finish(result="done"),
    ])
    result = session.invoke("meta.agent.spawn", task="work", checklist_ids=[value["id"]])
    assert result["result"] == "done"
    assert result["checklists"]["checklists"][0]["name"] == "delegated"
    assert session.checklists.execute("get", id=value["id"])["name"] == "original"


def test_a_spawned_child_keeps_the_parents_deny_list():
    seen: list[list[str]] = []

    def child_step(messages, tools):
        seen.append([t["function"]["name"] for t in tools])
        return ScriptedLLM.finish(result="done")

    session = Session(ScriptedLLM([]), registry=Registry(disabled=["planning.checklist.manage"]),
                      builtins=["meta", "checklist", "spawn"], policy=PolicyEngine(default_decision="allow"),
                      spawn_llm_factory=lambda model: ScriptedLLM([child_step]))
    assert session.invoke("meta.agent.spawn", task="work") == "done"
    assert "meta__tool__find" in seen[0] and "planning__checklist__manage" not in seen[0]


def test_shared_wire_fixture():
    document = json.loads((Path(__file__).parents[1] / "fixtures/checklists_v1.json").read_text(encoding="utf-8"))
    store = ChecklistStore.from_dict(document)
    assert store.to_dict() == document
    value = store.execute("get", id=document["checklists"][0]["id"])
    assert value["status"] == "in_progress"
    assert value["progress"] == {"total": 3, "pending": 0, "in_progress": 1, "blocked": 0,
                                 "completed": 1, "cancelled": 1, "remaining": 1, "fraction": .5}


def test_projection_budget_never_deletes_plan():
    from state_projection_loop.tokens import estimate_tokens
    cfg = Config.from_dict({"projection": {"window_tokens": 2000, "reserved_output_tokens": 200}})
    session = Session(ScriptedLLM(["ok"]), config=cfg)
    for i in range(12):
        session.checklists.execute("create", name=f"plan {i}", context_mode="full", items=[{"text": "x" * 500}] * 30)
    session.send("work")
    request = session.llm.requests[-1]
    assert estimate_tokens(request["messages"]) + session.projection.schema_tokens(request["tools"]) + 200 <= 2000
    assert len(session.checklists.execute("list")) == 12
    assert len(session.checklists.execute("list", mode="full")[0]["items"]) == 30


def test_memory_default_and_persisted_branch_rewind(tmp_path):
    from state_projection_loop import InMemoryLedger
    memory = Session(ScriptedLLM([]))
    memory.invoke("planning.checklist.manage", action="create", name="memory")
    assert isinstance(memory.ledger, InMemoryLedger)
    assert memory.config.persistence.ledger_directory is None
    assert not Session(ScriptedLLM([])).checklists.execute("list")
    cfg = Config.from_dict({"persistence": {"ledger_directory": str(tmp_path)}})
    session = Session(ScriptedLLM(["one", "two"]), config=cfg)
    session.invoke("planning.checklist.manage", action="create", name="persist")
    session.send("first")
    child, _ = session.branch()
    restarted_child = Session.resume_from_ledger(ScriptedLLM([]), child.run.id, config=cfg)
    assert restarted_child.checklists.to_dict() == session.checklists.to_dict()
    session.send("second")
    session.rewind(to_turn=1)
    restarted = Session.resume_from_ledger(ScriptedLLM([]), session.run.id, config=cfg)
    assert restarted.checklists.to_dict() == session.checklists.to_dict()


def test_ledger_failure_does_not_apply_edit():
    from state_projection_loop import InMemoryLedger

    class FailingLedger(InMemoryLedger):
        def append(self, run_id, type, data):
            if type == "checklists_changed":
                raise OSError("disk unavailable")
            return super().append(run_id, type, data)

    session = Session(ScriptedLLM([]), ledger=FailingLedger())
    with pytest.raises(RuntimeError, match="disk unavailable"):
        session.invoke("planning.checklist.manage", action="create", name="not committed")
    assert session.checklists.execute("list") == []


def test_native_schema_dedup_keeps_text_fallback():
    session = Session(ScriptedLLM([]))
    turn = TurnContext(config=session.config, registry=session.registry, ledger=session.ledger, run=session.run)
    kernel = session.projection.get("kernel")
    assert "Parameters (JSON Schema)" in kernel.render(turn)[0].content
    turn.api_tools = [c.api_schema() for c in session.registry.pinned()]
    assert "Parameters (JSON Schema)" not in kernel.render(turn)[0].content
    assert "planning.checklist.manage" in kernel.render(turn)[0].content
    turn.api_tools.pop()
    assert "Parameters (JSON Schema)" in kernel.render(turn)[0].content
