"""Category exploration checks; no embedding model or external services."""
import asyncio
import tempfile
from unittest import TestCase

from state_projection_loop import Config, Registry, ScriptedLLM, Session
from state_projection_loop.builtin.discovery import find_tools
from state_projection_loop.context import ToolContext

from _util import capability_dict


def make_registry():
    registry = Registry()
    for name, category in [("catalog.alpha", "docs/api"), ("catalog.beta", "docs/guide"),
                           ("other.gamma", "mail")]:
        registry.register(capability_dict(name, category=category, summary="Documentation tool",
                                          description="Detailed documentation usage", no_embed=True),
                          handler=lambda: "ok")
    registry.register(capability_dict("catalog.hidden", category="docs/api"), handler=lambda: "hidden")
    registry.disable("catalog.hidden")
    return registry


def browse(session, **args):
    args.setdefault("action", "search" if "query" in args else "list" if "category" in args else "categories")
    return session.invoke("tool_search", **args)


def test_pages_advance_across_queries_and_categories_without_loading_specs():
    session = Session(ScriptedLLM([]), registry=make_registry())
    first = browse(session, category="docs", k=1)
    assert first["tools"][0]["name"] == "catalog.alpha"
    assert first["remaining_matches"] == 1 and first["has_more"]
    assert "spec" not in first["tools"][0] and "signature" not in first["tools"][0]
    assert session.native_tools == []
    # A changed query and overlapping category share the same viewed set.
    second = browse(session, query="catalog Documentation", category="docs", k=1)
    assert second["tools"][0]["name"] == "catalog.beta"
    assert session.native_tools == []
    exhausted = browse(session, category="docs", k=1)
    assert exhausted["tools"] == [] and not exhausted["has_more"]
    assert next(c for c in exhausted["categories"] if c["category"] == "mail")["remaining"] == 1
    assert browse(session, category="mail")["tools"][0]["name"] == "other.gamma"


def test_only_describe_loads_a_tool_and_reject_unloads_it_until_reset():
    session = Session(ScriptedLLM([]), registry=make_registry())
    browse(session, category="docs")
    detail = browse(session, action="describe", name="catalog.alpha")
    assert "Detailed documentation usage" in detail["spec"]
    assert session.native_tools == ["catalog.alpha"]
    assert "catalog.alpha" in session.runtime.seen_specs
    browse(session, action="reject", name="catalog.alpha", reason="Wrong operation")
    assert session.native_tools == []
    with TestCase().assertRaisesRegex(RuntimeError, "rejected"):
        browse(session, action="describe", name="catalog.alpha")
    browse(session, action="reset")
    assert browse(session, category="docs")["tools"][0]["name"] == "catalog.alpha"


def test_restart_and_branch_keep_progress_and_a_new_request_starts_fresh():
    with tempfile.TemporaryDirectory() as directory:
        _check_restart_and_branch(directory)


def _check_restart_and_branch(directory):
    config = Config.from_dict({"persistence": {"ledger_directory": directory}})
    session = Session(ScriptedLLM([]), registry=make_registry(), config=config)
    browse(session, category="docs", k=1)
    resumed = Session.resume_from_ledger(ScriptedLLM(["ok"]), session.run.id,
                                         registry=make_registry(), config=config)
    branch, _ = resumed.branch()
    assert browse(branch, category="docs", k=1)["tools"][0]["name"] == "catalog.beta"
    assert browse(resumed, category="docs", k=1)["tools"][0]["name"] == "catalog.beta"
    resumed.send("次の依頼")
    assert browse(resumed, category="docs", k=1)["tools"][0]["name"] == "catalog.alpha"


def test_retry_of_the_same_command_returns_the_same_page():
    session = Session(ScriptedLLM([]), registry=make_registry())
    ctx = ToolContext(session=session, ledger=session.ledger, run=session.run,
                      registry=session.registry, search=session.search, command_id="retry")
    first = asyncio.run(find_tools(ctx, action="list", category="docs", k=1))
    assert asyncio.run(find_tools(ctx, action="list", category="docs", k=1)) == first
    assert sum(e.type == "tool_discovery" for e in session.ledger.iter_run(session.run.id)) == 1
    ctx.command_id = "next"
    assert asyncio.run(find_tools(ctx, action="list", category="docs", k=1))["tools"][0]["name"] == "catalog.beta"


def test_selection_after_the_last_snapshot_is_recovered_from_the_ledger():
    with tempfile.TemporaryDirectory() as directory:
        config = Config.from_dict({"persistence": {"ledger_directory": directory}})
        session = Session(ScriptedLLM([]), registry=make_registry(), config=config)
        # Invoke the handler directly to simulate a crash before the next snapshot.
        ctx = ToolContext(session=session, ledger=session.ledger, run=session.run,
                          registry=session.registry, search=session.search, command_id="describe")
        asyncio.run(find_tools(ctx, action="describe", name="catalog.alpha"))
        resumed = Session.resume_from_ledger(ScriptedLLM([]), session.run.id,
                                             registry=make_registry(), config=config)
        assert resumed.native_tools == ["catalog.alpha"]
        assert "catalog.alpha" in resumed.runtime.seen_specs
        ctx.command_id = "reject"
        asyncio.run(find_tools(ctx, action="reject", name="catalog.alpha"))
        resumed = Session.resume_from_ledger(ScriptedLLM([]), session.run.id,
                                             registry=make_registry(), config=config)
        assert resumed.native_tools == []


def test_updated_versions_reappear_and_disabled_tools_stay_hidden():
    registry = make_registry()
    session = Session(ScriptedLLM([]), registry=registry)
    browse(session, category="docs/api")
    updated = capability_dict("catalog.alpha", category="docs/api")
    updated["version"] = 2
    registry.register(updated, handler=lambda: "new")
    assert [t["name"] for t in browse(session, category="docs/api")["tools"]] == ["catalog.alpha"]
    with TestCase().assertRaisesRegex(RuntimeError, "disabled"):
        browse(session, action="describe", name="catalog.hidden")


def test_discovery_writes_are_sequential_even_with_multiple_calls_in_one_turn():
    from state_projection_loop.messages import Decision, ToolCall

    llm = ScriptedLLM([Decision(calls=[ToolCall(name="tool_search", arguments={"action": "list", "category": "docs", "k": 1}),
                                     ToolCall(name="tool_search", arguments={"action": "list", "category": "docs", "k": 1})]), "ok"])
    session = Session(llm, registry=make_registry())
    session.send("ドキュメント用のツールを探して")
    pages = [e.data["response"] for e in session.ledger.iter_run(session.run.id) if e.type == "tool_discovery"]
    assert [p["tools"][0]["name"] for p in pages] == ["catalog.alpha", "catalog.beta"]


if __name__ == "__main__":
    checks = [check for name, check in globals().copy().items() if name.startswith("test_")]
    for check in checks:
        check()
    print(f"{len(checks)} category discovery checks passed (no embeddings)")
