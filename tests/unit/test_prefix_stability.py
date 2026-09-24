"""The prompt a provider caches is the tools array, then the messages up to
the volatile tail. These tests pin the two things that used to move under
it: the native tool list (re-ranked every step) and the history boundaries
after a rewind or branch (left pointing at the old run's sequence numbers).
Asserted on what actually reaches the adapter."""
from __future__ import annotations

from state_projection_loop import Config, Registry, ScriptedLLM, Session, install_builtins
from state_projection_loop.context import TurnContext
from state_projection_loop.projection import Projection
from state_projection_loop.serialization import dumps

from _util import allow_all, capability_dict, echo_handler

LONG = "line\n" * 100


def _registry() -> Registry:
    registry = Registry()
    for name, words in (("weather.forecast.get", "weather forecast rain sunny"),
                        ("calendar.event.add", "calendar event schedule meeting"),
                        ("mail.message.send", "mail email message send"),
                        ("notes.note.write", "notes note write memo")):
        registry.register(capability_dict(name, description=f"{name} tool.", properties={"text": {"type": "string"}},
                                          embedding_text=words, effects=[("read", "workspace:*")]),
                          handler=echo_handler)
    return registry


def _stable_part(request: dict) -> tuple[str, list[str]]:
    """The tools array and every message before the trailing system block
    (working state, checklists, candidates: volatile by design), as bytes."""
    messages = list(request["messages"])
    while messages and messages[-1].role == "system":
        messages.pop()
    return dumps(request["tools"]), [dumps([m.role, m.content, m.tool_call_id, [c.to_dict() for c in m.tool_calls]])
                                     for m in messages]


class TestNativeToolsStayPut:
    def test_the_tools_array_only_ever_grows_at_the_end(self):
        """Candidates change with every message; the native list must not
        reorder or lose an entry because of it."""
        messages = ["weather forecast please", "schedule a calendar meeting", "rain tomorrow? weather",
                    "send an email message", "weather forecast again"]
        llm = ScriptedLLM([f"ok {i}" for i in range(len(messages))])
        session = Session(llm, registry=_registry(), policy=allow_all())
        for text in messages:
            session.send(text)
        sent = [[t["name"] for t in r["tools"]] for r in llm.requests]
        for before, after in zip(sent, sent[1:]):
            assert after[:len(before)] == before, f"{after} does not extend {before}"
        assert len(set(map(tuple, sent))) > 1, "the scenario must surface new tools along the way"
        assert sent[4] == sent[3], "a turn whose candidates were all offered already sends the same tools"

    def test_the_cached_prefix_is_byte_identical_when_only_candidates_change(self):
        llm = ScriptedLLM(["one", "two", "three"])
        session = Session(llm, registry=_registry(), policy=allow_all())
        session.send("weather forecast and calendar schedule, email and notes")  # surfaces every tool
        session.send("weather")
        session.send("calendar")
        parts = [_stable_part(r) for r in llm.requests]
        # The same tools, and each request's messages the previous ones plus the new turn.
        assert parts[0][0] == parts[1][0] == parts[2][0]
        for (_, before), (_, after) in zip(parts, parts[1:]):
            assert after[:len(before)] == before and len(after) > len(before)
        tail = [m.content for m in llm.requests[1]["messages"] if m.role == "system"][-1]
        assert tail.startswith("[Tool candidates")  # the ranking still reaches the model, at the tail

    def test_using_a_tool_does_not_move_it(self):
        llm = ScriptedLLM([
            ScriptedLLM.call("mail.message.send", text="a"), "sent",
            ScriptedLLM.call("weather.forecast.get", text="b"), "rain",
            ScriptedLLM.call("mail.message.send", text="c"), "sent again",
        ])
        config = Config.from_dict({"discovery": {"query_sources": []}})  # no candidates at all
        session = Session(llm, registry=_registry(), config=config, policy=allow_all())
        for text in ("one", "two", "three"):
            session.send(text)
        assert session.native_tools == ["mail.message.send", "weather.forecast.get"]
        last = [t["name"] for t in llm.requests[-1]["tools"]]
        assert last[-2:] == ["mail__message__send", "weather__forecast__get"]

    def test_the_list_is_capped_by_active_tools_least_recent_out(self):
        llm = ScriptedLLM([
            ScriptedLLM.call("mail.message.send", text="a"),
            ScriptedLLM.call("weather.forecast.get", text="b"),
            ScriptedLLM.call("mail.message.send", text="c"),
            ScriptedLLM.call("notes.note.write", text="d"),
            "done",
        ])
        config = Config.from_dict({"discovery": {"query_sources": [], "active_tools": 2}})
        session = Session(llm, registry=_registry(), config=config, policy=allow_all())
        session.send("go")
        assert session.native_tools == ["mail.message.send", "notes.note.write"]

    def test_a_resumed_run_sends_the_same_tools(self, tmp_path):
        config = Config.from_dict({"discovery": {"query_sources": []},
                                   "persistence": {"ledger_directory": str(tmp_path)}})
        first_llm = ScriptedLLM([ScriptedLLM.call("meta.tool.find", query="calendar schedule meeting"), "found"])
        first = Session(first_llm, registry=_registry(), config=config, policy=allow_all())
        first.send("find me a calendar tool")
        assert "calendar.event.add" in first.native_tools

        llm = ScriptedLLM(["again"])
        resumed = Session.resume_from_ledger(llm, first.run.id, config=config, registry=_registry(),
                                             policy=allow_all())
        assert resumed.native_tools == first.native_tools
        resumed.send("next")
        assert llm.requests[0]["tools"] == first_llm.requests[-1]["tools"]

    def test_rewind_puts_back_the_tools_that_turn_began_with(self):
        llm = ScriptedLLM([
            ScriptedLLM.call("mail.message.send", text="a"), "sent",
            ScriptedLLM.call("weather.forecast.get", text="b"), "rain",
        ])
        config = Config.from_dict({"discovery": {"query_sources": []}})
        session = Session(llm, registry=_registry(), config=config, policy=allow_all())
        session.send("one")
        session.send("two")
        assert session.native_tools == ["mail.message.send", "weather.forecast.get"]
        session.rewind(to_turn=1)
        assert session.native_tools == ["mail.message.send"]

    def test_the_window_drops_the_least_recently_used_schema_first(self):
        """The array is in first-sent order, so its first entry is not the
        one that matters least; the recency the session hands over is."""
        registry = _registry()
        mail, notes = (registry.get(n).tool_spec() for n in ("mail.message.send", "notes.note.write"))
        room = max(Projection([]).schema_tokens([mail]), Projection([]).schema_tokens([notes]))
        ctx = TurnContext(config=Config(), registry=registry,
                          tool_recency=["notes__note__write", "mail__message__send"])
        Projection([], window_tokens=room).render(ctx, api_tools=[mail, notes])
        assert [t["name"] for t in ctx.api_tools] == ["mail__message__send"]


def _tiered(**compression) -> Config:
    return Config.from_dict({"compression": {"full_window": 2, **compression},
                             "compaction": {"trigger_ratio": 0}})


class TestRewindCarriesTheHistoryBoundaries:
    def test_resending_after_a_rewind_reproduces_the_original_request(self):
        """Rewinding to turn t and sending the same message again must render
        exactly what turn t rendered the first time. The copied history is
        renumbered; a verbatim point left at the old run's number lands past
        it and compresses everything that should have been verbatim."""
        replies = [f"reply {i} {LONG}" for i in range(6)]
        llm = ScriptedLLM(replies + [replies[5]])
        session = Session(llm, config=_tiered())
        for i in range(6):
            session.send(f"msg {i}")
        session.rewind(to_turn=5)
        session.send("msg 5")
        assert llm.requests[-1]["messages"] == llm.requests[5]["messages"]

    def test_the_fold_point_is_carried_too(self):
        session = Session(ScriptedLLM([f"reply {i}" for i in range(4)]), config=_tiered())
        for i in range(4):
            session.send(f"msg {i}")
        events = [e for e in session.ledger.iter_run(session.run.id) if e.type in ("user_input", "model_response")]
        # As if a fold had absorbed the first exchange before turn 3 began.
        checkpoint = next(e for e in session.ledger.iter_run(session.run.id)
                          if e.type == "checkpoint" and e.sequence > events[6].sequence)
        checkpoint.data["working_state"]["folded_sequence"] = events[1].sequence
        checkpoint.data["working_state"]["verbatim_sequence"] = events[4].sequence
        session.rewind(to_turn=3)
        kept = [e for e in session.ledger.iter_run(session.run.id) if e.type in ("user_input", "model_response")]
        assert session.working_state.folded_sequence == kept[1].sequence
        assert session.working_state.verbatim_sequence == kept[4].sequence

    def test_rewinding_twice_restores_the_earlier_turns_state(self):
        """The first rewind used to copy only the messages, so a second
        rewind found the first rewind's checkpoint for every turn."""
        llm = ScriptedLLM([ScriptedLLM.call("state.goal.set", text="A"), "ok",
                           ScriptedLLM.call("state.goal.set", text="B"), "ok", "r2"])
        session = Session(llm, policy=allow_all())
        install_builtins(session.registry, ["state"])
        for text in ("t0", "t1", "t2"):
            session.send(text)
        session.rewind(to_turn=2)
        assert session.working_state.goal == "B"
        session.rewind(to_turn=1)
        assert session.working_state.goal == "A"

    def test_a_branch_can_be_rewound(self):
        llm = ScriptedLLM([ScriptedLLM.call("state.goal.set", text="A"), "ok",
                           ScriptedLLM.call("state.goal.set", text="B"), "ok"])
        session = Session(llm, policy=allow_all())
        install_builtins(session.registry, ["state"])
        session.send("t0")
        session.send("t1")
        branch, _ = session.branch()
        branch.rewind(to_turn=1)
        assert branch.working_state.goal == "A"

    def test_a_branch_renders_its_history_as_the_parent_does(self):
        replies = [f"reply {i} {LONG}" for i in range(7)]
        llm = ScriptedLLM(replies + ["same", "same"])
        session = Session(llm, config=_tiered())
        for i in range(7):
            session.send(f"msg {i}")
        branch, _ = session.branch()
        session.send("next")
        branch.send("next")
        assert llm.requests[-1]["messages"] == llm.requests[-2]["messages"]
