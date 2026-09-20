"""Regression tests for verified bugs in the leaf modules (artifacts,
serialization, json_schema, llm, events, memory).

Each test is named for the bug it pins; the Dart port carries the mirror of
this file at ``test/unit/regressions_test.dart``.
"""
from __future__ import annotations

import json

from state_projection_loop.artifacts import ArtifactStore
from state_projection_loop.events import JsonlLedger
from state_projection_loop.json_schema import validate_value
from state_projection_loop.llm import FINISH_NAME, extract_finish, parse_text_tool_calls
from state_projection_loop.memory import JsonlMemoryStore
from state_projection_loop.messages import Decision, ToolCall
from state_projection_loop.serialization import dumps


class TestArtifactIdIsNotAPath:
    """B1: ``aid`` is model-controlled and reaches the filesystem."""

    def test_an_id_can_never_address_another_run(self, tmp_path):
        victim = ArtifactStore("run_X", directory=tmp_path)
        record = victim.put({"secret": "s3cret"})
        attacker = ArtifactStore("run_OTHER", directory=tmp_path)
        assert attacker.exists(f"../run_X/{record.id}") is False
        assert attacker.resolve_args({"$artifact": f"../run_X/{record.id}"}) == {
            "$artifact": f"../run_X/{record.id}"
        }
        # The run that owns it still resolves it.
        assert victim.exists(record.id) is True

    def test_separators_absolute_paths_and_empty_ids_are_rejected(self, tmp_path):
        store = ArtifactStore("run_X", directory=tmp_path)
        for aid in ("..", "../x", "a/b", "a\\b", "/etc/passwd", "C:/Windows/x", "", "art_x.json"):
            assert store.exists(aid) is False, aid
            assert "Error: unknown artifact" in store.peek(aid)

    def test_a_non_string_id_is_a_miss_not_a_crash(self, tmp_path):
        # Dart's types rule this out; Python's do not, and `exists` is a
        # total predicate in both.
        assert ArtifactStore("run_X", directory=tmp_path).exists(7) is False


class TestDumpsNonFinite:
    """B2: NaN/Infinity broke the two ports in opposite directions."""

    def test_non_finite_floats_become_null(self):
        assert dumps({"s": float("nan")}) == '{"s":null}'
        assert dumps([float("inf"), float("-inf")]) == "[null,null]"
        assert dumps(float("nan")) == "null"

    def test_the_output_is_still_valid_json(self):
        # `json.loads` accepts the NaN token; a strict decoder (Dart's) does
        # not, which is the whole point.
        assert json.loads(dumps({"a": [float("nan")], "b": {"c": float("inf")}})) == {
            "a": [None], "b": {"c": None}
        }


class TestFencedToolCallParsing:
    """B3: a fenced block need not decode to an object."""

    def test_a_non_object_block_is_ignored(self):
        for body in ("[1,2]", "7", '"x"', "null", "true"):
            cleaned, calls = parse_text_tool_calls(f"before\n```tool_call\n{body}\n```\nafter")
            assert calls == []
            assert cleaned == "before\n\nafter"


class TestMalformedSchema:
    """B4: a bad capability spec must degrade, not crash."""

    def test_a_non_list_enum_is_ignored(self):
        assert validate_value({"enum": "abc"}, "a") is None

    def test_a_non_string_type_is_ignored(self):
        assert validate_value({"type": 7}, "x") is None

    def test_a_string_required_names_its_characters(self):
        assert validate_value({"required": "ab"}, {}) == 'arguments: missing required property "a"'


class TestEnumMatching:
    """B5: object and array enum members must compare structurally."""

    def test_structural_members_match(self):
        assert validate_value({"enum": [{"a": 1}]}, {"a": 1}) is None
        assert validate_value({"enum": [[1, 2]]}, [1, 2]) is None
        assert validate_value({"enum": [True]}, 1) is None  # the documented == wart

    def test_a_different_structure_still_fails(self):
        assert validate_value({"enum": [{"a": 1}]}, {"a": 2}) is not None
        assert validate_value({"enum": [[1, 2]]}, [1, 3]) is not None


class TestExistsIsTotal:
    """B6: ``exists`` is a predicate, not a parser."""

    def test_a_corrupt_or_foreign_file_reads_as_absent(self, tmp_path):
        store = ArtifactStore("run_X", directory=tmp_path)
        run_dir = tmp_path / "run_X"
        run_dir.mkdir(parents=True)
        (run_dir / "art_TORN.json").write_text('{"id":"art_TORN","run', encoding="utf-8")
        (run_dir / "art_FOREIGN.json").write_text('{"hello":1}', encoding="utf-8")
        assert store.exists("art_TORN") is False
        assert store.exists("art_FOREIGN") is False
        assert "Error: unknown artifact" in store.peek("art_TORN")


class TestTornJsonlLines:
    """B7: an unbuffered append can be cut in half by a crash."""

    def test_the_ledger_still_replays(self, tmp_path):
        ledger = JsonlLedger(tmp_path)
        ledger.append("run_1", "notice", {"text": "kept"})
        with (tmp_path / "run_1.jsonl").open("a", encoding="utf-8") as f:
            f.write('{"id":"evt_x","run_id":"run')
        assert [e.data["text"] for e in JsonlLedger(tmp_path).iter_run("run_1")] == ["kept"]

    def test_the_memory_store_still_constructs(self, tmp_path):
        path = tmp_path / "memory.jsonl"
        JsonlMemoryStore(path).save("tabs over spaces", ["style"])
        with path.open("a", encoding="utf-8") as f:
            f.write('{"id":"note_x","te')
        assert [n.text for n in JsonlMemoryStore(path).search("tabs", 5)] == ["tabs over spaces"]

    def test_an_unknown_note_field_is_ignored(self, tmp_path):
        path = tmp_path / "memory.jsonl"
        path.write_text('{"id":"n1","text":"hi there","tags":[],"ts":1.0,"future":true}\n',
                        encoding="utf-8")
        assert len(JsonlMemoryStore(path).search("hi", 5)) == 1


class TestExtractFinishCopies:
    """B8: the caller keeps its Decision."""

    def test_the_argument_is_left_alone(self):
        original = Decision(text="bye", calls=[ToolCall(name=FINISH_NAME, arguments={"result": "done"})])
        returned = extract_finish(original)
        assert returned is not original
        assert returned.finish is True and returned.result == "done" and returned.calls == []
        assert original.finish is False and original.result is None and len(original.calls) == 1

    def test_a_decision_without_finish_passes_straight_through(self):
        original = Decision(calls=[ToolCall(name="demo.echo", arguments={})])
        assert extract_finish(original) is original


class TestSpecaFindings:
    """Four defects the speca spec-driven audit surfaced, each reproduced
    before it was fixed. The first and third apply to both ports."""

    def test_a_line_torn_mid_character_does_not_break_resume(self, tmp_path):
        from state_projection_loop.events import JsonlLedger
        ledger = JsonlLedger(str(tmp_path))
        ledger.append("run_1", "user_input", {"text": "hello"})
        ledger.append("run_1", "user_input", {"text": "こんにちは世界"})
        path = tmp_path / "run_1.jsonl"
        # A crash mid-append can truncate inside a multi-byte character, so
        # the bytes do not decode at all — the JSON guard never even sees it.
        path.write_bytes(path.read_bytes()[:-12])
        assert len(list(ledger.iter_run("run_1"))) == 1

    def test_deeply_nested_json_is_not_a_tool_call(self):
        from state_projection_loop.llm import parse_text_tool_calls
        # json.loads raises RecursionError, not JSONDecodeError, and the body
        # is model-controlled: it must not escape the adapter.
        parse_text_tool_calls("```tool_call\n" + "[" * 20000 + "]" * 20000 + "\n```")

    def test_a_number_inside_a_longer_number_does_not_ground_it(self):
        from state_projection_loop.compression import ungrounded
        assert ungrounded("order 942 was cancelled", "we looked at commit 8942") == ["942"]
        assert ungrounded("v1.2 shipped", "we tagged v1.23 last week") == ["v1.2"]
        # Punctuation is still a boundary, so a real path stays grounded.
        assert ungrounded("see src/main.py", "edited a/src/main.py:42 today") == []

    def test_identical_calls_in_one_batch_hit_the_repeat_cap(self):
        from state_projection_loop import Config, Registry, ScriptedLLM, Session
        from state_projection_loop.messages import Decision, ToolCall
        from state_projection_loop.policy import PolicyEngine

        ran: list[str] = []

        def handler(x: str) -> str:
            ran.append(x)
            return "ok"

        registry = Registry()
        registry.register({
            "name": "demo.read.thing", "category": "demo",
            "spec": {"description": "Read a thing.", "parameters": {
                "type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}},
            "effects": [{"kind": "read", "resource": "workspace:*"}],
        }, handler=handler)
        config = Config()

        def run(arguments) -> int:
            ran.clear()
            llm = ScriptedLLM([
                Decision(text="go", calls=[
                    ToolCall(name="demo.read.thing", arguments=arguments(i), id=f"c{i}")
                    for i in range(8)]),
                ScriptedLLM.finish(result="done"),
            ])
            Session(llm, kernel="k", registry=registry, config=config,
                    policy=PolicyEngine(default_decision="allow")).run_job("go")
            return len(ran)

        # Read-only calls are buffered and run concurrently, so the
        # result-keyed loop guard could never see them repeat.
        assert run(lambda i: {"x": "same"}) == config.limits.max_repeats
        assert run(lambda i: {"x": f"v{i}"}) == 8
