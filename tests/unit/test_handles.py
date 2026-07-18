"""Value store & handles (§8.3, I7): refs, previews, peek, argument resolution."""
from __future__ import annotations

from state_projection_loop import ValueStore
from state_projection_loop.tokens import estimate_tokens


class TestPutAndRef:
    def test_ids_are_sequential(self):
        store = ValueStore()
        assert store.put("a").id == "$h1"
        assert store.put("b").id == "$h2"

    def test_ref_text_contains_type_size_preview(self):
        store = ValueStore()
        record = store.put([{"title": f"result {i}"} for i in range(50)], source="web_search")
        ref = store.ref_text(record)
        assert record.id in ref
        assert "list" in ref
        assert "len=50" in ref
        assert "from web_search" in ref
        assert "preview:" in ref
        assert estimate_tokens(ref) < 200

    def test_tail_preview(self):
        store = ValueStore()
        record = store.put("\n".join(f"line {i}" for i in range(200)))
        ref = store.ref_text(record, preview="tail", preview_tokens=30)
        assert "line 199" in ref

    def test_roundtrip(self):
        store = ValueStore()
        value = {"a": [1, 2, {"b": "ok"}]}
        record = store.put(value)
        assert store.get(record.id) is value


class TestPeek:
    def test_line_range(self):
        store = ValueStore()
        hid = store.put("\n".join(f"line {i}" for i in range(1, 21))).id
        out = store.peek(hid, range_="3-5")
        assert out.splitlines() == ["3: line 3", "4: line 4", "5: line 5"]

    def test_key_path(self):
        store = ValueStore()
        hid = store.put({"items": [{"name": "alpha"}, {"name": "beta"}]}).id
        assert store.peek(hid, range_="items[1].name") == "beta"

    def test_query_with_context(self):
        store = ValueStore()
        hid = store.put("\n".join(["aaa", "needle here", "bbb", "ccc"])).id
        out = store.peek(hid, query="needle")
        assert "2: needle here" in out
        assert "1: aaa" in out and "3: bbb" in out

    def test_query_no_match(self):
        store = ValueStore()
        hid = store.put("nothing to see").id
        assert "No lines matching" in store.peek(hid, query="zzz")

    def test_unknown_handle_lists_known(self):
        store = ValueStore()
        store.put("x")
        out = store.peek("$h99")
        assert "unknown handle" in out
        assert "$h1" in out

    def test_truncation_hint(self):
        store = ValueStore()
        hid = store.put("word " * 5000).id
        out = store.peek(hid, max_tokens=100)
        assert "truncated" in out
        assert estimate_tokens(out) <= 140

    def test_bad_key_path(self):
        store = ValueStore()
        hid = store.put({"a": 1}).id
        assert "cannot resolve" in store.peek(hid, range_="b.c")


class TestResolveArgs:
    def test_deep_resolution(self):
        store = ValueStore()
        record = store.put({"data": [1, 2, 3]})
        args = {"input": record.id, "nested": {"also": record.id}, "list": [record.id, "keep"]}
        resolved = store.resolve_args(args)
        assert resolved["input"] == {"data": [1, 2, 3]}
        assert resolved["nested"]["also"] == {"data": [1, 2, 3]}
        assert resolved["list"] == [{"data": [1, 2, 3]}, "keep"]

    def test_unknown_or_embedded_handles_left_alone(self):
        store = ValueStore()
        assert store.resolve_args("$h42") == "$h42"  # unknown -> untouched
        record = store.put("v")
        assert store.resolve_args(f"see {record.id} inside") == f"see {record.id} inside"
