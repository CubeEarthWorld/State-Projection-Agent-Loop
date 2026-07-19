"""Artifact store: structured references never confuse literal strings
(P0-6), previews, peek, run namespacing, move()."""
from __future__ import annotations

from state_projection_loop.artifacts import ArtifactStore, is_ref, ref


class TestReferenceForm:
    def test_is_ref(self):
        assert is_ref({"$artifact": "art_x"}) is True
        assert is_ref("$h1") is False
        assert is_ref({"$artifact": "art_x", "extra": 1}) is False
        assert is_ref({"other": "art_x"}) is False

    def test_bare_string_never_resolved(self):
        store = ArtifactStore("run_1")
        record = store.put("hello world")
        # A literal string that happens to equal the artifact id must NOT
        # be resolved — only the structured {"$artifact": ...} form is.
        resolved = store.resolve_args({"text": record.id})
        assert resolved == {"text": record.id}

    def test_structured_ref_is_resolved(self):
        store = ArtifactStore("run_1")
        record = store.put({"a": 1})
        resolved = store.resolve_args({"payload": ref(record.id)})
        assert resolved == {"payload": {"a": 1}}

    def test_nested_resolution(self):
        store = ArtifactStore("run_1")
        record = store.put([1, 2, 3])
        resolved = store.resolve_args({"outer": {"inner": [ref(record.id), "literal"]}})
        assert resolved == {"outer": {"inner": [[1, 2, 3], "literal"]}}

    def test_unknown_ref_passed_through(self):
        store = ArtifactStore("run_1")
        resolved = store.resolve_args({"x": ref("art_missing")})
        assert resolved == {"x": ref("art_missing")}


class TestPreviewAndPeek:
    def test_ref_text_contains_id_type_and_preview(self):
        store = ArtifactStore("run_1")
        record = store.put("x" * 5000, source="demo.tool")
        text = store.ref_text(record, preview_tokens=10)
        assert record.id in text
        assert "str" in text
        assert "demo.tool" in text

    def test_peek_query_returns_matching_lines_with_context(self):
        store = ArtifactStore("run_1")
        record = store.put("a\nb\nneedle\nc\nd")
        out = store.peek(record.id, query="needle")
        assert "needle" in out

    def test_peek_range_by_line(self):
        store = ArtifactStore("run_1")
        record = store.put("l1\nl2\nl3\nl4")
        out = store.peek(record.id, range_="2-3")
        assert "l2" in out and "l3" in out and "l1" not in out

    def test_peek_unknown_id(self):
        store = ArtifactStore("run_1")
        out = store.peek("art_missing")
        assert "unknown artifact" in out


class TestNamespacing:
    def test_move_creates_new_id_in_target_store(self):
        parent = ArtifactStore("run_parent")
        child = ArtifactStore("run_child")
        child_record = child.put({"result": 42}, source="child.tool")
        moved = parent.move(child_record)
        assert moved.id != child_record.id
        assert parent.get(moved.id) == {"result": 42}
        assert not parent.exists(child_record.id)
