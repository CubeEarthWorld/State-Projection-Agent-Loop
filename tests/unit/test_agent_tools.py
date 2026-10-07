"""Checks for the pinned agent tools; no models or external APIs."""
import tempfile
from pathlib import Path
from unittest import TestCase

from state_projection_loop import PolicyEngine, ScriptedLLM, Session


def test_pinned_tools_and_file_operations():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        llm = ScriptedLLM(["ok"])
        session = Session(llm, workspace_root=root, policy=PolicyEngine(default_decision="allow"))
        session.send("hello")
        resident = {tool["name"] for tool in llm.requests[0]["tools"]}
        assert {"read", "bash", "edit", "write", "grep", "find", "ls"} <= resident
        assert not any("meta" in name or "." in name for name in resident)
        session.invoke("write", path="src/demo.txt", content="alpha\nalpha\nbeta\n")
        assert session.invoke("ls") == ["src/"]
        assert session.invoke("find", pattern="*.txt") == ["src/demo.txt"]
        assert session.invoke("grep", pattern="alpha", path="src")[0] == {
            "path": "src/demo.txt", "line": 1, "text": "alpha"}
        before = session.invoke("read", path="src/demo.txt")
        with TestCase().assertRaisesRegex(RuntimeError, "matched 2") as error:
            session.invoke("edit", path="src/demo.txt", old_text="alpha", new_text="gamma")
        assert "1: alpha" in str(error.exception) and "2: alpha" in str(error.exception)
        assert "Add surrounding text" in str(error.exception)
        assert session.invoke("read", path="src/demo.txt") == before
        session.invoke("edit", path="src/demo.txt", old_text="alpha", new_text="gamma", replace_all=True)
        assert session.invoke("read", path="src/demo.txt") == "gamma\ngamma\nbeta\n"
        with TestCase().assertRaisesRegex(RuntimeError, "escapes"):
            session.invoke("read", path="../outside.txt")
        assert session.invoke("bash", command="printf agent-tool-check").endswith("agent-tool-check")
        assert session.invoke("bash", command="printf '\\343\\201\\202'").endswith("あ")


def test_read_ranges_and_edit_preserve_unread_file_content():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        content = "".join(f"line {i}\n" for i in range(1, 2101))
        (root / "large.txt").write_text(content, encoding="utf-8")
        session = Session(ScriptedLLM([]), workspace_root=root, policy=PolicyEngine(default_decision="allow"))
        first = session.invoke("read", path="large.txt")
        assert "line 2000\n" in first and "offset=2001" in first
        assert "art_" not in first and "line 2100" not in first
        assert session.invoke("read", path="large.txt", offset=2099, limit=2) == "line 2099\nline 2100\n"
        with TestCase().assertRaisesRegex(RuntimeError, "minimum"):
            session.invoke("read", path="large.txt", offset=0)
        # A failed ambiguous edit stays intact; a unique retry succeeds.
        (root / "duplicate.txt").write_text("first\nvalue\nsecond\nvalue\n", encoding="utf-8")
        with TestCase().assertRaisesRegex(RuntimeError, "matched 2"):
            session.invoke("edit", path="duplicate.txt", old_text="value", new_text="changed")
        session.invoke("edit", path="duplicate.txt", old_text="second\nvalue", new_text="second\nchanged")
        assert (root / "duplicate.txt").read_text() == "first\nvalue\nsecond\nchanged\n"
        session.invoke("edit", path="large.txt", old_text="line 2100\n", new_text="last line\n")
        assert (root / "large.txt").read_text() == content.replace("line 2100\n", "last line\n")


if __name__ == "__main__":
    test_pinned_tools_and_file_operations()
    test_read_ranges_and_edit_preserve_unread_file_content()
    print("Pinned agent tool checks passed")
