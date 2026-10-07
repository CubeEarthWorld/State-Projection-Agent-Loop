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
        with TestCase().assertRaisesRegex(RuntimeError, "matched 2"):
            session.invoke("edit", path="src/demo.txt", old_text="alpha", new_text="gamma")
        assert session.invoke("read", path="src/demo.txt") == before
        session.invoke("edit", path="src/demo.txt", old_text="alpha", new_text="gamma", replace_all=True)
        assert session.invoke("read", path="src/demo.txt") == "gamma\ngamma\nbeta\n"
        with TestCase().assertRaisesRegex(RuntimeError, "escapes"):
            session.invoke("read", path="../outside.txt")
        assert session.invoke("bash", command="printf agent-tool-check").endswith("agent-tool-check")
        assert session.invoke("bash", command="printf '\\343\\201\\202'").endswith("あ")


if __name__ == "__main__":
    test_pinned_tools_and_file_operations()
    print("Pinned agent tool checks passed")
