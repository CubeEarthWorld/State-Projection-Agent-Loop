"""Exercise the coding CLI without provider packages, API keys, or network."""
import json
from pathlib import Path
import sys

from state_projection_loop import ScriptedLLM


def test_cli_retains_source_and_finishes_without_extra_turns(tmp_path, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    import agent_cli

    source = "".join(f"source line {i}\n" for i in range(1500))
    (tmp_path / "source.txt").write_text(source, encoding="utf-8")

    def finish(messages, tools):
        assert any(m.role == "tool" and source in m.text() for m in messages)
        return ScriptedLLM.finish("verified")

    class FakeAdapter(ScriptedLLM):
        def __init__(self, *args, **kwargs):
            super().__init__([ScriptedLLM.call("read", path="source.txt")]
                             + [ScriptedLLM.call("ls") for _ in range(14)] + [finish])

    monkeypatch.setattr(agent_cli, "MeasuredAdapter", FakeAdapter)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")
    monkeypatch.setattr(sys, "argv", ["agent_cli", "verify source", "--cwd", str(tmp_path), "--json"])
    assert agent_cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "COMPLETED" and result["result"] == "verified"
    assert result["model_calls"] == 16
