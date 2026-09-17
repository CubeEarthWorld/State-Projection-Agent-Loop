"""Regenerate spec/fixtures/*.json from the Python implementation.

The fixtures are the cross-language contract: both packages' test suites
read the same files and must produce the same outputs. Python is the
reference implementation, so the expectations are generated from it — run
this after deliberately changing one of the covered functions, and never to
paper over an unexplained diff.

    python spec/generate_fixtures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from state_projection_loop.capability import synthesize_signature, to_api_name  # noqa: E402
from state_projection_loop.compression import (  # noqa: E402
    content_hash,
    head_tail_truncate,
    strip_noise,
    summarize_text,
)
from state_projection_loop.policy import glob_match  # noqa: E402
from state_projection_loop.runtime import apply_defaults, validate_args  # noqa: E402
from state_projection_loop.serialization import dumps  # noqa: E402
from state_projection_loop.tokens import estimate_tokens  # noqa: E402

TEXTS = [
    "",
    "hello",
    "L0\nL1\n",
    "日本語🎌テスト",
    "a" * 300,
    "\n".join(f"line {i}" for i in range(30)),
    "ERROR: boom\r\n  at frame 1\r\n  at frame 2\r\n",
    "diff --git a/x b/x\nindex 1234567..89abcde 100644\n--- a/x\n+++ b/x\nreal line\n",
    "\x1b[31mred\x1b[0m plain",
]

GLOBS = [
    ("bcd", "[^a]*"), ("^bc", "[^a]*"), ("abc", "[!a]*"), ("bbc", "[!a]*"),
    ("a\nb", "a*b"), ("a.c", "a?c"), ("x[y", "x[[]y"), ("a]b", "a[]]b"),
    ("web.search.query", "web.*"), ("web.search.query", "web.search.query"),
    ("state.goal.set", "state.*"), ("planning.checklist.manage", "state.*"),
]

SIGNATURES = [
    ("demo.tool.run", {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "default": 8},
            "mode": {"enum": ["a", "b"]},
            "cat": {"type": ["string", "null"]},
            "flag": {"type": "boolean", "default": True},
            "anything": {},
        },
        "required": ["query"],
    }),
    ("meta.tool.find", {"type": "object", "properties": {}}),
]

# Validation messages are a self-repair prompt sent to the model, so the
# wording is part of the contract, not an implementation detail.
VALIDATION = [
    ({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}, {}),
    ({"type": "object", "properties": {"a": {"type": "string"}}}, {"a": 42}),
    ({"type": "object", "properties": {"a": {"type": "string"}}}, {"a": None}),
    ({"type": "object", "properties": {"a": {"type": ["string", "null"]}}}, {"a": 1.5}),
    ({"type": "object", "properties": {"a": {"enum": ["x", "y"]}}}, {"a": "z"}),
    ({"type": "object", "properties": {"a": {"type": "integer", "minimum": 1}}}, {"a": 0}),
    ({"type": "object", "properties": {"a": {"type": "integer", "maximum": 9}}}, {"a": 10}),
    ({"type": "object", "properties": {"a": {"type": "string", "maxLength": 2}}}, {"a": "abc"}),
    ({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": False},
     {"a": "ok", "b": 1, "c": 2}),
    ({"type": "object", "properties": {"a": {"type": "array", "items": {"type": "integer"}}}},
     {"a": [1, "two"]}),
    ({"type": "object", "properties": {"a": {"type": "string", "default": "d"}}}, {}),
]

def projection_scenario() -> dict:
    """One whole turn, as the model receives it: the pin that a refactor of
    either port changed no model-visible text. Ids are given, so the output
    is deterministic."""
    from state_projection_loop import Config, Registry, ScriptedLLM, Session
    from state_projection_loop.messages import Decision, ToolCall
    from state_projection_loop.policy import PolicyEngine

    def cap(name, **kw):
        return {"name": name, "category": kw.pop("category", "demo"),
                "spec": {"description": kw.pop("description"), "parameters": kw.pop("parameters")},
                "discovery": kw, "effects": [{"kind": "read", "resource": "workspace:*"}]}

    warehouse = {"type": "object", "properties": {"warehouse": {"type": "string"}}, "required": ["warehouse"]}
    registry = Registry()
    registry.register(cap("demo.echo.say", description="Echo the text back. Useful for tests.",
                          parameters={"type": "object", "properties": {"text": {"type": "string", "default": "hi"}}},
                          pinned=True, kernel_note="Use demo.echo.say to repeat text."),
                      handler=lambda text="hi": f"echo: {text}")
    registry.register(cap("inventory.stock.get", category="inventory", description="在庫数を返す。Returns the stock count.",
                          parameters=warehouse, embedding_text="在庫 stock warehouse inventory"),
                      handler=lambda warehouse: {"warehouse": warehouse, "stock": 42})
    registry.register(cap("inventory.stock.audit", category="inventory", description="Audit the stock of a warehouse.",
                          parameters=warehouse, require_spec=True),
                      handler=lambda warehouse: "audited")
    llm = ScriptedLLM([
        Decision(text="checking", calls=[ToolCall(name="inventory.stock.get", arguments={"warehouse": "tokyo"}, id="c1"),
                                         ToolCall(name="inventory.stock.audit", arguments={"warehouse": 7}, id="c2")]),
        ScriptedLLM.finish(result="42"),
    ])
    session = Session(llm, kernel="You are a stock agent.", registry=registry,
                      config=Config.from_dict({"mode": "job"}), policy=PolicyEngine(default_decision="allow"),
                      seed={"goal": "report tokyo stock", "confirmed_facts": ["tokyo is a warehouse"],
                            "decisions": [{"text": "use inventory tools", "reason": "they are authoritative"}],
                            "flags": {"urgent": True}})
    session.run_job("How much stock does the tokyo warehouse have?")
    request = llm.requests[-1]
    return {
        "messages": [
            {"role": m.role, "content": m.content, "tool_call_id": m.tool_call_id, "name": m.name,
             "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in m.tool_calls]}
            for m in request["messages"]
        ],
        "tools": request["tools"],
    }


JSON_VALUES = [
    {"a": 1, "b": [1, 2, {"c": None}]},
    {"日本語": "🎌", "n": 1.5, "t": True},
    [],
    {},
]


def main() -> None:
    out = ROOT / "spec" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)

    (out / "compression.json").write_text(json.dumps({
        "content_hash": [{"text": t, "expected": content_hash(t)} for t in TEXTS],
        "strip_noise": [{"text": t, "expected": strip_noise(t)} for t in TEXTS],
        "summarize_text": [{"text": t, "expected": summarize_text(t)} for t in TEXTS],
        "head_tail_truncate": [
            {"text": t, "max_lines": n, "expected": head_tail_truncate(t, n)}
            for t in TEXTS for n in (4, 7, 10, 40)
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (out / "policy_glob.json").write_text(json.dumps({
        "glob_match": [
            {"value": v, "pattern": p, "expected": glob_match(v, p)}
            for v, p in GLOBS
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (out / "capability.json").write_text(json.dumps({
        "synthesize_signature": [
            {"name": n, "parameters": p, "expected": synthesize_signature(n, p)}
            for n, p in SIGNATURES
        ],
        "api_name": [
            {"name": n, "expected": to_api_name(n)}
            for n in ("meta.tool.find", "planning.checklist.manage", "a.b.c.d.e")
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (out / "validation.json").write_text(json.dumps({
        "validate_args": [
            {"schema": s, "arguments": a, "expected": validate_args(s, a)}
            for s, a in VALIDATION
        ],
        "apply_defaults": [
            {"schema": s, "arguments": a, "expected": apply_defaults(s, a)}
            for s, a in VALIDATION
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (out / "serialization.json").write_text(json.dumps({
        "dumps": [{"value": v, "expected": dumps(v)} for v in JSON_VALUES],
        "estimate_tokens": [{"value": t, "expected": estimate_tokens(t)} for t in TEXTS],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (out / "projection.json").write_text(
        json.dumps(projection_scenario(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(list(out.glob('*.json')))} fixture files to {out}")


if __name__ == "__main__":
    main()
