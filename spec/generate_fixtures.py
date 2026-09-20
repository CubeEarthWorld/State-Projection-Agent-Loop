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
from state_projection_loop.json_schema import apply_defaults, validate_args  # noqa: E402
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
    # Every cross-port divergence found so far lived in a gap in this list:
    # the two ports agreed on all nine cases above and disagreed the moment
    # a line ended in anything but "\n". Keep the awkward inputs here.
    "diff --git a/f b/f\r\nindex 111..222 100644\r\nrest\r\n",  # CRLF throughout
    "index 1234567..89abcde 100644\rtail",                      # lone \r
    "a\vb\vc\vd\ve",                                            # \v is a Python line break
    "h1\x0cf2\x1c3\x1d4\x1e5\x856 7 8",               # the rest of them
    "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 astral 🎌🇯🇵 surrogate pairs",                       # non-BMP, UTF-16 vs code points
    "HTTP status: 200 OK\n" + "\n".join(f"body {i}" for i in range(50)),
    "exit code 0\n" + "\n".join(f"body {i}" for i in range(50)),
    "Meeting at 14:30 with Bob\n" + "\n".join(f"note {i}" for i in range(50)),
    "at foo.js:12\n" + "\n".join(f"frame {i}" for i in range(50)),
    "\n".join("x" for _ in range(41)),  # truncating this one would lengthen it
]

GLOBS = [
    ("bcd", "[^a]*"), ("^bc", "[^a]*"), ("abc", "[!a]*"), ("bbc", "[!a]*"),
    ("a\nb", "a*b"), ("a.c", "a?c"), ("x[y", "x[[]y"), ("a]b", "a[]]b"),
    ("web.search.query", "web.*"), ("web.search.query", "web.search.query"),
    ("state.goal.set", "state.*"), ("planning.checklist.manage", "state.*"),
    # `?` counts one code point, not one UTF-16 unit, and a reversed range
    # matches nothing rather than raising — Dart got both wrong.
    ("🎌", "?"), ("🎌", "*"), ("a🎌b", "a?b"),
    ("ab", "[!z-a]"), ("ab", "[z-a]"), ("a", "[b-a]"), ("-", "[a-]"),
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
    # An enum member that is an object or an array has to compare by value.
    # Dart's `List.contains` compares collections by identity, so it rejected
    # everything here until the validator learned to compare structurally.
    ({"type": "object", "properties": {"a": {"enum": [{"k": 1}, {"k": 2}]}}}, {"a": {"k": 1}}),
    ({"type": "object", "properties": {"a": {"enum": [{"k": 1}]}}}, {"a": {"k": 9}}),
    ({"type": "object", "properties": {"a": {"enum": [[1, 2], [3]]}}}, {"a": [1, 2]}),
    ({"type": "object", "properties": {"a": {"enum": [[1, 2]]}}}, {"a": [2, 1]}),
    ({"type": "object", "properties": {"a": {"enum": [None, "x"]}}}, {"a": None}),
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
    # Float rendering is part of the contract: every tool schema, artifact
    # body and ledger row carries numbers through `dumps`. Python's repr
    # goes scientific outside 1e-4 .. 1e16 and pads the exponent to two
    # digits; Dart's `toString` does neither, so `1e-7` and `1e20` came out
    # differently in the two ports until Dart got a matching formatter.
    {"tiny": 1e-7, "small": 1e-6, "edge_lo": 1e-5, "fixed_lo": 1e-4},
    {"big": 1e20, "edge_hi": 1e16, "fixed_hi": 1e15, "huge": 1e100},
    {"neg": -1.5e20, "denormal": 5e-324, "max": 1.7976931348623157e308},
    {"whole": 2.0, "third": 0.3333333333333333, "zero": 0.0, "negzero": -0.0},
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
