"""The cross-language contract.

These fixtures are read by both packages' test suites. A change here that
is not mirrored in the other language is exactly the kind of silent drift
that produced two different content hashes, two different glob dialects and
two different truncation rules — each of which only showed up in
production. Regenerate with `python spec/generate_fixtures.py`, never by
editing a JSON file to match new behaviour.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from state_projection_loop.capability import synthesize_signature, to_api_name
from state_projection_loop.compression import (
    content_hash,
    head_tail_truncate,
    strip_noise,
    summarize_text,
)
from state_projection_loop.policy import glob_match
from state_projection_loop.serialization import dumps
from state_projection_loop.tokens import estimate_tokens

FIXTURES = Path(__file__).resolve().parents[2] / "spec" / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def cases(name: str, key: str) -> list:
    return load(name)[key]


class TestCompression:
    @pytest.mark.parametrize("case", cases("compression", "content_hash"))
    def test_content_hash(self, case):
        assert content_hash(case["text"]) == case["expected"]

    @pytest.mark.parametrize("case", cases("compression", "strip_noise"))
    def test_strip_noise(self, case):
        assert strip_noise(case["text"]) == case["expected"]

    @pytest.mark.parametrize("case", cases("compression", "summarize_text"))
    def test_summarize_text(self, case):
        assert summarize_text(case["text"]) == case["expected"]

    @pytest.mark.parametrize("case", cases("compression", "head_tail_truncate"))
    def test_head_tail_truncate(self, case):
        assert head_tail_truncate(case["text"], case["max_lines"]) == case["expected"]


class TestPolicyGlob:
    @pytest.mark.parametrize("case", cases("policy_glob", "glob_match"))
    def test_glob_match(self, case):
        assert glob_match(case["value"], case["pattern"]) is case["expected"]


class TestCapability:
    @pytest.mark.parametrize("case", cases("capability", "synthesize_signature"))
    def test_synthesize_signature(self, case):
        assert synthesize_signature(case["name"], case["parameters"]) == case["expected"]

    @pytest.mark.parametrize("case", cases("capability", "api_name"))
    def test_api_name(self, case):
        assert to_api_name(case["name"]) == case["expected"]


class TestSerialization:
    @pytest.mark.parametrize("case", cases("serialization", "dumps"))
    def test_dumps(self, case):
        assert dumps(case["value"]) == case["expected"]

    @pytest.mark.parametrize("case", cases("serialization", "estimate_tokens"))
    def test_estimate_tokens(self, case):
        assert estimate_tokens(case["value"]) == case["expected"]


class TestProjection:
    """One whole turn as the model receives it, shared with the Dart port:
    no refactor of either package may change a byte of it."""

    def test_projection_matches_the_golden_turn(self):
        import sys

        sys.path.insert(0, str(FIXTURES.parent))
        from generate_fixtures import projection_scenario

        assert projection_scenario() == load("projection")


class TestBundledDefinitions:
    """The bundled definitions are data, shipped as package data and shared
    with the Dart port. A missing or malformed file is a packaging bug that
    only shows up when a session is constructed."""

    @pytest.mark.parametrize("name", ["meta", "spawn", "state", "checklist"])
    def test_loads(self, name):
        from state_projection_loop.builtin.defs import load

        assert load(name)

    def test_every_bundled_capability_has_a_handler(self):
        from state_projection_loop import Registry
        from state_projection_loop import install_builtins

        registry = Registry()
        install_builtins(registry, ["meta", "checklist", "spawn", "state"])

        assert list(registry)
        for capability in registry:
            assert capability.execution.handler is not None, (
                f"{capability.name} would fail at call time with no_handler"
            )


class TestValidation:
    """Validation messages are a self-repair prompt sent to the model, so
    the wording is part of the contract, not an implementation detail."""

    @pytest.mark.parametrize("case", cases("validation", "validate_args"))
    def test_validate_args(self, case):
        from state_projection_loop.runtime import validate_args

        assert validate_args(case["schema"], case["arguments"]) == case["expected"]

    @pytest.mark.parametrize("case", cases("validation", "apply_defaults"))
    def test_apply_defaults(self, case):
        from state_projection_loop.runtime import apply_defaults

        assert apply_defaults(case["schema"], case["arguments"]) == case["expected"]
