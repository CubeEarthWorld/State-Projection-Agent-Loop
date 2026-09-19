"""The package's one JSON Schema validator."""
from __future__ import annotations

import pytest

from state_projection_loop.json_schema import apply_defaults, validate_args, validate_value


class TestValidator:
    """Hand-written cases: an oracle independent of the generated fixtures."""

    SCHEMA = {
        "type": "object",
        "properties": {
            "q": {"type": "string", "minLength": 2},
            "n": {"type": "integer", "minimum": 1, "maximum": 10},
            "mode": {"enum": ["a", "b"]},
            "items": {"type": "array", "items": {"type": "string"}},
            "opt": {"type": ["string", "null"]},
        },
        "required": ["q"],
        "additionalProperties": False,
    }

    def test_accepts_valid(self):
        assert validate_value(self.SCHEMA, {"q": "ok", "n": 5, "mode": "a",
                                            "items": ["x"], "opt": None}) is None

    @pytest.mark.parametrize("args,fragment", [
        ({}, "required"),
        ({"q": "ok", "n": "5"}, "expected type"),
        ({"q": "ok", "n": 0}, "minimum"),
        ({"q": "ok", "n": 11}, "maximum"),
        ({"q": "x"}, "minLength"),
        ({"q": "ok", "mode": "c"}, "not one of"),
        ({"q": "ok", "items": ["x", 1]}, "expected type"),
        ({"q": "ok", "zzz": 1}, "unexpected properties"),
        ({"q": "ok", "n": True}, "expected type"),
    ])
    def test_rejects_invalid(self, args, fragment):
        assert fragment in validate_value(self.SCHEMA, args)

    def test_validate_args_agrees(self):
        assert validate_args(self.SCHEMA, {"q": "ok"}) is None
        assert validate_args(self.SCHEMA, {"q": 1}) is not None
        assert validate_args(self.SCHEMA, "not a dict") is not None

    def test_apply_defaults(self):
        schema = {"type": "object", "properties": {"k": {"type": "integer", "default": 7}}}
        assert apply_defaults(schema, {}) == {"k": 7}
        assert apply_defaults(schema, {"k": 1}) == {"k": 1}
