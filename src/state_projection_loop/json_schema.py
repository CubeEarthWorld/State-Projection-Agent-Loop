"""JSON Schema validation: the one validator of this package.

Deliberately one small validator rather than ``jsonschema``: the error text
goes to the model as a self-repair prompt, and two different validators
meant this package and its Dart port rejected different arguments with
different wording for the same schema. The subset covers what a
tool-argument schema actually uses.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .serialization import dumps

#: One name -> predicate table serving both directions: "does this value
#: have this JSON type" and "what is this value's JSON type name". Order
#: matters for the second: the first match wins, so ``True`` names itself
#: "boolean" and ``1`` "integer".
_TYPES: dict[str, Callable[[Any], bool]] = {
    "null": lambda v: v is None,
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def _json_type_name(value: Any) -> str:
    """Name a value's type in the JSON Schema vocabulary.

    The message this feeds is a self-repair prompt sent to the model, so it
    names types the way the schema beside it does — and identically in the
    Dart port, which has no Python type names to fall back on.
    """
    return next((name for name, ok in _TYPES.items() if ok(value)), type(value).__name__)


def _type_ok(expected: Any, value: Any) -> bool:
    ok = _TYPES.get(expected) if isinstance(expected, str) else None
    return ok is None or ok(value)


def validate_value(schema: dict[str, Any], value: Any, path: str = "") -> Optional[str]:
    """Validate any JSON value against the JSON Schema subset a tool-argument
    schema actually uses; error message or None."""
    where = path or "arguments"
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_type_ok(x, value) for x in types):
            return f"{where}: expected type {dumps(t)}, got {_json_type_name(value)}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{where}: {dumps(value)} is not one of {dumps(schema['enum'])}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{where}: {value} is less than minimum {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{where}: {value} is greater than maximum {schema['maximum']}"
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            return f"{where}: shorter than minLength {schema['minLength']}"
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return f"{where}: longer than maxLength {schema['maxLength']}"
    if isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                return f"{where}: missing required property {dumps(req)}"
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value and isinstance(sub, dict):
                err = validate_value(sub, value[key], f"{where}.{key}")
                if err:
                    return err
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            if extra:
                return f"{where}: unexpected properties {dumps(sorted(extra))}"
    if isinstance(value, list):
        # Length before contents: the arguments are model-controlled, and
        # walking a million items to then reject the array on a handler's
        # own cap blocks the loop every other session shares.
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return f"{where}: more than maxItems {schema['maxItems']} entries"
        if "minItems" in schema and len(value) < schema["minItems"]:
            return f"{where}: fewer than minItems {schema['minItems']} entries"
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(value):
                err = validate_value(schema["items"], item, f"{where}[{i}]")
                if err:
                    return err
    if "anyOf" in schema:
        errs = []
        for sub in schema["anyOf"]:
            err = validate_value(sub, value, where)
            if err is None:
                break
            errs.append(err)
        else:
            return f"{where}: no anyOf branch matched ({'; '.join(errs)})"
    return None


def apply_defaults(schema: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Fill missing top-level arguments that declare a schema default."""
    out = dict(args)
    for key, sub in (schema.get("properties") or {}).items():
        if key not in out and isinstance(sub, dict) and "default" in sub:
            out[key] = sub["default"]
    return out


def validate_args(schema: dict[str, Any], args: Any) -> Optional[str]:
    """Return an error message, or None when the arguments pass."""
    if not isinstance(args, dict):
        return f"arguments must be a JSON object, got {_json_type_name(args)}"
    return validate_value(schema, args)
