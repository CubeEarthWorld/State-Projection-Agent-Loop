"""Capability contracts: naming, retry-safety/retries coupling, effects,
concurrency policy, decorator-based construction."""
from __future__ import annotations

import pytest

from state_projection_loop.capability import (
    Capability,
    ConcurrencyPolicy,
    Effect,
    build_capability_from_function,
    capability,
    validate_capability_name,
)


class TestNaming:
    def test_valid_names(self):
        for name in ("a.b", "a.b.c", "a.b.c.d", "a.b.c.d.e", "filesystem.file.read"):
            validate_capability_name(name)  # no raise

    def test_invalid_names(self):
        for name in ("singleword", "A.b", "a..b", "a.b.c.d.e.f", "a.B"):
            with pytest.raises(ValueError):
                validate_capability_name(name)

    def test_qualified_name(self):
        cap = Capability(name="demo.thing", version=3)
        assert cap.qualified_name == "demo.thing@3"


class TestRetrySafetyGate:
    def test_retries_requires_safe_retry_safety(self):
        from state_projection_loop.capability import CapabilityExecution

        with pytest.raises(ValueError, match="unsafe"):
            CapabilityExecution(retries=2, retry_safety="never_retry")
        with pytest.raises(ValueError, match="unsafe"):
            CapabilityExecution(retries=1, retry_safety="check_then_retry")

    def test_retries_allowed_for_pure_and_idempotent(self):
        from state_projection_loop.capability import CapabilityExecution

        CapabilityExecution(retries=2, retry_safety="pure")
        CapabilityExecution(retries=2, retry_safety="idempotent")


class TestConcurrencyPolicy:
    def test_exclusive_resource_requires_key(self):
        with pytest.raises(ValueError, match="resource_key"):
            ConcurrencyPolicy(mode="exclusive_resource")
        ConcurrencyPolicy(mode="exclusive_resource", resource_key="db:accounts")

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            ConcurrencyPolicy(mode="whenever")


class TestIsPure:
    def test_no_effects_declared_is_not_pure(self):
        cap = Capability(name="demo.thing")
        assert cap.is_pure is False

    def test_none_effect_is_pure(self):
        cap = Capability(name="demo.thing", effects=[Effect(kind="none")])
        assert cap.is_pure is True

    def test_write_effect_is_not_pure(self):
        cap = Capability(name="demo.thing", effects=[Effect(kind="write", resource="workspace:*")])
        assert cap.is_pure is False


class TestDecorator:
    def test_bare_decorator_infers_name_from_function(self):
        @capability
        def demo_thing(x: int) -> int:
            """Do a thing.

            Args:
                x: the input
            """
            return x

        cap = demo_thing.__spal_capability__
        assert cap.name == "demo.thing"
        assert cap.spec.parameters["properties"]["x"]["description"] == "the input"
        assert cap.spec.parameters["required"] == ["x"]

    def test_decorator_with_options(self):
        @capability(name="demo.write", retry_safety="idempotent", effects=[("write", "workspace:*")],
                    timeout_s=5.0)
        def demo_write(path: str) -> str:
            return path

        cap = demo_write.__spal_capability__
        assert cap.execution.retry_safety == "idempotent"
        assert cap.effects == [Effect(kind="write", resource="workspace:*")]
        assert cap.execution.timeout_s == 5.0

    def test_optional_param_gets_default_and_ctx_excluded(self):
        from state_projection_loop.capability import ToolContext

        def handler(ctx: ToolContext, path: str, verbose: bool = False) -> str:
            return path

        cap = build_capability_from_function(handler, name="demo.op")
        assert "ctx" not in cap.spec.parameters["properties"]
        assert cap.spec.parameters["properties"]["verbose"]["default"] is False
        assert cap.wants_ctx is True


class TestProjections:
    def test_card_and_spec_text(self):
        cap = build_capability_from_function(
            lambda path: path, name="demo.read",
            summary="reads a thing",
        )
        assert "demo.read" in cap.card_text()
        assert "demo.read@1" in cap.spec_text()

    def test_api_schema_shape(self):
        cap = build_capability_from_function(lambda x: x, name="demo.op")
        schema = cap.api_schema()
        assert schema["type"] == "function"
        # dots are encoded ("__") for provider-safe function names — most
        # native-function-calling providers (OpenAI included) reject "."
        assert schema["function"]["name"] == "demo__op" == cap.api_name
        assert "parameters" in schema["function"]

    def test_api_name_round_trips_through_registry(self):
        from state_projection_loop.capability import from_api_name, to_api_name

        assert to_api_name("demo.op") == "demo__op"
        assert from_api_name("demo__op") == "demo.op"
