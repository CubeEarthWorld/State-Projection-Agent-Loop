"""Deterministic runtime (spec §6, §8, invariant I5).

The LLM only decides *what* to do; validation, retries, timeouts,
parallelism, output policies and budgets are enforced here in code.

JSON Schema validation uses ``jsonschema`` when installed and falls back to
a built-in mini validator otherwise (keeps the core pure-Python for
embedded environments such as Ren'Py).
"""
from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import Config
from .handles import ValueStore, serialize_value, truncate_to_tokens
from .hooks import Hooks
from .logger import SessionLogger
from .messages import ToolCall
from .projection import TurnContext
from .registry import Registry
from .tokens import estimate_tokens
from .tooldef import ToolContext, ToolDef

try:
    import jsonschema as _jsonschema
except ImportError:  # pragma: no cover - exercised via _mini_validate tests
    _jsonschema = None


# ---------------------------------------------------------------------------
# Validation (§6)
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _type_ok(expected: str, value: Any) -> bool:
    py = _TYPE_MAP.get(expected)
    if py is None:
        return True
    if expected in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, py)


def _mini_validate(schema: dict[str, Any], value: Any, path: str = "") -> Optional[str]:
    """Minimal JSON Schema subset validator (fallback when jsonschema is absent)."""
    where = path or "arguments"
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_type_ok(x, value) for x in types):
            return f"{where}: expected type {t}, got {type(value).__name__}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{where}: {value!r} is not one of {schema['enum']}"
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
                return f"{where}: missing required property {req!r}"
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value and isinstance(sub, dict):
                err = _mini_validate(sub, value[key], f"{where}.{key}")
                if err:
                    return err
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            if extra:
                return f"{where}: unexpected properties {sorted(extra)}"
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            err = _mini_validate(schema["items"], item, f"{where}[{i}]")
            if err:
                return err
    if "anyOf" in schema:
        errs = []
        for sub in schema["anyOf"]:
            err = _mini_validate(sub, value, where)
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
    """Return an error message, or None when the arguments pass (§6 step 2)."""
    if not isinstance(args, dict):
        return f"arguments must be a JSON object, got {type(args).__name__}"
    if _jsonschema is not None:
        try:
            _jsonschema.validate(instance=args, schema=schema)
            return None
        except _jsonschema.ValidationError as exc:
            loc = ".".join(str(p) for p in exc.absolute_path) or "arguments"
            return f"{loc}: {exc.message}"
        except _jsonschema.SchemaError as exc:
            return f"tool schema itself is invalid: {exc.message}"
    return _mini_validate(schema, args)


# ---------------------------------------------------------------------------
# Results & budget
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    call: ToolCall
    ok: bool
    value: Any = None
    error: Optional[str] = None
    observation: str = ""
    handle: Optional[str] = None
    elapsed_s: float = 0.0
    final: bool = False  # set by done()


@dataclass
class BudgetState:
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    started: float = field(default_factory=time.time)

    def note_usage(self, prompt: int, completion: int, cfg: Config) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        b = cfg.budget
        self.cost += prompt / 1000 * b.cost_per_1k_input + completion / 1000 * b.cost_per_1k_output

    def exceeded(self, cfg: Config) -> Optional[str]:
        b = cfg.budget
        if b.max_steps is not None and self.steps >= b.max_steps:
            return f"max_steps ({b.max_steps}) reached"
        total = self.prompt_tokens + self.completion_tokens
        if b.max_tokens is not None and total >= b.max_tokens:
            return f"max_tokens ({b.max_tokens}) reached (used ~{total})"
        if b.max_cost is not None and self.cost >= b.max_cost:
            return f"max_cost ({b.max_cost}) reached (spent ~{self.cost:.4f})"
        if b.max_seconds is not None and time.time() - self.started >= b.max_seconds:
            return f"max_seconds ({b.max_seconds}) reached"
        return None


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class Runtime:
    def __init__(
        self,
        registry: Registry,
        store: ValueStore,
        config: Config,
        hooks: Hooks,
        logger: SessionLogger,
    ) -> None:
        self.registry = registry
        self.store = store
        self.config = config
        self.hooks = hooks
        self.logger = logger
        # Tools whose full spec has already been projected into the
        # conversation (pinned specs live in the kernel → pre-seeded by the
        # session). Used by the require_spec gate (§6).
        self.seen_specs: set[str] = set()
        self._consecutive_validation_failures: dict[str, int] = {}

    # -- public -------------------------------------------------------------

    async def execute(self, calls: list[ToolCall], turn: TurnContext, ctx: ToolContext) -> list[ToolResult]:
        """Validate and run a batch of calls (§8.1).

        ``parallel_safe`` calls run concurrently; the rest run sequentially.
        Results are returned in the original call order.
        """
        results: list[Optional[ToolResult]] = [None] * len(calls)
        runnable: list[tuple[int, ToolDef, dict[str, Any]]] = []
        for i, call in enumerate(calls):
            pre = self._pre_check(call)
            if isinstance(pre, ToolResult):
                results[i] = pre
            else:
                runnable.append((i, pre[0], pre[1]))

        parallel = [(i, t, a) for i, t, a in runnable if t.execution.parallel_safe]
        serial = [(i, t, a) for i, t, a in runnable if not t.execution.parallel_safe]

        # parallel_safe calls run concurrently; only then do the mutating
        # (non-parallel-safe) calls run, one at a time, in call order.
        tasks = {
            i: asyncio.create_task(self._run_one(t, calls[i], a, ctx))
            for i, t, a in parallel
        }
        for i, task in tasks.items():
            results[i] = await task
        for i, tool, args in serial:
            results[i] = await self._run_one(tool, calls[i], args, ctx)

        final: list[ToolResult] = []
        for i, result in enumerate(results):
            assert result is not None
            for hook in self.hooks.after_execute:
                replacement = hook(calls[i], result, turn)
                if replacement is not None:
                    result = replacement
            self.logger.log(
                "execute",
                tool=calls[i].name,
                ok=result.ok,
                error=result.error,
                handle=result.handle,
                elapsed_s=round(result.elapsed_s, 4),
            )
            final.append(result)
        return final

    # -- pre-checks: unknown tool / require_spec / validation ----------------

    def _pre_check(self, call: ToolCall) -> ToolResult | tuple[ToolDef, dict[str, Any]]:
        tool = self.registry.get(call.name)
        if tool is None:
            toc = self.registry.toc_text()
            return ToolResult(
                call=call,
                ok=False,
                error="unknown_tool",
                observation=(
                    f"Error: tool {call.name!r} is not registered. "
                    f"Tool index: {toc or '(empty)'}. Use find_tools(query) to locate the right tool."
                ),
            )

        if tool.discovery.require_spec and call.name not in self.seen_specs:
            self.seen_specs.add(call.name)
            return ToolResult(
                call=call,
                ok=False,
                error="require_spec",
                observation=(
                    f"Tool {call.name!r} requires its full spec to be reviewed before first use. "
                    f"The spec follows — verify your arguments against it and call again.\n"
                    + tool.spec_text()
                ),
            )

        args = call.arguments if isinstance(call.arguments, dict) else {}
        if call.raw_arguments is not None and not args:
            error: Optional[str] = f"arguments were not valid JSON: {call.raw_arguments[:200]!r}"
        else:
            args = apply_defaults(tool.spec.parameters, args)
            error = validate_args(tool.spec.parameters, args)

        if error is not None:
            n = self._consecutive_validation_failures.get(call.name, 0) + 1
            self._consecutive_validation_failures[call.name] = n
            limit = self.config.limits.max_validation_retries
            if n > limit:
                observation = (
                    f"Validation failed {n} times in a row for {call.name!r}; giving up on this call "
                    f"(limit {limit}). Last error: {error}. Try a different tool or approach."
                )
            else:
                self.seen_specs.add(call.name)
                observation = (
                    f"Validation error calling {call.name!r}: {error}\n"
                    "The call was NOT executed. The full spec follows — fix the arguments and retry.\n"
                    + tool.spec_text()
                )
            return ToolResult(call=call, ok=False, error=f"validation: {error}", observation=observation)

        self._consecutive_validation_failures[call.name] = 0
        return tool, args

    # -- execution ----------------------------------------------------------

    async def _run_one(self, tool: ToolDef, call: ToolCall, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        handler = tool.execution.handler
        if handler is None:
            return ToolResult(
                call=call, ok=False, error="no_handler",
                observation=f"Error: tool {tool.name!r} has no executable handler registered.",
            )
        resolved = self.store.resolve_args(args) if tool.execution.resolve_handles else args
        attempts = max(1, tool.execution.retries + 1)
        start = time.time()
        last_error = ""
        for attempt in range(attempts):
            try:
                value = await asyncio.wait_for(
                    self._invoke(handler, tool, resolved, ctx),
                    timeout=tool.execution.timeout_s,
                )
                elapsed = time.time() - start
                observation, handle = self._observation_for(tool, value)
                return ToolResult(
                    call=call, ok=True, value=value,
                    observation=observation, handle=handle, elapsed_s=elapsed,
                )
            except asyncio.TimeoutError:
                last_error = f"timed out after {tool.execution.timeout_s}s"
            except Exception as exc:  # noqa: BLE001 — tool errors become observations
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts - 1:
                await asyncio.sleep(min(0.5 * (attempt + 1), 2.0))
        elapsed = time.time() - start
        return ToolResult(
            call=call, ok=False, error=last_error, elapsed_s=elapsed,
            observation=(
                f"Error executing {tool.name!r} ({attempts} attempt(s)): {last_error}. "
                "The call failed; adjust and retry or use another tool."
            ),
        )

    @staticmethod
    async def _invoke(handler: Any, tool: ToolDef, args: dict[str, Any], ctx: ToolContext) -> Any:
        kwargs = dict(args)
        if tool.wants_ctx:
            kwargs = {"ctx": ctx, **kwargs}
        if inspect.iscoroutinefunction(handler):
            return await handler(**kwargs)
        return await asyncio.to_thread(handler, **kwargs)

    # -- output policy (§8.3, I7) -------------------------------------------

    def _observation_for(self, tool: ToolDef, value: Any) -> tuple[str, Optional[str]]:
        text = serialize_value(value)
        policy = tool.execution.output_policy
        threshold = policy.max_inline_tokens or self.config.handles.inline_threshold_tokens
        tokens = estimate_tokens(text)
        if tokens <= threshold:
            return text if text else "(empty result)", None
        if policy.overflow == "truncate":
            return truncate_to_tokens(text, threshold) + "\n…[truncated by output_policy]", None
        record = self.store.put(value, source=tool.name)
        ref = self.store.ref_text(
            record,
            preview=policy.preview,
            preview_tokens=self.config.handles.preview_tokens,
        )
        return ref + "\nUse peek(handle, query=..., range=...) to inspect further.", record.id
