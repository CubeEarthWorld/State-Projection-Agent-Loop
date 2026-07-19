"""Deterministic runtime: validate → authorize → execute → record.

The LLM only decides *what* to do; validation, retries, timeouts, ordering,
policy authorization and output shaping are enforced here in code.

Two correctness properties this module exists to guarantee, both violated
by naive "batch of tool calls" runtimes:

* **Order** (P0-1): calls execute in the model's stated order by default.
  The only concurrency allowed is a run of *adjacent* calls whose
  capabilities declare no write/external effects — reads never race a
  write, and a write never jumps ahead of an earlier read or write. There
  is no cross-batch dependency solver; that complexity is deliberately out
  of scope (see the design spec's "later" list).
* **Idempotency** (P0-2): a capability may only be auto-retried by this
  runtime if its ``retry_safety`` is ``pure`` or ``idempotent`` —
  :class:`~state_projection_loop.capability.CapabilityExecution` refuses to
  even construct with ``retries > 0`` otherwise. A timeout is recorded as
  outcome ``unknown``, never silently treated as ``failed``: we cannot tell
  whether a synchronous handler's underlying effect completed after the
  awaiting task gave up on it, and collapsing that distinction is exactly
  what lets non-idempotent operations double-fire.

JSON Schema validation uses ``jsonschema`` when installed and falls back to
a built-in mini validator otherwise (keeps the core pure-Python for
embedded environments).
"""
from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from .artifacts import ArtifactStore, serialize_value, truncate_to_tokens
from .capability import Capability, Effect, ToolContext
from .config import Config
from .messages import ToolCall
from .policy import PolicyEngine
from .projection import TurnContext
from .registry import Registry
from .run import Command, Run
from .tokens import estimate_tokens

try:
    import jsonschema as _jsonschema
except ImportError:  # pragma: no cover - exercised via _mini_validate tests
    _jsonschema = None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "string": str, "integer": int, "number": (int, float), "boolean": bool,
    "array": list, "object": dict, "null": type(None),
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
    """Return an error message, or None when the arguments pass."""
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
# Results
# ---------------------------------------------------------------------------

OUTCOMES = ("ok", "failed", "unknown", "denied", "waiting_approval")


@dataclass
class ToolResult:
    call: ToolCall
    ok: bool
    value: Any = None
    error: Optional[str] = None
    observation: str = ""
    artifact_id: Optional[str] = None
    elapsed_s: float = 0.0
    outcome: str = "ok"  # one of OUTCOMES
    command_id: Optional[str] = None


@dataclass
class ExecuteBatchResult:
    """Result of one call to :meth:`Runtime.execute`.

    ``halted`` is true when a call in the batch required approval: the
    run has already been transitioned to ``WAITING_FOR_APPROVAL`` and its
    ``pending_calls`` holds everything from that point on (inclusive) for
    :meth:`Runtime.resume_pending` to continue once approved. Calls after a
    halt point are never even validated — order is preserved by construction.
    """

    results: list[ToolResult] = field(default_factory=list)
    halted: bool = False


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

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
    def __init__(self, registry: Registry, store: ArtifactStore, config: Config) -> None:
        self.registry = registry
        self.store = store
        self.config = config
        # Capabilities whose full spec has already been projected into the
        # conversation (pinned specs live in the kernel → pre-seeded by the
        # session). Used by the require_spec gate.
        self.seen_specs: set[str] = set()
        self._consecutive_validation_failures: dict[str, int] = {}

    # -- public ---------------------------------------------------------------

    async def execute(
        self, calls: list[ToolCall], turn: TurnContext, ctx: ToolContext, run: Run, policy: PolicyEngine,
    ) -> ExecuteBatchResult:
        """Validate, authorize and run a batch of calls, in order (P0-1).

        A contiguous run of calls whose capabilities declare no write/
        external effects may execute concurrently; anything else runs one
        at a time, strictly in the order the model asked for it.
        """
        results: list[ToolResult] = []
        buffer: list[tuple[ToolCall, Capability, dict[str, Any]]] = []

        async def flush() -> None:
            if not buffer:
                return
            if len(buffer) == 1:
                call, cap, args = buffer[0]
                results.append(await self._execute_one(cap, args, ctx, run, call))
            else:
                tasks = [self._execute_one(cap, args, ctx, run, call) for call, cap, args in buffer]
                results.extend(await asyncio.gather(*tasks))
            buffer.clear()

        for idx, call in enumerate(calls):
            pre = self._pre_check(call)
            if isinstance(pre, ToolResult):
                await flush()
                results.append(pre)
                continue
            capability, args = pre
            decision = policy.evaluate(capability, args)
            if decision.decision == "deny":
                await flush()
                results.append(ToolResult(
                    call=call, ok=False, outcome="denied", error=decision.reason,
                    observation=f"Denied by policy ({decision.layer}): {decision.reason}",
                ))
                continue
            if decision.decision == "require_approval":
                await flush()
                command = run.new_command(capability.qualified_name, args, capability.execution.retry_safety)
                run.pending_calls = list(calls[idx:])
                run.request_approval(command, capability.effects, decision.reason,
                                      policy_revision=policy.revision,
                                      expires_in_s=self.config.limits.approval_expires_s)
                results.append(ToolResult(
                    call=call, ok=False, outcome="waiting_approval", error="approval_required",
                    observation=f"Approval required: {decision.reason}", command_id=command.id,
                ))
                return ExecuteBatchResult(results=results, halted=True)
            if self._is_read_only(capability):
                buffer.append((call, capability, args))
            else:
                await flush()
                results.append(await self._execute_one(capability, args, ctx, run, call))
        await flush()
        return ExecuteBatchResult(results=results, halted=False)

    async def resume_pending(
        self, run: Run, ctx: ToolContext, policy: PolicyEngine, turn: TurnContext,
    ) -> ExecuteBatchResult:
        """Continue a run's ``pending_calls`` after its approval was resolved.

        The first pending call already has a :class:`~state_projection_loop.run.Command`
        (created when approval was requested) and is executed directly,
        reusing its ``command_id`` — no re-validation, no re-authorization,
        so an approved command cannot silently get a different idempotency
        key on retry. The remaining calls go back through the normal
        ``execute`` path.
        """
        pending = run.pending_calls
        if not pending:
            return ExecuteBatchResult(results=[], halted=False)
        first_call = pending[0]
        resolved = run.last_resolved_approval
        approved = run.commands.get(resolved.command_id) if resolved and resolved.resolution == "approved" else None
        if resolved is not None and resolved.resolution == "denied":
            denied_command = run.commands.get(resolved.command_id)
            observation = f"Approval denied: {denied_command.capability_name if denied_command else first_call.name} was not executed."
            run.pending_calls = []
            return ExecuteBatchResult(
                results=[ToolResult(call=first_call, ok=False, outcome="denied", error="approval_denied",
                                     observation=observation,
                                     command_id=denied_command.id if denied_command else None)],
                halted=False,
            )
        capability = self.registry.get(approved.capability_name) if approved else self.registry.get(first_call.name)
        results: list[ToolResult] = []
        if capability is None:
            results.append(ToolResult(call=first_call, ok=False, outcome="failed", error="unknown_capability",
                                       observation=f"Error: capability {first_call.name!r} no longer registered."))
        else:
            args = approved.arguments if approved else (first_call.arguments if isinstance(first_call.arguments, dict) else {})
            results.append(await self._execute_one(capability, args, ctx, run, first_call, command=approved))
        run.pending_calls = []
        rest = await self.execute(pending[1:], turn, ctx, run, policy)
        results.extend(rest.results)
        if rest.halted:
            return ExecuteBatchResult(results=results, halted=True)
        return ExecuteBatchResult(results=results, halted=False)

    # -- pre-checks: unknown capability / require_spec / validation ---------

    def _pre_check(self, call: ToolCall) -> ToolResult | tuple[Capability, dict[str, Any]]:
        capability = self.registry.get(call.name)
        if capability is None:
            toc = self.registry.toc_text()
            return ToolResult(
                call=call, ok=False, outcome="failed", error="unknown_capability",
                observation=(
                    f"Error: capability {call.name!r} is not registered. "
                    f"Tool index: {toc or '(empty)'}. Use find_tools(query) to locate the right one."
                ),
            )

        if capability.discovery.require_spec and capability.name not in self.seen_specs:
            self.seen_specs.add(capability.name)
            return ToolResult(
                call=call, ok=False, outcome="failed", error="require_spec",
                observation=(
                    f"Capability {call.name!r} requires its full spec to be reviewed before first use. "
                    f"The spec follows — verify your arguments against it and call again.\n"
                    + capability.spec_text()
                ),
            )

        args = call.arguments if isinstance(call.arguments, dict) else {}
        if call.raw_arguments is not None and not args:
            error: Optional[str] = f"arguments were not valid JSON: {call.raw_arguments[:200]!r}"
        else:
            args = apply_defaults(capability.spec.parameters, args)
            error = validate_args(capability.spec.parameters, args)

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
                self.seen_specs.add(capability.name)
                observation = (
                    f"Validation error calling {call.name!r}: {error}\n"
                    "The call was NOT executed. The full spec follows — fix the arguments and retry.\n"
                    + capability.spec_text()
                )
            return ToolResult(call=call, ok=False, outcome="failed", error=f"validation: {error}",
                               observation=observation)

        self._consecutive_validation_failures[call.name] = 0
        return capability, args

    @staticmethod
    def _is_read_only(capability: Capability) -> bool:
        # Mirrors PolicyEngine.evaluate: undeclared effects are treated as
        # the most restrictive kind, so an author who forgot to declare
        # effects doesn't also get free parallel execution.
        effects = capability.effects or [Effect(kind="external", resource="undeclared:*")]
        return all(e.kind in ("none", "read") for e in effects)

    # -- execution ------------------------------------------------------------

    async def _execute_one(
        self, capability: Capability, args: dict[str, Any], ctx: ToolContext, run: Run, call: ToolCall,
        command: Optional[Command] = None,
    ) -> ToolResult:
        if command is None:
            command = run.new_command(capability.qualified_name, args, capability.execution.retry_safety)
        call_ctx = replace(ctx, command_id=command.id)

        handler = capability.execution.handler
        if handler is None:
            run.record_outcome(command, "failed", error="no_handler")
            return ToolResult(
                call=call, ok=False, outcome="failed", error="no_handler", command_id=command.id,
                observation=f"Error: capability {capability.name!r} has no executable handler registered.",
            )
        resolved = self.store.resolve_args(args) if capability.execution.resolve_handles else args
        attempts = max(1, capability.execution.retries + 1)
        start = time.time()
        last_error = ""
        last_outcome = "failed"
        for attempt in range(attempts):
            command.attempts += 1
            try:
                value = await asyncio.wait_for(
                    self._invoke(handler, capability, resolved, call_ctx),
                    timeout=capability.execution.timeout_s,
                )
                elapsed = time.time() - start
                observation, artifact_id = self._observation_for(capability, value)
                run.record_outcome(command, "ok", result_ref=artifact_id)
                return ToolResult(
                    call=call, ok=True, value=value, outcome="ok", command_id=command.id,
                    observation=observation, artifact_id=artifact_id, elapsed_s=elapsed,
                )
            except asyncio.TimeoutError:
                # We cannot confirm whether the underlying effect completed
                # after the awaiting task gave up — never collapse this into
                # "failed" (P0-2). A retry only proceeds below if the
                # capability's retry_safety already permits blind retries.
                last_error = f"timed out after {capability.execution.timeout_s}s"
                last_outcome = "unknown"
            except Exception as exc:  # noqa: BLE001 — capability errors become observations
                last_error = f"{type(exc).__name__}: {exc}"
                last_outcome = "failed"
            if attempt < attempts - 1:
                await asyncio.sleep(min(0.5 * (attempt + 1), 2.0))
        elapsed = time.time() - start
        run.record_outcome(command, last_outcome, error=last_error)
        return ToolResult(
            call=call, ok=False, error=last_error, outcome=last_outcome, elapsed_s=elapsed,
            command_id=command.id,
            observation=(
                f"{'Timed out' if last_outcome == 'unknown' else 'Error'} executing {capability.name!r} "
                f"({attempts} attempt(s)): {last_error}. "
                + ("Outcome is UNKNOWN — do not blindly retry a non-idempotent action; check state first."
                   if last_outcome == "unknown" else "The call failed; adjust and retry or use another tool.")
            ),
        )

    @staticmethod
    async def _invoke(handler: Any, capability: Capability, args: dict[str, Any], ctx: ToolContext) -> Any:
        kwargs = dict(args)
        if capability.wants_ctx:
            kwargs = {"ctx": ctx, **kwargs}
        if inspect.iscoroutinefunction(handler):
            return await handler(**kwargs)
        return await asyncio.to_thread(handler, **kwargs)

    # -- output policy --------------------------------------------------------

    def _observation_for(self, capability: Capability, value: Any) -> tuple[str, Optional[str]]:
        text = serialize_value(value)
        policy = capability.execution.output_policy
        threshold = policy.max_inline_tokens or self.config.artifacts.inline_threshold_tokens
        tokens = estimate_tokens(text)
        if tokens <= threshold:
            return text if text else "(empty result)", None
        if policy.overflow == "truncate":
            return truncate_to_tokens(text, threshold) + "\n…[truncated by output_policy]", None
        record = self.store.put(value, source=capability.name)
        ref_text = self.store.ref_text(
            record, preview=policy.preview, preview_tokens=self.config.artifacts.preview_tokens,
        )
        return ref_text + f"\nUse peek(artifact={{\"$artifact\": \"{record.id}\"}}, query=..., range=...) to inspect further.", record.id
