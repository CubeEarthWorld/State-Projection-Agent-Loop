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

JSON Schema validation uses one small built-in validator (``_mini_validate``)
— see :func:`validate_args` for why that is deliberate rather than a
fallback.
"""
from __future__ import annotations

import asyncio
import inspect
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .artifacts import ArtifactStore, serialize_value
from .capability import Capability, Effect, ToolContext
from .compression import content_hash
from .config import Config
from .llm import FINISH_NAME
from .messages import Decision, Message, ToolCall
from .policy import PolicyEngine
from .registry import Registry
from .run import Command, Question, Run
from .serialization import dumps
from .tokens import estimate_tokens, truncate_to_tokens

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "string": str, "integer": int, "number": (int, float), "boolean": bool,
    "array": list, "object": dict, "null": type(None),
}


def _json_type_name(value: Any) -> str:
    """Name a value's type in the JSON Schema vocabulary.

    The message this feeds is a self-repair prompt sent to the model, so it
    names types the way the schema beside it does — and identically in the
    Dart port, which has no Python type names to fall back on.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _type_ok(expected: str, value: Any) -> bool:
    py = _TYPE_MAP.get(expected)
    if py is None:
        return True
    if expected in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, py)


def _mini_validate(schema: dict[str, Any], value: Any, path: str = "") -> Optional[str]:
    """The JSON Schema subset a tool-argument schema actually uses."""
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
                err = _mini_validate(sub, value[key], f"{where}.{key}")
                if err:
                    return err
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            if extra:
                return f"{where}: unexpected properties {dumps(sorted(extra))}"
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


def validate_value(schema: dict[str, Any], value: Any) -> Optional[str]:
    """Validate any JSON value against a schema; error message or None."""
    return _mini_validate(schema, value)


def validate_args(schema: dict[str, Any], args: Any) -> Optional[str]:
    """Return an error message, or None when the arguments pass.

    Deliberately one small validator rather than ``jsonschema``: the error
    text goes to the model as a self-repair prompt, and two different
    validators meant this package and its Dart port rejected different
    arguments with different wording for the same schema. The subset covers
    what a tool-argument schema actually uses.
    """
    if not isinstance(args, dict):
        return f"arguments must be a JSON object, got {_json_type_name(args)}"
    return _mini_validate(schema, args)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

OUTCOMES = ("ok", "failed", "unknown", "denied", "waiting_approval", "waiting_user")

# Outcomes whose result arrives later (approval, answer): nothing is recorded
# for the call until then, so the decision stays out of the projection as a
# whole — see pair_tool_calls.
WAITING_OUTCOMES = frozenset({"waiting_approval", "waiting_user"})
_DIGITS = re.compile(r"\d")


@dataclass
class ToolResult:
    call: ToolCall
    ok: bool
    value: Any = None
    error: Optional[str] = None
    observation: str = ""
    artifact_id: Optional[str] = None
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

    def note_decision(self, decision: Decision, messages: list[Message], api_tools: list[dict], cfg: Config) -> None:
        """Account one model turn: the adapter's reported usage when it has
        one, otherwise an estimate from what was sent and what came back."""
        if decision.usage is not None:
            self.note_usage(decision.usage.prompt_tokens, decision.usage.completion_tokens, cfg)
            return
        completion = estimate_tokens(decision.text)
        for call in decision.calls:
            arguments = call.raw_arguments if call.raw_arguments is not None else call.arguments
            completion += 6 + estimate_tokens(call.name) + estimate_tokens(arguments)
        # Adapters normalize finish(result) out of calls before returning.
        if decision.finish:
            completion += 6 + estimate_tokens(FINISH_NAME) + estimate_tokens({"result": decision.result})
        self.note_usage(estimate_tokens(messages) + estimate_tokens(api_tools), completion, cfg)

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
    # No artifact store of its own: it uses ctx.store, the session's current
    # one. A resumed run installs a fresh store for its new run id, and a
    # second copy captured here would silently keep writing artifacts the
    # session (and therefore meta.artifact.peek) could no longer read.
    def __init__(self, registry: Registry, config: Config) -> None:
        self.registry = registry
        self.config = config
        # Capabilities whose full spec has already been projected into the
        # conversation. Used by the require_spec gate; pinned capabilities
        # are exempt because their spec is always in the kernel section.
        self.seen_specs: set[str] = set()
        self._consecutive_validation_failures: dict[str, int] = {}
        # Loop guard memory: (capability, arguments hash, result tag) of the
        # last ``limits.repeat_window`` executed calls in this run.
        self._recent: list[tuple[str, str, str]] = []

    def reset(self) -> None:
        """Forget what this runtime learned from the conversation so far.
        Called on rewind: the specs shown and the failures counted are in
        the discarded history, so a capability must not start the new
        timeline already one strike from "giving up"."""
        self.seen_specs.clear()
        self._consecutive_validation_failures.clear()
        self._recent.clear()

    # -- loop guard -----------------------------------------------------------

    @staticmethod
    def _args_hash(args: dict[str, Any]) -> str:
        return content_hash(dumps(args))

    def _loop_guard(self, call: ToolCall, capability: Capability, args: dict[str, Any]) -> Optional[ToolResult]:
        """Refuse a call the model keeps repeating with identical arguments
        when every repeat failed, or (for anything but a pure read) every
        repeat returned the same result. Polling a pure read for a change is
        legitimate and stays allowed."""
        limit = self.config.limits.max_repeats
        if limit <= 0:
            return None
        args_hash = self._args_hash(args)
        tags = [tag for name, ahash, tag in self._recent if name == capability.name and ahash == args_hash]
        if len(tags) < limit:
            return None
        why: Optional[str] = None
        if sum(1 for t in tags if t.startswith("err:")) >= limit:
            why = f"failed identically {len(tags)} times"
        elif not (self.is_read_only(capability) and capability.execution.retry_safety == "pure"):
            if max(tags.count(t) for t in set(tags)) >= limit:
                why = f"returned the same result {len(tags)} times"
        if why is None:
            return None
        return ToolResult(
            call=call, ok=False, outcome="failed", error="loop_guard",
            observation=(
                f"Loop guard: \"{capability.name}\" with these exact arguments {why}. "
                "It was not executed again; change the arguments or the approach."
            ),
        )

    def _remember(self, capability: Capability, args: dict[str, Any], result: ToolResult) -> None:
        if result.outcome in WAITING_OUTCOMES:
            return
        tag = (content_hash(serialize_value(result.value)) if result.ok
               else "err:" + content_hash(_DIGITS.sub("", result.error or "")))
        self._recent.append((capability.name, self._args_hash(args), tag))
        del self._recent[:-self.config.limits.repeat_window or None]

    async def _run(
        self, capability: Capability, args: dict[str, Any], ctx: ToolContext, run: Run, call: ToolCall,
        command: Optional[Command] = None,
    ) -> ToolResult:
        result = await self._execute_one(capability, args, ctx, run, call, command=command)
        self._remember(capability, args, result)
        return result

    # -- public ---------------------------------------------------------------

    async def execute(
        self, calls: list[ToolCall], ctx: ToolContext, run: Run, policy: PolicyEngine,
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
                results.append(await self._run(cap, args, ctx, run, call))
            else:
                tasks = [self._run(cap, args, ctx, run, call) for call, cap, args in buffer]
                results.extend(await asyncio.gather(*tasks))
            buffer.clear()

        for idx, call in enumerate(calls):
            pre = self._pre_check(call)
            if isinstance(pre, ToolResult):
                await flush()
                results.append(pre)
                continue
            capability, args = pre
            tripped = self._loop_guard(call, capability, args)
            if tripped is not None:
                await flush()
                results.append(tripped)
                continue
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
            if self.is_read_only(capability):
                buffer.append((call, capability, args))
            else:
                await flush()
                results.append(await self._run(capability, args, ctx, run, call))
                if results[-1].outcome == "waiting_user":
                    run.pending_calls = list(calls[idx + 1:])
                    return ExecuteBatchResult(results=results, halted=True)
        await flush()
        return ExecuteBatchResult(results=results, halted=False)

    async def resume_pending(
        self, run: Run, ctx: ToolContext, policy: PolicyEngine,
    ) -> ExecuteBatchResult:
        """Continue a run's ``pending_calls`` after its approval was resolved
        or its question answered.

        After an approval the first pending call already has a :class:`~state_projection_loop.run.Command`
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
        run.last_resolved_approval = None  # consumed here; must not leak into a later pause
        approved = run.commands.get(resolved.command_id) if resolved and resolved.resolution == "approved" else None
        if resolved is not None and resolved.resolution == "denied":
            denied_command = run.commands.get(resolved.command_id)
            denied_name = denied_command.capability_name if denied_command else first_call.name
            run.pending_calls = []
            # Every parked call needs its own result: the denial cancels the
            # rest of the decision too, and a call left without one would
            # take the whole decision out of the projection.
            results = [ToolResult(
                call=first_call, ok=False, outcome="denied", error="approval_denied",
                observation=f"Approval denied: {denied_name} was not executed.",
                command_id=denied_command.id if denied_command else None,
            )]
            results += [
                ToolResult(call=call, ok=False, outcome="denied", error="approval_denied",
                           observation=f"Not executed: the approval for {denied_name} was denied.")
                for call in pending[1:]
            ]
            return ExecuteBatchResult(results=results, halted=False)
        run.pending_calls = []
        if approved is None:
            # Parked behind a question, not an approval: nothing here was
            # checked yet, so every call takes the normal path.
            return await self.execute(pending, ctx, run, policy)
        capability = self.registry.get(approved.capability_name)
        if capability is None:
            results = [ToolResult(call=first_call, ok=False, outcome="failed", error="unknown_capability",
                                  observation=f"Error: capability \"{first_call.name}\" no longer registered.")]
        else:
            results = [await self._run(capability, approved.arguments, ctx, run, first_call, command=approved)]
            if results[-1].outcome == "waiting_user":
                run.pending_calls = list(pending[1:])
                return ExecuteBatchResult(results=results, halted=True)
        rest = await self.execute(pending[1:], ctx, run, policy)
        return ExecuteBatchResult(results=results + rest.results, halted=rest.halted)

    # -- pre-checks: unknown capability / require_spec / validation ---------

    def _pre_check(self, call: ToolCall) -> ToolResult | tuple[Capability, dict[str, Any]]:
        capability = self.registry.get(call.name)
        if capability is None:
            toc = self.registry.toc_text()
            # Never point at a search tool that is itself absent or disabled:
            # a capability the model cannot reach must not be advertised.
            hint = (
                " Use meta.tool.find(query) to locate the right one."
                if "meta.tool.find" in self.registry else ""
            )
            return ToolResult(
                call=call, ok=False, outcome="failed", error="unknown_capability",
                observation=(
                    f"Error: capability \"{call.name}\" is not registered. "
                    f"Tool index: {toc or '(empty)'}.{hint}"
                ),
            )

        # A pinned capability's full spec is already in the kernel section,
        # so the gate is satisfied by construction — no pre-seeding needed.
        needs_spec = capability.discovery.require_spec and not capability.discovery.pinned
        if needs_spec and capability.name not in self.seen_specs:
            self.seen_specs.add(capability.name)
            return ToolResult(
                call=call, ok=False, outcome="failed", error="require_spec",
                observation=(
                    f"Capability \"{call.name}\" requires its full spec to be reviewed before first use. "
                    f"The spec follows — verify your arguments against it and call again.\n"
                    + capability.spec_text()
                ),
            )

        args = call.arguments if isinstance(call.arguments, dict) else {}
        if call.raw_arguments is not None and not args:
            error: Optional[str] = f"arguments were not valid JSON: \"{call.raw_arguments[:200]}\""
        else:
            args = apply_defaults(capability.spec.parameters, args)
            error = validate_args(capability.spec.parameters, args)

        if error is not None:
            n = self._consecutive_validation_failures.get(call.name, 0) + 1
            self._consecutive_validation_failures[call.name] = n
            limit = self.config.limits.max_validation_retries
            if n > limit:
                observation = (
                    f"Validation failed {n} times in a row for \"{call.name}\"; giving up on this call "
                    f"(limit {limit}). Last error: {error}. Try a different tool or approach."
                )
            else:
                self.seen_specs.add(capability.name)
                observation = (
                    f"Validation error calling \"{call.name}\": {error}\n"
                    "The call was NOT executed. The full spec follows — fix the arguments and retry.\n"
                    + capability.spec_text()
                )
            return ToolResult(call=call, ok=False, outcome="failed", error=f"validation: {error}",
                               observation=observation)

        self._consecutive_validation_failures[call.name] = 0
        return capability, args

    @staticmethod
    def is_read_only(capability: Capability) -> bool:
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
        call_ctx = ctx.for_command(command.id)

        handler = capability.execution.handler
        if handler is None:
            run.record_outcome(command, "failed", error="no_handler")
            return ToolResult(
                call=call, ok=False, outcome="failed", error="no_handler", command_id=command.id,
                observation=f"Error: capability \"{capability.name}\" has no executable handler registered.",
            )
        resolved = ctx.store.resolve_args(args) if capability.execution.resolve_handles else args
        attempts = max(1, capability.execution.retries + 1)
        last_error = ""
        last_outcome = "failed"
        for attempt in range(attempts):
            command.attempts += 1
            try:
                value = await asyncio.wait_for(
                    self._invoke(handler, capability, resolved, call_ctx),
                    timeout=capability.execution.timeout_s,
                )
                if isinstance(value, Question):
                    # The command stays pending until Session.answer completes it.
                    run.ask_question(command, call.id, value)
                    return ToolResult(
                        call=call, ok=False, outcome="waiting_user", error="question_pending",
                        observation=f"Question pending: {value.text}", command_id=command.id,
                    )
                observation, artifact_id = self._observation_for(capability, value, ctx.store)
                run.record_outcome(command, "ok", result_ref=artifact_id)
                return ToolResult(
                    call=call, ok=True, value=value, outcome="ok", command_id=command.id,
                    observation=observation, artifact_id=artifact_id,
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
        run.record_outcome(command, last_outcome, error=last_error)
        return ToolResult(
            call=call, ok=False, error=last_error, outcome=last_outcome,
            command_id=command.id,
            observation=(
                f"{'Timed out' if last_outcome == 'unknown' else 'Error'} executing \"{capability.name}\" "
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

    def _observation_for(
        self, capability: Capability, value: Any, store: ArtifactStore,
    ) -> tuple[str, Optional[str]]:
        text = serialize_value(value)
        policy = capability.execution.output_policy
        threshold = policy.max_inline_tokens or self.config.artifacts.inline_threshold_tokens
        tokens = estimate_tokens(text)
        if tokens <= threshold:
            return text if text else "(empty result)", None
        if policy.overflow == "truncate":
            return truncate_to_tokens(text, threshold) + "\n…[truncated by output_policy]", None
        record = store.put(value, source=capability.name)
        ref_text = store.ref_text(
            record, preview=policy.preview, preview_tokens=self.config.artifacts.preview_tokens,
        )
        if "meta.artifact.peek" in self.registry:
            ref_text += (
                f'\nUse meta.artifact.peek(artifact={{"$artifact": "{record.id}"}}, '
                "query=..., range=...) to inspect further."
            )
        return ref_text, record.id
