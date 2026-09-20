"""Deterministic runtime: validate → authorize → execute → record.

The LLM only decides *what* to do; validation, retries, timeouts, ordering,
policy authorization and output shaping are enforced here in code.

Two correctness properties this module exists to guarantee, both violated
by naive "batch of tool calls" runtimes:

* **Order**: calls execute in the model's stated order by default.
  The only concurrency allowed is a run of *adjacent* calls whose
  capabilities declare no write/external effects — reads never race a
  write, and a write never jumps ahead of an earlier read or write. There
  is no cross-batch dependency solver; that complexity is deliberately out
  of scope (see the design spec's "later" list).
* **Idempotency**: a capability may only be auto-retried by this
  runtime if its ``retry_safety`` is ``pure`` or ``idempotent`` —
  :class:`~state_projection_loop.capability.CapabilityExecution` refuses to
  even construct with ``retries > 0`` otherwise. A timeout is recorded as
  outcome ``unknown``, never silently treated as ``failed``: we cannot tell
  whether a synchronous handler's underlying effect completed after the
  awaiting task gave up on it, and collapsing that distinction is exactly
  what lets non-idempotent operations double-fire.

"""
from __future__ import annotations

import asyncio
import inspect
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .artifacts import ArtifactStore, serialize_value
from .capability import Capability
from .context import ToolContext
from .compression import content_hash
from .config import Config
from .json_schema import apply_defaults, validate_args
from .llm import FINISH_NAME
from .messages import Decision, Message, ToolCall
from .policy import PolicyEngine
from .registry import Registry
from .run import Command, Question, Run
from .serialization import dumps
from .tokens import estimate_tokens, truncate_to_tokens

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

# Outcomes whose result arrives later (approval, answer): nothing is recorded
# for the call until then, so the decision stays out of the projection as a
# whole — see pair_tool_calls.
WAITING_OUTCOMES = frozenset({"waiting_approval", "waiting_user"})
_DIGITS = re.compile(r"\d")


@dataclass
class ToolResult:
    call: ToolCall
    value: Any = None
    error: Optional[str] = None
    observation: str = ""
    artifact_id: Optional[str] = None
    outcome: str = "ok"  # ok | failed | unknown | denied | waiting_approval | waiting_user
    command_id: Optional[str] = None
    # ``value`` serialized once, for the observation and the loop guard's
    # result tag. Internal bookkeeping, not part of the result's identity.
    serialized: Optional[str] = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return self.outcome == "ok"


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

    def to_dict(self) -> dict[str, Any]:
        """What a snapshot keeps. Not ``started``: the wall clock restarts
        with the process that resumes the run."""
        return {"steps": self.steps, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "cost": self.cost}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BudgetState":
        return cls(steps=d.get("steps", 0), prompt_tokens=d.get("prompt_tokens", 0),
                   completion_tokens=d.get("completion_tokens", 0), cost=d.get("cost", 0.0))

    def note_usage(self, prompt: int, completion: int, cfg: Config) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        b = cfg.budget
        self.cost += prompt / 1000 * b.cost_per_1k_input + completion / 1000 * b.cost_per_1k_output

    def note_decision(self, decision: Decision, messages: list[Message], api_tools: list[dict], cfg: Config) -> None:
        """Account one model turn: a step, and the adapter's reported usage
        when it has one, otherwise an estimate from what was sent and what
        came back."""
        self.steps += 1
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

@dataclass
class Hooks:
    """Host code around each tool call, run after policy has authorised it.

    ``before_call(capability, args, ctx)`` may return replacement arguments
    (validated like the model's) or a string, which rejects the call with
    that text as the observation — a correctness gate like validation, not
    a second grant path: nothing a hook returns can allow a denied call.
    ``after_call(capability, args, result, ctx)`` may return a replacement
    observation (redaction, an attached diff). Every intervention is a
    ``hook_intervened`` ledger event.
    """

    before_call: Optional[Callable[[Capability, dict[str, Any], ToolContext], Any]] = None
    after_call: Optional[Callable[[Capability, dict[str, Any], "ToolResult", ToolContext], Any]] = None


class Runtime:
    # No artifact store of its own: it uses ctx.store, the session's current
    # one. A resumed run installs a fresh store for its new run id, and a
    # second copy captured here would silently keep writing artifacts the
    # session (and therefore meta.artifact.peek) could no longer read.
    def __init__(self, registry: Registry, config: Config, *, hooks: Optional[Hooks] = None) -> None:
        self.registry = registry
        self.config = config
        self.hooks = hooks or Hooks()
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

    def _loop_guard(self, call: ToolCall, capability: Capability, args_hash: str) -> Optional[ToolResult]:
        """Refuse a call the model keeps repeating with identical arguments
        when every repeat failed, or (for anything but a pure read) every
        repeat returned the same result. Polling a pure read for a change is
        legitimate and stays allowed."""
        limit = self.config.limits.max_repeats
        if limit <= 0:
            return None
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
            call=call, outcome="failed", error="loop_guard",
            observation=(
                f"Loop guard: \"{capability.name}\" with these exact arguments {why}. "
                "It was not executed again; change the arguments or the approach."
            ),
        )

    def _remember(self, capability: Capability, args_hash: str, result: ToolResult) -> None:
        if result.outcome in WAITING_OUTCOMES:
            return
        tag = (content_hash(result.serialized or "") if result.ok
               else "err:" + content_hash(_DIGITS.sub("", result.error or "")))
        self._recent.append((capability.name, args_hash, tag))
        del self._recent[:max(0, len(self._recent) - self.config.limits.repeat_window)]

    async def _run(
        self, capability: Capability, args: dict[str, Any], ctx: ToolContext, run: Run, call: ToolCall,
        command: Optional[Command] = None, args_hash: Optional[str] = None,
    ) -> ToolResult:
        result = await self._execute_one(capability, args, ctx, run, call, command=command)
        self._remember(capability, args_hash if args_hash is not None else self._args_hash(args), result)
        return result

    # -- public ---------------------------------------------------------------

    async def execute(
        self, calls: list[ToolCall], ctx: ToolContext, run: Run, policy: PolicyEngine,
    ) -> ExecuteBatchResult:
        """Validate, authorize and run a batch of calls, in order.

        A contiguous run of calls whose capabilities declare no write/
        external effects may execute concurrently; anything else runs one
        at a time, strictly in the order the model asked for it.
        """
        results: list[ToolResult] = []
        buffer: list[tuple[ToolCall, Capability, dict[str, Any], str]] = []
        # A batch supersedes any approval resolved before it: pending_calls is
        # about to belong to this batch, and a leftover request would send
        # resume_pending looking for the wrong call's command.
        run.last_resolved_approval = None

        async def flush(idx: int) -> bool:
            """Run the buffered read-only calls concurrently. True when one of
            them parked the run on a question: like the sequential path, the
            rest of the batch (from ``idx``) waits for the answer."""
            if not buffer:
                return False
            flushed = await asyncio.gather(
                *(self._run(cap, args, ctx, run, call, args_hash=h) for call, cap, args, h in buffer))
            results.extend(flushed)
            buffer.clear()
            if not any(r.outcome == "waiting_user" for r in flushed):
                return False
            run.pending_calls = list(calls[idx:])
            return True

        for idx, call in enumerate(calls):
            pre = self._pre_check(call)
            if isinstance(pre, ToolResult):
                if await flush(idx):
                    return ExecuteBatchResult(results=results, halted=True)
                results.append(pre)
                continue
            capability, args = pre
            args_hash = self._args_hash(args)
            tripped = self._loop_guard(call, capability, args_hash)
            if tripped is not None:
                if await flush(idx):
                    return ExecuteBatchResult(results=results, halted=True)
                results.append(tripped)
                continue
            decision = policy.evaluate(capability, args)
            if decision.decision == "deny":
                if await flush(idx):
                    return ExecuteBatchResult(results=results, halted=True)
                results.append(ToolResult(
                    call=call, outcome="denied", error=decision.reason,
                    observation=f"Denied by policy ({decision.layer}): {decision.reason}",
                ))
                continue
            if decision.decision == "require_approval":
                if await flush(idx):
                    return ExecuteBatchResult(results=results, halted=True)
                command = run.new_command(capability.qualified_name, args, capability.execution.retry_safety)
                run.pending_calls = list(calls[idx:])
                run.request_approval(command, capability.effects, decision.reason,
                                      policy_revision=policy.revision,
                                      expires_in_s=self.config.limits.approval_expires_s)
                results.append(ToolResult(
                    call=call, outcome="waiting_approval", error="approval_required",
                    observation=f"Approval required: {decision.reason}", command_id=command.id,
                ))
                return ExecuteBatchResult(results=results, halted=True)
            if self.is_read_only(capability):
                buffer.append((call, capability, args, args_hash))
            else:
                if await flush(idx):
                    return ExecuteBatchResult(results=results, halted=True)
                results.append(await self._run(capability, args, ctx, run, call, args_hash=args_hash))
                if results[-1].outcome == "waiting_user":
                    run.pending_calls = list(calls[idx + 1:])
                    return ExecuteBatchResult(results=results, halted=True)
        halted = await flush(len(calls))
        return ExecuteBatchResult(results=results, halted=halted)

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
                call=first_call, outcome="denied", error="approval_denied",
                observation=f"Approval denied: {denied_name} was not executed.",
                command_id=denied_command.id if denied_command else None,
            )]
            results += [
                ToolResult(call=call, outcome="denied", error="approval_denied",
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
            results = [ToolResult(call=first_call, outcome="failed", error="unknown_capability",
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
                call=call, outcome="failed", error="unknown_capability",
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
                call=call, outcome="failed", error="require_spec",
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
            return ToolResult(call=call, outcome="failed", error=f"validation: {error}",
                               observation=observation)

        self._consecutive_validation_failures[call.name] = 0
        return capability, args

    @staticmethod
    def is_read_only(capability: Capability) -> bool:
        return all(e.kind in ("none", "read") for e in capability.planned_effects)

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
                call=call, outcome="failed", error="no_handler", command_id=command.id,
                observation=f"Error: capability \"{capability.name}\" has no executable handler registered.",
            )
        def note_hook(stage: str, key: str, value: Any) -> None:
            run.ledger.append(run.id, "hook_intervened", {"command_id": command.id, "stage": stage, key: value})

        if self.hooks.before_call is not None:
            try:
                verdict = self.hooks.before_call(capability, args, call_ctx)
            except Exception as exc:  # noqa: BLE001 — a broken hook rejects its own call, not the batch
                verdict = f"{type(exc).__name__}: {exc}"
            if isinstance(verdict, dict):
                error = validate_args(capability.spec.parameters, verdict)
                verdict = f"hook returned invalid arguments: {error}" if error else verdict
            if isinstance(verdict, dict):
                note_hook("before", "arguments", verdict)
                args = verdict
            elif verdict is not None:
                note_hook("before", "rejected", str(verdict))
                run.record_outcome(command, "failed", error="hook_rejected")
                return ToolResult(call=call, outcome="failed", error="hook_rejected", command_id=command.id,
                                  observation=f"Rejected by hook: {verdict}")
        result = await self._attempts(capability, args, ctx, run, call, command, handler, call_ctx)
        if self.hooks.after_call is not None and result.outcome not in WAITING_OUTCOMES:
            try:
                replacement = self.hooks.after_call(capability, args, result, call_ctx)
            except Exception as exc:  # noqa: BLE001 — the call already ran; keep its real observation
                note_hook("after", "error", f"{type(exc).__name__}: {exc}")
                replacement = None
            if isinstance(replacement, str):
                note_hook("after", "observation", replacement)
                result.observation = replacement
        return result

    async def _attempts(
        self, capability: Capability, args: dict[str, Any], ctx: ToolContext, run: Run, call: ToolCall,
        command: Command, handler: Any, call_ctx: ToolContext,
    ) -> ToolResult:
        resolved = ctx.store.resolve_args(args) if capability.execution.resolve_handles else args
        attempts = max(1, capability.execution.retries + 1)
        last_error = ""
        last_outcome = "failed"
        started = time.monotonic()

        def elapsed_ms() -> int:
            return int((time.monotonic() - started) * 1000)

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
                        call=call, outcome="waiting_user", error="question_pending",
                        observation=f"Question pending: {value.text}", command_id=command.id,
                    )
                serialized = serialize_value(value)
                observation, artifact_id = self._observation_for(capability, serialized, value, ctx.store)
                run.record_outcome(command, "ok", result_ref=artifact_id, duration_ms=elapsed_ms())
                return ToolResult(
                    call=call, value=value, outcome="ok", command_id=command.id,
                    observation=observation, artifact_id=artifact_id, serialized=serialized,
                )
            except asyncio.TimeoutError:
                # We cannot confirm whether the underlying effect completed
                # after the awaiting task gave up — never collapse this into
                # "failed". A retry only proceeds below if the
                # capability's retry_safety already permits blind retries.
                last_error = f"timed out after {capability.execution.timeout_s}s"
                last_outcome = "unknown"
            except Exception as exc:  # noqa: BLE001 — capability errors become observations
                last_error = f"{type(exc).__name__}: {exc}"
                last_outcome = "failed"
            if attempt < attempts - 1:
                await asyncio.sleep(min(0.5 * (attempt + 1), 2.0))
        run.record_outcome(command, last_outcome, error=last_error, duration_ms=elapsed_ms())
        return ToolResult(
            call=call, error=last_error, outcome=last_outcome,
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
        # The context is the first parameter whatever it is called, so it
        # goes in positionally.
        positional = (ctx,) if capability.wants_ctx else ()
        if inspect.iscoroutinefunction(handler):
            return await handler(*positional, **args)
        return await asyncio.to_thread(handler, *positional, **args)

    # -- output policy --------------------------------------------------------

    def _observation_for(
        self, capability: Capability, text: str, value: Any, store: ArtifactStore,
    ) -> tuple[str, Optional[str]]:
        """``text`` is ``value`` already serialized by the caller."""
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
