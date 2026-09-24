"""The agent loop: project → decide → validate → authorize → execute →
record → continue/wait/complete.

``Session`` is the conversation container; ``Run`` (``session.run``) is the
unit of resumable execution it drives. Everything the model sees each turn
is re-projected from the Event Ledger with fidelity-graded compression —
there is no separately maintained conversation list. The ledger IS the
truth; the projection is a disposable window over it.

Two correctness properties enforced here that a naive loop gets wrong:

* completion (``Decision.finish``) is validated *before* any
  side-effecting call in the same decision executes; a decision that mixes
  the two is rejected outright, never partially honored.
* at most one in-flight turn per session. A second concurrent
  ``asend``/``arun_job`` raises :class:`ConcurrencyError` immediately
  instead of interleaving state.
"""
from __future__ import annotations

import asyncio
import copy
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .artifacts import ArtifactStore
from .builtin import DEFAULT_BUILTINS, install_builtins
from .checklists import ChecklistStore
from .compaction import FOLD_INSTRUCTIONS, apply_fold_delta, parse_fold_reply
from .config import Config
from .discovery import ScoredTool, ToolSearch
from .embeddings import EmbeddingBackend
from .events import Event, EventLedger, InMemoryLedger, JsonlLedger, ObservedLedger, Snapshot, RENDERABLE_TYPES, renderable
from .ids import new_id
from .llm import FINISH_SPEC, LLMAdapter, extract_finish
from .memory import JsonlMemoryStore, MemoryStore
from .messages import ASSISTANT, Decision, Message, SYSTEM, ToolCall, USER
from .policy import PolicyEngine
from .context import TurnContext
from .projection import Projection, Section, build_default_sections
from .registry import Registry
from .run import TERMINAL_STATES, ApprovalRequest, PendingQuestion, Run, RunStateError
from .json_schema import validate_value
from .runtime import WAITING_OUTCOMES, BudgetState, Hooks, Runtime
from .working_state import WORKING_STATE_FIELDS, WorkingState


class ConcurrencyError(RuntimeError):
    """Raised when a second turn is attempted on a session with one already
    in flight. Sessions are single-writer by design; run concurrent
    conversations as separate Sessions."""


def _ensure_no_running_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        "Session.send/run_job cannot be called from inside an event loop; "
        "use asend/arun_job instead"
    )


def _last_reason(session: "Session") -> str:
    """Why a run ended, from the last transition it recorded."""
    reason = ""
    for event in session.ledger.iter_run(session.run.id):
        if event.type == "run_state_changed":
            reason = event.data.get("reason") or ""
    return reason


def _make_ledger(config: Config) -> EventLedger:
    if config.persistence.ledger_directory:
        return JsonlLedger(config.persistence.ledger_directory)
    return InMemoryLedger()


class Session:
    def __init__(
        self,
        llm: LLMAdapter,
        *,
        kernel: str = "",
        config: Optional[Config] = None,
        registry: Optional[Registry] = None,
        seed: Optional[dict[str, Any]] = None,
        policy: Optional[PolicyEngine] = None,
        embedder: Optional[EmbeddingBackend] = None,
        sections: Optional[list[Section]] = None,
        spawn_llm_factory: Optional[Callable[[Optional[str]], LLMAdapter]] = None,
        ledger: Optional[EventLedger] = None,
        builtins: Iterable[str] = DEFAULT_BUILTINS,
        on_event: Optional[Callable[[Event], None]] = None,
        on_delta: Optional[Callable[[str, str], None]] = None,
        hooks: Optional[Hooks] = None,
        memory: Optional[MemoryStore] = None,
        _restored: Optional[Snapshot] = None,
    ) -> None:
        """``_restored`` is :meth:`resume_from_ledger`'s way in: the session
        continues that snapshot's run instead of starting one."""
        self.config = config or Config()
        self.llm = llm
        self.spawn_llm_factory = spawn_llm_factory
        self.registry = registry if registry is not None else Registry()
        install_builtins(self.registry, builtins)

        base = ledger if ledger is not None else _make_ledger(self.config)
        self.ledger = base if on_event is None else ObservedLedger(base, on_event)
        if _restored is None:
            self.session_id = new_id("session")
            self.run = Run(new_id("run"), self.session_id, self.ledger)
        else:
            self.run = Run.from_snapshot_state(_restored.run_id, self.ledger, _restored.state)
            self.session_id = self.run.session_id

        self.policy = policy if policy is not None else self._default_policy()
        # Cross-session notes (the `memory` pack). Beside the ledger when the
        # session persists, in process memory otherwise.
        ledger_dir = self.config.persistence.ledger_directory
        self.memory = memory if memory is not None else JsonlMemoryStore(
            Path(ledger_dir) / "memory.jsonl" if ledger_dir else None)

        artifacts_dir = Path(self.config.artifacts.directory) if self.config.artifacts.directory else None
        self.store = ArtifactStore(self.run.id, directory=artifacts_dir)
        self.search = ToolSearch(self.registry, embedder=embedder, vector=self.config.discovery.vector)

        # What branch() hands to the new session: code, not state, so it is
        # passed on rather than rebuilt from defaults.
        self._branch_args = dict(kernel=kernel, sections=sections, builtins=builtins, on_event=on_event,
                                 hooks=hooks, on_delta=on_delta, memory=self.memory)
        if sections is None:
            sections = build_default_sections(
                self.config.projection.sections, kernel_text=kernel,
            )
        self.projection = Projection(sections, window_tokens=self.config.projection.window_tokens)
        self.runtime = Runtime(self.registry, self.config, hooks=hooks)
        # ``on_delta(source, text)`` sees assistant text as it streams in
        # (source "model") and tool progress from ctx.emit (source "tool").
        # Delivery only: the ledger still records whole turns.
        self.on_delta = on_delta
        self._inflight: Optional["asyncio.Future[Decision]"] = None

        if _restored is None:
            # Typed fields go through the same parser snapshots use, so a
            # seeded `decisions` becomes RecordedDecision objects rather than
            # raw dicts that blow up on the next to_dict(). Anything else is
            # app-specific state and lands in `extra`, the documented escape hatch.
            seed = dict(seed or {})
            self.working_state = WorkingState.from_dict(
                {k: v for k, v in seed.items() if k in WORKING_STATE_FIELDS})
            self.working_state.extra.update(
                {k: v for k, v in seed.items() if k not in WORKING_STATE_FIELDS})
            self.budget = BudgetState()
        else:
            self.working_state = WorkingState.from_dict(_restored.state.get("working_state") or {})
            # Plans changed after the snapshot are in the ledger.
            for event in self.ledger.iter_run(self.run.id, after=_restored.sequence):
                if event.type == "checklists_changed":
                    self.working_state.checklists = ChecklistStore.from_dict(event.data["checklists"])
            self.budget = BudgetState.from_dict(_restored.state.get("budget") or {})

        # The non-pinned tools whose schemas go out natively, in the order
        # each was first sent, and the same names least recently used or
        # offered first. Pinned capabilities are added by _api_tools straight
        # from the registry, so they are never tracked here and can never be
        # evicted. Part of the snapshot: the tools array is the front of
        # most providers' cached prefix, so a resumed run must send the same one.
        self._native: list[str] = []
        self._recency: "OrderedDict[str, None]" = OrderedDict()
        if _restored is not None:
            self._restore_tools(_restored.state.get("tools"))
        self._interrupted = False
        # This run's sub-agents, until they are collected. A blocking spawn
        # adds and removes them around its own command; a background one
        # leaves them here for the loop head to tend.
        self.children: list["Session"] = []
        # The task driving this session unattended (a background sub-agent).
        # None means nobody is running its loop right now.
        self._driver: Optional["asyncio.Future[None]"] = None
        self._idle_turns = 0
        self._budget_grace_used = False
        self._lock = asyncio.Lock()
        if _restored is None:
            self.ledger.append(self.run.id, "run_state_changed", {"from": "RUNNING", "to": "RUNNING", "reason": "created"})
            self._snapshot()
        else:
            self._reattach_background()

    @staticmethod
    def _default_policy() -> PolicyEngine:
        engine = PolicyEngine(default_decision="require_approval")
        engine.apply_preset("auto_safe")
        return engine

    # -- public API -------------------------------------------------------------

    @property
    def checklists(self) -> ChecklistStore:
        return self.working_state.checklists

    @property
    def conversation(self) -> list[Message]:
        """Derived view of renderable ledger events as Messages. Read-only;
        the ledger is the source of truth, this is a convenience accessor."""
        return [message for _, message in renderable(self.ledger, self.run.id)]

    def send(self, content: Any) -> Any:
        """``content`` is a string, or a list of content parts (``{"type":
        "text", ...}``, ``{"type": "image_url", ...}``) passed through to the
        adapter as the user message."""
        _ensure_no_running_loop()
        return asyncio.run(self._parked(self.asend(content)))

    async def asend(self, content: Any) -> Any:
        async with self._guarded():
            self.ledger.append(self.run.id, "user_input", {"text": content})
            self._checkpoint()
            return await self._loop()

    # A job is started the way a chat turn is; ``config.mode`` is what makes
    # it run until finish(result).
    run_job = send
    arun_job = asend

    def interrupt(self) -> None:
        """Stop after the current step. A model call still waiting for the
        provider is cancelled outright; a tool that is already running
        finishes, so its outcome is recorded. Running sub-agents stop at
        their own next step boundary, still ``RUNNING`` and resumable."""
        self._interrupted = True
        inflight = self._inflight
        if inflight is not None:
            inflight.get_loop().call_soon_threadsafe(inflight.cancel)
        for child in list(self.children):
            child.interrupt()

    def cancel(self, reason: str = "cancelled") -> None:
        """End this run for good, sub-agents and all. For abandoning a run
        parked on an approval or a question - :meth:`interrupt` only stops a
        loop that is moving."""
        self._cancel_children(reason)
        self.run.cancel(reason)
        self._snapshot()

    async def park(self) -> None:
        """Stop every sub-agent at its next step boundary and wait for it.

        A parked child is ``RUNNING`` in the ledger with a snapshot at the
        boundary it stopped at, so the next turn - or the next process -
        picks it up. The synchronous API parks before the event loop it made
        is torn down; an async host calls this before it exits."""
        for child in list(self.children):
            child.interrupt()
        for child in list(self.children):
            driver = child._driver
            if driver is not None:
                await asyncio.gather(driver, return_exceptions=True)
            await child.park()
            # The interrupt was ours and it has done its job. Leaving the
            # flag set would eat the child's first step when it resumes.
            child._interrupted = False

    async def _parked(self, coro: Any) -> Any:
        """Nothing outlives the loop ``asyncio.run`` made for one call."""
        try:
            return await coro
        finally:
            await self.park()

    def _drive(self, task: Optional[Any] = None) -> None:
        """Run this session's loop unattended: the one way a sub-agent moves
        without its parent awaiting it. Fresh with ``task``, otherwise a
        resume. A no-op unless the run is RUNNING and nobody drives it."""
        if self._driver is not None or self.run.state != "RUNNING":
            return

        async def go() -> None:
            try:
                await (self.arun_job(task) if task is not None else self.aresume())
            except asyncio.CancelledError:
                raise  # the loop is going away; the run stays RUNNING, resumable
            except Exception as exc:  # noqa: BLE001 - a driver must not lose its error
                if self.run.state not in TERMINAL_STATES:
                    self.cancel(f"{type(exc).__name__}: {exc}")
            finally:
                self._driver = None

        self._driver = asyncio.ensure_future(go())

    def _cancel_children(self, reason: str) -> None:
        for child in list(self.children):
            child.interrupt()
            if child.run.state not in TERMINAL_STATES:
                child.cancel(reason)
        self.children.clear()

    def _fail(self, reason: str) -> None:
        """Fail this run. Never leaves a sub-agent running behind it."""
        self._cancel_children(reason)
        self.run.fail(reason)

    def _tend_children(self) -> None:
        """Collect and re-drive sub-agents, once per step.

        The loop head is the only safe place: it is always after a batch has
        been applied and snapshotted and before the next projection, so a
        notice can never land between an assistant's tool calls and their
        results - a message sequence no provider accepts."""
        for child in list(self.children):
            if child.run.state == "WAITING_FOR_USER":
                child.cancel("a sub-agent has no user to ask")
            if child.run.state in TERMINAL_STATES:
                self.budget.note_usage(child.budget.prompt_tokens,
                                       child.budget.completion_tokens, self.config)
                detail = "" if child.run.state == "COMPLETED" else f" ({_last_reason(child)})"
                self.notice(
                    f"[runtime] sub-agent {child.run.id} finished: {child.run.state}{detail}. "
                    f'Call meta.agent.join(run_ids=["{child.run.id}"]) for its result.',
                    child_run_id=child.run.id,
                )
                self.children.remove(child)
            else:
                child._drive()

    def _reattach_background(self) -> None:
        """After a restart, pick background sub-agents back up out of the
        ledger. One already announced by a notice was collected; the rest are
        this run's again, and the next loop head drives or reports them."""
        from .builtin.meta import child_session

        announced = {e.data["child_run_id"] for e in self.ledger.iter_run(self.run.id)
                     if e.type == "notice" and e.data.get("child_run_id")}
        for event in self.ledger.iter_run(self.run.id):
            if event.type != "run_spawned" or not event.data.get("background"):
                continue
            command = self.run.commands.get(event.data["command_id"])
            specs = (command.arguments.get("tasks") or []) if command else []
            for spec, run_id in zip(specs, event.data["child_run_ids"]):
                snapshot = self.ledger.load_snapshot(run_id)
                if run_id in announced or snapshot is None:
                    continue
                self.children.append(child_session(self, spec, 1, restored=snapshot))

    def notice(self, text: str, **data: Any) -> None:
        """Put out-of-band text into the run's context.

        For what the host did outside the loop and the model must still know
        about: a command the user typed that ran locally, a skill loaded on
        demand (``session.notice(session.invoke("skill.foo.load"))``), a file
        that was attached. It renders as a system message, costs no turn and
        calls no model — what the *user* said goes through :meth:`send`.
        """
        self.ledger.append(self.run.id, "notice", {"text": text, **data})

    def add_section(self, section: Section, *, before: str = "candidates") -> None:
        self.projection.insert_before(before, section)

    # -- approval lifecycle -------------------------------------------------

    def resolve_approval(self, decision: str) -> ApprovalRequest:
        return self.run.resolve_approval(decision, current_policy_revision=self.policy.revision)

    def answer(self, text: str) -> PendingQuestion:
        """Answer the question the model asked through ``meta.user.ask``; the
        run is ``RUNNING`` again and continues with :meth:`resume`."""
        question = self.run.answer(text)
        self._observe(question.call_id, "meta.user.ask", text)
        self._snapshot()
        return question

    @property
    def _pending(self) -> Any:
        return self.run.pending_approval or self.run.pending_question

    def resume(self) -> Any:
        _ensure_no_running_loop()
        return asyncio.run(self._parked(self.aresume()))

    async def aresume(self) -> Any:
        async with self._guarded():
            if self.run.state != "RUNNING":
                raise RunStateError(f"Run {self.run.id} is not resumable from state {self.run.state}")
            batch = await self.runtime.resume_pending(self.run, self._context(), self.policy)
            self._apply_batch(batch)
            self._snapshot()
            if batch.halted:
                return self._pending
            return await self._loop()

    # -- direct invocation ---------------------------------------------------

    def invoke(self, capability_name: str, **arguments: Any) -> Any:
        _ensure_no_running_loop()
        return asyncio.run(self._parked(self.ainvoke(capability_name, **arguments)))

    async def ainvoke(self, capability_name: str, **arguments: Any) -> Any:
        async with self._guarded():
            call = ToolCall(name=capability_name, arguments=arguments)
            batch = await self.runtime.execute([call], self._context(), self.run, self.policy)
            self._apply_batch(batch, record=False)
            self._snapshot()
            if batch.halted:
                return self._pending
            result = batch.results[0]
            if not result.ok:
                raise RuntimeError(result.observation or result.error or "invoke failed")
            return result.value

    # -- branching -------------------------------------------------------------

    def branch(self, *, at_message: Optional[int] = None) -> tuple["Session", list[str]]:
        new_session = Session(
            self.llm, config=copy.deepcopy(self.config), registry=self.registry, embedder=self.search.embedder,
            spawn_llm_factory=self.spawn_llm_factory, policy=self.policy, **self._branch_args,
        )
        new_session.working_state = copy.deepcopy(self.working_state)
        new_session._native = list(self._native)
        new_session._recency = OrderedDict(self._recency)
        # The first `cut` messages, and the checkpoints among them so the
        # branch can be rewound like any other run.
        events = [event for event, _ in renderable(self.ledger, self.run.id)]
        cut = len(events) if at_message is None else at_message
        kept = len(events[:cut])
        stop = events[kept].sequence if kept < len(events) else None
        moved = new_session._copy_events([
            event for event in self.ledger.iter_run(self.run.id)
            if (stop is None or event.sequence < stop)
            and (event.type in RENDERABLE_TYPES or event.type == "checkpoint")
        ])
        _carry_boundaries(new_session.working_state, moved, new_session._next_sequence())
        new_session.ledger.append(new_session.run.id, "branch_created", {
            "parent_run_id": self.run.id, "parent_session_id": self.session_id, "at_message": cut,
        })
        new_session._snapshot()
        return new_session, self._irreversible_effects()

    def rewind(self, *, to_turn: int) -> list[str]:
        """Destructive rewind: cancel the current run and replace it in-place
        with a new run containing only events up to ``to_turn`` (counted in
        user-input turns, 0-indexed). The session continues as if everything
        after that turn never happened.

        Returns a list of irreversible external effects that already executed
        and cannot be undone (e.g. a sent email). The caller should surface
        these to the user.

        Unlike :meth:`branch`, this mutates the session: the old run is
        cancelled, working_state and the native tool list are restored from
        the checkpoint at the rewind point, and the budget is reset. The kept
        events (checkpoints included) are renumbered in the new run, and the
        restored history boundaries are re-pointed at them, so the kept
        history renders exactly as it did before.

        ``to_turn`` must name an existing turn: past the last one there is no
        checkpoint to restore from, and rewinding anyway would keep the whole
        history while resetting the working state to an empty one.
        """
        # One pass: the effects committed before the to_turn-th user input,
        # the renderable events to keep, and the working state from the
        # checkpoint written right after it.
        irreversible: list[str] = []
        kept: list[Event] = []  # the renderable events, and the checkpoints among them
        checkpoint: dict[str, Any] = {}  # the one written right after the to_turn-th input
        user_count = 0
        cut = False
        for event in self.ledger.iter_run(self.run.id):
            if not cut and event.type == "user_input":
                cut = user_count == to_turn
                user_count += 1
            if not cut:
                if event.type in RENDERABLE_TYPES or event.type == "checkpoint":
                    kept.append(event)
                elif (effect := self._effect_notice(event)) is not None:
                    irreversible.append(effect)
            elif event.type == "checkpoint":
                checkpoint = event.data
                break
        if not cut:
            raise ValueError(
                f"rewind(to_turn={to_turn}) is out of range: this run has {user_count} user turn(s)"
            )

        old_run_id = self.run.id
        self.ledger.append(old_run_id, "rewound", {
            "to_turn": to_turn, "kept_messages": sum(1 for e in kept if e.type in RENDERABLE_TYPES),
        })
        if self.run.state not in TERMINAL_STATES:
            self.cancel(f"rewound to turn {to_turn}")

        self.run = Run(new_id("run"), self.session_id, self.ledger)
        # The kept checkpoints come along too, so this run can be rewound
        # again to any turn it still has.
        moved = self._copy_events(kept)
        restored_ws = WorkingState.from_dict(checkpoint.get("working_state") or {})
        _carry_boundaries(restored_ws, moved, self._next_sequence())
        self.working_state = restored_ws
        # The native tools go back to what they were when that turn began;
        # a checkpoint written without them (an older one) starts empty.
        self._restore_tools(checkpoint.get("tools"))
        self._checkpoint()

        self.budget = BudgetState()
        self._idle_turns = 0
        self._budget_grace_used = False
        self.runtime.reset()
        self._snapshot()

        return irreversible

    def _copy_events(self, events: list[Event]) -> list[tuple[int, int]]:
        """Append ``events`` to the current run, in order, and return each
        one's ``(old, new)`` sequence. The copies are numbered afresh, so a
        checkpoint copied along has its history boundaries re-pointed at them."""
        moved: list[tuple[int, int]] = []
        for event in events:
            data = dict(event.data)
            if event.type == "checkpoint" and isinstance(data.get("working_state"), dict):
                state = WorkingState.from_dict(data["working_state"])
                _carry_boundaries(state, moved, self._next_sequence())
                data["working_state"] = {**data["working_state"], "verbatim_sequence": state.verbatim_sequence,
                                         "folded_sequence": state.folded_sequence}
            moved.append((event.sequence, self.ledger.append(self.run.id, event.type, data).sequence))
        return moved

    def _next_sequence(self) -> int:
        return self.ledger.last_sequence(self.run.id) + 1

    def _irreversible_effects(self) -> list[str]:
        """External effects this run already committed — a sent email, a
        pushed commit. Neither branching nor rewinding can undo them, so both
        report them (:meth:`rewind` collects its own, up to the cut point).
        """
        return [effect for event in self.ledger.iter_run(self.run.id)
                if (effect := self._effect_notice(event)) is not None]

    def _effect_notice(self, event: Event) -> Optional[str]:
        """The note for one already-committed external effect, or None when
        the event is not one."""
        if event.type != "command_completed":
            return None
        command = self.run.commands.get(event.data.get("command_id", ""))
        if command is None:
            return None
        capability = self.registry.get(command.capability_name.rsplit("@", 1)[0])
        # planned_effects, not effects: a capability that declared none is
        # treated as external everywhere else (policy, runtime), and it is
        # exactly the one whose handler might have done something real
        # without anyone noticing. Reading the raw list here reported it as
        # safe to rewind past.
        if capability is None or not any(e.kind == "external" for e in capability.planned_effects):
            return None
        return f"{capability.qualified_name} (command {command.id}) already ran and cannot be undone"

    # -- process-restart resume ------------------------------------------------

    @classmethod
    def resume_from_ledger(
        cls, llm: LLMAdapter, run_id: str, *, config: Optional[Config] = None, **session_args: Any,
    ) -> "Session":
        """Continue a persisted run in a new process. ``session_args`` are
        :class:`Session`'s own (``kernel``, ``registry``, ``policy``,
        ``sections``, ``builtins``, ...): they are code, not state, so the
        caller passes what the first process passed."""
        config = config or Config()
        if not config.persistence.ledger_directory:
            raise RunStateError("resume_from_ledger requires config.persistence.ledger_directory")
        ledger = JsonlLedger(config.persistence.ledger_directory)
        snapshot = ledger.load_snapshot(run_id)
        if snapshot is None:
            raise RunStateError(f"No snapshot found for run {run_id!r}; nothing to resume")
        return cls(llm, config=config, ledger=ledger, _restored=snapshot, **session_args)

    def _snapshot(self) -> None:
        state = {
            "working_state": self.working_state.to_dict(),
            "budget": self.budget.to_dict(),
            "tools": self._tools_state(),
            **self.run.to_snapshot_state(),
        }
        self.ledger.save_snapshot(Snapshot(
            run_id=self.run.id, sequence=self.ledger.last_sequence(self.run.id), ts=time.time(), state=state,
        ))

    # -- concurrency guard ---------------------------------------------

    def _guarded(self):
        if self._lock.locked():
            raise ConcurrencyError(
                f"Session {self.session_id} (run {self.run.id}) already has a turn in flight; "
                "concurrent send()/run_job()/resume()/invoke() calls are not allowed on one session"
            )
        return self._lock

    # -- loop -----------------------------------------------------------------

    async def _loop(self) -> Any:
        while True:
            if self._interrupted:
                self._interrupted = False
                self.ledger.append(self.run.id, "run_state_changed",
                                    {"from": self.run.state, "to": self.run.state, "reason": "interrupted"})
                # Snapshot at the boundary we stopped at, so a parked
                # sub-agent resumes here and not from an older step.
                self._snapshot()
                return self._last_text(ASSISTANT) or "[interrupted]"

            self._tend_children()
            stop = self._enforce_budget()
            if stop is not None:
                self._snapshot()
                return stop

            ctx, messages = self._project()
            # Both model calls of a step are inside the guard: interrupting
            # the fold must return a value like every other interrupt path,
            # not raise CancelledError at the caller.
            try:
                if await self._fold():
                    # From scratch: shrinking the first rendering consumed its
                    # candidates and schemas, and the fold may have moved the goal.
                    ctx, messages = self._project()
                self.ledger.append(self.run.id, "projection_compiled", {
                    "tokens": self.projection.last_message_tokens, "messages": len(messages),
                    "candidates": [s.tool.name for s in ctx.candidates],
                })
                started = time.monotonic()
                decision = extract_finish(await self._complete(messages, ctx.api_tools or None))
            except asyncio.CancelledError:
                if not self._interrupted:
                    raise
                continue  # interrupt() cancelled the call; the loop head records it
            self.budget.note_decision(decision, messages, ctx.api_tools, self.config)
            for call in decision.calls:
                call.name = self.registry.resolve_api_name(call.name)
            self.ledger.append(self.run.id, "model_response", {
                "text": decision.text, "finish": decision.finish,
                "calls": [c.to_dict() for c in decision.calls],
                "usage": decision.usage.to_dict() if decision.usage else None,
                "latency_ms": int((time.monotonic() - started) * 1000),
            })

            if decision.finish and decision.calls:
                self.ledger.append(self.run.id, "decision_validated", {
                    "ok": False, "reason": "finish combined with tool calls in the same decision",
                })
                for call in decision.calls:
                    self._observe(call.id, call.name,
                                   "Rejected: cannot call finish(result) together with other tools in the "
                                   "same decision. Call finish(result) alone once you are done.")
                continue

            if decision.finish:
                # A run must never reach a terminal state with sub-agent work
                # outstanding: that is how results go missing and how
                # "stopped" sessions leave agents running. Collect whatever
                # finished during the call first, so the only thing that
                # bounces a finish is work that is genuinely unfinished.
                self._tend_children()
                running = [c.run.id for c in self.children]
                if running:
                    self.ledger.append(self.run.id, "decision_validated", {
                        "ok": False, "reason": f"background sub-agents still running: {', '.join(running)}",
                    })
                    self.notice(
                        f"[runtime] finish(result) rejected: sub-agents {', '.join(running)} are still "
                        "running. Call meta.agent.join to wait for them (cancel=true to stop them), "
                        "then finish."
                    )
                    continue
                schema = self.config.result_schema
                error = validate_value(schema, decision.result) if schema else None
                if error is not None:
                    self.ledger.append(self.run.id, "decision_validated", {"ok": False, "reason": f"result_schema: {error}"})
                    self.notice(f"[runtime] finish(result) rejected: {error}. Fix the result and call finish again.")
                    continue
                self.ledger.append(self.run.id, "decision_validated", {"ok": True, "finish": True})
                if self.config.mode == "job":
                    self.run.complete(decision.result)
                    self._snapshot()
                    return self.run.result
                return decision.result if decision.result is not None else decision.text

            if not decision.calls:
                outcome = self._handle_text_only(decision)
                if outcome is not _CONTINUE:
                    self._snapshot()
                    return outcome
                continue

            self._idle_turns = 0
            self.ledger.append(self.run.id, "decision_validated", {"ok": True, "finish": False})
            batch = await self.runtime.execute(decision.calls, ctx, self.run, self.policy)
            self._apply_batch(batch)
            self._snapshot()
            if batch.halted:
                return self._pending

    async def _complete(self, messages: list[Message], tools: Optional[list[dict]]) -> Decision:
        """One model call under ``config.model``: a timeout, retries with
        backoff, cancellation by :meth:`interrupt`. Every failed attempt is a
        ``model_call_failed`` event; when the last one fails, a job run fails
        and the exception reaches the caller."""
        cfg = self.config.model
        # Only a host that observes deltas asks the adapter to stream.
        streaming = {"on_delta": lambda text: self.on_delta("model", text)} if self.on_delta else {}
        for attempt in range(1, cfg.retries + 2):
            self._inflight = asyncio.ensure_future(self.llm.complete(messages, tools, **streaming))
            try:
                return await asyncio.wait_for(self._inflight, timeout=cfg.timeout_s)
            except Exception as exc:  # noqa: BLE001 — every provider error is one attempt
                error = (f"timed out after {cfg.timeout_s}s" if isinstance(exc, asyncio.TimeoutError)
                         else f"{type(exc).__name__}: {exc}")
                self.ledger.append(self.run.id, "model_call_failed", {"attempt": attempt, "error": error})
                if attempt > cfg.retries:
                    if self.config.mode == "job" and self.run.state not in TERMINAL_STATES:
                        self._fail(f"model_error: {error}")
                    self._snapshot()
                    raise
                await asyncio.sleep(cfg.backoff_s * attempt)
            finally:
                self._inflight = None
        raise AssertionError("unreachable")

    def _step_tiers(self) -> None:
        """Move the history's verbatim point forward in steps: only when the
        verbatim tail has grown to four times ``full_window`` is it cut back
        to ``full_window``. Between steps the rendering of every older message
        is unchanged, so the prompt prefix stays byte-identical and a
        provider's cache keeps hitting; a step is one deliberate rebuild."""
        keep = self.config.compression.full_window
        history = renderable(self.ledger, self.run.id)
        tail = sum(1 for event, _ in history if event.sequence >= self.working_state.verbatim_sequence)
        if keep <= 0 or tail <= 4 * keep:
            return
        self.working_state.verbatim_sequence = history[-keep][0].sequence

    def _project(self) -> tuple[TurnContext, list[Message]]:
        self._step_tiers()
        ctx = self._context()
        cfg = self.config.projection
        tools = self._api_tools(ctx)
        messages = self.projection.render(
            ctx, api_tools=tools, reserved_tokens=cfg.reserved_output_tokens + cfg.provider_overhead_tokens)
        self._settle_tools(ctx, tools)
        return ctx, messages

    async def _fold(self) -> bool:
        """Compaction: when the prompt exceeds ``compaction.trigger_ratio`` of
        the window, fold history older than the full-fidelity window into the
        working state with one model call. Returns True when the projection
        must be re-rendered."""
        ratio = self.config.compaction.trigger_ratio
        if ratio <= 0:
            return False
        # The ratio applies to the room the render actually has for
        # messages and schemas: the window less the reserved output. Measured
        # against the whole window it is unreachable once the reserve
        # exceeds the slack, and measured with the reserve counted it fires
        # every turn of a small window.
        cfg = self.config.projection
        room = cfg.window_tokens - cfg.reserved_output_tokens - cfg.provider_overhead_tokens
        used = self.projection.last_message_tokens + self.projection.last_schema_tokens
        if used <= ratio * room:
            return False
        # Fold from the ledger, never from the projection: what masking
        # cleared from the prompt is exactly what a fold must still read.
        # The region is everything before the verbatim point and after the
        # last fold, so a fold happens at most once per step of the point,
        # when the prefix is being rebuilt anyway, and always has a step's
        # worth of messages to work on. Forcing the point down to fold
        # sooner produced a fold every turn under a window the verbatim
        # tail alone overflows, each one a model call and a cache rebuild.
        foldable = [(e, m) for e, m in renderable(self.ledger, self.run.id)
                    if self.working_state.folded_sequence < e.sequence < self.working_state.verbatim_sequence]
        if not foldable:
            return False
        transcript = "\n".join(f"{m.role}: {m.content}" for _, m in foldable)
        prompt = [Message(role=SYSTEM, content=FOLD_INSTRUCTIONS), Message(role=USER, content=transcript)]
        decision = await self._complete(prompt, None)
        self.budget.note_decision(decision, prompt, [], self.config)
        delta = parse_fold_reply(decision.text)
        before = self.working_state.to_dict()
        error = ("reply was not a JSON object" if delta is None
                 else apply_fold_delta(self.working_state, delta, transcript=transcript))
        if error is not None:
            self.notice(f"[runtime] compaction skipped: {error}")
            return False
        self.working_state.folded_sequence = foldable[-1][0].sequence
        self.ledger.append(self.run.id, "state_folded", {
            "through_sequence": self.working_state.folded_sequence, "before": before, "delta": delta,
        })
        # A fold is the one place the working state changes without a tool
        # call behind it, and the event records `before` + `delta` rather
        # than the result — replaying it would need the transcript the
        # grounding check ran against. Snapshot instead, or a restart before
        # the next one silently loses everything the fold merged.
        self._snapshot()
        return True

    def _apply_batch(self, batch, *, record: bool = True) -> None:
        for result in batch.results:
            # A call parked on an approval has no result yet. Recording a
            # placeholder observation would either be overwritten by the real
            # one on resume (two results for one call) or stand in for a call
            # that never ran; instead the whole decision stays out of the
            # projection until it completes — see pair_tool_calls.
            if record and result.outcome not in WAITING_OUTCOMES:
                self._observe(result.call.id, result.call.name, result.observation, ok=result.ok)
            if result.ok:
                self._activate(result.call.name)

    # -- loop helpers -----------------------------------------------------------

    def _enforce_budget(self) -> Optional[Any]:
        reason = self.budget.exceeded(self.config)
        if reason is None:
            return None
        if not self._budget_grace_used:
            self._budget_grace_used = True
            hint = " or call finish(result)" if self.config.mode == "job" else ""
            self.notice(f"[runtime] Budget exceeded: {reason}. Wrap up now with a final answer{hint}.")
            return None
        if self.config.mode == "job":
            if self.run.state not in TERMINAL_STATES:
                self._fail(f"budget_stop: {reason}")
            return self.run.result if self.run.result is not None else self._last_text(ASSISTANT)
        return self._last_text(ASSISTANT) or "[budget exhausted]"

    def _context(self) -> TurnContext:
        return TurnContext(
            config=self.config, registry=self.registry, ledger=self.ledger, run=self.run,
            working_state=self.working_state, session=self, store=self.store, search=self.search,
            candidates=self._layer2_candidates(), emit=self._emit,
        )

    def _emit(self, text: str) -> None:
        if self.on_delta is not None:
            self.on_delta("tool", text)

    def _layer2_candidates(self) -> list[ScoredTool]:
        sources = {"last_user_message": lambda: self._last_text(USER),
                   "last_model_thought": lambda: self._last_text(ASSISTANT),
                   "goal_if_exists": lambda: self.working_state.goal}
        query = "\n".join(
            text for text in (sources[s]() for s in self.config.discovery.query_sources if s in sources) if text
        )
        if not query:
            return []
        pinned = {c.name for c in self.registry.pinned()}
        return self.search.search(query, k=self.config.discovery.k, layer=2, exclude=pinned)

    def _last_text(self, role: str) -> str:
        for _, message in reversed(renderable(self.ledger, self.run.id)):
            if message.role == role and isinstance(message.content, str) and message.content:
                return message.content
        return ""

    def _api_tools(self, ctx: TurnContext) -> list[dict]:
        """The native schemas for this step: pinned ones in registry order,
        then every other native tool in the order it was first sent, then
        ``finish`` in job mode.

        Most providers render the tools array ahead of the whole
        conversation, so it is the front of the cached prefix: a list that
        changes changes everything after it. Nothing here reorders it — not
        this step's candidate ranking (that is the candidates section's
        job, at the tail), not recency. A candidate offered for the first
        time is appended and then stays, so it is callable natively as
        before, and a step whose candidates were all offered already sends
        the same bytes as the step before. The list shrinks only when it
        outgrows ``discovery.active_tools`` (one deliberate rebuild) or the
        window forces it.
        """
        pinned = [c.name for c in self.registry.pinned()]
        offered = [s.tool.name for s in ctx.candidates]
        for name in offered:
            self._activate(name)
        self._evict(keep=set(offered))
        ctx.tool_recency = [c.api_name for c in map(self.registry.get, self._recency) if c is not None]
        schemas = [capability.tool_spec() for capability in map(self.registry.get, dict.fromkeys(pinned + self._native))
                   if capability is not None]
        if self.config.mode == "job":
            schemas.append(FINISH_SPEC)
        return schemas

    def _settle_tools(self, ctx: TurnContext, sent: list[dict]) -> None:
        """Forget the native tools the window made this step leave out, so
        the next step appends them again instead of re-inserting them
        mid-list. A tool left out because it is disabled for now keeps its
        place and returns there when it is enabled again."""
        names = {t.get("name") for t in ctx.api_tools}
        for tool in sent:
            if tool.get("name") not in names:
                name = self.registry.resolve_api_name(tool.get("name", ""))
                if name in self._recency:
                    self._native.remove(name)
                    del self._recency[name]

    def _activate(self, name: str) -> None:
        capability = self.registry.get(name)
        if capability is not None and capability.discovery.pinned:
            return  # always sent, never tracked
        if name not in self._recency:
            self._native.append(name)
        self._recency[name] = None
        self._recency.move_to_end(name)

    def _evict(self, keep: set[str]) -> None:
        """Hold the native tools other than this step's candidates to
        ``discovery.active_tools``, dropping the least recently used or offered."""
        over = sum(1 for name in self._native if name not in keep) - self.config.discovery.active_tools
        for name in [n for n in self._recency if n not in keep][:max(over, 0)]:
            self._native.remove(name)
            del self._recency[name]

    def activate(self, names: Iterable[str]) -> None:
        """Mark tools used so their schemas are sent natively from the next
        step on. A tool not yet in the native list joins it at the end."""
        for name in names:
            self._activate(name)

    @property
    def native_tools(self) -> list[str]:
        """The non-pinned tools whose schemas are sent natively, in the
        order they appear in the tools array (first sent first)."""
        return list(self._native)

    def _tools_state(self) -> dict[str, list[str]]:
        return {"native": list(self._native), "recency": list(self._recency)}

    def _restore_tools(self, state: Any) -> None:
        state = state if isinstance(state, dict) else {}
        self._native = [str(n) for n in state.get("native") or []]
        order = [str(n) for n in state.get("recency") or [] if n in self._native]
        self._recency = OrderedDict.fromkeys([n for n in self._native if n not in order] + order)

    def _handle_text_only(self, decision) -> Any:
        if self.config.mode == "chat":
            return decision.text
        self._idle_turns += 1
        if self._idle_turns > self.config.limits.max_idle_turns:
            self.ledger.append(self.run.id, "run_state_changed",
                                {"from": self.run.state, "to": self.run.state, "reason": "gave_up_text_only"})
            return decision.text
        self.notice(
            "[runtime] No tool was called. Continue working with tools, "
            "or call finish(result) to finish the job."
        )
        return _CONTINUE

    def _observe(self, call_id: str, name: str, text: str, *, ok: bool = True) -> None:
        self.ledger.append(self.run.id, "observation", {"call_id": call_id, "name": name, "text": text, "ok": ok})

    def _checkpoint(self) -> None:
        self.ledger.append(self.run.id, "checkpoint", {
            "working_state": self.working_state.to_dict(), "tools": self._tools_state(),
        })


_CONTINUE = object()  # sentinel: the loop should keep going


def _carry_boundaries(state: WorkingState, moved: list[tuple[int, int]], next_sequence: int) -> None:
    """Re-point ``state``'s history boundaries after its events were copied
    into another run under new sequence numbers (``moved``: ``(old, new)``
    pairs, oldest first). The verbatim point moves to the first copy at or
    after it (``next_sequence`` when none is, so every copy stays older), and
    the fold point to the last copy at or before it. Left as they were, the
    old run's numbers mean different messages in the new run: a verbatim
    point past the copies compressed the whole kept history."""
    state.verbatim_sequence = next((new for old, new in moved if old >= state.verbatim_sequence), next_sequence)
    state.folded_sequence = max((new for old, new in moved if old <= state.folded_sequence), default=0)
