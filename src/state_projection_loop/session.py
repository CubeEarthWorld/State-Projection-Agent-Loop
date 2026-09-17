"""The agent loop: project → decide → validate → authorize → execute →
record → continue/wait/complete.

``Session`` is the conversation container; ``Run`` (``session.run``) is the
unit of resumable execution it drives. Everything the model sees each turn
is re-projected from the Event Ledger with fidelity-graded compression —
there is no separately maintained conversation list. The ledger IS the
truth; the projection is a disposable window over it.

Two correctness properties enforced here that a naive loop gets wrong:

* **P0-3** — completion (``Decision.finish``) is validated *before* any
  side-effecting call in the same decision executes; a decision that mixes
  the two is rejected outright, never partially honored.
* **P0-4** — at most one in-flight turn per session. A second concurrent
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
from .events import Event, EventLedger, InMemoryLedger, JsonlLedger, ObservedLedger, Snapshot, RENDERABLE_TYPES, event_to_message
from .ids import new_id
from .llm import FINISH_SCHEMA, LLMAdapter, extract_finish
from .messages import ASSISTANT, Message, SYSTEM, ToolCall, USER
from .policy import PolicyEngine
from .context import TurnContext
from .projection import Projection, Section, build_default_sections
from .registry import Registry
from .run import ApprovalRequest, PendingQuestion, Run, RunStateError
from .json_schema import validate_value
from .runtime import WAITING_OUTCOMES, BudgetState, Runtime
from .tokens import estimate_tokens
from .working_state import WORKING_STATE_FIELDS, WorkingState



class ConcurrencyError(RuntimeError):
    """Raised when a second turn is attempted on a session with one already
    in flight (P0-4). Sessions are single-writer by design; run concurrent
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

        artifacts_dir = Path(self.config.artifacts.directory) if self.config.artifacts.directory else None
        self.store = ArtifactStore(self.run.id, directory=artifacts_dir)
        self.search = ToolSearch(self.registry, embedder=embedder, vector=self.config.discovery.vector)

        self._kernel_text = kernel
        if sections is None:
            sections = build_default_sections(
                self.config.projection.sections, kernel_text=kernel,
            )
        self.projection = Projection(sections, window_tokens=self.config.projection.window_tokens)
        self.runtime = Runtime(self.registry, self.config)

        if _restored is None:
            # Typed fields go through the same parser snapshots use, so a
            # seeded `decisions` becomes RecordedDecision objects rather than
            # raw dicts that blow up on the next to_dict(). Anything else is
            # app-specific state and lands in `extra`, the documented escape hatch.
            seed = dict(seed or {})
            known = {k: v for k, v in seed.items() if k in WORKING_STATE_FIELDS}
            self.working_state = WorkingState.from_dict(known)
            self.working_state.extra.update(
                {k: v for k, v in seed.items() if k not in WORKING_STATE_FIELDS}
            )
            self.budget = BudgetState()
        else:
            self.working_state = WorkingState.from_dict(_restored.state.get("working_state") or {})
            # Plans changed after the snapshot are in the ledger.
            for event in self.ledger.iter_run(self.run.id, after=_restored.sequence):
                if event.type == "checklists_changed":
                    self.working_state.checklists = ChecklistStore.from_dict(event.data["checklists"])
            self.budget = BudgetState.from_dict(_restored.state.get("budget") or {})

        # Recently used non-pinned tools (an LRU). Pinned capabilities are
        # added by _api_tools straight from the registry, so they are never
        # tracked here and can never be evicted.
        self._active: "OrderedDict[str, None]" = OrderedDict()
        self._interrupted = False
        self._idle_turns = 0
        self._budget_grace_used = False
        self._lock = asyncio.Lock()
        if _restored is None:
            self.ledger.append(self.run.id, "run_state_changed", {"from": "RUNNING", "to": "RUNNING", "reason": "created"})
            self._snapshot()

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
        msgs: list[Message] = []
        for event in self.ledger.iter_run(self.run.id):
            msg_dict = event_to_message(event)
            if msg_dict is not None:
                msgs.append(Message.from_dict(msg_dict))
        return msgs

    def send(self, text: str) -> Any:
        _ensure_no_running_loop()
        return asyncio.run(self.asend(text))

    async def asend(self, text: str) -> Any:
        async with self._guarded():
            self.ledger.append(self.run.id, "user_input", {"text": text})
            self._checkpoint()
            return await self._loop()

    def run_job(self, task: str) -> Any:
        _ensure_no_running_loop()
        return asyncio.run(self.arun_job(task))

    async def arun_job(self, task: str) -> Any:
        async with self._guarded():
            self.ledger.append(self.run.id, "user_input", {"text": task})
            self._checkpoint()
            return await self._loop()

    def interrupt(self) -> None:
        self._interrupted = True

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
        return asyncio.run(self.aresume())

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
        return asyncio.run(self.ainvoke(capability_name, **arguments))

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
            self.llm, kernel=self._kernel_text, config=copy.deepcopy(self.config), registry=self.registry,
            embedder=getattr(self.search, "embedder", None),
            spawn_llm_factory=self.spawn_llm_factory, policy=self.policy,
        )
        new_session.working_state = copy.deepcopy(self.working_state)
        renderable = [e for e in self.ledger.iter_run(self.run.id) if e.type in RENDERABLE_TYPES]
        cut = len(renderable) if at_message is None else at_message
        for event in renderable[:cut]:
            new_session.ledger.append(new_session.run.id, event.type, dict(event.data))
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
        cancelled, working_state is restored from the checkpoint at the rewind
        point, and the budget is reset.
        """
        irreversible = self._irreversible_effects(up_to_turn=to_turn)
        # One pass: keep renderable events before the to_turn-th user input,
        # and restore the working state from the checkpoint written right
        # after it.
        kept_renderable: list[Event] = []
        restored_ws = WorkingState()
        user_count = 0
        cut = False
        for event in self.ledger.iter_run(self.run.id):
            if not cut and event.type == "user_input":
                cut = user_count == to_turn
                user_count += 1
            if not cut:
                if event.type in RENDERABLE_TYPES:
                    kept_renderable.append(event)
            elif event.type == "checkpoint":
                restored_ws = WorkingState.from_dict(event.data.get("working_state") or {})
                break

        old_run_id = self.run.id
        self.ledger.append(old_run_id, "rewound", {"to_turn": to_turn, "kept_messages": len(kept_renderable)})
        if self.run.state not in ("COMPLETED", "FAILED", "CANCELLED"):
            self.run.cancel(f"rewound to turn {to_turn}")

        self.run = Run(new_id("run"), self.session_id, self.ledger)
        for event in kept_renderable:
            self.ledger.append(self.run.id, event.type, dict(event.data))
        self.ledger.append(self.run.id, "checkpoint", {"working_state": restored_ws.to_dict()})

        self.working_state = restored_ws
        self.budget = BudgetState()
        self._idle_turns = 0
        self._budget_grace_used = False
        self._active = OrderedDict()
        self.runtime.reset()
        self._snapshot()

        return irreversible

    def _irreversible_effects(self, *, up_to_turn: Optional[int] = None) -> list[str]:
        """External effects this run already committed — a sent email, a
        pushed commit. Neither branching nor rewinding can undo them, so both
        report them; ``up_to_turn`` stops the scan at the cut point.
        """
        notices: list[str] = []
        user_count = 0
        for event in self.ledger.iter_run(self.run.id):
            if event.type == "user_input" and up_to_turn is not None:
                if user_count >= up_to_turn:
                    break
                user_count += 1
            if event.type != "command_completed":
                continue
            command = self.run.commands.get(event.data.get("command_id", ""))
            if command is None:
                continue
            capability = self.registry.get(command.capability_name.rsplit("@", 1)[0])
            if capability and any(e.kind == "external" for e in capability.effects):
                notices.append(f"{capability.qualified_name} (command {command.id}) already ran and cannot be undone")
        return notices

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
            **self.run.to_snapshot_state(),
        }
        self.ledger.save_snapshot(Snapshot(
            run_id=self.run.id, sequence=self.ledger.last_sequence(self.run.id), ts=time.time(), state=state,
        ))

    # -- concurrency guard (P0-4) ---------------------------------------------

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
                return self._last_assistant_text() or "[interrupted]"

            stop = self._enforce_budget()
            if stop is not None:
                self._snapshot()
                return stop

            ctx, messages = self._project()
            if await self._fold(ctx, messages):
                # From scratch: shrinking the first rendering consumed its
                # candidates and schemas, and the fold may have moved the goal.
                ctx, messages = self._project()
            self.ledger.append(self.run.id, "projection_compiled", {
                "tokens": estimate_tokens(messages), "messages": len(messages),
                "candidates": [s.tool.name for s in ctx.candidates],
            })

            decision = extract_finish(await self.llm.complete(messages, ctx.api_tools or None))
            self.budget.note_decision(decision, messages, ctx.api_tools, self.config)
            for call in decision.calls:
                call.name = self.registry.resolve_api_name(call.name)
            self.budget.steps += 1
            self.ledger.append(self.run.id, "model_response", {
                "text": decision.text, "finish": decision.finish,
                "calls": [{"name": c.name, "arguments": c.arguments, "id": c.id} for c in decision.calls],
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
                schema = self.config.result_schema
                error = validate_value(schema, decision.result) if schema else None
                if error is not None:
                    self.ledger.append(self.run.id, "decision_validated", {"ok": False, "reason": f"result_schema: {error}"})
                    self._notice(f"[runtime] finish(result) rejected: {error}. Fix the result and call finish again.")
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

    def _project(self) -> tuple[TurnContext, list[Message]]:
        ctx = self._context()
        cfg = self.config.projection
        messages = self.projection.render(
            ctx, api_tools=self._api_tools(ctx),
            reserved_tokens=cfg.reserved_output_tokens + cfg.provider_overhead_tokens,
        )
        return ctx, messages

    async def _fold(self, ctx: TurnContext, messages: list[Message]) -> bool:
        """Compaction: when the prompt exceeds ``compaction.trigger_ratio`` of
        the window, fold history older than the full-fidelity window into the
        working state with one model call. Returns True when the projection
        must be re-rendered."""
        ratio = self.config.compaction.trigger_ratio
        if ratio <= 0:
            return False
        used = estimate_tokens(messages) + self.projection.schema_tokens(ctx.api_tools)
        if used <= ratio * self.config.projection.window_tokens:
            return False
        events = [e for e in self.ledger.iter_run(self.run.id) if e.type in RENDERABLE_TYPES]
        keep = self.config.compression.full_window
        foldable = [e for e in events[:max(0, len(events) - keep)] if e.sequence > self.working_state.folded_sequence]
        if not foldable:
            return False
        lines = []
        for e in foldable:
            m = event_to_message(e)
            if m is not None:
                lines.append(f"{m['role']}: {m.get('content', '')}")
        prompt = [Message(role=SYSTEM, content=FOLD_INSTRUCTIONS), Message(role=USER, content="\n".join(lines))]
        decision = await self.llm.complete(prompt)
        self.budget.steps += 1
        self.budget.note_decision(decision, prompt, [], self.config)
        delta = parse_fold_reply(decision.text)
        before = self.working_state.to_dict()
        error = "reply was not a JSON object" if delta is None else apply_fold_delta(self.working_state, delta)
        if error is not None:
            self._notice(f"[runtime] compaction skipped: {error}")
            return False
        self.working_state.folded_sequence = foldable[-1].sequence
        self.ledger.append(self.run.id, "state_folded", {
            "through_sequence": self.working_state.folded_sequence, "before": before, "delta": delta,
        })
        return True

    def _apply_batch(self, batch, *, record: bool = True) -> None:

        for result in batch.results:
            # A call parked on an approval has no result yet. Recording a
            # placeholder observation would either be overwritten by the real
            # one on resume (two results for one call) or stand in for a call
            # that never ran; instead the whole decision stays out of the
            # projection until it completes — see pair_tool_calls.
            if record and result.outcome not in WAITING_OUTCOMES:
                self._observe(result.call.id, result.call.name, result.observation)
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
            self._notice(f"[runtime] Budget exceeded: {reason}. Wrap up now with a final answer{hint}.")
            return None
        if self.config.mode == "job":
            if self.run.state not in ("COMPLETED", "FAILED", "CANCELLED"):
                self.run.fail(f"budget_stop: {reason}")
            return self.run.result if self.run.result is not None else self._last_assistant_text()
        return self._last_assistant_text() or "[budget exhausted]"

    def _context(self) -> TurnContext:
        return TurnContext(
            config=self.config, registry=self.registry, ledger=self.ledger, run=self.run,
            working_state=self.working_state, session=self, store=self.store, search=self.search,
            candidates=self._layer2_candidates(),
        )

    def _layer2_candidates(self) -> list[ScoredTool]:
        query = "\n".join(q for q in self._candidate_queries() if q)
        if not query:
            return []
        pinned = {c.name for c in self.registry.pinned()}
        return self.search.search(query, k=self.config.discovery.k, layer=2, exclude=pinned)

    def _candidate_queries(self) -> list[str]:
        parts: list[str] = []
        for source in self.config.discovery.query_sources:
            if source == "last_user_message":
                parts.append(self._last_text(USER))
            elif source == "last_model_thought":
                parts.append(self._last_text(ASSISTANT))
            elif source == "goal_if_exists":
                parts.append(self.working_state.goal)
        return parts

    def _last_text(self, role: str) -> str:
        events = [e for e in self.ledger.iter_run(self.run.id) if e.type in RENDERABLE_TYPES]
        for event in reversed(events):
            msg_dict = event_to_message(event)
            if msg_dict and msg_dict.get("role") == role:
                content = msg_dict.get("content", "")
                if isinstance(content, str) and content:
                    return content
        return ""

    def _last_assistant_text(self) -> str:
        return self._last_text(ASSISTANT)

    def _api_tools(self, ctx: TurnContext) -> list[dict]:
        names: "OrderedDict[str, None]" = OrderedDict()
        for capability in self.registry.pinned():
            names[capability.name] = None
        for scored in ctx.candidates:
            names[scored.tool.name] = None
        for name in self._active:
            names[name] = None
        schemas = [self.registry.get(n).api_schema() for n in names if n in self.registry]
        if self.config.mode == "job":
            schemas.append(FINISH_SCHEMA)
        return schemas

    def _activate(self, name: str) -> None:
        self._active[name] = None
        self._active.move_to_end(name)
        while len(self._active) > self.config.discovery.active_tools:
            self._active.popitem(last=False)

    def activate(self, names: Iterable[str]) -> None:
        """Mark tools recently used so their schemas are sent natively next turn."""
        for name in names:
            self._activate(name)

    def _handle_text_only(self, decision) -> Any:
        if self.config.mode == "chat":
            return decision.text
        self._idle_turns += 1
        if self._idle_turns > self.config.limits.max_idle_turns:
            self.ledger.append(self.run.id, "run_state_changed",
                                {"from": self.run.state, "to": self.run.state, "reason": "gave_up_text_only"})
            return decision.text
        self._notice(
            "[runtime] No tool was called. Continue working with tools, "
            "or call finish(result) to finish the job."
        )
        return _CONTINUE

    def _observe(self, call_id: str, name: str, text: str) -> None:
        self.ledger.append(self.run.id, "observation", {"call_id": call_id, "name": name, "text": text})

    def _notice(self, text: str) -> None:
        self.ledger.append(self.run.id, "notice", {"text": text})

    def _checkpoint(self) -> None:
        self.ledger.append(self.run.id, "checkpoint", {"working_state": self.working_state.to_dict()})


class _Continue:
    """Sentinel: the loop should keep going."""


_CONTINUE = _Continue()
