"""The agent loop: project → decide → validate → authorize → execute →
record → continue/wait/complete.

``Session`` is the conversation container; ``Run`` (``session.run``) is the
unit of resumable execution it drives. Everything the model sees each turn
is re-projected from the Event Ledger's derived state (working state +
conversation), never accumulated ad hoc.

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
from typing import Any, Callable, Optional

from .artifacts import ArtifactStore
from .builtin.meta import ensure_meta_tools
from .capability import Capability, ToolContext
from .compaction import Compactor
from .config import Config
from .discovery import ScoredTool, ToolSearch
from .embeddings import EmbeddingBackend
from .events import EventLedger, InMemoryLedger, JsonlLedger, Snapshot
from .ids import new_id
from .llm import FINISH_SCHEMA, LLMAdapter, extract_finish
from .messages import ASSISTANT, Message, OBSERVATION, SYSTEM, USER
from .policy import PolicyEngine
from .projection import Projection, Section, TurnContext, build_default_sections
from .registry import Registry
from .run import ApprovalRequest, Run, RunStateError
from .runtime import BudgetState, Runtime
from .tokens import estimate_tokens
from .working_state import WorkingState

_ACTIVE_TOOL_CAP = 48


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
        extra_sections: Optional[dict[str, Section]] = None,
        summarizer: Optional[LLMAdapter] = None,
        spawn_llm_factory: Optional[Callable[[Optional[str]], LLMAdapter]] = None,
        ledger: Optional[EventLedger] = None,
    ) -> None:
        self.config = config or Config()
        self.llm = llm
        self.summarizer = summarizer
        self.spawn_llm_factory = spawn_llm_factory
        self.registry = registry if registry is not None else Registry()
        ensure_meta_tools(self.registry)

        self.session_id = new_id("session")
        self.ledger = ledger if ledger is not None else _make_ledger(self.config)
        self.run = Run(new_id("run"), self.session_id, self.ledger)

        self.policy = policy if policy is not None else self._default_policy()
        self.hooks_on_change: list[Callable[[str], None]] = []

        artifacts_dir = Path(self.config.artifacts.directory) if self.config.artifacts.directory else None
        self.store = ArtifactStore(self.run.id, directory=artifacts_dir)
        self.search = ToolSearch(self.registry, embedder=embedder, vector=self.config.discovery.vector)

        self._kernel_text = kernel
        pinned = self.registry.pinned()
        if sections is None:
            sections = build_default_sections(
                self.config.projection.sections, kernel_text=kernel, pinned=pinned, extra=extra_sections,
            )
        self.projection = Projection(sections, window_tokens=self.config.projection.window_tokens)
        self.runtime = Runtime(self.registry, self.store, self.config)
        self.runtime.seen_specs.update(c.name for c in pinned)  # kernel carries their specs
        self.compactor = Compactor(self.config, self._resolve_summarizer())

        self.working_state = WorkingState()
        for key, value in (seed or {}).items():
            if hasattr(self.working_state, key):
                setattr(self.working_state, key, value)
            else:
                self.working_state.extra[key] = value

        self.conversation: list[Message] = []
        self.budget = BudgetState()

        self._active: "OrderedDict[str, None]" = OrderedDict((c.name, None) for c in pinned)
        self._interrupted = False
        self._idle_turns = 0
        self._budget_grace_used = False
        self._lock = asyncio.Lock()
        self.ledger.append(self.run.id, "run_state_changed", {"from": "RUNNING", "to": "RUNNING", "reason": "created"})

    @staticmethod
    def _default_policy() -> PolicyEngine:
        """Out-of-the-box posture: effect-free calls (state/meta tools) and
        workspace reads run automatically; everything else — including any
        custom capability with a write/external effect — requires an
        explicit grant or approval. Callers building a real deployment are
        expected to pass their own :class:`PolicyEngine`."""
        engine = PolicyEngine(default_decision="require_approval")
        engine.apply_preset("auto_safe")
        return engine

    def _resolve_summarizer(self) -> Optional[LLMAdapter]:
        if self.summarizer is not None:
            return self.summarizer
        if self.config.compaction.model == "none":
            return None
        return self.llm

    # -- public API -------------------------------------------------------------

    def send(self, text: str) -> Any:
        """Chat mode: one user message in, the final text reply (or a
        pending :class:`~state_projection_loop.run.ApprovalRequest`) out."""
        _ensure_no_running_loop()
        return asyncio.run(self.asend(text))

    async def asend(self, text: str) -> Any:
        async with self._guarded():
            self.conversation.append(Message(role=USER, content=text))
            self.ledger.append(self.run.id, "user_input", {"text": text})
            return await self._loop()

    def run_job(self, task: str) -> Any:
        """Job mode: run until finish(result), budget exhaustion, interrupt,
        or an approval pause."""
        _ensure_no_running_loop()
        return asyncio.run(self.arun_job(task))

    async def arun_job(self, task: str) -> Any:
        async with self._guarded():
            self.conversation.append(Message(role=USER, content=task))
            self.ledger.append(self.run.id, "user_input", {"text": task})
            return await self._loop()

    def interrupt(self) -> None:
        """Request the loop to stop at the next iteration boundary."""
        self._interrupted = True

    def add_section(self, section: Section, *, before: str = "candidates") -> None:
        self.projection.insert_before(before, section)

    # -- approval lifecycle -------------------------------------------------

    def resolve_approval(self, decision: str) -> ApprovalRequest:
        """Approve or deny the run's pending approval. Does not resume
        execution — call :meth:`resume`/:meth:`aresume` afterward."""
        return self.run.resolve_approval(decision, current_policy_revision=self.policy.revision)

    def resume(self) -> Any:
        _ensure_no_running_loop()
        return asyncio.run(self.aresume())

    async def aresume(self) -> Any:
        """Continue a run paused at ``WAITING_FOR_APPROVAL`` (now resolved)
        or freshly reconstructed via :meth:`Session.resume_from_ledger`."""
        async with self._guarded():
            if self.run.state != "RUNNING":
                raise RunStateError(f"Run {self.run.id} is not resumable from state {self.run.state}")
            turn = self._new_turn()
            batch = await self.runtime.resume_pending(self.run, self._tool_context(), self.policy, turn)
            self._apply_batch(batch)
            self._snapshot()
            if batch.halted:
                return self.run.pending_approval
            return await self._loop()

    # -- direct invocation (bypasses the model, still goes through the same
    #    validate/authorize/execute/record pipeline) -------------------------

    def invoke(self, capability_name: str, **arguments: Any) -> Any:
        _ensure_no_running_loop()
        return asyncio.run(self.ainvoke(capability_name, **arguments))

    async def ainvoke(self, capability_name: str, **arguments: Any) -> Any:
        from .messages import ToolCall

        async with self._guarded():
            turn = self._new_turn()
            call = ToolCall(name=capability_name, arguments=arguments)
            batch = await self.runtime.execute([call], turn, self._tool_context(), self.run, self.policy)
            self._apply_batch(batch, record_conversation=False)
            self._snapshot()
            if batch.halted:
                return self.run.pending_approval
            result = batch.results[0]
            if not result.ok:
                raise RuntimeError(result.observation or result.error or "invoke failed")
            return result.value

    # -- branching (non-destructive rewind) ----------------------------------

    def branch(self, *, at_message: Optional[int] = None) -> tuple["Session", list[str]]:
        """Create a new Session sharing history up to ``at_message`` (default:
        current end). The parent's ledger is never modified or truncated —
        this only ever adds a new run whose own ledger starts with a
        ``branch_created`` event pointing at the parent.

        Returns ``(new_session, irreversible_effects)`` where the second
        element lists external effects already committed by the parent run
        that this branch cannot undo (e.g. a sent email, a git push).
        """
        cut = len(self.conversation) if at_message is None else at_message
        new_session = Session(
            self.llm, kernel=self._kernel_text, config=copy.deepcopy(self.config), registry=self.registry,
            embedder=getattr(self.search, "embedder", None), summarizer=self.summarizer,
            spawn_llm_factory=self.spawn_llm_factory, policy=self.policy,
        )
        new_session.conversation = list(self.conversation[:cut])
        new_session.working_state = copy.deepcopy(self.working_state)
        new_session.ledger.append(new_session.run.id, "branch_created", {
            "parent_run_id": self.run.id, "parent_session_id": self.session_id, "at_message": cut,
        })
        return new_session, self._irreversible_effects()

    def _irreversible_effects(self) -> list[str]:
        notices: list[str] = []
        for event in self.ledger.iter_run(self.run.id):
            if event.type != "command_completed":
                continue
            command = self.run.commands.get(event.data.get("command_id", ""))
            if command is None:
                continue
            capability_name = command.capability_name.rsplit("@", 1)[0]
            capability = self.registry.get(capability_name)
            if capability and any(e.kind == "external" for e in capability.effects):
                notices.append(f"{capability.qualified_name} (command {command.id}) already ran and cannot be undone")
        return notices

    # -- process-restart resume ------------------------------------------------

    @classmethod
    def resume_from_ledger(
        cls, llm: LLMAdapter, run_id: str, *, config: Optional[Config] = None,
        registry: Optional[Registry] = None, policy: Optional[PolicyEngine] = None,
        embedder: Optional[EmbeddingBackend] = None, summarizer: Optional[LLMAdapter] = None,
        spawn_llm_factory: Optional[Callable[[Optional[str]], LLMAdapter]] = None,
    ) -> "Session":
        """Reconstruct a Session from its last snapshot in a NEW process.

        This is what makes a ``WAITING_FOR_APPROVAL`` run survive a process
        restart (P1-2): the caller (a new process, possibly hours later)
        loads the ledger directory, finds the snapshot, and gets back a
        Session ready for :meth:`resolve_approval` + :meth:`resume`.
        """
        config = config or Config()
        if not config.persistence.ledger_directory:
            raise RunStateError("resume_from_ledger requires config.persistence.ledger_directory")
        ledger = JsonlLedger(config.persistence.ledger_directory)
        snapshot = ledger.load_snapshot(run_id)
        if snapshot is None:
            raise RunStateError(f"No snapshot found for run {run_id!r}; nothing to resume")

        session = cls(
            llm, config=config, registry=registry, policy=policy, embedder=embedder,
            summarizer=summarizer, spawn_llm_factory=spawn_llm_factory, ledger=ledger,
        )
        session.run = Run.from_snapshot_state(run_id, ledger, snapshot.state)
        session.session_id = snapshot.state.get("session_id", session.session_id)
        session.working_state = WorkingState.from_dict(snapshot.state.get("working_state") or {})
        session.conversation = [Message.from_dict(d) for d in snapshot.state.get("conversation") or []]
        budget_data = snapshot.state.get("budget") or {}
        session.budget = BudgetState(
            steps=budget_data.get("steps", 0), prompt_tokens=budget_data.get("prompt_tokens", 0),
            completion_tokens=budget_data.get("completion_tokens", 0), cost=budget_data.get("cost", 0.0),
        )
        session.store = ArtifactStore(
            session.run.id, directory=Path(config.artifacts.directory) if config.artifacts.directory else None,
        )
        return session

    def _snapshot(self) -> None:
        state = {
            "session_id": self.session_id,
            "working_state": self.working_state.to_dict(),
            "conversation": [m.to_dict() for m in self.conversation],
            "budget": {
                "steps": self.budget.steps, "prompt_tokens": self.budget.prompt_tokens,
                "completion_tokens": self.budget.completion_tokens, "cost": self.budget.cost,
            },
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

            self._maybe_compact()

            turn = self._new_turn()
            api_tools = self._api_tools(turn)
            turn.dedupe_candidate_cards = self.config.projection.dedupe_candidate_cards_against_schemas
            reserved = self.config.projection.reserved_output_tokens + self.config.projection.provider_overhead_tokens
            messages = self.projection.render(turn, api_tools=api_tools, reserved_tokens=reserved)
            self.ledger.append(self.run.id, "projection_compiled", {
                "tokens": estimate_tokens(messages), "messages": len(messages),
                "candidates": [s.tool.name for s in turn.candidates],
            })

            decision = extract_finish(self.llm.complete(messages, api_tools or None))
            for call in decision.calls:
                call.name = self.registry.resolve_api_name(call.name)
            self.budget.steps += 1
            self._note_usage(decision, messages)
            self.ledger.append(self.run.id, "model_response", {
                "text": decision.text[:2000], "finish": decision.finish,
                "calls": [{"name": c.name, "arguments": c.arguments} for c in decision.calls],
            })

            self.conversation.append(
                Message(role=ASSISTANT, content=decision.text, tool_calls=list(decision.calls))
            )

            if decision.finish and decision.calls:
                # P0-3: reject outright, execute nothing.
                self.ledger.append(self.run.id, "decision_validated", {
                    "ok": False, "reason": "finish combined with tool calls in the same decision",
                })
                for call in decision.calls:
                    self._observe(call.id, call.name,
                                   "Rejected: cannot call finish(result) together with other tools in the "
                                   "same decision. Call finish(result) alone once you are done.")
                continue

            if decision.finish:
                self.ledger.append(self.run.id, "decision_validated", {"ok": True, "finish": True})
                if self.config.mode == "job":
                    self.run.complete(decision.result)
                    self._snapshot()
                    return self.run.result
                # Chat mode: finish() is just an alternate way to answer this
                # turn — the run itself stays RUNNING so the conversation
                # can continue with the next send().
                return decision.result if decision.result is not None else decision.text

            if not decision.calls:
                outcome = self._handle_text_only(decision)
                if outcome is not _CONTINUE:
                    self._snapshot()
                    return outcome
                continue

            self._idle_turns = 0
            self.ledger.append(self.run.id, "decision_validated", {"ok": True, "finish": False})
            batch = await self.runtime.execute(decision.calls, turn, self._tool_context(), self.run, self.policy)
            self._apply_batch(batch)
            self._snapshot()
            if batch.halted:
                return self.run.pending_approval

    def _apply_batch(self, batch, *, record_conversation: bool = True) -> None:
        for result in batch.results:
            if record_conversation:
                self._observe(result.call.id, result.call.name, result.observation)
            if result.ok:
                self._activate(result.call.name)

    # -- loop helpers -----------------------------------------------------------

    def _enforce_budget(self) -> Optional[Any]:
        """On overrun, grant exactly one grace turn to wrap up, then stop
        deterministically."""
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

    def _maybe_compact(self) -> None:
        window = self.config.projection.window_tokens
        if not self.compactor.should_compact(self.conversation, window):
            return
        folded, remaining = self.compactor.fold(self.conversation, self.working_state)
        if folded:
            self.conversation = remaining
            self.ledger.append(self.run.id, "state_folded", {"working_state": self.working_state.to_dict()})

    def _new_turn(self) -> TurnContext:
        turn = TurnContext(
            config=self.config, registry=self.registry, conversation=self.conversation,
            working_state=self.working_state, session=self, store=self.store, step=self.budget.steps,
        )
        turn.candidates = self._layer2_candidates()
        return turn

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
        for message in reversed(self.conversation):
            if message.role == role and message.text():
                return message.text()
        return ""

    def _last_assistant_text(self) -> str:
        return self._last_text(ASSISTANT)

    def _api_tools(self, turn: TurnContext) -> list[dict]:
        """Native tool schemas for this turn: pinned + candidates + recently
        activated. Bounded, so tool schemas never approach O(N)."""
        names: "OrderedDict[str, None]" = OrderedDict()
        for capability in self.registry.pinned():
            names[capability.name] = None
        for scored in turn.candidates:
            names[scored.tool.name] = None
        for name in self._active:
            names[name] = None
        schemas = [self.registry.get(n).api_schema() for n in names if n in self.registry]
        if self.config.mode == "job":
            # finish() is not a registered Capability — it's the formal
            # completion signal handled directly by the loop (P0-3) — but
            # native tool-calling providers still need its schema to offer it.
            schemas.append(FINISH_SCHEMA)
        return schemas

    def _activate(self, name: str) -> None:
        self._active[name] = None
        self._active.move_to_end(name)
        pinned = {c.name for c in self.registry.pinned()}
        while len(self._active) > _ACTIVE_TOOL_CAP:
            for candidate in self._active:
                if candidate not in pinned:
                    del self._active[candidate]
                    break
            else:
                break

    def _activate_tools(self, names: list[str]) -> None:
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
        """Append a tool observation — structurally distinct role."""
        self.conversation.append(Message(role=OBSERVATION, content=text, tool_call_id=call_id, name=name))

    def _notice(self, text: str) -> None:
        self.conversation.append(Message(role=SYSTEM, content=text))

    def _note_usage(self, decision, messages: list[Message]) -> None:
        if decision.usage is not None:
            self.budget.note_usage(decision.usage.prompt_tokens, decision.usage.completion_tokens, self.config)
        else:
            self.budget.note_usage(estimate_tokens(messages), estimate_tokens(decision.text), self.config)

    def _tool_context(self) -> ToolContext:
        return ToolContext(
            session=self, registry=self.registry, store=self.store, working_state=self.working_state,
            config=self.config, search=self.search, ledger=self.ledger, run=self.run,
        )


class _Continue:
    """Sentinel: the loop should keep going."""


_CONTINUE = _Continue()
