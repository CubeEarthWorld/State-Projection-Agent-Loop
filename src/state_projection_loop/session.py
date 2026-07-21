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
from typing import Any, Callable, Optional

from .artifacts import ArtifactStore
from .builtin.meta import ensure_meta_tools
from .capability import Capability, ToolContext
from .config import Config
from .discovery import ScoredTool, ToolSearch
from .embeddings import EmbeddingBackend
from .events import EventLedger, InMemoryLedger, JsonlLedger, Snapshot, RENDERABLE_TYPES, event_to_message
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
        spawn_llm_factory: Optional[Callable[[Optional[str]], LLMAdapter]] = None,
        ledger: Optional[EventLedger] = None,
    ) -> None:
        self.config = config or Config()
        self.llm = llm
        self.spawn_llm_factory = spawn_llm_factory
        self.registry = registry if registry is not None else Registry()
        ensure_meta_tools(self.registry)

        self.session_id = new_id("session")
        self.ledger = ledger if ledger is not None else _make_ledger(self.config)
        self.run = Run(new_id("run"), self.session_id, self.ledger)

        self.policy = policy if policy is not None else self._default_policy()

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
        self.runtime.seen_specs.update(c.name for c in pinned)

        self.working_state = WorkingState()
        for key, value in (seed or {}).items():
            if hasattr(self.working_state, key):
                setattr(self.working_state, key, value)
            else:
                self.working_state.extra[key] = value

        self.budget = BudgetState()

        self._active: "OrderedDict[str, None]" = OrderedDict((c.name, None) for c in pinned)
        self._interrupted = False
        self._idle_turns = 0
        self._budget_grace_used = False
        self._lock = asyncio.Lock()
        self.ledger.append(self.run.id, "run_state_changed", {"from": "RUNNING", "to": "RUNNING", "reason": "created"})

    @staticmethod
    def _default_policy() -> PolicyEngine:
        engine = PolicyEngine(default_decision="require_approval")
        engine.apply_preset("auto_safe")
        return engine

    # -- public API -------------------------------------------------------------

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
            return await self._loop()

    def interrupt(self) -> None:
        self._interrupted = True

    def add_section(self, section: Section, *, before: str = "candidates") -> None:
        self.projection.insert_before(before, section)

    # -- approval lifecycle -------------------------------------------------

    def resolve_approval(self, decision: str) -> ApprovalRequest:
        return self.run.resolve_approval(decision, current_policy_revision=self.policy.revision)

    def resume(self) -> Any:
        _ensure_no_running_loop()
        return asyncio.run(self.aresume())

    async def aresume(self) -> Any:
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

    # -- direct invocation ---------------------------------------------------

    def invoke(self, capability_name: str, **arguments: Any) -> Any:
        _ensure_no_running_loop()
        return asyncio.run(self.ainvoke(capability_name, **arguments))

    async def ainvoke(self, capability_name: str, **arguments: Any) -> Any:
        from .messages import ToolCall

        async with self._guarded():
            turn = self._new_turn()
            call = ToolCall(name=capability_name, arguments=arguments)
            batch = await self.runtime.execute([call], turn, self._tool_context(), self.run, self.policy)
            self._apply_batch(batch, record=False)
            self._snapshot()
            if batch.halted:
                return self.run.pending_approval
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
        irreversible = self._irreversible_effects_up_to(to_turn)
        all_events = list(self.ledger.iter_run(self.run.id))
        renderable = [e for e in all_events if e.type in RENDERABLE_TYPES]
        checkpoints = [e for e in all_events if e.type == "checkpoint"]

        user_turns_seen = 0
        cut_index = len(renderable)
        for i, event in enumerate(renderable):
            if event.type == "user_input":
                if user_turns_seen == to_turn:
                    cut_index = i
                    break
                user_turns_seen += 1

        kept_renderable = renderable[:cut_index]

        restored_ws = WorkingState()
        user_count = 0
        looking_for_checkpoint = False
        for event in all_events:
            if event.type == "user_input":
                if user_count == to_turn:
                    looking_for_checkpoint = True
                user_count += 1
            elif event.type == "checkpoint" and looking_for_checkpoint:
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
        self._active = OrderedDict((c.name, None) for c in self.registry.pinned())
        self.runtime.seen_specs = {c.name for c in self.registry.pinned()}
        self.runtime._consecutive_validation_failures = {}

        return irreversible

    def _irreversible_effects_up_to(self, to_turn: int) -> list[str]:
        notices: list[str] = []
        user_count = 0
        for event in self.ledger.iter_run(self.run.id):
            if event.type == "user_input":
                if user_count >= to_turn:
                    break
                user_count += 1
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
        embedder: Optional[EmbeddingBackend] = None,
        spawn_llm_factory: Optional[Callable[[Optional[str]], LLMAdapter]] = None,
    ) -> "Session":
        config = config or Config()
        if not config.persistence.ledger_directory:
            raise RunStateError("resume_from_ledger requires config.persistence.ledger_directory")
        ledger = JsonlLedger(config.persistence.ledger_directory)
        snapshot = ledger.load_snapshot(run_id)
        if snapshot is None:
            raise RunStateError(f"No snapshot found for run {run_id!r}; nothing to resume")

        session = cls(
            llm, config=config, registry=registry, policy=policy, embedder=embedder,
            spawn_llm_factory=spawn_llm_factory, ledger=ledger,
        )
        session.run = Run.from_snapshot_state(run_id, ledger, snapshot.state)
        session.session_id = snapshot.state.get("session_id", session.session_id)
        session.working_state = WorkingState.from_dict(snapshot.state.get("working_state") or {})
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
            batch = await self.runtime.execute(decision.calls, turn, self._tool_context(), self.run, self.policy)
            self._apply_batch(batch)
            self._snapshot()
            if batch.halted:
                return self.run.pending_approval

    def _apply_batch(self, batch, *, record: bool = True) -> None:
        for result in batch.results:
            if record:
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

    def _new_turn(self) -> TurnContext:
        turn = TurnContext(
            config=self.config, registry=self.registry, ledger=self.ledger, run_id=self.run.id,
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

    def _api_tools(self, turn: TurnContext) -> list[dict]:
        names: "OrderedDict[str, None]" = OrderedDict()
        for capability in self.registry.pinned():
            names[capability.name] = None
        for scored in turn.candidates:
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
        self.ledger.append(self.run.id, "observation", {"call_id": call_id, "name": name, "text": text})

    def _notice(self, text: str) -> None:
        self.ledger.append(self.run.id, "notice", {"text": text})

    def _checkpoint(self) -> None:
        self.ledger.append(self.run.id, "checkpoint", {"working_state": self.working_state.to_dict()})

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
