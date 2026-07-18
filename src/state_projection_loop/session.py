"""The agent loop (spec §2.3): render → decide → execute → commit.

The Session wires the three nouns (Registry, Projection, Runtime) together
and owns the mutable truth: conversation, summary, state, value store,
budget. Everything the model sees each turn is re-projected from here.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any, Callable, Optional

from .builtin.meta import ensure_meta_tools
from .compaction import Compactor
from .config import Config
from .discovery import ScoredTool, ToolSearch
from .embeddings import EmbeddingBackend
from .handles import ValueStore
from .hooks import Hooks
from .llm import LLMAdapter
from .logger import SessionLogger
from .messages import ASSISTANT, Message, OBSERVATION, SYSTEM, USER
from .projection import Projection, Section, TurnContext, build_default_sections
from .registry import Registry
from .runtime import BudgetState, Runtime
from .tokens import estimate_tokens
from .tooldef import ToolContext

_ACTIVE_TOOL_CAP = 48


def _ensure_no_running_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        "Session.send/run_job cannot be called from inside an event loop; "
        "use asend/arun_job instead"
    )


class Session:
    def __init__(
        self,
        llm: LLMAdapter,
        *,
        kernel: str = "",
        config: Optional[Config] = None,
        registry: Optional[Registry] = None,
        seed: Optional[dict[str, Any]] = None,
        hooks: Optional[Hooks] = None,
        embedder: Optional[EmbeddingBackend] = None,
        sections: Optional[list[Section]] = None,
        extra_sections: Optional[dict[str, Section]] = None,
        summarizer: Optional[LLMAdapter] = None,
        spawn_llm_factory: Optional[Callable[[Optional[str]], LLMAdapter]] = None,
    ) -> None:
        self.config = config or Config()
        self.llm = llm
        self.summarizer = summarizer
        self.spawn_llm_factory = spawn_llm_factory
        self.registry = registry if registry is not None else Registry()
        ensure_meta_tools(self.registry, mode=self.config.mode)

        self.state: dict[str, Any] = dict(seed or {})
        self.store = ValueStore()
        self.hooks = hooks or Hooks()
        self.logger = SessionLogger(self.config.log_path)
        self.search = ToolSearch(self.registry, embedder=embedder, vector=self.config.discovery.vector)

        pinned = self.registry.pinned()
        if sections is None:
            sections = build_default_sections(
                self.config.projection.sections,
                kernel_text=kernel,
                pinned=pinned,
                extra=extra_sections,
            )
        self.projection = Projection(sections, window_tokens=self.config.projection.window_tokens)
        self.runtime = Runtime(self.registry, self.store, self.config, self.hooks, self.logger)
        self.runtime.seen_specs.update(t.name for t in pinned)  # kernel carries their specs
        self.compactor = Compactor(self.config, self._resolve_summarizer())

        self.conversation: list[Message] = []
        self.summary: list[str] = []
        self.budget = BudgetState()

        self._active: "OrderedDict[str, None]" = OrderedDict((t.name, None) for t in pinned)
        self._interrupted = False
        self._done: Optional[tuple[Any]] = None
        self._idle_turns = 0
        self._budget_grace_used = False

    def _resolve_summarizer(self) -> Optional[LLMAdapter]:
        if self.summarizer is not None:
            return self.summarizer
        if self.config.compaction.model == "none":
            return None
        return self.llm

    # -- public API ---------------------------------------------------------

    def send(self, text: str) -> str:
        """Chat mode: one user message in, the final text reply out."""
        _ensure_no_running_loop()
        return asyncio.run(self.asend(text))

    async def asend(self, text: str) -> str:
        self.conversation.append(Message(role=USER, content=text))
        self.logger.log("user", text=text)
        return await self._loop()

    def run_job(self, task: str) -> Any:
        """Job mode: run until done(result), budget exhaustion, or interrupt."""
        _ensure_no_running_loop()
        return asyncio.run(self.arun_job(task))

    async def arun_job(self, task: str) -> Any:
        self._done = None
        self.conversation.append(Message(role=USER, content=task))
        self.logger.log("job", task=task)
        return await self._loop()

    def interrupt(self) -> None:
        """Request the loop to stop at the next iteration boundary."""
        self._interrupted = True

    def add_section(self, section: Section, *, before: str = "candidates") -> None:
        self.projection.insert_before(before, section)

    # -- loop ---------------------------------------------------------------

    async def _loop(self) -> Any:
        while True:
            if self._interrupted:
                self._interrupted = False
                self.logger.log("interrupted")
                return self._last_assistant_text() or "[interrupted]"

            stop = self._enforce_budget()
            if stop is not None:
                return stop

            self._maybe_compact()

            turn = self._new_turn()
            messages = self.projection.render(turn)
            api_tools = self._api_tools(turn)
            self.logger.log(
                "render",
                tokens=estimate_tokens(messages),
                messages=len(messages),
                candidates=[s.tool.name for s in turn.candidates],
            )

            decision = self.llm.complete(messages, api_tools or None)
            self.budget.steps += 1
            self._note_usage(decision, messages)
            self.logger.log(
                "decide",
                text=decision.text[:2000],
                calls=[{"name": c.name, "arguments": c.arguments} for c in decision.calls],
            )

            self.conversation.append(
                Message(role=ASSISTANT, content=decision.text, tool_calls=list(decision.calls))
            )

            if decision.calls and (block := self._run_after_decide(decision, turn)):
                for call in decision.calls:
                    self._observe(call.id, call.name, block.text())
                self.logger.log("hook_block", reason=block.reason)
                continue

            if not decision.calls:
                outcome = self._handle_text_only(decision)
                if outcome is not _CONTINUE:
                    return outcome
                continue

            self._idle_turns = 0
            results = await self.runtime.execute(decision.calls, turn, self._tool_context())
            for result in results:
                self._observe(result.call.id, result.call.name, result.observation)
                if result.ok:
                    self._activate(result.call.name)

            if self._done is not None:
                self.logger.log("done")
                return self._done[0]

    # -- loop helpers -------------------------------------------------------

    def _enforce_budget(self) -> Optional[Any]:
        """Budget enforcement (§8.1): on overrun, grant exactly one grace
        turn to wrap up, then stop deterministically."""
        reason = self.budget.exceeded(self.config)
        if reason is None:
            return None
        if not self._budget_grace_used:
            self._budget_grace_used = True
            hint = " or call done(result)" if self.config.mode == "job" else ""
            self._notice(f"[runtime] Budget exceeded: {reason}. Wrap up now with a final answer{hint}.")
            return None
        self.logger.log("budget_stop", reason=reason)
        if self.config.mode == "job":
            return self._done[0] if self._done else self._last_assistant_text()
        return self._last_assistant_text() or "[budget exhausted]"

    def _maybe_compact(self) -> None:
        window = self.config.projection.window_tokens
        if not self.compactor.should_compact(self.conversation, window):
            return
        entry, remaining = self.compactor.fold(self.conversation)
        if entry:
            self.summary.append(entry)
            self.conversation = remaining
            self.logger.log("compact", summary_tokens=estimate_tokens(entry), kept=len(remaining))

    def _new_turn(self) -> TurnContext:
        turn = TurnContext(
            config=self.config,
            registry=self.registry,
            conversation=self.conversation,
            summary=self.summary,
            state=self.state,
            session=self,
            store=self.store,
            step=self.budget.steps,
        )
        turn.candidates = self._layer2_candidates()
        return turn

    def _layer2_candidates(self) -> list[ScoredTool]:
        query = "\n".join(q for q in self._candidate_queries() if q)
        if not query:
            return []
        pinned = {t.name for t in self.registry.pinned()}
        return self.search.search(query, k=self.config.discovery.k, layer=2, exclude=pinned)

    def _candidate_queries(self) -> list[str]:
        parts: list[str] = []
        for source in self.config.discovery.query_sources:
            if source == "last_user_message":
                parts.append(self._last_text(USER))
            elif source == "last_model_thought":
                parts.append(self._last_text(ASSISTANT))
            elif source == "goal_if_exists":
                parts.append(str(self.state.get("goal", "")))
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
        activated. Bounded, so tool schemas never approach O(N) (I1)."""
        names: "OrderedDict[str, None]" = OrderedDict()
        for tool in self.registry.pinned():
            names[tool.name] = None
        for scored in turn.candidates:
            names[scored.tool.name] = None
        for name in self._active:
            names[name] = None
        return [self.registry.get(n).api_schema() for n in names if n in self.registry]

    def _activate(self, name: str) -> None:
        self._active[name] = None
        self._active.move_to_end(name)
        pinned = {t.name for t in self.registry.pinned()}
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

    def _run_after_decide(self, decision, turn):
        for hook in self.hooks.after_decide:
            block = hook(decision, turn)
            if block is not None:
                return block
        return None

    def _handle_text_only(self, decision) -> Any:
        if self.config.mode == "chat":
            return decision.text
        self._idle_turns += 1
        if self._idle_turns > self.config.limits.max_idle_turns:
            self.logger.log("job_gave_up_text", text=decision.text[:500])
            return decision.text
        self._notice(
            "[runtime] No tool was called. Continue working with tools, "
            "or call done(result) to finish the job."
        )
        return _CONTINUE

    def _observe(self, call_id: str, name: str, text: str) -> None:
        """Append a tool observation — structurally distinct role (I6)."""
        self.conversation.append(
            Message(role=OBSERVATION, content=text, tool_call_id=call_id, name=name)
        )

    def _notice(self, text: str) -> None:
        self.conversation.append(Message(role=SYSTEM, content=text))
        self.logger.log("notice", text=text)

    def _note_usage(self, decision, messages: list[Message]) -> None:
        if decision.usage is not None:
            self.budget.note_usage(
                decision.usage.prompt_tokens, decision.usage.completion_tokens, self.config
            )
        else:
            self.budget.note_usage(
                estimate_tokens(messages), estimate_tokens(decision.text), self.config
            )

    def _tool_context(self) -> ToolContext:
        return ToolContext(
            session=self,
            registry=self.registry,
            store=self.store,
            state=self.state,
            config=self.config,
            search=self.search,
            logger=self.logger,
        )

    def _finish(self, result: Any) -> None:
        self._done = (result,)


class _Continue:
    """Sentinel: the loop should keep going."""


_CONTINUE = _Continue()
