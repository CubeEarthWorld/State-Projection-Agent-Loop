# state-projection-loop

**v0.4: Built-in planning checklists.** Every session includes one unified
`planning.checklist.manage` tool for named ULID plans, item edits, progress,
context visibility and JSON handoff. Plans use the same in-memory or JSONL
storage as the conversation and survive history compression and completion.
See [the checklist specification and Python/Dart examples](docs/checklists.md).

**State-Projection Agent Loop** — a vendor-agnostic, resumable LLM agent
runtime built on two principles:

> Truth lives *outside* the context. Every turn, the prompt is re-rendered as
> a minimal, disposable **projection** of that truth.
>
> The LLM proposes; it never decides. Execution order, idempotency, policy
> authorization, and what actually happened are guaranteed by code, not by
> the model's good behavior.

Conventional agent loops conflate the transcript, the source of truth, and
the model input — which structurally causes O(N) tool preloading, batches
that run out of the model's stated order, non-idempotent actions that
double-fire on a timeout, and "what happened?" being unanswerable after the
fact. This runtime decomposes that trinity and makes execution a real state
machine: an append-only **Event Ledger** is the only source of truth; a
**Run** can pause for approval, survive a process restart, and resume
exactly where it left off.

日本語のシナリオ例(ゲームマスター/カスタマーサポート/コーディングエージェント)は
[`examples/`](examples/) にあります。

## The package is LLM-agnostic

`state_projection_loop` depends on **no LLM provider SDK**. It defines only
a one-method `LLMAdapter` Protocol (`async complete(messages, tools) -> Decision`)
and a scripted test double (`ScriptedLLM`) for deterministic tests. Talking
to a real model — OpenAI, Anthropic, DeepSeek, a local server, anything — is
entirely your own adapter, implementing that Protocol however you like.
Reference implementations (`OpenAICompatAdapter`, `AnthropicAdapter`,
`OpenAICompatEmbedding`, `LlamaCppEmbedding`) live in
[`examples/llm_adapters.py`](examples/llm_adapters.py) — copy and adapt them
freely; they are examples, not a package API with a stability contract.

## Architecture

```
Registry ──▶ Projection ──▶ LLM ──▶ Validate ──▶ Authorize ──▶ Execute ──▶ Record ──▶ Continue/Wait/Complete
(capabilities)  (render)   (decide)              (Policy)      (Runtime)   (Ledger)
```

| Component | Responsibility |
|---|---|
| **Registry** | Versioned `Capability` ledger: dotted names (`filesystem.file.read`), JSON schema, declared effects, retry safety, categories, epochs, external `ToolProvider`s |
| **Projection** | Ordered sections (`fixed` / `append` / `epoch` / `volatile`) rendered into the per-turn prompt; window budget includes native tool schemas and reserved output tokens |
| **PolicyEngine** | The sole owner of execution permission — layered `absolute > admin > developer > workspace > session > llm`; a higher layer's `deny` can never be relaxed by a lower one |
| **Runtime** | Schema validation & self-repair, in-order execution (only adjacent read-only calls run concurrently), retry-safety-gated retries, `OUTCOME_UNKNOWN` on timeout, output-size artifacts |
| **Run** | The state machine: `RUNNING / WAITING_FOR_APPROVAL / WAITING_FOR_USER / COMPLETED / FAILED / CANCELLED` |
| **EventLedger** | Append-only log of everything that happened; `Session`/`Run` state is *derived* from it, never the other way around |
| **ArtifactStore** | Large results live outside the context, referenced as `{"$artifact": "art_..."}` — never a bare string, so ordinary data can never be misread as a reference |
| **WorkingState** | Structured goal / facts / decisions(+reasons) / open questions / next actions — compaction *merges* into it instead of re-summarizing prose |

## Capability awareness in layers

With 1,000 registered capabilities, per-turn overhead stays **under 3k
tokens** (enforced by an acceptance test) instead of ~150k for full-spec
preloading:

| Layer | What | Cost |
|---|---|---|
| 0 | Pinned capabilities — full spec resident in the kernel | opt-in |
| 1 | TOC — category names + counts, epoch-cached | ≤100 tk |
| 2 | Auto candidates — vector+BM25+tag search, top-k cards injected each turn | ~300 tk |
| 3 | `meta.tool.find` — the model searches the registry itself (fallback) | +1 loop |

Every registered capability stays reachable even with vectors disabled.

## Install

```bash
pip install state-projection-loop                  # core: jsonschema only
pip install "state-projection-loop[dev]"            # + pytest for running the test suite
pip install "state-projection-loop[examples]"       # + openai/anthropic/llama-cpp-python
                                                     #   (only needed to run examples/llm_adapters.py)
```

The core has a single dependency (`jsonschema`) and even falls back to a
built-in mini validator when it's absent. The sync API (`session.send`,
`session.run_job`) hides asyncio entirely.

## Quickstart

```python
from state_projection_loop import Session, capability
from examples.llm_adapters import OpenAICompatAdapter  # your own adapter is just as valid

@capability(name="inventory.stock.get", category="inventory",
            embedding_text="在庫 いくつ 残り stock",
            retry_safety="pure", effects=[("none", "*")])
def get_stock(warehouse: str) -> dict:
    """倉庫の在庫数を返す。

    Args:
        warehouse: 倉庫名(例: 東京, 大阪)
    """
    return {"warehouse": warehouse, "stock": 42}

session = Session(OpenAICompatAdapter(model="...", api_key="...", base_url="..."),
                   kernel="あなたは在庫管理アシスタント。答えたら finish(result) を呼ぶ。")
session.registry.register(get_stock)
print(session.run_job("東京倉庫の在庫はいくつ?"))
```

Capabilities can equally be declared as JSON dicts
(`registry.register({...}, handler=fn)`) with cards, effects, retry safety,
usage notes, examples, and output policy — see
[`src/state_projection_loop/capability.py`](src/state_projection_loop/capability.py)
and [`examples/customer_support/tools.py`](examples/customer_support/tools.py).

## Execution correctness the runtime guarantees

- **Order**: calls execute in the model's stated order by default. Only a
  contiguous run of capabilities that declare no write/external effect may
  execute concurrently — a write never jumps ahead of an earlier read, and a
  capability that forgets to declare its effects is treated as the most
  restrictive kind, not the safest.
- **Idempotency**: a capability may only be auto-retried if `retry_safety`
  is `pure` or `idempotent` — declaring `retries > 0` otherwise is a
  construction-time error. A timeout is recorded as outcome `unknown`, never
  silently `failed`: the runtime cannot know whether the underlying effect
  completed after it gave up waiting, and collapsing that distinction is
  exactly what lets non-idempotent operations double-fire.
- **Completion**: `finish(result)` is a formal property of the model's
  decision, not a capability routed through the runtime — a decision that
  combines `finish` with other tool calls is rejected outright, nothing in
  it executes.
- **Concurrency**: at most one turn in flight per `Session`; a second
  concurrent `asend`/`arun_job`/`resume`/`invoke` raises `ConcurrencyError`
  immediately instead of interleaving state.
- **Self-repair**: invalid arguments are *not executed*; the model receives
  the validation error plus the full spec as an observation and retries.
  `require_spec: true` forces a spec review before a dangerous capability's
  first use.
- **Artifacts**: results above `max_inline_tokens` are stored outside the
  context and projected as a preview card, referenced as
  `{"$artifact": "art_..."}`. A bare string that happens to equal an
  artifact id is never resolved — only the structured reference form is.

## Policy: the LLM proposes, code decides

```python
from state_projection_loop import Session
from state_projection_loop.policy import PolicyEngine, Rule

policy = PolicyEngine(default_decision="require_approval")
policy.apply_preset("auto_safe")          # effect-free calls + workspace reads run automatically
policy.set_scope("network_access", "deny", layer="admin")   # a lower layer can never relax this
policy.add_rule("workspace", Rule(decision="allow", capability_pattern="fs.*"))

session = Session(llm, policy=policy)
result = session.send("...")
if session.run.state == "WAITING_FOR_APPROVAL":
    session.resolve_approval("approved")   # or "denied"
    result = session.resume()
```

Evaluation order is fixed: `absolute > admin > developer > workspace > session > llm`.
The most restrictive matching rule wins across layers — a `deny` at any
layer can never be relaxed by one below it. An LLM-proposed safety
assessment (`policy.set_llm_safety_mode("advisory" | "approval_routing")`)
can escalate toward approval but can never grant a bare `allow` or issue the
final `deny` by itself.

## Resumable runs

A `WAITING_FOR_APPROVAL` run survives a process restart:

```python
# process 1
session = Session(llm, config=Config.from_dict(
    {"mode": "job", "persistence": {"ledger_directory": "./runs"}}))
session.run_job("delete the old backups")
run_id = session.run.id   # paused: WAITING_FOR_APPROVAL

# process 2 (hours later, no reference to the first Session)
restored = Session.resume_from_ledger(llm, run_id, config=cfg, registry=registry)
restored.resolve_approval("approved")
result = restored.resume()
```

Every projection, decision, policy verdict, command start/outcome, approval,
and run-state change is an `Event` in the append-only ledger
(`InMemoryLedger` by default, `JsonlLedger` when `persistence.ledger_directory`
is set). `Session` state is a *derived* view of that ledger, recoverable from
Events + a periodic `Snapshot`.

## Rewinding without losing history

```python
branch, irreversible = session.branch(at_message=6)
```

Past events are never deleted or mutated — `branch()` starts a new `Run`
that shares conversation/working-state up to the cut point. `irreversible`
lists effects the parent run already committed (anything with a declared
`external` effect) that the branch cannot undo — a sent email or a git push
stays sent/pushed regardless of which branch you're on now.

## Working state (structured, not prose)

```python
session = Session(llm, kernel=GM_KERNEL, builtins=["meta", "checklist", "state"],
                  seed={"goal": "escape the dungeon", "extra": {"flags": {}}})
# the "state" pack: state.goal.set / state.fact.add / state.decision.record / state.extra.* + [Working state] view
```

`WorkingState` is a finite record — goal, acceptance criteria, constraints,
confirmed facts, `(decision, reason)` pairs, open questions, next actions,
artifact refs, plus a free-form `extra` dict for app-specific state. It is
projected every turn and checkpointed per user turn, so a decision's reason
recorded twenty turns ago is still there verbatim after the conversation
itself has been compressed out of the window. The original messages are
never lost — they stay in the Event Ledger, searchable via
`meta.history.search` even after being folded out of the live projection.

## Bundled tools: packs on, names off

Bundled tools come in **packs** and there is one switch for them:

| Pack | Tools | Default |
|---|---|---|
| `meta` | `meta.tool.find`, `meta.artifact.peek`, `meta.history.search` | on |
| `checklist` | `planning.checklist.manage` | on |
| `state` | `state.goal.set`, `state.fact.add`, … (9 tools) | off |
| `spawn` | `meta.agent.spawn` | off |

```python
Session(llm)                                     # meta + checklist
Session(llm, builtins=["meta", "state"])        # no checklist, with state tools
Session(llm, builtins=())                        # nothing bundled at all
install_builtins(registry, ["spawn"])            # same operation on a registry you built yourself
```

`install_builtins` is idempotent and a name the registry already resolves is
left alone, so your own definition of `meta.tool.find` wins over the bundled
one. An unknown pack name raises at construction.

Per-tool control is the registry's deny-list, which works for bundled and
developer capabilities alike:

```python
registry = Registry(disabled=["planning.checklist.manage", "debug/*"])
session = Session(llm, registry=registry)

session.registry.disable("my.dangerous.tool")   # mid-session, e.g. per sub-agent
session.registry.enable("my.dangerous.tool")
```

Entries match a capability name, a category, or a category prefix
(`"cat/*"`) — the same rule `subset()` uses, so `subset()` is the allow-list
and `disable()` the deny-list. A capability is reachable iff it is registered
and not denied; there is no third state.

A disabled capability is gone from **every** surface the model can see: the
native tool schemas, the pinned specs and runtime notes in the kernel, the
tool index, layer-2 candidates, `meta.tool.find`, and execution (it fails as
`unknown_capability`). Both `Registry.__iter__` and `Registry.get()` skip
disabled entries and everything else derives from those two, so there is no
surface left to leak through. The deny-list is by *name*, not by registered
object, so installing a pack again cannot bring a denied tool back.

### Kernel notes belong to the tool, not the kernel

Any **pinned** capability may carry `discovery.kernel_note`, one sentence of
standing guidance that appears under "[Runtime notes]" while the capability
is reachable. The bundled tools use it (that is where "use
`planning.checklist.manage` to plan multi-step work" comes from), and so can
yours. Only pinned capabilities contribute, so the kernel stays bounded by
the pin set you chose, never by registry size.

## Standard agent features (each one is a switch)

| Feature | Switch | What it does |
|---|---|---|
| Clarifying questions | pack `ask` | `meta.user.ask(question, choices?)` pauses the run in `WAITING_FOR_USER`; `send()`/`run_job()` return a `PendingQuestion`, the host calls `session.answer(text)` then `session.resume()`. The answer is the tool's result; the pause survives a restart like an approval does. |
| Loop guard | `limits.max_repeats` (3; `0` off) | An identical call (same name and arguments) that failed identically, or returned the same result, `max_repeats` times within the last `limits.repeat_window` (8) calls is not executed again; the model gets a "loop guard" observation. Pure reads may still be polled. |
| Structured job output | `result_schema` | In job mode `finish(result)` is validated against the JSON Schema; a failing result is bounced back like an argument error. |
| Observers | `Session(on_event=fn)` | `fn(event)` fires after every ledger append. Read-only by contract (the veto point stays the policy engine); an observer that raises is ignored. |
| Compaction | `compaction.trigger_ratio` (`0` off) | When the prompt exceeds the ratio of the window, one extra model call folds history older than the full-fidelity window into `WorkingState` as a schema-validated JSON delta (`state_folded` keeps the pre-fold state); folded events then render as one-line summaries. Deterministic compression stays on regardless. |
| Skills | `skill_capability(name, text, summary=...)` | Progressive disclosure for instructions: a skill is a capability `skill.<name>.load`, so it rides the TOC, candidates and `meta.tool.find` with no second index. |
| Toolkits | `install_toolkits(registry, root, shell=True)` | Root-confined `filesystem.file.list/read/write` and `shell.command.run` with declared effects; never installed unless you ask. |

```python
session = Session(llm, builtins=["meta", "checklist", "ask"], on_event=print,
                  config=Config.from_dict({"compaction": {"trigger_ratio": 0.8}}))
session.registry.register(skill_capability("deploy", DEPLOY_STEPS, summary="How to deploy the service"))
install_toolkits(session.registry, "./workspace")

reply = session.send("Release the service")
if session.run.state == "WAITING_FOR_USER":     # the model asked something
    session.answer(input(reply.text))
    reply = session.resume()
```

## Sub-agents (opt-in)

```python
session = Session(llm, builtins=["meta", "checklist", "spawn"])
# the model can now: spawn(task=..., tool_scope=["web.*"], max_steps=15)
```

Parent and child share **only** the task string and the result; a child's
artifacts must be explicitly moved into the parent's namespace.

## Swappable everything

- **Models**: `LLMAdapter` Protocol — bring your own; see
  `examples/llm_adapters.py` for `OpenAICompatAdapter` / `AnthropicAdapter`
  reference implementations. `ScriptedLLM` drives deterministic tests.
- **Embeddings**: `EmbeddingBackend` Protocol. The package ships only
  `HashingEmbedding` (dependency-free, deterministic); real embedding
  backends (`OpenAICompatEmbedding`, `LlamaCppEmbedding`) are examples too.
- **Capability sources**: `ToolProvider` Protocol — sync external
  capability servers into the registry mid-session.
- **Sections**: subclass `Section` (`render`, optional `shrink`) and pass the
  full list via `Session(sections=...)` or splice one in with
  `session.add_section(...)`. When the window overflows, sections are asked
  to `shrink` from last to first, so order is also priority: put what you
  can most afford to lose last. Built-ins shrink candidates, then the
  checklist view, then the oldest history.
- **Persistence**: `EventLedger` Protocol — `InMemoryLedger` for tests,
  `JsonlLedger` for disk; implement your own for a real database.

## Configuration (defaults shown)

```python
Config.from_dict({
  "mode": "chat",                          # or "job" (finish(result) ends the run)
  "result_schema": None,                   # job mode: JSON Schema finish(result) must satisfy
  "compaction": {"trigger_ratio": 0.0},    # 0 = off; e.g. 0.8 folds history into WorkingState
  "projection": {
      "sections": ["kernel", "toc", "history", "working_state", "checklists", "candidates"],
      "window_tokens": 30000, "reserved_output_tokens": 1024, "provider_overhead_tokens": 0,
      "dedupe_candidate_cards_against_schemas": True,
  },
  "discovery": {"vector": "auto", "k": 8, "toc": True, "active_tools": 48,
                "query_sources": ["last_user_message", "last_model_thought", "goal_if_exists"]},
  "compression": {"full_window": 6, "compressed_window": 24, "summary_window": 60,
                  "compressed_max_lines": 80, "observation_max_lines": 40},
  "budget": {"max_steps": 50, "max_tokens": None, "max_cost": None, "max_seconds": None,
             "cost_per_1k_input": 0.0, "cost_per_1k_output": 0.0},
  "artifacts": {"inline_threshold_tokens": 800, "preview_tokens": 120, "directory": None},
  "limits": {"max_validation_retries": 2, "max_idle_turns": 3, "approval_expires_s": 3600.0,
             "max_repeats": 3, "repeat_window": 8},
  "persistence": {"ledger_directory": None},
})
```

Cross-session memory is specified but not built; see [docs/roadmap.md](docs/roadmap.md).

## Changes in 0.5 (pre-1.0: breaking, no aliases)

- `Session(builtins=...)` / `install_builtins()` replace `ensure_meta_tools`,
  `ensure_checklist_tool`, `install_state`, `install_spawn`.
- One `ToolContext` for sections and handlers replaces `TurnContext`.
- `Section.shrink` replaces the hard-coded overflow ladder; `extra_sections`
  is gone (pass `sections=`).
- `discovery.kernel_note` replaces the hard-coded runtime-note table.
- `Registry.categories()` returns `(total, pinned)`; `categories_with_pinned`,
  `register_many`, `Message.meta`, `ToolResult.elapsed_s`, four never-emitted
  event types and the unused `WAITING_FOR_USER` state are removed.
- `Session.activate()` and `Runtime.reset()` are public; `discovery.active_tools`
  replaces a hard-coded LRU size.
- Handlers receive `ToolContext`; sections receive its superset `TurnContext`
  (candidates, native schemas, dedupe flag), so projection state never reaches a tool.
- Shrinking now counts native schemas: a dropped candidate takes its schema with it,
  and the least recently used non-pinned schema is dropped as a last resort.
- New standard features, each optional: `ask` pack, loop guard, `result_schema`,
  `on_event` observers, compaction, `skill_capability`, `install_toolkits`.
  `WAITING_FOR_USER` is back, now in use.

## Examples & scenarios

| Scenario | Code | What it shows |
|---|---|---|
| Customer support (web) | [`examples/customer_support/`](examples/customer_support/) | manual search over real sample manuals, human escalation with full transcript, chart cards, `require_spec` gating |
| Game master (TRPG) | [`examples/game_master/`](examples/game_master/) | BGM/image/expression presentation, dice, full working-state goal/flags/variables, drift-free `[Working state]` view |
| Coding agent | [`examples/coding_agent/`](examples/coding_agent/) | sandboxed file tools, real test runs, red→green fix cycle, job-mode `finish(result)` |

Run the live ones (needs `.env` — see `.env.example`; requires
`pip install "state-projection-loop[examples]"`):

```bash
python -m examples.quickstart
python -m examples.customer_support.run_live
python -m examples.game_master.run_live
python -m examples.coding_agent.run_live
```

## Testing

```bash
pip install -e ".[dev,examples]"
pytest tests --ignore=tests/integration     # offline: unit + acceptance + scenarios
```

`tests/acceptance/test_p0_p1_acceptance.py` is the redesign's own checklist:
execution order, finish-vs-side-effects rejection, idempotency/OUTCOME_UNKNOWN,
concurrency isolation, artifact-reference safety, schema-aware budgeting,
approval survives a simulated process restart, policy layering, ledger-based
reproducibility, non-destructive branching, irreversible-effect surfacing,
and per-command traceability. `tests/acceptance/test_acceptance.py` covers
the broader baseline: ≤3k-token overhead at 1,000 registered capabilities,
full reachability with vectors off, the validation→spec→retry self-repair
path, and a working default-config chat agent. Scenario tests drive the
*real* tools (files, subprocesses, manuals) with a scripted model.

Live integration tests (real LLM API + optional GGUF embedding download):

```bash
cp .env.example .env     # fill in LLM_API_KEY (or DEEPSEEK_API_KEY), set SPAL_RUN_LIVE=1
pytest tests/integration -q
```

**Never commit `.env`** — it is gitignored; keys stay out of the repository.

## License

MIT
