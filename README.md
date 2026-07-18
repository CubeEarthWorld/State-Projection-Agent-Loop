# state-projection-loop

**State-Projection Agent Loop** — a next-generation, vendor-agnostic LLM agent
runtime built on one principle:

> Truth lives *outside* the context. Every turn, the prompt is re-rendered as a
> minimal, disposable **projection** of that truth.

Conventional agent loops conflate three things — the append-only transcript,
the source of truth, and the model input — which structurally causes O(N) tool
preloading, O(N²) history re-billing, context pollution by intermediate data,
and goal drift. This runtime decomposes that trinity: register thousands of
tools and pay only a few hundred tokens per turn for tool awareness.

日本語のシナリオ例(ゲームマスター/カスタマーサポート/コーディングエージェント)は
[`examples/`](examples/) にあります。

## Architecture: 3 nouns, 4 verbs

```
Registry ──▶ Projection ──▶ LLM ──▶ Runtime ──▶ commit ──▶ (loop)
 (tools)      (render)     (decide)  (execute)
```

| Noun | Responsibility |
|---|---|
| **Registry** | Tool ledger: JSON metadata + Python handlers, categories, epochs, external `ToolProvider`s |
| **Projection** | Ordered sections (`fixed` / `append` / `epoch` / `volatile`) rendered into the per-turn prompt, window budget enforced |
| **Runtime** | Everything deterministic: schema validation & self-repair, retries, timeouts, parallelism, handles, budgets, structured logs |

## Tool awareness in 4 layers

With 1,000 registered tools, per-turn tool overhead stays **under 3k tokens**
(enforced by an acceptance test) instead of ~150k for full-spec preloading:

| Layer | What | Cost |
|---|---|---|
| 0 | Pinned tools — full spec resident in the kernel | opt-in |
| 1 | TOC — category names + counts, epoch-cached | ≤100 tk |
| 2 | Auto candidates — vector+BM25+tag search over the registry, top-k cards injected each turn | ~300 tk |
| 3 | `find_tools` — the model searches the ledger itself (fallback) | +1 loop |

Every registered tool stays reachable even with vectors disabled (invariant I10).

## Install

```bash
pip install state-projection-loop            # core (pure Python + jsonschema)
pip install state-projection-loop[openai]    # OpenAI-compatible adapter (DeepSeek etc.)
pip install state-projection-loop[embeddings]  # GGUF embeddings via llama-cpp-python
```

The core has a single dependency (`jsonschema`) and even falls back to a
built-in mini validator when it's absent — it can be vendored into embedded
Python environments (e.g. Ren'Py) with zero binary dependencies. The sync API
(`session.send`) hides asyncio entirely.

## Quickstart

```python
from state_projection_loop import Session, tool
from state_projection_loop.adapters import DeepSeekAdapter  # or any OpenAI-compatible API

@tool(category="inventory", embedding_text="在庫 いくつ 残り stock")
def get_stock(warehouse: str) -> dict:
    """倉庫の在庫数を返す。

    Args:
        warehouse: 倉庫名(例: 東京, 大阪)
    """
    return {"warehouse": warehouse, "stock": 42}

session = Session(DeepSeekAdapter(), kernel="あなたは在庫管理アシスタント。")
session.registry.register(get_stock)
print(session.send("東京倉庫の在庫はいくつ?"))
```

Tools can equally be declared as JSON dicts (`registry.register({...}, handler=fn)`)
with cards, usage notes, examples, discovery hints and execution policy — see
[the tool schema](src/state_projection_loop/tooldef.py) and
[`examples/customer_support/tools.py`](examples/customer_support/tools.py).

## What the runtime gives you for free

- **Self-repair** (§6): invalid arguments are *not executed*; the model receives
  the validation error plus the full spec as an observation and retries.
  `require_spec: true` forces a spec review before a dangerous tool's first use.
- **Handles** (§8.3): results above `max_inline_tokens` are stored outside the
  context and projected as `$hN` references (type + size + preview). The model
  inspects them with `peek(handle, query=..., range=...)`; tools accept `$hN`
  as arguments and the runtime resolves them. Big data never transits the context.
- **Budgets**: `max_steps / max_tokens / max_cost / max_seconds` — on overrun the
  model gets exactly one grace turn to wrap up.
- **Parallelism**: `parallel_safe` calls run concurrently; mutating calls run
  sequentially, in order.
- **Compaction** (§10): when the conversation exceeds `window_tokens × 0.8`, the
  older half is folded into a first-person summary under **contract v1**
  (chronology, decisions *with reasons*, unfinished intentions, user constraints
  verbatim, raw data replaced by handles).
- **Observation labeling** (I6): tool results always occupy a structurally
  distinct role — a prompt-injection *mitigation*, documented as such.
- **Structured logs**: every render/decide/execute/commit event, JSONL-ready.

## State is just tools (optional)

```python
from state_projection_loop import Session, install_state

session = Session(llm, kernel=GM_KERNEL, seed={"goal": "escape the dungeon", "flags": {}})
install_state(session)   # state_set/get/delete, set_goal, set_flag + [State] view
```

The `state_view` section re-projects the goal and flags **every turn**, which
prevents goal drift structurally. A support bot simply doesn't install it —
the core is identical (invariant I11: the default config alone is a complete
chat agent).

## Control-flow extension (hooks)

Two constrained interception points — enough for human-in-the-loop approval
and guardrails without forking the loop:

```python
from state_projection_loop import Hooks, HookBlock

def approval_gate(decision, turn):
    if any(c.name == "delete_everything" for c in decision.calls):
        return HookBlock(reason="human approval required")

session = Session(llm, hooks=Hooks(after_decide=[approval_gate]))
# after_execute hooks can transform observations (e.g. redact secrets)
```

## Sub-agents (opt-in)

```python
from state_projection_loop import install_spawn
install_spawn(session.registry)
# the model can now: spawn(task=..., tool_scope=["web/*"], max_steps=15)
```

Parent and child share **only** the task string and the result (invariant I9).

## Swappable everything

- **Models**: `LLMAdapter` protocol. Bundled: `OpenAICompatAdapter` /
  `DeepSeekAdapter` (native function calling + fenced-JSON text fallback).
  `ScriptedLLM` drives deterministic tests.
- **Embeddings**: `EmbeddingBackend` protocol. Bundled: `HashingEmbedding`
  (dependency-free) and `LlamaCppEmbedding` (GGUF; defaults to
  [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
  community GGUF with proper query/document prompt prefixes).
- **Tool sources**: `ToolProvider` protocol — sync external tool servers into
  the registry mid-session; the epoch-cached TOC follows automatically.
- **Sections**: implement the `Section` protocol and insert it anywhere the
  invariants allow.

## Configuration (defaults shown)

```python
Config.from_dict({
  "mode": "chat",                          # or "job" (done(result) ends the loop)
  "projection": {"sections": ["kernel", "toc", "summary", "conversation", "candidates"],
                 "window_tokens": 30000},
  "discovery": {"vector": "auto", "k": 8, "toc": True,
                "query_sources": ["last_user_message", "last_model_thought", "goal_if_exists"]},
  "compaction": {"trigger_ratio": 0.8, "model": "same", "contract": "v1"},
  "budget": {"max_steps": 50, "max_tokens": None, "max_cost": None, "max_seconds": None},
  "handles": {"inline_threshold_tokens": 800},
  "limits": {"max_validation_retries": 2},
})
```

## Examples & scenarios

| Scenario | Code | What it shows |
|---|---|---|
| Customer support (web) | [`examples/customer_support/`](examples/customer_support/) | manual search over real sample manuals, human escalation with full transcript, chart cards, `require_spec` gating |
| Game master (TRPG) | [`examples/game_master/`](examples/game_master/) | BGM/image/expression presentation tools, dice, full goal/flag/variable state, drift-free `[State]` view |
| Coding agent | [`examples/coding_agent/`](examples/coding_agent/) | sandboxed file tools, real test runs, red→green fix cycle |

Run the live ones (needs `.env`, see below):

```bash
python examples/quickstart.py
python -m examples.customer_support.run_live
python -m examples.game_master.run_live
python -m examples.coding_agent.run_live
```

## Testing

```bash
pip install -e ".[dev,openai]"
pytest tests --ignore=tests/integration     # offline: unit + acceptance + scenarios
```

The offline suite includes the spec's acceptance criteria: (a) ≤3k-token tool
overhead at 1,000 registered tools, (b) full reachability with vectors off,
(c) the validation→spec→retry self-repair path, (d) default-config chat.
Scenario tests drive the *real* tools (files, subprocesses, manuals) with a
scripted model.

Live integration tests (real DeepSeek API + GGUF embedding download):

```bash
cp .env.example .env     # fill in DEEPSEEK_API_KEY, set SPAL_RUN_LIVE=1
pip install -e ".[embeddings]"
pytest tests/integration -q
```

**Never commit `.env`** — it is gitignored; keys stay out of the repository.

## Spec conformance & deliberate deviations

Implements the 射影ループ設計仕様書 v1.0 including invariants I1–I11, and the
two fixes identified in its review:

1. **Hooks** (`after_decide` / `after_execute`) — constrained middleware; the
   hook API cannot violate the projection invariants.
2. **Epoch cache class + TOC as its own section + `ToolProvider`** — the kernel
   stays immutable (I4) while the tool index may change mid-session; external
   tool sources plug in without touching the core.

Also included from the review's "small fixes" list: a per-iteration interrupt
check (`session.interrupt()`), and multimodal-ready message content.

## Known limitations (accepted, per spec §16)

Candidate quality depends on query phrasing (layers 1/3 remain as fallback);
summary updates invalidate part of the prefix cache (amortized by rarity);
lazy specs cannot catch wrong-but-schema-valid arguments (use `require_spec`
or pinning); observation labeling mitigates but does not prevent prompt
injection.

## License

MIT
