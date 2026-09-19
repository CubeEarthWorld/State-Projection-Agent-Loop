# History compression

Every turn re-renders the prompt from the Event Ledger. The ledger keeps
everything; the projection decides how much of it the model sees. This
document is the contract for that decision, shared by the Python and Dart
ports and pinned by `spec/fixtures/projection.json`.

## Design goals, in order

1. **Nothing the task needs is lost.** What a later step most often needs
   to look back at is: what the user asked for, what went wrong, what was
   decided and why. Those survive first.
2. **Deterministic first, model second.** The tiers below need no model
   call. An optional LLM *fold* into the structured `WorkingState` is a
   second layer, and its output is validated before it is trusted.
3. **Byte-stable prefixes.** A provider's prompt cache only hits when the
   prompt's prefix is identical to the last turn's. Compression that
   rewrites old messages a little every turn buys nothing and costs a full
   cache rebuild per turn; here old messages change only at deliberate
   steps.
4. **No task or language assumptions.** Tiers are chosen by a message's
   role and its position, never by its content's meaning; error detection
   uses structural traces (exit codes, stack frames) and a small
   multilingual word list, and a *failed* call is kept readable whatever
   its text says.

## The tiers

`WorkingState.verbatim_sequence` is the ledger sequence from which history
renders verbatim. Older messages are tiered by their distance from that
point, counted in messages of every role:

| distance from the point | user | assistant | tool result |
|---|---|---|---|
| tail (at or after the point) | verbatim | verbatim | verbatim |
| next `compressed_window` (24) | verbatim | noise stripped, head+tail (`compressed_max_lines`) | **masked**: first line + size — unless the call failed or reports an error, then head+tail (`observation_max_lines`) |
| next `summary_window` (60) | verbatim | first meaningful line + size | first meaningful line + size |
| older | verbatim | dropped | dropped |
| folded (≤ `folded_sequence`) | verbatim | dropped | dropped |

`pair_tool_calls` then removes any decision whose results are not all
present (and vice versa), so a tier boundary never leaves a provider with
an unanswered tool call.

The masked form of a tool result is deliberate: a result the model already
acted on rarely needs re-reading, and clearing it is where most of the
savings are (the finding of several 2025 agent-trajectory studies is that
masking old observations matches LLM summarization on task success at
roughly half the cost). What does need re-reading — an error — stays
readable. Large results are artifacts anyway; their reference survives the
mask and `meta.artifact.peek` can always go back.

## When the point moves

`Session._step_tiers` runs before every projection. The verbatim point
moves only when the tail has grown to **four times `full_window`**, and
then it is cut back to `full_window` messages. Between steps, every older
message renders exactly as it did the turn before, so the rendered prefix
is byte-identical and grows only at the end. A step is one deliberate
cache rebuild every few turns, instead of one every turn.

The point is part of the working state, so it survives a restart and
`rewind` restores it with the rest.

## The fold

With `compaction.trigger_ratio > 0`, a prompt that exceeds that fraction of
the *room* — the window less `reserved_output_tokens` and provider
overhead, which is what the render actually budgets messages and schemas
against — triggers one model call that folds history into the working
state as a JSON delta (`facts_add`, `decisions_add`, `questions_add`,
`questions_resolve`, `next_actions`). Measured against the whole window the
trigger is unreachable as soon as the reserve exceeds the slack (a 4k
window with the default 1k reserve never folded); measured with the
reserve counted it fires nearly every turn. Three rules keep it honest:

- **Fold from the ledger, never from the projection.** What masking
  cleared from the prompt is exactly what the fold must still read. The
  region is everything between the last fold and the verbatim point, so
  the fold changes only what the tiers had already stopped rendering in
  full, and it happens at most once per step of the point — the moment
  the prefix is rebuilt anyway — with a step's worth of messages to read.
  Between steps an over-full prompt is handled by the deterministic
  shrink, which drops from the front what the next fold will read from
  the ledger. (Forcing the point down to fold sooner gave a fold every
  turn under a window the verbatim tail alone overflows: a model call and
  a cache rebuild per turn, and recall fell rather than rose.)
- **Shape is validated** with the same JSON Schema validator as tool
  arguments.
- **Entries are grounded.** An entry that names an identifier, path or
  number the transcript never mentions is the mark of an invented fact; it
  is dropped and listed under `ungrounded` in the `state_folded` event.
  Plain-language entries need no grounding.

Folded messages keep only the user's words in the prompt; their substance
lives in the working state, and their text stays in the ledger for
`meta.history.search`.

## Measured

`evals/compression_eval.py` drives a 30-turn tool-using conversation
against a real model, then asks questions whose answers sit at known
depths, graded by exact substring. With `deepseek-flash`, a 12k window and
two seeds:

| arm | accuracy | user-stated facts | prompt tokens / turn | cache hit ratio |
|---|---|---|---|---|
| previous design (tiers by age, every message rewritten each turn) | 33% | 0% | 2,546 | 31% |
| this design (tiers by distance, stepped point, masked results) | 67% | 100% | 2,372 | 79% |

The remaining misses are a fact buried in a tool result thirty turns back
(masked, by design; the fold is the layer that keeps such facts) and one
ambiguous correction. Run the eval yourself; the numbers above are a
snapshot, not a guarantee.

## Configuration

```python
"compression": {"full_window": 6, "compressed_window": 24, "summary_window": 60,
                "compressed_max_lines": 80, "observation_max_lines": 40},
"compaction":  {"trigger_ratio": 0.0},   # 0 = no fold
```

Windows count messages of every role. `full_window` is the smallest
verbatim tail; the tail grows to four times that before a step.
