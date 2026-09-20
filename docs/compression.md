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

With `compaction.trigger_ratio > 0` (the default is 0.75), a prompt that exceeds that fraction of
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

`evals/compression_eval.py` drives a 30-turn tool-using conversation of one
of five shapes (`evals/scenarios.py`: records lookup with corrections, a
coding loop with failing tests, support tickets, a dice game with an
inventory, a long contract read section by section), in English or
Japanese, against a real model. It then asks questions whose answers sit
at known depths, graded by exact substring, and reports what compression
is for: prompt tokens per turn and the provider's prompt-cache hit ratio.
Numbers below are `deepseek-flash`, two seeds per cell; a snapshot, not a
guarantee. `--offline` runs the same matrix against a scripted stand-in
for the model: no recall, but tokens per turn and a prefix-stability
proxy for the cache ratio, free, so a change to the tiers is checked for
size and byte-stability before the live matrix is paid for.

**Tiers alone, 12k window.** Recall per cell, and the cache hit ratio,
which is 0.85–0.88 in every cell and both languages: the prefix stays
byte-identical whatever the task or script.

| task | en | ja |
|---|---|---|
| records | 0.67 | 0.75 |
| coding | 0.80 | 0.80 |
| support | 0.70 | 0.80 |
| game | 0.90 | 0.70 |
| document | 0.75 | 1.00 |
| all | 0.76 | 0.81 |

Against the previous design on the records task (tiers by age, every
message rewritten each turn): recall 0.33 → 0.67, user-stated facts 0 →
100%, cache hit ratio 0.31 → 0.86.

**Where the misses are.** By question kind over all cells (6k window,
tiers alone): facts the user stated 0.83, the latest tool result 0.95,
corrections 0.69, abstaining when nothing was stated 1.00 — and a fact
buried in a tool result older than the compressed window 0.38, the text of
the first error 0.00. Those two are what masking discards by design; they
are the fold's job, not the tiers'. The remaining misses are model
behaviour on a verbatim user message (the Japanese game cell answers "the
first roll" with the first roll whose result is still visible).

**The fold.** Under a 6k window across all ten cells, `--fold 0.75`
against the same cells with tiers alone (`evals/results/matrix_6k.json`
vs `matrix_6k_fold.json`):

| runs | recall | cache hit ratio | prompt tokens / turn |
|---|---|---|---|
| where a fold fired (all Japanese cells, both document cells) | 0.75 → 0.98 | 0.81 → 0.66 | 3,573 → 3,613 |
| where none fired (English cells under the estimator's threshold) | unchanged projection | 0.86 | 2,676 |

By question kind, folds move exactly what masking discards: a fact in a
tool result older than the compressed window 0.38 → 0.75, the first
error's text 0.00 → 0.50, and nothing else got worse (latest tool result
0.95 → 1.00, user-stated facts 0.83 → 1.00). The cost is the cache: every
fold is a prefix rebuild. The first fold design (force the point down
whenever nothing older was left to fold) folded every turn of the
document task (19–28 folds in 30 turns) and recall there fell from 0.88 to
0.50 while the cache ratio fell from 0.90 to 0.56 — that is why a fold
now happens only at a step of the point (`matrix_6k_fold_forced.json`
keeps that arm).

## Configuration

```python
"compression": {"full_window": 6, "compressed_window": 24, "summary_window": 60,
                "compressed_max_lines": 80, "observation_max_lines": 40},
"compaction":  {"trigger_ratio": 0.75},  # 0 = no fold
```

Windows count messages of every role. `full_window` is the smallest
verbatim tail; the tail grows to four times that before a step.
