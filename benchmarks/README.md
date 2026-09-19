# Benchmark

Does the layered capability projection actually cost less than conventional
full-spec preloading, and does it cost anything in quality to find out?

## Method

Both arms run the **same `Session`**, the same tasks, the same model, the same
tools. Only how capabilities reach the model differs:

| arm | configuration |
|---|---|
| `preload` | every capability `pinned=True`, discovery off (`toc=False, k=0`). The conventional agent loop: every full spec, every turn. |
| `spal` | nothing pinned; TOC + BM25/tag candidate cards, `meta.tool.find` as fallback. |

Holding the runtime fixed is the point. A hand-written baseline loop would only
prove that two different harnesses are different.

Four tasks, graded deterministically (no LLM judge): a single lookup, a
two-step chain where the second call depends on the first, a fetch-then-sum,
and one whose wording deliberately shares no keywords with the tool that
answers it (`"deliveries running behind schedule"` ->
`logistics.shipment.delayed_count`) to stress discovery recall.

Registry scale `N` is swept with synthetic filler capabilities of three
parameter shapes, so spec sizes are realistic rather than uniform.

## Running it

```bash
# structural half - no API key, no spend, deterministic
python -m benchmarks.bench_offline --sizes 25,200,1000

# live half - needs LLM_API_KEY (DeepSeek) or ANTHROPIC_API_KEY in .env
python -m benchmarks.bench --provider deepseek --sizes 25,200,1000 --repeat 3
python -m benchmarks.bench --provider anthropic --model claude-haiku-4-5

# harness self-check, offline
python -m benchmarks.test_bench
```

## Results: deepseek-flash, 2026-09-19

72 runs, 3 per cell, temperature 0, medians reported.

| N | arm | success | recall | turns | prompt tk/turn | sec |
|--:|---|--:|--:|--:|--:|--:|
| 25 | preload | 10/11 | 12/12 | 2.0 | 6,596 | 2.3 |
| 25 | **spal** | **12/12** | 12/12 | 2.0 | **2,982** | 2.0 |
| 200 | preload | 11/11 | 12/12 | 2.0 | 35,486 | 4.0 |
| 200 | **spal** | 11/12 | 12/12 | 2.0 | **3,076** | 2.1 |
| 1000 | preload | 10/11 | 12/12 | 2.0 | 167,714 | 7.6 |
| 1000 | **spal** | **12/12** | 12/12 | 2.0 | **3,080** | 2.4 |

Success excludes three hard API errors (see Caveats); recall counts whether
every capability the task needs was actually called.

**Context cost is flat, not linear.** 2,982 -> 3,080 tokens/turn from N=25 to
N=1000 (+3%), against 6,596 -> 167,714 (+2,443%). The ratio is 2.2x / 11.5x /
54.5x. The package's `<3k tokens at 1,000 capabilities` claim holds: the
offline estimator predicted 2,748 and the provider billed 3,080.

**Discovery costs no extra round-trips.** Turn count is identical in every
cell. The predicted `meta.tool.find` tax did not materialise: BM25 put the
right tool in the native schema list on turn 1, every time, including the
keyword-mismatched task at N=1000 with vector search off.

**No quality cost, and none of the failures are about context.** Every single
non-successful run called the correct capability and then fumbled the final
answer - arithmetic slips on the summing task, a rephrased number on another.
Zero failures are attributable to either context strategy.

**Latency tracks context size** - 3.2x at N=1000 - and is not rescued by
caching the way cost is.

## Caveats

- **Caching flatters the cost gap's opposite direction.** 96% of the grid's
  prompt tokens were DeepSeek cache hits (preload 96%, spal 84%). Priced at a
  typical 10x cache discount the cost gap is ~13x, not the ~25x that raw
  token volume suggests. Three identical repeats per cell inflate the hit
  rate well above what varied production traffic would see.
- **The `preload` arm is slightly harsher than a true conventional loop.**
  Pinning also renders a `[Pinned tools]` card index into the kernel - 37k
  tokens at N=1000, ~22% of that arm's prompt - which a conventional loop
  does not pay. Native-schema-only preloading is 131,350 tokens at N=1000,
  so the fair structural ratio is ~48x rather than ~55x.
- **Three runs died on a provider error**, all in the `preload` arm:
  `reasoning_content in the thinking mode must be passed back`. That is a bug
  in `examples/llm_adapters.py` (it captures `reasoning_content` but never
  echoes it), not a property of the arm. They are excluded from success
  counts. The clustering in one arm is unexplained at n=3.
- **n=3, four tasks, one model, synthetic fillers.** Recall 12/12 is a floor,
  not a measurement of the recall ceiling. A registry whose capabilities
  overlap semantically would be a much harder discovery test than this one.
- `deepseek-flash`, not a frontier model. Bigger models may need less help
  finding tools, which would narrow the recall margin and leave only the
  token result.
- Offline token figures use the package's own `estimate_tokens`, not a
  provider tokenizer. It came within 0.4% of the billed count on the
  preload arm and under-read the spal arm by ~11%.

Raw data: `results-deepseek.json`, `results-offline.json`, `run.log`.
