# Phase 1 — Core Pipeline: Progress

Branch: `phase-1-core-pipeline`
Plan: [docs/superpowers/plans/2026-09-27-phase-1-core-pipeline.md](docs/superpowers/plans/2026-09-27-phase-1-core-pipeline.md)
Spec: [docs/superpowers/specs/2026-09-27-rag-from-scratch-design.md](docs/superpowers/specs/2026-09-27-rag-from-scratch-design.md)

Status key: ⬜ not started · 🔄 in progress · ✅ complete

| # | Task | Status | Commits | Notes |
|---|------|--------|---------|-------|
| 1 | Scaffolding, Config, test fixtures | ✅ | `1ae1bad` | 9 tests pass; review clean |
| 2 | HTML text extraction | ✅ | `77f3c41`..`b497946` | 31 tests pass; 3 review rounds, all findings fixed |
| 3 | Corpus fetch script + fetch corpus | ✅ | `cdd5aee`, `ae41c4f` | 38 docs, all titles verified; 46 tests |
| 4 | Document loader | ✅ | `e6943af`, `e138637`, `f209fad` | 16 tests; all malformed-metadata shapes now actionable |
| 5 | Token-aware chunking | ✅ | `f72b505` | 18 tests; 5,116 chunks, offsets independently verified |
| 6 | Embeddings | ✅ | `dfde3c1` | 9 fast + 4 slow; review clean, no findings |
| 7 | Similarity and top-k | ✅ | `7fd1286` | 16 tests; review approved, 2 Minor |
| 8 | Vector store | ✅ | `1937bf1` | 13 tests; 6-18 ms search over 5,116 chunks; review approved |
| 9 | Trace | ✅ | `673a407` | 8 tests; review approved |
| 10 | Gemini client (cache + retry) | ✅ | `31a1fba`, `920eea4` | 16 tests; live call + cache verified |
| 11 | Prompt template and generation | ✅ | `ff64a4f` | 16 tests; grounding verified against off-corpus question |
| 12 | Pipeline | ✅ | `a093547` | 16 tests; full index built, end-to-end ask() works |
| 13 | CLI | ✅ | `bf41261`, `461c227` | 18 tests; Windows encoding crash fixed |
| 14 | README + no-frameworks guard | ✅ | `adc41d3`, `9f9c811`, `536dfb1` | 211 total; Quick start run end to end |
| — | Final whole-branch review | ✅ | `82d347a`..`858497d` | 5 Important + minors, all fixed |

**Phase 1 complete.** 246 tests passing, plus 4 slow tests against the real model.
Final review found 5 Important issues; all are fixed.

Working system: 38 documents → 5,116 chunks → 9.0 MB index; search 2–4 ms;
grounded answers with citations; `--no-llm` runs with no API key.

```
python -m rag index
python -m rag ask "How does ColBERT score a document?" --trace
```

Detail on findings, decisions and deviations:
[docs/superpowers/phase-1-notes.md](docs/superpowers/phase-1-notes.md)

---

# Phase 2 — Query Translation: Progress

Branch: `phase-2-query-translation` (stacked on `phase-1-core-pipeline`; neither merged to `main` yet)
Plan: [docs/superpowers/plans/2026-09-27-phase-2-query-translation.md](docs/superpowers/plans/2026-09-27-phase-2-query-translation.md)

| # | Task | Status | Commits | Notes |
|---|------|--------|---------|-------|
| 1 | Trace depth, score labels, translation steps | ✅ | `ff3ab86` | 254 tests; review clean |
| 2 | Reciprocal rank fusion + best-score merge | ✅ | `3d2347e` | 269 tests; RRF math hand-verified |
| 3 | Strategy protocol, context, direct | ✅ | `28e5166` | 282 tests; no import cycle |
| 4 | Multi-query | ✅ | `4153a81` | 295 tests; beats direct on a real question |
| 5 | RAG-Fusion | ✅ | `87e661e` | 301 tests |
| 6 | Step-back | ✅ | `d0a60a1` | 307 tests |
| 7 | HyDE | ✅ | `8d41135`, `4c19628` | 321 tests; incl. parser fixes |
| 8 | Decomposition (recursive + independent) | ✅ | `2c90cff` | 334 tests; nesting verified |
| 9 | CLI `--strategy`, rendering, README | ✅ | `0ec17a5`, `dbd3eeb`, `1e4ba4f` | 352 tests |
| — | Final whole-branch review | 🔄 | | |

### Notes for the Phase 3 plan

Two design constraints the final review surfaced. Both would distort the benchmark if
missed, so fold them into the Phase 3 plan rather than discovering them in the numbers.

- **Stage names are not comparable across strategies.** Four strategies emit a
  `translate` stage that is LLM-only; decomposition emits one `decompose` stage that
  swallows retrieval *and* N generations. `total_ms` is comparable; per-stage timings
  are not. Do not build a per-stage latency table across strategies.
- **`top_k` is used as both the per-query retrieval depth and the final truncation**, so
  RAG-Fusion fuses five 5-item lists down to 5. That structurally compresses the
  difference between `rag-fusion` and `multi-query` — the benchmark could report "no
  difference" for reasons of plumbing rather than method. Phase 3 should give the
  per-query depth its own knob, larger than the final k.
- **Benchmark should hard-fail on any "degraded to direct" note**, not just trust exit
  codes, so a silent LLM outage cannot be measured as "no technique helps".

### Phase 2 carried items

- **Task 9, Minor x2** — `format_trace` prints `Strategy:` twice in verbose output when
  translation steps exist; `--decomposition-mode` is silently ignored when the strategy
  is not decomposition, which is inconsistent next to `--no-llm`, a flag that *is*
  validated. Both cosmetic, both deferred to the final review's fix pass.
- **Step-back took four attempts** — first line (searched the preamble), last candidate
  (searched the sign-off), shape/`?` (defeated by a preamble phrased as a question),
  and finally chatter-filter + shape with a fallback. Nine reply shapes now pass. The
  code documents the ceiling: the principled fix is structured output, which Phase 4
  introduces for logical routing. Step-back should adopt it then rather than accruing
  more rules.

- **Task 8, Minor** — recursive decomposition accumulates prior Q/A pairs into each
  later sub-question's prompt with no length cap; the "two sentences at most" limit is
  prompt-only, not enforced. Harmless at the default `max_sub_questions=3` (~2 pairs),
  unbounded in principle. `max_sub_questions` is not exposed on the CLI, so nothing can
  reach the bad case today. Revisit if it ever is.

- **Task 6, IMPORTANT — FIXED.** `step_back.py` searched the model's preamble and
  discarded the real question. The first fix attempt (take the *last* candidate) traded
  one failure for another: preamble worked, sign-off broke — it assumed the model obeys
  the prompt, which is what caused the original bug. Now identifies the question by
  *shape* (prefers a line ending in `?`, strips markdown emphasis, falls back to the
  longest candidate). Verified against six reply shapes including preamble-only,
  sign-off-only, both, and bolded.

- **Tasks 4/5, Important — FIXED** (`8d41135`) — `parse_query_list` kept the model's
  preamble and sign-off as search queries. Confirmed: a reply wrapped in "Here are five
  queries:" / "I hope these help!" yielded two junk queries. Harmful for RAG-Fusion in
  particular, where junk results fuse in as independent evidence. Now prefers marked
  lines when any marker is present. Same commit normalised near-duplicate rewrite
  detection (casefold + whitespace) and truncated rewrites to `n`.
- **Task 3, Minor** — `build_answer_prompt`'s docstring lost Phase 1's note explaining
  that `str.format` only scans the template, so chunk text containing `{}` cannot
  inject. Behaviour is unchanged; the comment is what stops someone reintroducing the
  bug. Restore in a later batch.
- **RESOLVED (was Task 3 ⚠️)** — verified all three multi-query degradation paths
  (no LLM, `LLMError`, unusable reply) leave `queries=[question]` and zero translation
  steps. No stale rewrites on a degraded trace.

### First evidence a strategy helps

Question: *"What problem does HyDE solve?"* against the real 5,116-chunk index.

- `direct` retrieved 4 `hyde` chunks and still answered: *"there is no mention of the
  specific problem that HyDE is designed to solve."*
- `multi-query` generated 5 rewrites (including *"zero shot dense retrieval failure
  modes"*) and answered: *"HyDE solves the difficulty of creating effective fully
  zero-shot dense retrieval systems when no relevance label is available [5]."*

One question is an anecdote, not a measurement — Phase 3's benchmark is what settles
whether this holds. But it is the first sign the technique does something real.
