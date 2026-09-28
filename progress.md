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
| — | Final whole-branch review | ✅ | `e70aac2`..`795b521` | 3 Important + all Minors fixed; 368 tests |

### Phase 2 final review — resolved

- **IMPORTANT — HyDE's safety net was decorative.** Its docstring claimed searching the
  question alongside the hypothetical document meant a bad hypothetical degrades rather
  than replaces the result. Measured: question scores 0.40–0.33, hypothetical 0.79–0.74,
  and `merge_best_score` took **0/5** from the question. `include_question=True` did
  nothing. Now fuses by rank; **2/5** final chunks come from question-only hits.
  The irony worth keeping: `rag_fusion.py`'s own docstring explains that lists from
  different queries are not on a shared scale, and HyDE is where that is most true.
- **IMPORTANT — a missing API key silently made every strategy `direct`, exit 0.**
  Confirmed by moving `.env` aside. Phase 3's benchmark would have measured this as
  "no technique helps". Now exits 1 for any non-direct strategy; `direct` still
  degrades usefully.
- **IMPORTANT — decomposition's `trace.queries` listed the original question**, which
  is never searched. Broke the invariant that `queries` is exactly what was embedded.
  Now lists only the sub-questions.

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

---

# Phase 3 — Evaluation Harness: Progress

Branch: `phase-3-evaluation` (stacked on `phase-2-query-translation`; none merged to `main`)
Plan: [docs/superpowers/plans/2026-09-27-phase-3-evaluation-harness.md](docs/superpowers/plans/2026-09-27-phase-3-evaluation-harness.md)

| # | Task | Status | Commits | Notes |
|---|------|--------|---------|-------|
| 1 | Retrieval depth vs top_k; optional generation | ✅ | `cc994e6` | 374 tests; fusion now differs from union |
| 2 | Gold loading, quote → span | ✅ | `905f672`, `76ca130` | multi-quote schema |
| 3 | Span → overlapping chunks | ✅ | `5b8826e` | overlap, not containment |
| 4 | Recall@k, MRR, nDCG | ✅ | `9ce76fa`, `40a22ea` | + doc-precision |
| 5 | Ten gold questions | ✅ | `b8f57be` | 10 questions, 17 quotes, 10 docs |
| 6 | Benchmark runner, CLI, README | ✅ | `c58970e`..`03e055f` | 439 tests |
| — | Final whole-branch review | ✅ | `1aa1c3c`..`0e1da3a` | 3 Important + minors fixed; 450 tests |

### Phase 3 carried items

- **Task 1, Minor (test coverage, not a defect)** — the implementer said no existing
  tests needed adjusting because `retrieval_depth=20` exceeds the fixture corpus's
  chunk count. The conclusion was right, the mechanism wasn't: fixture behaviour *did*
  change (per-query candidates went from ~3 to ~11), but no existing assertion checks
  chunk *identity* — only counts, ranks and a score inequality. So the strategy tests
  never exercised the depth-vs-k distinction at all, which is exactly what the plan
  suspected. Not worth adding identity assertions to fixture tests (brittle); the real
  coverage arrives with Task 6's benchmark, which measures identity by construction.

### Phase 3 — recorded deviations

- **The plan's gold-set acceptance bar was not met, and was overridden.** Task 5 Step 7
  said: "A mean Recall@5 around 0.4-0.8 is a healthy gold set. If it is near 0.0,
  something is wrong with the spans rather than with retrieval; investigate before
  proceeding." Measured: **0.050**, both before and after the multi-span fix.

  Investigated rather than ignored. Retrieval is not broken: relevant chunks rank 4th
  to 64th out of 5,116, and for several questions the top-5 are all from the correct
  paper and genuinely answer the question — the gold set simply names some answering
  passages, not all. Multi-span widened the relevant set but could not move Recall@5,
  because with 5,116 chunks and 1-4 relevant per question the metric has a low ceiling
  regardless of gold-set quality.

  **The bar itself was wrong.** It was written before any measurement existed, and it
  assumed a smaller corpus or broader relevance than this project has. Superseded by:
  chunk metrics at k=20 (Recall@20 = 0.325, real headroom) plus document-precision@5
  (0.560, spread 0.0-1.0, so it discriminates). Recording it here because a future
  reader should not have to wonder why an explicit gate was never satisfied.

- **Gold-set mix is thin on decomposition.** The plan suggested 2 two-fact questions;
  only `colbertv2-tradeoff` is clearly one. Actual mix: ~4 direct, 3 vocabulary
  mismatch, 2 general/step-back, 1 decomposition. Worth correcting when Phase 7 grows
  the set to 30, not worth redoing now.

- **Queued Minor** — `evaluation/gold.py`: a `quotes` list containing a non-string
  raises a bare `TypeError` from `str.count`, with no question id, breaking the
  module's otherwise-consistent "every error names the question" contract.

- **Queued Minor** — the plan's own "A refinement to the spec's gold-set format"
  section still describes the single-quote model and was never updated for multi-quote.

### Phase 3 — the latency column was measuring the wrong thing

The first published table had a `Mean ms` column reporting rag-fusion at 94 ms and
multi-query at 42,680 ms. rag-fusion is not 450x faster. Both build their rewrite
prompt from the same `MULTI_QUERY_TEMPLATE`, so they share an LLM cache key, and
`STRATEGY_NAMES` happens to order multi-query first — it paid for every rewrite and
rag-fusion got them free. Reordering the tuple would have swapped which looked fast.

Replaced with two order-independent columns: **LLM calls** per question (counted
logically, cache hit or not) and **Mean ms (warm)** (each strategy warmed before being
timed). Verified by running the two strategies in both orders and getting identical
numbers.

The retrieval scores were unaffected — all four score columns are bit-identical before
and after. What changed is the cost story, and one claim that rested on it: the
"decomposition costs ~2000x direct" line was an artifact of cold-cache timing. The
real premium is 3.4 LLM calls per question versus 0-1, and 110 ms warm versus 42 ms.
Still the most expensive strategy, for the smallest recall gain, but by 2.6x rather
than three orders of magnitude.

### Phase 3 final review — resolved

Verdict was "merge after fixes"; the reviewer hunted for a third scoring defect and
found none, verifying each suspect empirically rather than by argument.

- **IMPORTANT — `Mean ms (warm)` excluded the merging the README said it measured.**
  Confirmed: a rag-fusion run recorded only `translate, embed, search`. Fixed by
  timing the merge in all four fusing strategies, which makes the claim true rather
  than deleting it. Decomposition's merge already sits inside its `decompose` stage.
- **IMPORTANT — the benchmark scored a disk index against freshly recomputed chunks.**
  Chunk ids are positional, so a corpus change after indexing would have produced
  colliding ids over different text and a silently wrong table. Clean today; fixed by
  scoring against `store.chunks`, which also removed duplicated work.
- **IMPORTANT — `score_strategy` had no test**, despite producing every published
  number. Now covered end to end against hand-computed values.
- **README honesty.** "Bit-identical across runs" was cache determinism, not method
  determinism — `.cache/` is gitignored, so a clean clone re-samples the model and
  gets different numbers. The table is one draw over 10 questions with no variance
  estimate, and the README now says so. Also corrected a false claim in the Phase 3
  plan that multi-quote had created Recall@5 headroom; it did not (0.050 before and
  after) — what created headroom was measuring at k=20 and adding the document metric.

All four score columns are byte-identical before and after every fix.

---

# Phase 4 — Routing and Query Construction: Progress

Branch: `phase-4-routing` (stacked on `phase-3-evaluation`; none merged to `main`)
Plan: [docs/superpowers/plans/2026-09-28-phase-4-routing-and-query-construction.md](docs/superpowers/plans/2026-09-28-phase-4-routing-and-query-construction.md)

| # | Task | Status | Commits | Notes |
|---|------|--------|---------|-------|
| 1 | Topic metadata | ✅ | `f3cec5b` | 9/9/8/8/4 across five topics |
| 2 | Index carries document metadata | ✅ | `e4799f2` | 461 tests; index rebuilt |
| 3 | Structured output on the LLM | ✅ | `f2298e7` | |
| 4 | MetadataFilter + mask compilation | ✅ | `c7518c2` | |
| 5 | Apply the mask before top-k | ✅ | `02a55d0` | 495 tests; k results, not survivors | |
| 6 | Query construction | ✅ | `9204195` | | |
| 7 | Logical + semantic routing | ✅ | `8176111` | | |
| 8 | Pipeline wiring | ✅ | `6ae4a12`, `d4f415d` | 527 tests; exemplar routing 7/9 | | widens the benchmark degradation check |
| 9 | CLI, step-back via structured output, README | ✅ | `396d98e`..`2eb8b46` | 519 tests (8 prose tests deleted) | | deletes the prose heuristics |
| — | Final whole-branch review | ✅ | `0b21186`..`82f677c` | 4 Important fixed; routing measured; 543 tests |

### Semantic routing: descriptions lose to exemplars

Measured over nine questions with known intent, routing by cosine against short
*descriptions of question types* got 5/9 right — "What is late interaction?" and
"Define reciprocal rank fusion." both routed to `comparison`. The descriptions are
not the problem: they sit 0.18-0.38 cosine apart, comfortably separated.

The problem is what is being compared. Matching a question against a description of
a question *type* asks the embedder for intent; MiniLM embeds topic. Replacing the
descriptions with **exemplar questions** ("What is X? What does X mean? Define X.")
matches question-shaped text against question-shaped text and scores **7/9**.

That is the same insight HyDE rests on — embed something shaped like what you are
searching for — applied to routing rather than retrieval. Folded into Task 8.

### Step-back after structured output: churn, not regression

The conversion moved step-back's row — Recall@20 0.475 to 0.383 — which reads as a
regression and is not one. Scored question by question, **five of ten changed and
they moved both ways**: `raptor-clustering` 0.33->0.67 and `cot-limits` 0.00->0.50
improved; `selfrag-tokens` 1.00->0.00, `crag-quality` 1.00->0.50 and `hyde-problem`
0.50->0.25 worsened. Net negative, but with half the set flipping in both directions
and each question worth up to 0.1, n=10 cannot distinguish it from chance.

One explanation was tested and disproved: the JSON prompt does return longer questions,
and the old prompt carried a brevity instruction the new one dropped — but adding it
back leaves the score at exactly 0.383. Verbosity is not the cause.

Kept anyway. Structured output replaced four successive attempts to parse one line of
prose, each beaten by a reply shape the last had not anticipated. An unmeasurable
retrieval difference is worth a parsing path that cannot silently pick the wrong line.

### Phase 4 carried items

- **Sentinel coverage, queued for Task 9.** Sweeping every `trace.note` in `rag/` for
  failure-shaped wording found two without the `degraded` sentinel the benchmark now
  hard-fails on:
  - `decomposition.py` — "sub-question failed, continuing without it". A genuine
    partial degradation: the strategy carries on with fewer sub-answers, so its
    benchmark numbers would be quietly based on less work than the method specifies.
    **This is the exact blind spot the Phase 3 final review called latent** and it is
    still open. Add the sentinel.
  - `generation.py` — "generation failed". Not a degradation: there is no fallback,
    the answer is simply absent, and the benchmark runs with `generate=False` so it
    never fires there. Leave it, but say so in a comment so the next sweep does not
    re-raise it.
  The deeper point for the final review: the sentinel is a contract enforced only by
  convention, and it has now been missed twice by two different authors.

- **Task 7, IMPORTANT — FIXED** (`d4f415d`). `logical_route` has three failure paths; two note
  "degraded to searching everything", the third — the model returning topics that are
  all hallucinated or an empty list — notes "chose nothing valid; searching
  everything", with no `degraded`. Confirmed by grepping the notes. Task 8 widens the
  benchmark's check to that single word, so this one failure mode would go uncounted:
  routing could silently fall through on every question and the benchmark would read
  it as "routing does not help". Exactly the failure Phase 2 already hit once. One-line
  fix; queued behind Task 8, which is editing the same file.

- **Tasks 3-5, Minor** — `_check_shape` treats a bool as a valid `integer`, since
  `isinstance(True, int)` is True in Python. Latent only: no schema in the project
  declares `integer` or `boolean` yet. Worth fixing before one does.
- **Tasks 3-5, Minor** — `_extract_json_object`'s greedy `\{.*\}` spans first brace to
  last, so a reply mentioning a brace before the real object fails to parse. It fails
  *safe* (raises, caller degrades) rather than parsing the wrong object; the cost is
  spurious degradation, not wrong output.
- **Tasks 3-5, Minor** — author matching is case-insensitive substring, right for
  "Khattab" against "Khattab and Zaharia" but over-broad for a one- or two-character
  constraint. A note for Task 6's `build_filter`, which is the only producer.

- **Task 1, IMPORTANT — FIXED** (`natural_questions` -> `retrieval-models`, index rebuilt). `natural_questions` is classified
  `evaluation-benchmarks`, but the document is Lee et al.'s ORQA paper, "Latent
  Retrieval for Weakly Supervised Open Domain Question Answering" — it *proposes* a
  retrieval method (pre-training a retriever with an Inverse Cloze Task), and belongs
  in `retrieval-models` beside DPR and Contriever, which it precedes methodologically.
  Confirmed by reading the document. The `doc_id` is a Phase 1 misnomer: the title was
  verified against the content back then and matched, but nothing ever checked the
  *filename* against the content, and the name is what misled the classification.
  Moving it gives 9/10/9/8/7, still under half. Requires an index rebuild, so queued
  behind Tasks 3-5.
- **Task 1, Minor (leave)** — `hyde` sits in `prompting-reasoning` though it produces
  embeddings for dense retrieval. Defensible: the bucket means "LLM-prompted query
  transformation" alongside step-back and query-rewriting, not "retrieval model
  families". `rankgpt` in `evaluation-benchmarks` is similarly borderline and similarly
  acceptable.

### Phase 4 final review — resolved

Verdict was "merge after fixes". Four Important, all fixed and verified.

- **`GeminiLLM.structured` poisoned its own cache permanently.** It cached the reply
  *before* validating it, so a malformed response was written under a deterministic
  key and re-read forever — reproduced across a fresh client making zero API calls.
  With the `degraded` sentinel now fatal, one bad reply would have aborted the
  benchmark permanently until someone cleared `.cache/` by hand. Fixed by evicting the
  entry on validation failure.
- **The sentinel was missed five times, by four authors**, including in a file whose
  own docstring promised it. Now mechanical: `Trace.degraded(what, fallback)` writes
  the word, `benchmark.py` imports the same constant instead of restating it, and an
  AST test walks every `.note()` call in `rag/` against a three-entry allowlist with
  written reasons. Six sites converted.
- **Routing shipped unmeasured — my error.** The plan promised a `--route` benchmark
  column; the final task's dispatch (which I wrote) asked instead for a paragraph
  explaining its absence. The gold set measures routing for free: every question names
  a `doc_id`, every document carries a `topic`.
- **`SemanticRouter` was rebuilt per question**, re-embedding its exemplars on every
  call while its docstring claimed otherwise.

### Does routing help? Measured: no.

| Recall | Abstention | Mean topics chosen | Available | Questions |
|---:|---:|---:|---:|---:|
| 0.900 | 0.000 | 1.30 | 5 | 10 |

The router is not gaming the metric: it never abstains and picks 1.3 topics of 5, so
0.900 is real narrowing that mostly finds the right collection.

End to end it still does not pay. Mean Recall@20 across the six strategies falls from
0.394 to 0.378 with `--route`: up for `hyde`, flat for two, down for three. On a corpus
of 5,116 chunks, plain top-k already finds most answers, so the one question in ten
where a 90%-accurate router excludes the right topic costs more than narrowing saves.

**The single miss is probably my labelling, not the router.** `longcontext-position`
asks "does it matter where in a long prompt the relevant passage sits?"; the gold
document is `lost_in_middle`, which I filed under `evaluation-benchmarks`, and the
router chose `prompting-reasoning` — a defensible reading. Deliberately **not**
reclassified: changing the partition after seeing which question the router missed is
fitting labels to the metric, which is worse than an honestly earned 0.900.

---

# Phase 5 — Indexing Techniques: Progress

Branch: `phase-5-indexing` (stacked on `phase-4-routing`; none merged to `main` yet)
Plan: [docs/superpowers/plans/2026-09-28-phase-5-indexing-techniques.md](docs/superpowers/plans/2026-09-28-phase-5-indexing-techniques.md)

| # | Task | Status | Commits | Notes |
|---|------|--------|---------|-------|
| 1 | Synthetic node conventions (`level`, `-1` spans) | ✅ | `327ace5` | 550 tests; real index still loads, 0 synthetic |
| 2 | K-means in NumPy | ✅ | `35b30ad`, `0d13bb3` | 565 tests; determinism verified across processes |
| 3 | Summarisation | ✅ | `dee1fcb` | 574 tests; LLM disk cache confirmed empirically |
| 4 | Multi-representation indexing | ✅ | `2a69a25` | 588 tests; docstore round-trips, legacy index loads |
| 5 | RAPTOR tree building | ✅ | `738d122` | 602 tests; 3 invariants verified independently |
| 6 | Index modes | ✅ | `2f16596` | 613 tests; per-mode paths, no-llm guard raises |
| 7 | Retrieval against the new indexes | ✅ | `34e2de6` | 620 tests; expansion covers all six strategies |
| 8 | Build, measure, write up | 🔄 | `ce63de3` | 623 tests; `--index` flag done, builds running |
| — | Final whole-branch review | ⬜ | | |

**Key decision.** RAPTOR and multi-representation build their own index files
(`data/index-raptor.npz`, `data/index-multirep.npz`). The spec asks for one
`VectorStore` spanning all levels, which holds *inside* the RAPTOR index — but the
benchmark scores gold spans against `store.chunks`, so adding summary nodes to the
default index would move all six baseline rows for reasons unrelated to the
strategies. Each technique is measured as its own run via `--index`.

**Expected measurement artifact.** The gold set marks chunks overlapping an answering
span. A multi-representation index contains no such chunks — only 38 summaries — so
Recall@20 may be exactly 0 there. That is the metric failing to see the technique, not
the technique failing; `DocPrec@5` is the only column that means anything for it.

### Open finding: silent document loss in multi-representation

The first multirep build indexed **36 of 38 documents**. `ircot` and `sbert` were
dropped and *nothing recorded it* — `build_index` never passes a `trace` to
`build_multi_representation`, so its `trace.degraded(...)` branch is dead code in
production. The CLI printed "indexed 36 documents", which looks fine unless you
happen to know it should be 38.

Both documents summarise successfully on retry, so the cause was transient
(rate-limiting during the 38-call burst) and the existing backoff was not enough.
The same hole applies to RAPTOR: failed cluster summaries are skipped with no
record either.

Being fixed before any number is reported: a partial index must announce itself and
record the loss in `meta`, because an index quietly missing 5% of the corpus is
exactly the "plausible nonsense, no error" failure `_index_meta` exists to prevent.
