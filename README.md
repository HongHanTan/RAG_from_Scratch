# RAG from Scratch

A retrieval-augmented generation pipeline built without a RAG framework. No
LangChain, no LlamaIndex, no vector database, no `sentence-transformers`. The
text splitter, the embedding pooling, the similarity math and the prompt
assembly are all written out. `tests/test_no_frameworks.py` enforces this by
failing the suite if any of them is ever imported.

All seven phases are complete, through the 30-question final benchmark
below. See the roadmap at the end, and the design spec for the full plan:
[`docs/superpowers/specs/2026-09-27-rag-from-scratch-design.md`](docs/superpowers/specs/2026-09-27-rag-from-scratch-design.md).

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env        # add your GOOGLE_API_KEY
python -m rag index
python -m rag ask "How does ColBERT score a document?" --trace
```

Indexing embeds the full corpus on CPU and takes about two minutes. Retrieval
works without an API key:

```bash
python -m rag ask "What problem does HyDE solve?" --no-llm
```

## How it works

| Stage | What happens | Where |
|---|---|---|
| Load | 38 documents on RAG and retrieval, fetched from ar5iv and committed | `rag/loader.py` |
| Chunk | Sliding window of 200 tokens with 50 overlap, over token ids | `rag/chunking.py` |
| Embed | all-MiniLM-L6-v2, mean pooling over the attention mask, L2 normalised | `rag/embedding.py` |
| Store | Dense `(n_chunks, 384)` float32 NumPy matrix | `rag/store.py` |
| Retrieve | Cosine similarity as one matmul, stable top-k | `rag/similarity.py` |
| Generate | Numbered context injected into a prompt template, sent to Gemini | `rag/prompts.py` |

Every stage writes into one `Trace` object (`rag/trace.py`), which is what
`--trace` prints and what the benchmark and dashboard will read in later phases.

The 38 documents total 3,064,396 characters and chunk into 5,116 pieces. The
resulting index (`data/index.npz`) is 9.0 MB. Search over those 5,116 chunks
takes 2–4 ms — brute-force cosine similarity, no approximate index. Embedding
the full corpus from scratch takes about 130 seconds on CPU.

## What grounding actually looks like

Asking a question the corpus answers returns a cited answer, retrieved almost
entirely from the two ColBERT papers in the corpus:

```
$ python -m rag ask "How does ColBERT score a document?" --trace
Based on the provided context, ColBERT scores documents using its
late-interaction operator [5], which is implemented via MaxSim operations [4, 5].

Sources:
  [1] colbertv2:41  score 0.580  chars 27782-28606
  [2] colbert:65    score 0.552  chars 41769-42501
  [3] colbert:46    score 0.539  chars 30381-30972
  [4] colbert:14    score 0.535  chars 9371-10214
  [5] colbert:41    score 0.530  chars 27035-27992

Timings:
  embed          96.2 ms
  search          3.2 ms
  generate     1565.7 ms
  total        1665.1 ms
```

(source excerpts and the full prompt sent, also printed by `--trace`, are
omitted above for length)

Ask the same question again and generation drops from ~1.5 s to ~2 ms,
served from the on-disk LLM cache — and the trace says so explicitly, rather
than silently reporting a generate time 1000x smaller and letting the reader
assume Gemini is just fast:

```
  generate        2.0 ms
  ...
note: generation served from cache
```

The system also declines rather than inventing an answer. No document in this
corpus explains reciprocal rank fusion, and the model says so instead of
answering from prior knowledge:

```
$ python -m rag ask "What is reciprocal rank fusion?"
The provided context does not contain information about reciprocal rank fusion.

Sources:
  [1] ir_llm_survey (chunk 81)
  [2] ir_llm_survey (chunk 87)
  [3] ir_llm_survey (chunk 88)
  [4] ir_llm_survey (chunk 348)
  [5] rankgpt (chunk 93)
```

The same happens with a question the underlying model certainly knows the
answer to, if asked without retrieval — the prompt forbids using anything
outside the retrieved context:

```
$ python -m rag ask "What is the capital of Australia and what is its population?"
The provided context does not contain information about the capital of
Australia or its population.
```

## Three things worth pointing out

**Chunking is token-aware, not character-aware.** MiniLM truncates at 256
tokens. A character-based splitter produces chunks whose tails the model
silently discards — no error, just quietly worse retrieval. Sliding over token
ids instead means every chunk fits by construction.

**Embeddings are 384-dimensional, not 3-dimensional.** Diagrams draw vector
space in 3D because 3D is drawable. Nothing here assumes three dimensions.

**Search is exact brute force, and that is a deliberate limit.** Scoring every
query against every chunk is one `(n_queries, 384) @ (384, n_chunks)` matmul,
which at a few thousand chunks takes a few milliseconds — faster than the
overhead an approximate index would add. Vectors are L2-normalised once when
the store is built rather than on every query; re-normalising 5,116 unit rows
per search cost more than the matmul itself, and removing it took search from
6.6 ms to 2.7 ms. It is O(n) per query. Past roughly a million vectors, FAISS
or HNSW becomes the right answer.

## Strategies

Phase 2 adds five query translation strategies on top of the `direct`
baseline, selected with `--strategy` (default `direct`) and shown with
`--queries` or `--trace`. Every strategy but `direct` needs an LLM to produce
its rewrites — combining any of them with `--no-llm` is rejected rather than
silently degrading to plain retrieval, and a translation strategy whose LLM
cannot be built at all (missing key, bad auth, exhausted quota) fails the run
rather than silently degrading to `direct`.

- **`direct`** — embeds the question as-is and searches once. Not a
  translation strategy itself; the baseline every one of the five is compared
  against.
- **`multi-query`** — asks the LLM for several differently-worded rewrites of
  the question, searches each, and merges results by best cosine score across
  all of them.
- **`rag-fusion`** — the same rewrites as `multi-query`, but merged by
  Reciprocal Rank Fusion instead of best score, so a chunk that ranks
  consistently well across queries outranks one that scores highest in only
  one.
- **`step-back`** — asks the LLM for one more general question about the
  underlying concept, and searches both the original and the general question.
  The general question is obtained via structured output (a JSON `{"question":
  "..."}` field), not parsed out of prose — see
  [Routing and query construction](#routing-and-query-construction) below.
- **`hyde`** — asks the LLM to write a short hypothetical passage that would
  answer the question, and searches with that passage's embedding (plus the
  original question, by default) on the idea that a fake answer is closer in
  embedding space to a real answer than the question is.
- **`decomposition`** — splits the question into sub-questions, answers each
  from its own retrieval, and feeds the sub-answers to the final prompt as
  working notes. Two modes via `--decomposition-mode`: `recursive` carries
  each sub-answer into the next sub-question's prompt (for parts that depend
  on each other), `independent` answers every sub-question on its own and
  concatenates them (cheaper, correct when the parts don't depend on each
  other).

Worked example — the rewrites `multi-query` actually generated for one
question, real output, not illustrative text:

```
$ python -m rag ask "How does ColBERT score a document?" --strategy multi-query --queries
ColBERT estimates relevance by having each query embedding interact with all
document embeddings via a MaxSim operator...

Strategy: multi-query
  query: ColBERT late interaction scoring mechanism explanation
  query: How does ColBERT compute relevance scores between query and document
  query: ColBERT max-similarity operator vector math
  query: Document ranking algorithm in ColBERT retrieval model
  query: ColBERT token embeddings scoring process

Sources:
  [1] colbert (chunk 38)
  [2] colbert (chunk 75)
  [3] colbert (chunk 46)
  [4] colbert (chunk 13)
  [5] colbertv2 (chunk 41)
```

Compare that to `--strategy direct` on the same question, which returns
`colbertv2:41, colbert:65, colbert:46, colbert:14, colbert:41` — three of the
five chunks differ. **Nothing here is measured.** The strategies retrieve a
different set of chunks than direct retrieval, and sometimes chunks that
direct's top-k misses entirely (as above), but nothing in this phase says
whether that set is *better* — more relevant, more sufficient for a correct
answer — only that it's different. Phase 3's evaluation harness (Recall@k,
MRR, nDCG against a gold set) is what will actually answer that question. Any
claim that these strategies "help" before Phase 3 runs would be a guess
wearing a lab coat.

## Benchmark

Retrieval — not answer quality — measured against the **30-question** gold set
(`evaluation/gold.json`), using `python -m evaluation.benchmark`. Phase 7 grew
the set from 10 questions to 30. Ten of the new ones are **cross-document**:
their answer needs passages from two or three papers, which the old gold
schema could not even express. All twenty were written, given a
**pre-registered prediction** of which strategy should win them, and
committed *before* anything was measured — the commit order is the evidence,
and no question was edited after its result was seen.

| Strategy | Recall@20 | MRR@20 | nDCG@20 | DocPrec@5 | LLM calls | Mean ms (warm) | Questions |
|---|---:|---:|---:|---:|---:|---:|---:|
| hyde | 0.512 | 0.335 | 0.310 | 0.533 | 1.0 | 123 | 30 |
| rag-fusion | 0.376 | 0.167 | 0.193 | 0.480 | 1.0 | 91 | 30 |
| multi-query | 0.314 | 0.202 | 0.199 | 0.460 | 1.0 | 90 | 30 |
| step-back | 0.300 | 0.166 | 0.162 | 0.393 | 1.0 | 53 | 30 |
| direct | 0.293 | 0.174 | 0.166 | 0.440 | 0.0 | 27 | 30 |
| decomposition | 0.278 | 0.154 | 0.153 | 0.480 | 3.2 | 62 | 30 |

### What 20 more questions did to the Phase 2-6 conclusions

This is the most useful thing in this section: it says how far the
10-question benchmark every earlier phase rested on could actually be
trusted. Same index, same code, same metrics — only the gold set changed.

| Strategy | R@20 (10q) | R@20 (30q) | MRR (10q) | MRR (30q) | DocPrec@5 (10q) | DocPrec@5 (30q) |
|---|---:|---:|---:|---:|---:|---:|
| hyde | 0.558 | 0.512 | 0.358 | 0.335 | 0.660 | 0.533 |
| rag-fusion | 0.392 | 0.376 | 0.140 | 0.167 | 0.560 | 0.480 |
| multi-query | 0.375 | 0.314 | 0.190 | 0.202 | 0.400 | 0.460 |
| step-back | 0.383 | 0.300 | 0.136 | 0.166 | 0.400 | 0.393 |
| direct | 0.325 | 0.293 | 0.103 | 0.174 | 0.560 | 0.440 |
| decomposition | 0.333 | 0.278 | 0.194 | 0.154 | 0.600 | 0.480 |

**Everything got harder.** Mean `Recall@20` over the six strategies falls
0.394 to 0.346 and mean `DocPrec@5` 0.530 to 0.464. That is expected: the 20
new questions were written to separate the techniques, a third of them need
passages from several papers, and three of them no strategy answers at all
(below). Mean `MRR@20` actually rises slightly, 0.187 to 0.200.

**What still holds.** HyDE leads every column, by a margin nothing else comes
near: +0.219 `Recall@20` over `direct` on 30 questions against +0.233 on 10.
That was the only difference the 10-question table said was likely to survive
a larger set, and it did.

**Four Phase 2-6 claims that no longer hold:**

1. *"multi-query's `DocPrec@5` (0.400) is worse than direct's (0.560)"* —
   **reversed.** On 30 questions multi-query is 0.460 against direct's 0.440.
   The crowding effect that paragraph explained was a 10-question artifact.
2. *"Step-back sits in between: better recall than direct"* — **gone.**
   0.300 against 0.293 is a tie, not an ordering; on 10 questions the gap was
   +0.058.
3. *"decomposition spends a mean of 3.4 model calls per question ... for a
   Recall@20 gain of only 0.008 over direct"* — **the sign flipped.**
   Decomposition is now *last of six* and 0.015 *below* plain retrieval, at
   3.2 calls a question. The cost case against it is stronger than Phase 3
   could state.
4. *Direct retrieval ranks worst.* On 10 questions `direct` had the lowest
   `MRR@20` of all six (0.103). On 30 it is **third** (0.174), ahead of
   rag-fusion, step-back and decomposition. Only HyDE and multi-query now
   rank better than doing nothing. Read with (1) and (2): outside HyDE, query
   translation on this corpus is closer to a wash than the 10-question table
   suggested, not further from it.

The rank order of the middle four reshuffled completely (10q: rag-fusion,
step-back, multi-query, decomposition, direct — 30q: rag-fusion, multi-query,
step-back, direct, decomposition), which is exactly what "treat rank order
among the four middling strategies as unreliable" predicted. The warning was
right; the numbers it warned about were not.

### The prediction scorecard: 6 of 20

`python -m evaluation.benchmark --predictions` scores the `expects` field
every new question carries — the strategy its author predicted would win it,
written down before any of it was run. A question where every strategy ties
counts as a miss, because the prediction distinguished nothing there.

```
predictions correct: 6/20 (30.0%)
```

| Question | Predicted | Actual winner | Hit |
|---|---|---|---|
| chain-of-thought-definition | direct | tie (all 1.000) | no |
| devlin-masking-rate | direct | hyde, multi-query, rag-fusion | no |
| dual-encoder-scaling | direct | tie (all 1.000) | no |
| frozen-versus-trained-lm | decomposition | hyde | no |
| llm-instead-of-annotators | hyde | five-way tie including hyde | **yes** |
| long-context-cost | hyde | hyde, step-back | **yes** |
| multihop-one-step-retrieval | decomposition | hyde | no |
| naive-then-advanced-rag | decomposition | hyde | no |
| one-model-many-tasks | rag-fusion | rag-fusion | **yes** |
| pairwise-comparison-blowup | hyde | tie (all 0.000) | no |
| pre-2018-architecture | hyde | hyde, step-back | **yes** |
| query-transformation-family | multi-query | multi-query, rag-fusion | **yes** |
| reader-passage-fusion | multi-query | hyde | no |
| reasoning-without-checking | hyde | tie (all 0.000) | no |
| retriever-without-labels | hyde | decomposition | no |
| sparse-dense-mismatch | decomposition | four-way tie including decomposition | **yes** |
| t5-corpus-and-task-prefix | decomposition | hyde | no |
| when-counting-words-wins | hyde | tie (all 0.000) | no |
| zeroshot-generalisation-gap | rag-fusion | multi-query | no |
| zhou-scan-accuracy | direct | hyde, multi-query, rag-fusion | no |

**30% is a bad score, and reporting it is the point.** The gold set was
written by the person who built the system, with knowledge of what every
technique does — which is what lets it separate them and also exactly what
would let it be fitted to a flattering answer. The pre-registration makes
that risk measurable instead of hidden, and the measurement says the author's
model of these techniques is wrong most of the time. Five of the twenty ended
in an all-strategy tie and are counted as misses; of the fifteen questions
that had a winner at all, the prediction named it six times.

Read by predicted technique it is worse than the headline:

| Predicted | Hits | Predicted on |
|---|---:|---:|
| hyde | 3 | 7 |
| decomposition | 1 | 5 |
| direct | **0** | 4 |
| rag-fusion | 1 | 2 |
| multi-query | 1 | 2 |

**Every `direct` prediction failed.** Those four were the deliberately easy
vocabulary-match questions, written as baseline cases plain retrieval ought
to win. Two ended in a six-way tie at 1.000 — everything found the chunk, so
"direct wins" was unfalsifiable as posed — and on the other two `direct`
scored 0.500 while hyde, multi-query and rag-fusion scored 1.000. Plain
retrieval never *uniquely* won a question it was predicted to win.

Meanwhile HyDE won or co-won **11 of the 15 questions that had a winner at
all**, including five written to favour decomposition or direct. The
technique the author predicted least confidently is the one that keeps
happening.

**Three questions score 0.000 for every strategy**
(`pairwise-comparison-blowup`, `reasoning-without-checking`,
`when-counting-words-wins`) — all three written to reward HyDE by sharing
little vocabulary with the passage that answers them. They apparently share
too little: nothing retrieves their gold chunk in the top 20. They are left
in the set unedited. Rewriting a question after seeing it score zero is the
precise thing the pre-registration exists to prevent, and since they drag
every strategy down equally the comparison between strategies is unaffected.

**Phase 4 moved one row, and the sample cannot say which way.** (Every
number in this note is from the 10-question set it was measured on;
Phase 7's 30-question replacements are above.) Converting
`step-back` to structured output changes the exact text the model returns for
its general question, which changes what gets retrieved. `Recall@20` reads
0.383 where the free-form version read 0.475, `MRR@20` 0.136 against 0.196,
`nDCG@20` 0.170 against 0.225, `DocPrec@5` 0.400 against 0.460.

Those numbers look like a clean regression and they are not. Scoring both
prompts question by question, **five of the ten changed and they moved in
both directions** — `raptor-clustering` 0.33 to 0.67 and `cot-limits` 0.00 to
0.50 improved, while `selfrag-tokens` 1.00 to 0.00, `crag-quality` 1.00 to
0.50 and `hyde-problem` 0.50 to 0.25 got worse. The net is negative, but with
half the set flipping both ways and each question worth up to 0.1, ten
questions cannot distinguish that from chance.

One tempting explanation was tested and is wrong. The JSON-constrained prompt
does return longer, more textbook-phrased questions, and the old prompt ended
with "reply with the general question only, on one line" while the JSON one
has no brevity instruction. Adding that instruction back changes the score
not at all — 0.383 either way. Whatever drives the churn, verbosity is not
it.

The other five rows are bit-for-bit unchanged, which is what rules out a
routing- or construction-caused regression: both are off by default, so a
leak into the default path would have moved more than one row.

Structured output is kept regardless of the number. It replaced four
successive attempts to parse one line of prose out of a model reply, each of
which lost to a reply shape the previous one had not anticipated. Trading an
unmeasurable difference in retrieval for a parsing path that cannot silently
pick the wrong line is the right trade — and hand-tuning the new prompt until
the old number came back would be fitting the prompt to ten questions.

Converting also uncovered a real accounting gap: `_CallCountingLLM` only
intercepted `.generate()`, so step-back's move to `.structured()` bypassed
the counter's override via `__getattr__` and briefly reported `0.0` LLM
calls instead of `1.0` — silently understating its cost. Fixed by
intercepting `.structured()` too; the row above already reflects the fix.

Ran repeatedly (the full sweep, plus `multi-query` and `rag-fusion` run alone
in both orders — see below); every metric column, `LLM calls` included, was
bit-identical across runs. `Mean ms (warm)` moves a little run to run, as
wall-clock numbers do, but no longer with strategy order. Read it as an order
of magnitude, not a measurement: fusing a handful of short lists costs well
under a millisecond, which is below this machine's run-to-run noise, so the
small gaps between the middle four strategies are not meaningful.

**That determinism is the LLM cache's, not the method's.** `.cache/` (the
on-disk LLM cache these repeat runs share) is gitignored — it is not part of
this repository. A clean clone has no cache to warm: it re-samples the model
for every rewrite, hypothetical document, and sub-question, and gets
different text back each time. This table is one draw of those LLM outputs
over a 30-question gold set, with no variance estimate across draws. "Ran
repeatedly" above means repeated against *this machine's* warm cache, which
shows the columns are order-independent and re-run-stable — it does not mean
a second person cloning this repository and running the benchmark cold would
reproduce these exact numbers.

**Why the table no longer reports raw `Mean ms`:** an earlier version of this
table reported wall-clock time directly, and it was wrong in a way a
sceptical reader would find immediately — it showed `rag-fusion` at 94ms and
`multi-query` at 42680ms, a 450x gap. Both strategies build their rewrite
prompt from the same `MULTI_QUERY_TEMPLATE`, the same question and the same
`n`, so they produce an identical prompt and therefore an identical entry in
the on-disk LLM cache. `rag.strategies.STRATEGY_NAMES` lists `multi-query`
before `rag-fusion`, so multi-query was the one paying for every cache miss
and rag-fusion collected the free hits. Reordering that tuple would have
swapped which strategy "looked" 450x faster — the number measured cache
ordering, not the strategy.

It is replaced with two columns that cannot be perturbed by cache state or
strategy order:

- **`LLM calls`** — the mean number of `.generate()` calls a strategy makes
  per question, counted logically whether or not the cache served it (via a
  small counting wrapper in `evaluation/benchmark.py`, kept separate from
  `rag.llm.GeminiLLM.call_count`, which deliberately counts only real API
  attempts). This is the real cost driver: it is what you would still pay if
  the cache were empty, and it does not change with run order. The column
  counts query-translation calls only — the benchmark runs with
  `generate=False`, so it never pays for the answer-generation call every
  strategy, `direct` included, makes in production. `direct`'s `0.0` is the
  cost of *this benchmark's* retrieval step, not of answering a question with
  it.
- **`Mean ms (warm)`** — mean `total_ms` measured with every strategy's own
  cache already warm, by running the gold set twice per strategy and scoring
  only the second pass. It measures retrieval and orchestration cost —
  embedding, vector search, merging or fusing ranked lists — not the cost of
  an actual LLM round trip, and it is *not* comparable to a cold call's
  latency.

Confirmed empirically: running `python -m evaluation.benchmark --strategy
multi-query --strategy rag-fusion` and the same command with the two
`--strategy` flags reversed produces identical `LLM calls` (1.0 for each,
either order) — the property the old `Mean ms` column did not have.

`Recall@20`, `MRR@20` and `nDCG@20` are measured at retrieval depth 20, not
the answer prompt's `top_k=5` — at k=5 every strategy scores near zero (plain
retrieval measured 0.050) and there is no headroom to tell them apart.
`DocPrec@5` is reported separately, always at 5: it is the fraction of the
chunks that would actually reach the answer prompt that come from the
document holding the answer, and it is robust to the chunk-level gold set
being incomplete (see below), since it only checks which paper a chunk came
from, not which passage. Reading the first 5 of a k=20 ranked list as
`DocPrec@5` is only valid because `retrieval_depth` (20 by default) is at
least `k`: the first 5 of a k=20 run are byte-identical to what a k=5 run
would return today. That stops being true at any `--k` above `retrieval_depth`
— `--k 50`, for instance, silently raises the depth to 50, which changes what
each query retrieves before fusion and can change which chunks land in the
first 5. Read `DocPrec@5` as meaningful at the benchmark's default depth, not
as a claim that holds at every `--k`.

**What this actually shows:** only HyDE clearly beats plain retrieval, and it
does so on every column — recall, ranking, and document precision alike. The
other four translation strategies are not a clean win over doing nothing.
RAG-Fusion, multi-query and step-back each find slightly more relevant chunks
somewhere in the top 20 than direct retrieval does (0.376, 0.314 and 0.300
against 0.293), but only multi-query also ranks them better (`MRR@20` 0.202
against direct's 0.174), and only multi-query and rag-fusion also improve
`DocPrec@5`. Decomposition is worse than direct on every column. On this
corpus, with this gold set, query translation is one technique that works and
four that are a wash or a regression once ranking and document precision are
counted, not just "was the chunk somewhere in the top 20."

Decomposition's cost case rests on `LLM calls`, not milliseconds: with the
cache warm it costs 62ms per question, only ~2.3x direct's 27ms — a world
away from the roughly 2000x an earlier version of this table reported,
because that old figure was really measuring a cold LLM round trip, not
orchestration cost. The number that holds up is `LLM calls`: decomposition
spends a mean of 3.2 model calls per question (one to split the question,
plus one per sub-question it answers) against 1 for every other translation
strategy and 0 for direct — and on 30 questions it buys a `Recall@20`
*0.015 below* direct's, last of the six. On 10 questions this paragraph could
only say the 0.008 gain was within noise; on 30 the gain is gone. Several
extra round trips to the model, each with its own latency and price in
production even when nothing is cached, for retrieval no better than not
making them.

**Read this table narrowly:**

- **30 questions.** Three times the old set and still small. Each question is
  worth up to 0.033 of `Recall@20`, so differences of a few points remain
  noise; the reshuffle described above is what that looks like in practice.
  Treat the hyde-vs-everything-else gap as the one difference that has now
  survived a tripling of the sample, and the ordering of the other five as
  provisional.
- **One author.** The questions, the quotes *and* the `expects` predictions
  are all one person's, written with knowledge of how the corpus, the chunker
  and the six strategies behave. Pre-registering the predictions makes the
  resulting bias measurable — it does not remove it.
- **Retrieval only.** Nothing here measures whether the retrieved chunks
  produce a better final answer — that is not evaluated anywhere yet.
- **The chunk-level gold set is incomplete.** It names some answering
  passages per question, not all of them — hand-verifying every chunk that
  could answer a question against a 3M-character corpus is not workable. That
  understates `Recall@20`, `MRR@20` and `nDCG@20` for every strategy equally, so
  the relative comparison holds, but none of those three numbers is an
  absolute quality score. `DocPrec@5` does not have this problem, which is
  why it is reported alongside them.

## Routing and query construction

Phase 4 adds three opt-in flags on `rag ask` that narrow *what gets searched*
before a strategy decides *how*: `--route`, `--construct`, and
`--semantic-prompt`. All three degrade to a safe default and record a note
containing `degraded` if the model call fails or returns nothing usable —
the benchmark's `check_not_degraded` treats that word as fatal, so a silent
fallback can never be mistaken for a working feature.

- **`--route`** (logical routing) asks the LLM which of the corpus's topical
  collections could plausibly answer the question, via structured output
  (`ROUTE_SCHEMA`), and restricts search to those before the strategy runs.
  It needs the LLM and is rejected alongside `--no-llm`, the same way
  `--strategy` other than `direct` is. Corpus documents are partitioned into
  five topics, none holding more than half the 38 documents:

  | Topic | Documents |
  |---|---:|
  | `retrieval-models` | 10 |
  | `rag-systems` | 9 |
  | `prompting-reasoning` | 8 |
  | `evaluation-benchmarks` | 7 |
  | `foundations` | 4 |

- **`--construct`** (query construction) asks the LLM to infer a metadata
  filter from the question itself — a date bound, an author, a topic — also
  via structured output, and applies it as a mask over the candidate set
  *before* top-k selection runs, so a filtered search still returns a full k
  results when k documents survive the filter. It needs the LLM for the same
  reason `--route` does, and is rejected the same way.
- **`--semantic-prompt`** picks which answer-prompt variant to use (e.g.
  "definition" vs. "mechanism") by embedding the question and comparing it
  against embedded exemplar questions for each variant — no LLM call. It
  stays allowed with `--no-llm`, since it costs only an embedding.

### The demo, honestly

```
python -m rag ask "What did the RAPTOR paper say about clustering, from anything published before 2024?" --construct --trace
```

`--construct` correctly infers `published before 2024-01-01` from the
question and applies it. RAPTOR is dated 2024-01-31 in this corpus's
metadata, so that filter genuinely excludes the very paper the question
names — this is the filter working as specified, not a bug. The retrieved
chunks come from unrelated documents (MTEB and Instructor, both about
clustering in a different sense), and the model's answer says plainly that
"the provided context does not contain any information about the RAPTOR
paper" and does not answer the question. A question that asks for a
document explicitly excluded by its own date filter should come back empty
rather than quietly answering from the wrong source, and it does.

```
python -m rag ask "How does ColBERT score a document?" --route --trace
```

Routes to `retrieval-models` (shown in the trace as `route: search
retrieval-models`) and answers correctly from ColBERT and ColBERTv2 chunks.

```
python -m rag ask "How does ColBERT score a document?" --semantic-prompt --trace
```

Picks the `mechanism` prompt variant (`route: prompt mechanism (cosine
0.103)`) with no LLM call for the routing decision itself, and answers using
that prompt's "describe the mechanism step by step" framing.

```
python -m rag ask "q" --route --no-llm
```

Exits 1: `--no-llm cannot be combined with --route`.

### Exemplar questions, not descriptions of them

`--route` uses `GeminiLLM.structured` (the LLM case); `--semantic-prompt`
uses `SemanticRouter`, which needs no LLM at all. `SemanticRouter` was first
tried with `PROMPT_DESCRIPTIONS` — prose
descriptions of what each prompt variant is for — matched against the
question's embedding. That scored 5/9 on a small set of questions with known
correct routing: "What is late interaction?" and "Define reciprocal rank
fusion." both routed to the "comparison" variant instead of "definition",
because matching a question against a *description of an intent* asks the
embedder to encode intent, and a sentence embedder like MiniLM encodes topic
instead. Replacing the descriptions with `PROMPT_EXEMPLARS` — actual
question-shaped exemplars for each variant, not descriptions of them — scores
7/9: matching a question against other questions compares like with like.
This is the same insight HyDE rests on (embed something shaped like what
you're searching for, not a description of it) applied to routing instead of
retrieval.

### Is routing worth using? Measured, not asserted

This section used to argue that scoring `--route` against the gold set
"would answer a different question" than the strategy table above and leave
it at that. That argument doesn't survive contact with the gold set itself:
every one of its questions already names a `doc_id`, and every document
already carries a `topic` in `store.doc_meta` — so routing accuracy is
computable with zero new labelling. Did `logical_route` return a topic set
containing the topic of the document that actually holds the answer?
`evaluation/routing_eval.py` answers that directly, with no new gold data:

```
$ python -m evaluation.routing_eval   # 10-question gold set, Phase 4
| Recall | Abstention rate | Mean topics chosen | Topics available | Questions |
|---:|---:|---:|---:|---:|
| 0.900 | 0.000 | 1.30 | 5 | 10 |
```

Read all three numbers together, not the first alone — a router that always
returns `()` ("search everything") would also score 1.000 on recall while
doing nothing. This one doesn't do that: abstention rate is 0.000, and mean
topics chosen is 1.30 of the 5 available, close to the most aggressive
narrowing possible (1) rather than the least (5). It narrows hard and still
contains the right topic 9 times out of 10.

That is a different question from "does narrowing the candidate set before a
strategy searches make retrieval better or worse" — and `--route` on
`evaluation.benchmark` answers that one, at the cost of one extra
`.structured()` call per question (visible in `LLM calls` below):

```
$ python -m evaluation.benchmark --route
| Strategy | Recall@20 | MRR@20 | nDCG@20 | DocPrec@5 | LLM calls | Mean ms (warm) | Questions |
|---|---:|---:|---:|---:|---:|---:|---:|
| hyde | 0.592 | 0.261 | 0.292 | 0.620 | 2.0 | 148 | 10 |
| rag-fusion | 0.392 | 0.163 | 0.200 | 0.580 | 2.0 | 70 | 10 |
| multi-query | 0.375 | 0.197 | 0.211 | 0.460 | 2.0 | 70 | 10 |
| step-back | 0.350 | 0.125 | 0.152 | 0.320 | 2.0 | 45 | 10 |
| direct | 0.292 | 0.106 | 0.129 | 0.600 | 1.0 | 32 | 10 |
| decomposition | 0.267 | 0.107 | 0.129 | 0.560 | 4.4 | 83 | 10 |
```

Against the unrouted table above, `Recall@20` moves in every direction at
once: up for `hyde` (0.558 → 0.592), unchanged for `rag-fusion` and
`multi-query`, and down for `step-back`, `direct` and, worst, `decomposition`
(0.333 → 0.267). Averaged across all six strategies it drops slightly, 0.394
→ 0.378; `DocPrec@5`'s average is essentially flat, 0.530 → 0.523.
**Routing does not help retrieval on this corpus.** Its 90% topic recall
above is real, but the 10% it gets wrong excludes the gold document's topic
entirely for that question, and on a 38-document corpus that plain top-k
already answers reasonably well, that miss costs most strategies more recall
than narrowing the search space gains them. `hyde` is the one strategy that
improved; this gold set is too small to say why. This is a measured negative
result, reported the same way the four under-performing Phase 2 strategies
already are above — an honest "doesn't help here" is worth more than an
unmeasured "would help".

**This table is still the 10-question one, and that is a gap.** Phase 7
re-ran every other variant on 30 questions but could not re-run `--route`:
routing issues one `.structured()` call per question with a prompt that
depends only on the question, so the 20 new questions are 20 uncached API
calls, and the free tier's 500-requests-per-day quota was already spent on
the strategy sweeps (see below). The routed numbers above therefore describe
the old gold set only, and given how much the unrouted table moved between 10
and 30 questions they should not be assumed to carry over.

### `--construct` and `--semantic-prompt` are still not in the benchmark table

The trick that makes `--route` measurable — every gold question already
names a `doc_id`, every document already carries the one label being routed
on — doesn't extend to `--construct`. Query construction infers a date
bound, an author *or* a topic from the question, and there is no gold label
for "the right author" or "the right cutoff date" the way there is a
`topic` for every document; building one would mean labelling a second gold
set this project doesn't have. `--semantic-prompt` picks an answer-prompt
variant, not a candidate set, so it doesn't change what's retrieved at all
and has nothing for the strategy table to measure. Both stay off by default
in the benchmark for that reason, unlike `--route` above.

## Indexing techniques

Phases 1-4 all changed the *query*. These two change what goes **into** the
index, and they are measured the same way: same gold set, same metrics.

Each technique writes its own index file, selected with `--index-mode`:

```
python -m rag index --index-mode multirep   # 38 summaries + a docstore
python -m rag index --index-mode raptor     # 5,116 chunks + a 730-node summary tree
python -m rag ask "..." --index raptor --trace
python -m evaluation.benchmark --index raptor
```

They are separate files on purpose. The benchmark scores gold spans
positionally against `store.chunks`, so adding summary nodes to the default
index would move all six baseline rows for reasons unrelated to the
strategies. Keeping them apart is what makes the comparison below mean
anything — and the flat table reproduces its Phase 2-4 numbers to three
decimals, which is the check that it worked.

### Multi-representation: a 135x smaller index that retrieves better

Summarise each document with Gemini, embed the *summary*, and keep the full
text in a docstore. A summary matches, and the whole document is handed to
the generator. 38 documents, 38 nodes.

On the benchmark table it scores **0.000 Recall@20 for every strategy**. That
number is worthless, and it is worth explaining why rather than filing it as
a defeat:

- **Recall@20 is 0.000 by construction.** The gold set marks chunks whose
  character span overlaps an answering passage. A summary node is generated
  text with no position in any document, so it can never satisfy a span. The
  metric cannot see this technique at all.
- **DocPrec@5 is capped at 0.273.** The index holds exactly one node per
  document, so a question can contribute at most one slot per gold document
  it names — one for the twenty single-document questions, two or three for
  the cross-document ones, giving a ceiling of 41/150. Flat can reach 1.0.
  The measured 0.180-0.220 is close to that ceiling, not a score.

On the 30-question set (`python -m evaluation.benchmark --index multirep`,
five strategies — see the quota note below):

| Strategy | Recall@20 | MRR@20 | nDCG@20 | DocPrec@5 | LLM calls | Mean ms (warm) | Questions |
|---|---:|---:|---:|---:|---:|---:|---:|
| hyde | 0.000 | 0.000 | 0.000 | 0.220 | 1.0 | 188 | 30 |
| direct | 0.000 | 0.000 | 0.000 | 0.213 | 0.0 | 44 | 30 |
| step-back | 0.000 | 0.000 | 0.000 | 0.200 | 1.0 | 65 | 30 |
| rag-fusion | 0.000 | 0.000 | 0.000 | 0.193 | 1.0 | 129 | 30 |
| multi-query | 0.000 | 0.000 | 0.000 | 0.180 | 1.0 | 131 | 30 |

The honest comparison is at the document level — did retrieval surface the
right paper, and how high? A cross-document question counts as a hit when
*any* of its gold documents is found, at the best rank any of them reached.
Re-measured on all 30 questions:

| index | nodes | DocHit@1 | DocHit@5 | DocMRR |
|---|---:|---:|---:|---:|
| flat | 5,116 | 0.433 | 0.800 | 0.594 |
| **multirep** | **38** | **0.467** | **0.900** | **0.631** |
| raptor | 5,846 | 0.433 | 0.800 | 0.591 |

**An index 135x smaller finds the right document more often than flat
chunking**, and ranks it higher. That result survives the larger gold set:
multirep still leads all three columns, and its `DocHit@5` lead is now
visible where on 10 questions all three indexes were pinned at 1.000. A
document summary is a better search key than any one of its paragraphs,
because it describes the whole document rather than one corner of it.

What did *not* survive is the absolute level. On 10 questions every index
found the right paper inside the top 5 every time (`DocHit@5` 1.000 across
the board) and flat's `DocHit@1` was 0.600; on 30 it is 0.800 and 0.433.
**The claim "`DocHit@5` = 1.000" no longer holds for any index** — it was a
property of ten easy questions, not of the corpus.

The cost is precision within a document: multirep returns 44,000 characters
where flat returns 800, so it tells you *which paper* and leaves finding the
passage to the generator's context window.

### RAPTOR: the abstractions work, and they cost precision

K-means written out in NumPy (`rag/clustering.py` — importing `sklearn` would
defeat the exercise), applied recursively: cluster the chunk embeddings,
summarise each cluster, embed those summaries, cluster *those*. Three levels
over 5,116 chunks:

```
levels: {0: 5116, 1: 640, 2: 80, 3: 10}   # 730 summaries, 730 LLM calls
```

All levels live in one store, so a single search spans raw passages and
abstractions together.

**On the 30-question span-based gold set, RAPTOR loses across the board**
(five strategies — see the quota note below):

| Strategy | Recall@20 flat | raptor | MRR@20 flat | raptor | DocPrec@5 flat | raptor |
|---|---:|---:|---:|---:|---:|---:|
| hyde | 0.512 | 0.408 | 0.335 | 0.245 | 0.533 | 0.407 |
| rag-fusion | 0.376 | 0.343 | 0.167 | 0.154 | 0.480 | 0.373 |
| multi-query | 0.314 | 0.279 | 0.202 | 0.146 | 0.460 | 0.287 |
| step-back | 0.300 | 0.174 | 0.166 | 0.095 | 0.393 | 0.187 |
| direct | 0.293 | 0.277 | 0.174 | 0.153 | 0.440 | 0.373 |

Not one cell improves. The pattern is the Phase 5 one at a smaller
magnitude: `step-back` is hit hardest again (0.300 to 0.174), `direct` least
(0.293 to 0.277).

The mechanism is not mysterious. Summary nodes occupy top-k slots and can
never be credited, so every slot one takes is a slot a creditable chunk did
not get. Measured: **16.0% of top-20 slots** across the ten gold questions.

**But the abstraction layer does exactly what it is designed to do.** Split
the questions by kind, and the summaries activate on precisely the ones they
are for:

| question kind | share of top-20 that is a summary |
|---|---:|
| gold set (specific: "how does ColBERT score a document?") | 16.0% |
| broad ("what problem do most of these techniques share?") | **50.0%** |

Asked a broad question, half the retrieved context becomes abstractions —
and they are apt. `raptor:1:571` reads *"the passages collectively share a
central theme focused on evaluating, refining, and supplementing
retrieval-augmented generation systems"*, which is an answer to a question no
single 200-token chunk can answer.

#### Phase 5's excuse, tested — and it does not hold

Phase 5 closed this section by saying the fair reading was not "RAPTOR is
worse" but **"RAPTOR is built for a question this gold set does not
contain"**: ten hand-written questions each with one answering passage in one
paper, which is the case flat top-k already wins. Phase 7 wrote the missing
question type — ten cross-document questions whose answer needs passages from
two or three papers — and the right test is to compare the indexes **on that
subset alone**, using `direct` so no query rewriting confounds it:

| subset | n | flat Recall@20 | raptor Recall@20 | delta |
|---|---:|---:|---:|---:|
| cross-document | 10 | 0.197 | 0.146 | **-0.051** |
| single-document | 20 | 0.342 | 0.342 | +0.000 |

**RAPTOR loses on its own home ground, and only there.** On the twenty
single-document questions the two indexes are exactly tied — the summary
nodes cost nothing because they rarely displace anything that would have been
credited. On the ten cross-document questions, the case the technique was
designed for and the case Phase 5 said the gold set was missing, RAPTOR is
0.051 *worse*.

**This is a stronger negative result than Phase 5's**, and it should be read
as one. Phase 5 measured RAPTOR on questions it was not for and could
reasonably plead the gold set. That plea is now spent: the questions exist,
they were written and committed before any of this was measured, and the
technique still does not win them. The mechanism from Phase 5 is unchanged —
every top-k slot a summary node takes is a slot a creditable chunk did not
get — and a cross-document question does not rescue it, because the metric
still needs *character spans in the source papers*, and a summary of a
cluster is not one however apt it is.

The honest limit of this result: span-based recall may simply be the wrong
instrument for RAPTOR, in the same way it is the wrong instrument for
multi-representation. The abstraction-share measurement above (50% of
retrieved context on broad questions) says the summaries do activate and are
apt; what this project has never built is a metric that can credit an answer
assembled from an abstraction. Saying "RAPTOR loses under span recall, and
span recall cannot see what RAPTOR produces" is both halves of the finding,
and neither half cancels the other.

Levels 2 and 3 are almost never retrieved (2 hits of 200 on the gold set, 1
of 60 on broad questions). At 38 documents the tree is taller than the corpus
justifies — `raptor_max_depth: 2` would likely lose nothing.

### What building these actually cost

Worth recording, because the write-ups usually omit it. The RAPTOR build is
730 LLM calls, and getting them took three attempts and four bug fixes:

- **The first build silently indexed 36 of 38 documents.** Two summaries hit
  a transient rate limit and were skipped. `build_index` passed no `Trace`,
  so the `trace.degraded(...)` branch was dead code and nothing recorded it —
  the CLI printed "indexed 36 documents", which looks fine unless you know it
  should be 38. One of the two, `ircot`, was the gold document for a question
  multirep then "failed". Fixed: a partial index records the loss in its
  `meta`, prints it, and exits 1.
- **The second build produced a tree with no tree in it.** Quota exhaustion
  mid-run killed levels 2 and 3 entirely, leaving one flat layer of 463
  summaries. Recorded, this time, by the fix above.
- **The API hung rather than failing.** During a Gemini 3.5 overload one call
  returned after **691 seconds**; five retries of that is an hour per summary,
  and the build managed three calls in ninety minutes. Fixed with a 30s
  request timeout — calls then resolved in 8-45s.
- **Exponential backoff was shorter than the rate limit.** The free tier
  allows 15 requests/minute and its 429 carries a `retryDelay` of 35-51s;
  five exponential retries total 31s, so every retry fired inside the closed
  window. That cost 9 cluster summaries. Fixed by honouring the server's
  `retryDelay`.

None of these produced a stack trace. All four produced plausible output with
quietly wrong contents, which is the failure mode this project keeps finding
and the reason the `degraded` sentinel exists.

## Late interaction: ColBERT-style reranking buys recall, at 46x the latency

Dense retrieval compares one vector per query against one vector per chunk.
A passage that answers a question in a single clause is represented by the
average of every clause it contains, so the answering clause is diluted by
everything around it. Late interaction (ColBERT) keeps every token on both
sides and asks a narrower question — for each query term, how well does the
*best-matching* term in this passage match it?

That is MaxSim:

```
score(q, d) = sum over query tokens i of  max over doc tokens j of  (q_i . d_j)
```

Both sides are L2-normalised, so each inner product is a cosine and the max
picks one document token per query token. `rag/late_interaction.py` is the
whole implementation: `maxsim()` is a matrix product, a row-wise max and a
sum; `rerank()` rescores a dense shortlist with it. `Embedder.encode_tokens`
supplies the per-token matrices from the same MiniLM model with no pooling —
no new dependency — dropping padding via the attention mask and `[CLS]`/
`[SEP]` via the special-tokens mask. Special tokens are not a harmless
constant: they appear in every sequence, so a query's `[CLS]` matches every
document's `[CLS]` at near-1.0 and adds a *near*-constant term, which is
noise on the ranking rather than a shift.

Reranked scores carry `score_kind="maxsim"`, never mixed with cosine. A
MaxSim score sums one max per query token, so it lands around 5-20 where a
cosine lands around 0.5; printing both as "score" would make reranking look
like a tenfold quality jump.

```bash
python -m rag ask "How does ColBERT score a document?" --rerank
python -m evaluation.benchmark --index flat --rerank
```

Retrieval fetches `rerank_depth` (50) candidates, MaxSim rescores all of
them, and the best 20 are what the metrics see. Un-reranked, k=20, flat
index, 30 questions (the same five rows as the table at the top of this
README; `decomposition` is absent from both tables here — see the quota note
below):

| Strategy | Recall@20 | MRR@20 | nDCG@20 | DocPrec@5 | LLM calls | Mean ms (warm) | Questions |
|---|---:|---:|---:|---:|---:|---:|---:|
| hyde | 0.512 | 0.335 | 0.310 | 0.533 | 1.0 | 123 | 30 |
| rag-fusion | 0.376 | 0.167 | 0.193 | 0.480 | 1.0 | 91 | 30 |
| multi-query | 0.314 | 0.202 | 0.199 | 0.460 | 1.0 | 90 | 30 |
| step-back | 0.300 | 0.166 | 0.162 | 0.393 | 1.0 | 53 | 30 |
| direct | 0.293 | 0.174 | 0.166 | 0.440 | 0.0 | 27 | 30 |

Reranked (`--rerank`), same index, same gold set:

| Strategy | Recall@20 | MRR@20 | nDCG@20 | DocPrec@5 | LLM calls | Mean ms (warm) | Questions |
|---|---:|---:|---:|---:|---:|---:|---:|
| rag-fusion | 0.504 | 0.341 | 0.299 | 0.547 | 1.0 | 3542 | 30 |
| hyde | 0.483 | 0.332 | 0.284 | 0.533 | 1.0 | 3619 | 30 |
| multi-query | 0.424 | 0.311 | 0.271 | 0.547 | 1.0 | 3587 | 30 |
| direct | 0.410 | 0.299 | 0.252 | 0.500 | 0.0 | 3505 | 30 |
| step-back | 0.383 | 0.268 | 0.235 | 0.527 | 1.0 | 3537 | 30 |

Per-strategy `Recall@20` delta: `rag-fusion` +0.128, `direct` +0.117,
`multi-query` +0.110, `step-back` +0.083, `hyde` **-0.029**. Averaged over
the five, `Recall@20` 0.359 → 0.441, `MRR@20` 0.209 → 0.310, `nDCG@20`
0.206 → 0.268, `DocPrec@5` 0.461 → 0.531. `rag-fusion` overtakes `hyde`,
which had led every table in this README since Phase 2 — the one Phase 6
finding the larger gold set leaves standing unchanged.

**Phase 6's claim that "recall rises for every strategy" no longer holds.**
On 30 questions `hyde`'s recall *falls*, 0.512 → 0.483. The direction of
every other row is the same as Phase 6 reported and the magnitudes are
roughly half of it (mean +0.082 against +0.148); hyde alone crossed zero.

**What reranking does to `hyde` is nearly the opposite of what Phase 6
said.** That section reported hyde's recall edging up while its ranking
collapsed — `MRR@20` 0.358 → 0.242, `DocPrec@5` 0.660 → 0.580 — and
explained the collapse by MaxSim summing over a long generated query. On 30
questions hyde's `MRR@20` is 0.335 → 0.332 and its `DocPrec@5` 0.533 →
0.533: both flat. The ranking collapse was a 10-question artifact. What is
left is a recall loss, which that explanation does not cover. The safe
statement is the narrow one: **reranking helps every strategy except HyDE,
and on HyDE it does nothing useful.**

**The latency is the result too, not a footnote.** `Mean ms (warm)` goes from
27-123 ms to 3505-3619 ms — 77 ms to 3558 ms averaged over the five
strategies, about **46x**, and 130x for `direct`, which had nothing else to
pay for. Reranking runs a second forward pass over 50 chunks per question
with every token position kept, and that is the whole bill: it costs no extra
LLM call (`LLM calls` is identical in both tables), just local compute. The
per-strategy spread collapses because the reranker's ~3.5 s dwarfs whatever
the strategy itself was doing. (The two tables' timings come from separate
runs, so read the ratio, not the millisecond.)

**It is not inert:** measured directly on the 10-question set, **10 of 10
gold questions had their top-20 order changed** by reranking. A reranker that
reordered nothing would produce identical numbers and be indistinguishable
from one switched off, so this is checked rather than assumed.

**The confound, and the ablation that removes it.** `--rerank` changes two
things at once: it rescores with MaxSim, and it deepens the candidate pool
from 20 to 50, because reranking can only promote something the dense pass
already returned. For the five strategies that retrieve per sub-query and
merge, a deeper pool also changes *what gets fused* before reranking sees
it. So the table above cannot, by itself, say which of the two did the work.

Measuring it needs a control that holds the pool at 50 and varies only
whether MaxSim reorders it. This ablation is the Phase 6 one, on the
10-question set, and was not re-run in Phase 7:

| strategy | R@20 pool-50, no rerank | R@20 reranked | MaxSim alone |
|---|---:|---:|---:|
| rag-fusion | 0.375 | 0.650 | **+0.275** |
| multi-query | 0.375 | 0.558 | +0.183 |
| decomposition | 0.333 | 0.517 | +0.183 |
| step-back | 0.383 | 0.500 | +0.117 |
| direct | 0.325 | 0.425 | +0.100 |
| hyde | 0.558 | 0.600 | +0.042 |
| | | **mean** | **+0.150** |

**The pool contributes almost nothing on its own.** Deepening 20 to 50
without reranking leaves five of six strategies exactly where they started
(`direct` 0.325, `multi-query` 0.375, `step-back` 0.383, `hyde` 0.558,
`decomposition` 0.333 — all unchanged) and makes `rag-fusion` slightly
*worse*, 0.392 to 0.375. That is not surprising in hindsight: taking the
cosine top 20 out of a cosine-sorted 50 gives back the same 20, and for the
merge strategies the extra candidates mostly add noise to the fusion. The
entire gain is MaxSim.

A first attempt at this ablation was wrong in a way worth recording, because
the wrong version looked authoritative. It reranked a pool of 20 into 20
slots and reported `+0.000` for every strategy — an apparently decisive
result that MaxSim does nothing. But reranking 20 items into 20 slots is a
*permutation*, and Recall@20 is set-based, so it cannot move whatever the
reranker does. The number was a tautology, not a measurement. MRR, which is
order-sensitive, moved in that same run — which is what exposed it.

**Two limitations the technique carries as implemented here:**

- **This reranks a shortlist; it does not index every token.** A full
  late-interaction index stores one vector per token rather than per chunk —
  roughly 100x the storage of the pooled 5,116-chunk index — and would need
  its own approximate search over token vectors. Reranking a dense shortlist
  is what is actually deployed in practice, and it inherits the dense pass's
  ceiling: a chunk that never reaches the top 50 can never be promoted.
- **These are MiniLM token vectors, not trained ColBERT weights.**
  `all-MiniLM-L6-v2` was trained for pooled sentence similarity, not for
  per-token late interaction, and there is no learned linear projection down
  to ColBERT's usual 128 dimensions. A real ColBERT checkpoint is trained so
  that individual token vectors are discriminative; these are borrowed from a
  model optimised for their average. The gains above are from the mechanism
  despite the weights, not from a faithful ColBERT.

The standing caveats apply unchanged: 30 questions, retrieval only (no answer
quality), and a chunk-level gold set that is necessarily incomplete — which
is why `DocPrec@5` is reported beside the chunk metrics.

## What could not be measured, and why

Phase 7's final sweep hit Gemini's free-tier **daily** quota
(`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, 500 requests) partway
through. The flat 30-question sweep is complete, all six strategies, because
it ran first and paid for every uncached rewrite the twenty new questions
needed. What it could not pay for:

- **`decomposition` on the raptor, multirep and reranked tables.** Every
  other strategy builds its rewrite prompt from the question alone, so those
  prompts were already in the on-disk cache after the flat sweep and the
  three remaining sweeps cost zero API calls. Decomposition also *answers*
  each sub-question, and that prompt contains the retrieved context — which
  differs per index and changes again under reranking. Those are genuinely
  new calls, and there was no quota left for them. Its row is missing rather
  than estimated.
- **The `--route` table on 30 questions.** Routing issues one structured call
  per question; twenty of the thirty were uncached.

Nothing was allowed to degrade to fill the gap. `check_not_degraded` treats
a strategy that silently fell back to plain retrieval as fatal, the raptor
sweep raised it on decomposition's first uncached sub-question, and the run
stopped there rather than reporting plain retrieval as decomposition. Five
rows measured honestly are worth more than six with one of them quietly
wrong — that is the same judgement the `degraded` sentinel exists to enforce,
applied to the benchmark's own results.

## The retrieval inspector

Every number above is a summary. The inspector is the other view: one
question, one run, and everything the pipeline actually did with it.

```bash
pip install -e ".[dashboard]"
python -m rag dashboard                        # http://127.0.0.1:8000
python -m rag dashboard --port 8765 --index raptor
```

Five panels, all fed by two `GET` endpoints:

| Panel | Shows |
|---|---|
| Chunk table | rank, `score_kind` and raw score, document, chunk index, an excerpt |
| Translation trace | what the strategy rewrote the question into, step by step |
| Stage timings | a bar per stage from the same `Trace` the `--trace` flag prints |
| Vector projection | the retrieved chunks and the query, in 2D |
| Strategy comparison | the same question run through all six strategies |

The first four are views of a *single* run, so they share one request to
`/api/ask` — giving each its own endpoint would re-run retrieval three extra
times and leave four panels describing four different runs. Comparison
re-runs the question once per strategy and is much slower, so it is a second,
independent request to `/api/compare` and each panel clears its own loading
state: the chunk table paints while the LLM-backed strategies are still
thinking.

**The projection is 384 dimensions squeezed into 2, and must not be read as
the embedding space.** It is PCA via `np.linalg.svd` — centre the points,
keep the two directions of greatest variance — and those two directions
preserve as much spread as any plane can, which is a far weaker promise than
it looks. Distances in the scatter are not distances in the index, and
apparent clusters may be an artefact of the two axes that survived. The panel
says so on the page. The query marker is projected with the *same* transform
as the chunks rather than its own, because a marker placed by a different
transform would sit somewhere arbitrary relative to the points it is there to
be compared against.

**It is read-only and it binds `127.0.0.1`.** Every endpoint is a `GET`;
mutating verbs return 405. Nothing in `rag/dashboard/` writes an index, a
corpus or a cache entry — an inspector that can change the thing it inspects
is a worse tool. It has no authentication and it spends Gemini quota on every
strategy that rewrites a question, so it stays off the network unless you
pass `--host` and mean it.

No build step and no front-end framework: one `.html`, one `.css`, one `.js`
of plain JavaScript served from disk, with no `<script src="http...">`. A
framework fetched from a CDN is still a framework wrapper, and
`tests/test_dashboard_static.py` fails if one appears. FastAPI and uvicorn are
an optional extra, imported inside the `dashboard` branch of the CLI, so
`python -m rag ask` still runs with neither installed.

## Known limitations

**`publish_date` values are approximate.** The dates in `data/metadata.json`
were written from domain knowledge rather than read from the arXiv API, so
some may be revision dates rather than original submission dates. They are
adequate for the metadata filtering Phase 4 demonstrates, which only needs a
field that partitions the corpus, but they are not authoritative.

**The corpus is uneven.** It covers ColBERT, RAPTOR, DPR and HyDE well, and
has no document that explains reciprocal rank fusion at all — which is why the
refusal above is genuine rather than staged.

**Search is O(n) per query** and the whole index lives in memory. See the note
on brute-force search above.

**The gold set cannot score multi-representation, and penalises RAPTOR.** Its
answers are character spans, so a summary node is structurally uncreditable.
Both techniques are reported above on document-level metrics as well. Phase 7
added the cross-document questions Phase 5 said were missing, and RAPTOR
still loses on them (-0.051 `Recall@20` against flat on that subset alone);
what remains unbuilt is a metric that can credit an answer assembled *from*
an abstraction rather than quoted from a source paper.

**30 questions is still a small gold set, written by one person.** The
questions, the quotes and the `expects` predictions all come from the author
of the system. Pre-registering the predictions and reporting that only 6 of
20 held is what makes that bias visible; it does not remove it. Three
questions are answered by no strategy at all and are left in place unedited.

**Two tables are incomplete.** `decomposition` is missing from the raptor,
multirep and reranked tables, and the `--route` table is still the
10-question one. See "What could not be measured, and why" above.

## Configuration

The default LLM is `gemini-3.5-flash-lite` (`rag/config.py`), pinned to a
dated model rather than `gemini-flash-lite-latest` so that timings and answers
quoted here stay reproducible as the "latest" alias moves on. Override any
setting via `.env` or CLI flags; see `Config` for the full list.

`--retrieval-depth` (default 20) is how many chunks each individual query
retrieves before a multi-query strategy combines the lists — distinct from
`--k`, which is how many chunks survive into the answer prompt after that
combination; `python -m rag ask "..." --strategy hyde --retrieval-depth 30`
raises it for one run.

## Tests

```bash
python -m pytest              # fast: no network, no model, no API key
python -m pytest -m slow      # loads the real embedding model
```

`tests/test_no_frameworks.py` enforces the central claim: it fails if any
forbidden package is imported anywhere in the project.

## Roadmap

- [x] **Phase 1** — core pipeline
- [x] **Phase 2** — query translation: multi-query, RAG-Fusion, decomposition, step-back, HyDE
- [x] **Phase 3** — evaluation harness: Recall@k, MRR, nDCG, DocPrec@k
- [x] **Phase 4** — routing and query construction
- [x] **Phase 5** — multi-representation indexing and RAPTOR
- [x] **Phase 6** — ColBERT-style late interaction, and a retrieval inspector dashboard
- [x] **Phase 7** — full gold set and final benchmark table

Design: [`docs/superpowers/specs/2026-09-27-rag-from-scratch-design.md`](docs/superpowers/specs/2026-09-27-rag-from-scratch-design.md)
