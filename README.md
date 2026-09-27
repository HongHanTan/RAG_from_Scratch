# RAG from Scratch

A retrieval-augmented generation pipeline built without a RAG framework. No
LangChain, no LlamaIndex, no vector database, no `sentence-transformers`. The
text splitter, the embedding pooling, the similarity math and the prompt
assembly are all written out. `tests/test_no_frameworks.py` enforces this by
failing the suite if any of them is ever imported.

Phase 1 (core indexing, retrieval and generation) and Phase 2 (query
translation) are complete. See the roadmap below for what comes next, and the
design spec for the full plan:
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

Phase 3 measures retrieval — not answer quality — against a 10-question gold
set (`evaluation/gold.json`), using `python -m evaluation.benchmark`:

| Strategy | Recall@20 | MRR | nDCG@20 | DocPrec@5 | Mean ms | Questions |
|---|---:|---:|---:|---:|---:|---:|
| hyde | 0.558 | 0.358 | 0.315 | 0.660 | 32277 | 10 |
| step-back | 0.475 | 0.196 | 0.225 | 0.460 | 5346 | 10 |
| rag-fusion | 0.392 | 0.140 | 0.184 | 0.560 | 94 | 10 |
| multi-query | 0.375 | 0.190 | 0.208 | 0.400 | 42680 | 10 |
| decomposition | 0.333 | 0.194 | 0.187 | 0.600 | 105976 | 10 |
| direct | 0.325 | 0.103 | 0.137 | 0.560 | 53 | 10 |

Ran twice; every metric column was bit-identical between runs (`Mean ms` is
wall-clock and drops once the LLM cache is warm, which is expected and not a
determinism concern).

`Recall@20`, `MRR` and `nDCG@20` are measured at retrieval depth 20, not the
answer prompt's `top_k=5` — at k=5 every strategy scores near zero (plain
retrieval measured 0.050) and there is no headroom to tell them apart.
`DocPrec@5` is reported separately, always at 5: it is the fraction of the
chunks that would actually reach the answer prompt that come from the
document holding the answer, and it is robust to the chunk-level gold set
being incomplete (see below), since it only checks which paper a chunk came
from, not which passage.

**What this actually shows:** only HyDE clearly beats plain retrieval, and it
does so on every column — recall, ranking, and document precision alike. The
other four translation strategies are a mixed bag rather than a clean win.
RAG-Fusion and multi-query find more relevant chunks somewhere in the top 20
than direct retrieval does (higher Recall@20), but multi-query's `DocPrec@5`
(0.400) is *worse* than direct's (0.560): unioning five rewritten queries can
crowd the top 5 with chunks from the wrong paper even while surfacing more
correct chunks further down the list. Decomposition costs roughly 2000x
direct's latency (106s vs 53ms per question) for a Recall@20 gain of 0.008
over direct — not a result that would survive being spent in production.
Step-back sits in between: better recall than direct, but a lower `DocPrec@5`
too. On this corpus, with this gold set, query translation is not a uniform
win — most of it is a wash or a regression once ranking and document
precision are counted, not just "was the chunk somewhere in the top 20."

**Read this table narrowly:**

- **10 questions.** Differences of a few points are noise at this sample
  size; treat rank order among the four middling strategies as unreliable and
  the hyde-vs-everything-else gap as the only difference likely to survive a
  larger set.
- **Retrieval only.** Nothing here measures whether the retrieved chunks
  produce a better final answer — that is not evaluated anywhere yet.
- **The chunk-level gold set is incomplete.** It names some answering
  passages per question, not all of them — hand-verifying every chunk that
  could answer a question against a 3M-character corpus is not workable. That
  understates `Recall@20`, `MRR` and `nDCG@20` for every strategy equally, so
  the relative comparison holds, but none of those three numbers is an
  absolute quality score. `DocPrec@5` does not have this problem, which is
  why it is reported alongside them.
- **The gold set was written by the person who built the system**, which is
  a real bias: the questions and quotes were chosen with knowledge of how the
  corpus and chunker behave.

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

## Configuration

The default LLM is `gemini-3.5-flash-lite` (`rag/config.py`), pinned to a
dated model rather than `gemini-flash-lite-latest` so that timings and answers
quoted here stay reproducible as the "latest" alias moves on. Override any
setting via `.env` or CLI flags; see `Config` for the full list.

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
- [ ] **Phase 4** — routing and query construction
- [ ] **Phase 5** — multi-representation indexing and RAPTOR
- [ ] **Phase 6** — ColBERT-style late interaction, and a retrieval inspector dashboard
- [ ] **Phase 7** — full gold set and final benchmark table

Design: [`docs/superpowers/specs/2026-09-27-rag-from-scratch-design.md`](docs/superpowers/specs/2026-09-27-rag-from-scratch-design.md)
