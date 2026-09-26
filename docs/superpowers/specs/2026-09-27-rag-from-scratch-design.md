# RAG from Scratch — Design

Date: 2026-09-27
Status: Approved for planning

## Purpose

Build a retrieval-augmented generation system with no RAG framework — no LangChain,
no LlamaIndex, no vector database, no `sentence-transformers`. Every mechanic the
frameworks hide is written out: the text splitter, the embedding pooling, the
similarity math, the rank fusion, the prompt assembly.

The system implements the techniques described in `RAG.pdf`: the core
indexing/retrieval/generation loop, the three-stage advanced pipeline (query
translation, routing, query construction), two advanced indexing techniques
(multi-representation, RAPTOR), and ColBERT-style late interaction.

This is a portfolio piece. Its success condition is that an engineer reviewing the
repository concludes the author understands text vectorization, similarity search,
and prompt engineering without third-party abstractions. A benchmark table showing
which techniques measurably help is the strongest single artifact toward that end.

### A correction carried into the code

`RAG.pdf` repeatedly describes embeddings as living in "3D space". They do not —
the model used here produces 384 dimensions. The PDF's diagrams are 3D because 3D
is drawable. Nothing about the algorithms changes, but the code names this
dimension `dim`, and the dashboard's 2D scatter plot is labelled a PCA
*projection* rather than the space itself.

## Allowed dependencies

`numpy`, `torch`, `transformers` (embeddings only), `google-genai` (generation
only), `fastapi` + `uvicorn` (dashboard only), `pytest`. Standard library for
everything else. `scikit-learn` is installed in the environment but is not used;
k-means and PCA are written out.

LangChain is installed in the environment. It must not be imported anywhere in
this project.

## Architecture

### Module layout

```
rag/
  config.py               Config dataclass: model ids, chunk size/overlap, k, paths
  loader.py               data/corpus/*.txt + metadata.json -> Document
  chunking.py             token-aware sliding window -> Chunk
  embedding.py            AutoModel + mean pooling + L2 norm -> (n, 384) float32
  store.py                VectorStore: NumPy matrix + parallel metadata; .npz save/load
  similarity.py           cosine, top_k, reciprocal_rank_fusion, maxsim
  llm.py                  Gemini provider: generate() / structured(); retry + backoff + cache
  prompts.py              every prompt template, in one place
  trace.py                Trace / StageTiming dataclasses
  strategies/
    base.py               Strategy protocol
    direct.py             plain top-k retrieval
    multi_query.py
    rag_fusion.py
    decomposition.py      recursive (IR-CoT) and independent variants
    step_back.py
    hyde.py
  routing.py              logical (LLM structured output) + semantic (cosine over prompts)
  query_construction.py   NL -> MetadataFilter -> NumPy boolean mask
  indexing/
    multi_representation.py
    raptor.py
  colbert.py              token-level late interaction reranker
  generation.py           context assembly + answer prompt
  pipeline.py             wires the stages, returns a Trace
  __main__.py             CLI
evaluation/
  gold.json               ~30 questions tagged with expected chunks
  metrics.py              recall@k, mrr, ndcg
  benchmark.py            run all strategies, emit markdown table
web/
  app.py                  FastAPI, read-only over rag/
  static/                 index.html, app.js, style.css
scripts/
  fetch_corpus.py         download and clean 35-40 source documents
data/
  corpus/*.txt            committed
  metadata.json           source, title, publish_date, author, url
tests/
```

### Data flow for one question

```
question
  -> routing         select source subset                 (PDF Stage 2)
  -> construction    NL -> metadata filter -> mask         (PDF Stage 3)
  -> translation     1 question -> N queries               (PDF Stage 1)
  -> embed           N x 384 float32
  -> search          (N,384) @ (384,chunks), mask, top-k
  -> fuse            RRF across the N result lists
  -> generate        context + template -> Gemini
  -> Trace           every intermediate value, every timing
```

### Three load-bearing decisions

**`Trace` is the spine.** Each stage appends to one `Trace` object: rewritten
queries, retrieved chunks with scores, the exact prompt sent, and milliseconds per
stage. The CLI prints it under `--trace`, the dashboard renders it as panels, and
the benchmark reads scores off it. One representation, three consumers. This is
what lets the dashboard stay a thin read-only layer that owns no logic.

**Chunking is token-aware, not character-aware.** MiniLM's context window is 256
tokens. A character-based sliding window produces chunks the model silently
truncates, losing the tail of every long chunk with no error raised.
`chunking.py` therefore slides over `tokenizer.encode()` output — 200 tokens with
50 overlap — and decodes back to text. Chunk boundaries are recorded as both token
offsets and character offsets, so the dashboard can highlight source spans.

**`VectorStore` is a dense NumPy matrix, and that limit is documented rather than
hidden.** Search is one `(n_queries, 384) @ (384, n_chunks)` matmul: exact
brute-force KNN, no approximate index. At a few thousand chunks this is
sub-millisecond. The README states plainly that this is O(n) per query and that
FAISS or HNSW is the answer past roughly a million vectors.

## Corpus

`scripts/fetch_corpus.py` downloads 35–40 documents about RAG and its techniques —
arXiv papers and abstracts (RAG, HyDE, RAPTOR, ColBERT, Self-RAG, Step-Back
Prompting, IR-CoT) plus engineering blog posts — strips markup, and writes
`data/corpus/*.txt` alongside `data/metadata.json`.

The resulting text files are committed. Three reasons: the repository clones and
runs offline, the gold evaluation set stays pinned to a corpus that cannot drift,
and the fetch script still demonstrates real HTML and PDF parsing.

35–40 documents rather than ~20, because RAPTOR clustering needs enough material
to produce meaningful clusters. Raising this number later would invalidate the
gold set.

Metadata fields — `source`, `title`, `publish_date`, `author`, `url` — are chosen
so that query construction (Stage 3) has real constraints to map onto, including
the PDF's own `publish_date < 2024` example.

## Phases

Each phase is independently demoable and independently useful. Each also gets its
own implementation plan and its own execution pass; this document is the shared
spec they are planned against, not a single plan.

### Phase 1 — Core pipeline

`fetch_corpus.py`, loader, chunking, embedding, store, cosine + top-k, generation,
CLI. Delivers `python -m rag ask "..."` returning a grounded answer with cited
chunks.

Tests: chunk boundary and overlap arithmetic, mean pooling against hand-computed
values, cosine against a NumPy reference, top-k ordering including ties.

This phase alone satisfies the original project brief. Everything after it
implements `RAG.pdf`.

### Phase 2 — PDF Stage 1: query translation

The `Strategy` protocol and five strategies: multi-query, RAG-Fusion,
decomposition, step-back, HyDE. Selected with `--strategy`.

Decomposition implements both forms the PDF describes: the recursive IR-CoT form
where each sub-answer is concatenated into the next sub-question, and the
independent form where sub-answers are gathered and synthesised at the end. These
are a flag on one strategy.

Tests: RRF scores against a hand-computed worked example, query deduplication,
trace shape. The LLM is mocked, so the suite runs offline in seconds.

### Phase 3 — Evaluation harness

The package is named `evaluation/` rather than `eval/`, since `eval` is a Python
builtin and a module of that name reads as a mistake.

`evaluation/gold.json`: ~30 hand-written questions, each tagged with the chunk ids that
should be retrieved. `evaluation/metrics.py`: Recall@k, MRR, nDCG. `evaluation/benchmark.py`
runs every strategy across the gold set and emits a markdown table.

Placed third rather than last so that every technique in Phases 4–6 is measured
the day it lands, rather than evaluated in bulk at the end. The README's benchmark
table then grows across the commit history.

### Phase 4 — PDF Stages 2 and 3: routing and query construction

Logical routing: Gemini structured output against a JSON schema describing the
available sources. Semantic routing: cosine between the question embedding and a
set of embedded prompt descriptions, selecting the closest prompt.

Query construction: natural language to a `MetadataFilter` dataclass (source, date
range, author), compiled to a NumPy boolean mask applied before top-k rather than
after — so k results are returned, rather than however many of the top k survived
the filter.

Demo: "What did the RAPTOR paper say about clustering, from anything published
before 2024?" — the trace shows the inferred filter.

Tests: filter-to-mask correctness, schema validation, and the fallback path when
the model returns malformed JSON.

### Phase 5 — Indexing techniques

**Multi-representation.** Summarise each document with Gemini, embed the
summaries, keep a `doc_id -> full document` docstore. Retrieval matches against a
summary and returns the whole document for the generation context.

**RAPTOR.** K-means written in NumPy, applied recursively: cluster chunk
embeddings, summarise each cluster, embed the summaries, cluster those, until one
cluster remains or a depth cap is reached. Every level is stored in the same
`VectorStore` with a `level` metadata field, so a single search spans raw chunks
and abstractions together.

Demo: a high-level question such as "what problem do most of these techniques
share?", which flat top-k answers poorly and RAPTOR answers well — the PDF's
stated motivation, shown rather than asserted.

Tests: k-means convergence on synthetic separable clusters, recursion
termination, level metadata integrity.

### Phase 6 — ColBERT and the dashboard

**ColBERT** as a reranker. Dense retrieval fetches the top 50; token-level late
interaction rescores them to the top 5. Token embeddings keep all positions with
no pooling; the score is the sum over query tokens of the maximum similarity
against any document token — exactly the PDF's description.

Two limitations go in the README rather than being glossed. It reranks rather than
indexing every token of the corpus, because full late-interaction indexing costs
roughly 100x the storage and reranking is what is deployed in practice. And these
are MiniLM token vectors, not trained ColBERT weights; MiniLM was not trained for
late interaction, so the gain may be small or negative. Phase 3's benchmark
reports whatever it actually is. A measured negative result, explained, is a
stronger signal than an unmeasured technique.

**Dashboard**, built last, as a read-only FastAPI layer over the package plus one
static page of vanilla JavaScript. No React, no build step, no Streamlit — a
framework wrapper would cut against the premise of the project.

Panels:

- Chunk table: every retrieved chunk with raw cosine score, rank, source, index.
- Strategy comparison: the same question across strategies, side by side.
- Translation trace: rewritten queries, the HyDE hypothetical document, sub-questions and their intermediate answers.
- Stage timings: embed / search / generate, in milliseconds.
- Vector projection: chunk embeddings reduced to 2D via `np.linalg.svd`, with the question as a distinct marker. Labelled a projection of 384 dimensions, not the space itself.

Because the LLM-heavy strategies take seconds, each panel renders its own loading
state independently rather than blocking the page.

## Cross-cutting concerns

### Testing

Test-driven throughout: write the test, watch it fail, then implement.

The LLM is mocked by default, so the full suite runs offline with no API spend. A
separate `@pytest.mark.live` suite exercises real Gemini calls and is skipped
unless explicitly selected. The embedding model loads once per session via a
fixture; pure-math units — chunking, cosine, RRF, k-means, metrics — use a tiny
deterministic fake embedder and load no model at all.

### Error handling

**Every advanced strategy degrades to plain retrieval rather than crashing, and
the `Trace` records that it degraded.** If multi-query's rewrite call is rate
limited, retrieval proceeds on the original question and the trace says so
explicitly. Silent degradation would be worse than a crash, so it is recorded, not
swallowed.

LLM calls retry with exponential backoff on 429 and 5xx. Structured-output calls
validate against their schema and fall back to a permissive default when the model
returns malformed JSON.

### Caching

An on-disk cache keyed by `hash(prompt + model + params)` for LLM calls, and
`.npz` for the embedding matrix.

Without this, a single benchmark run is roughly 6 strategies (direct plus the
five translation strategies) x 30 questions x several calls each — on the order of 700 LLM calls, repeated on every metric
tweak. On Gemini's free tier that is the difference between a twenty-second
iteration loop and a rate-limited afternoon.

### Configuration and reproducibility

`GOOGLE_API_KEY` is read from the environment or a gitignored `.env`. One `Config`
dataclass holds model ids, chunk size, overlap, k, and paths; CLI flags override
it. Temperature is 0 wherever the API allows and k-means is seeded, so benchmark
numbers are reproducible rather than drifting between runs.

## Non-goals

No approximate nearest-neighbour index. No conversation memory or multi-turn. No
token streaming. No authentication or multi-user support. No Docker or deployment.
No reranking model beyond ColBERT. Each is reasonable to want; none strengthens
the signal this project exists to send.

## Risks

**Gemini free-tier rate limits** are the most likely source of friction. Mitigated
by the response cache, exponential backoff, and mocked tests — but limits should
be expected during benchmark runs.

**Gold set quality is manual work.** Thirty questions tagged with correct chunks
takes real effort, and a sloppy gold set produces confident but meaningless
metrics. Budget time for it in Phase 3 rather than rushing it.

**RAPTOR clustering quality depends on corpus size.** The 35–40 document target
addresses this, but if clusters still come out uninformative, the fallback is to
chunk more aggressively so there are more leaves to cluster.
