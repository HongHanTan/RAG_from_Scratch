# RAG from Scratch

A retrieval-augmented generation pipeline built without a RAG framework. No
LangChain, no LlamaIndex, no vector database, no `sentence-transformers`. The
text splitter, the embedding pooling, the similarity math and the prompt
assembly are all written out. `tests/test_no_frameworks.py` enforces this by
failing the suite if any of them is ever imported.

Phase 1 — the core indexing, retrieval and generation loop — is complete. See
the roadmap below for what comes next, and the design spec for the full plan:
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
- [ ] **Phase 2** — query translation: multi-query, RAG-Fusion, decomposition, step-back, HyDE
- [ ] **Phase 3** — evaluation harness: Recall@k, MRR, nDCG
- [ ] **Phase 4** — routing and query construction
- [ ] **Phase 5** — multi-representation indexing and RAPTOR
- [ ] **Phase 6** — ColBERT-style late interaction, and a retrieval inspector dashboard
- [ ] **Phase 7** — full gold set and final benchmark table

Design: [`docs/superpowers/specs/2026-09-27-rag-from-scratch-design.md`](docs/superpowers/specs/2026-09-27-rag-from-scratch-design.md)
