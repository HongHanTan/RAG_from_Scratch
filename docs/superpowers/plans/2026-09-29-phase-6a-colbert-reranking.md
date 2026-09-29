# Phase 6a: ColBERT-Style Late Interaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rerank dense-retrieval results with token-level late interaction, and measure honestly whether it helps.

**Architecture:** Dense retrieval fetches a deep pool (50); a reranker rescores each candidate with MaxSim — the sum over query tokens of the maximum similarity against any document token — and returns the top k reordered. Token embeddings come from the same MiniLM model with no pooling, so this reuses the existing `Embedder` rather than adding a dependency.

**Tech Stack:** Python 3.14, NumPy, `transformers` (already used), pytest. No new dependencies.

## Global Constraints

- **No RAG framework.** `langchain`, `llama_index`, `sentence_transformers`, `sklearn`, `bs4`, `requests`, `dotenv` are installed and MUST NOT be imported. `tests/test_no_frameworks.py` scans `rag/`, `scripts/`, `tests/`, `evaluation/` and `pyproject.toml`.
- **Allowed third-party:** `numpy`, `torch`, `transformers`, `google.genai`, `pytest`.
- **Determinism.** Model in `.eval()` under `torch.no_grad()`, stable sorts. Benchmark numbers must not drift.
- **Test-driven.** Write the test, run it, watch it fail for the expected reason, then implement.
- **Every unit test runs offline.** No API call, no model load, unless marked `@pytest.mark.slow` or `@pytest.mark.live`; both deselected by default.
- **Platform is Windows.** `pathlib`, explicit `encoding="utf-8"`.
- **Degrade, but record it.** Use `Trace.degraded(what, fallback)` — it writes the `degraded` sentinel the benchmark hard-fails on. `tests/test_degradation_notes.py` enforces this with an AST walk over every `.note()` call in `rag/` against an explicit allowlist.
- **Never set `PYTHONIOENCODING`** when running anything. A Windows console crash was once masked by exactly that.
- **Commit after every task.** Messages end with a blank line then exactly:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## What Phases 1–5 provide

| Thing | Signature |
|---|---|
| `rag.embedding.Embedder` | `.tokenizer` (fast), `.model` (HF `AutoModel`, `.eval()`), `._torch`, `.max_length`, `.device`, `.dim`, `.encode(texts, batch_size=32) -> np.ndarray` (L2-normalised float32) |
| `rag.embedding.mean_pool(hidden, mask)` | `(batch, seq, dim) x (batch, seq) -> (batch, dim)` |
| `rag.embedding.l2_normalize(matrix, eps=1e-12)` | normalises the **last** axis, so it works on `(n, dim)` and `(seq, dim)` alike |
| `rag.chunking.RetrievedChunk` | `chunk`, `score`, `rank` (1-based), `score_kind="cosine"` |
| `rag.chunking.Chunk` | `chunk_id`, `doc_id`, `index`, `text`, `token_start`, `token_end`, `char_start`, `char_end`, `level=0`, `.is_synthetic` |
| `rag.config.Config` | frozen; `top_k=5`, `retrieval_depth=20`, `max_seq_tokens=256`, validates `retrieval_depth >= top_k` in `__post_init__` |
| `rag.pipeline.ask(...)` | `(question, store, embedder, llm, config, k=None, strategy="direct", strategy_options=None, generate=True, route=False, construct=False, semantic_prompt=False) -> Trace` |
| `rag.trace.Trace` | `stage(name)`, `note(msg)`, `degraded(what, fallback)`, `.retrieved` |
| `tests.conftest` | `FakeEmbedder` (dim 8, CRC-seeded), `FakeTokenizer`, `FakeLLM`, `tiny_corpus` |

Current suite: 637 passed, 6 deselected. Indexes on disk: `data/index.npz` (5,116 flat), `data/index-multirep.npz` (38), `data/index-raptor.npz` (5,846).

Flat baseline, k=20, the table this phase must be comparable to:

| Strategy | Recall@20 | MRR@20 | nDCG@20 | DocPrec@5 |
|---|---:|---:|---:|---:|
| hyde | 0.558 | 0.358 | 0.315 | 0.660 |
| rag-fusion | 0.392 | 0.140 | 0.184 | 0.560 |
| step-back | 0.383 | 0.136 | 0.170 | 0.400 |
| multi-query | 0.375 | 0.190 | 0.208 | 0.400 |
| decomposition | 0.333 | 0.194 | 0.187 | 0.600 |
| direct | 0.325 | 0.103 | 0.137 | 0.560 |

## Design decisions, stated up front

**Reranking, not full late-interaction indexing.** Indexing every token of 5,116 chunks at 384 dimensions is roughly 100x the storage of the pooled index, and reranking a shortlist is what is actually deployed. The spec requires this limitation in the README rather than glossed.

**MiniLM token vectors, not trained ColBERT weights.** `all-MiniLM-L6-v2` was trained for pooled sentence similarity, not for late interaction. The gain may be small or negative. The benchmark reports whatever it is: a measured negative result, explained, is a stronger signal than an unmeasured technique.

**Special tokens are excluded from MaxSim.** `[CLS]` and `[SEP]` appear in every sequence, so a query's `[CLS]` matches every document's `[CLS]` at near-1.0 and adds a near-constant offset to every score. Near-constant is not constant, so it is noise on the ranking rather than a harmless shift. Padding is excluded too, via the attention mask.

**`score_kind="maxsim"`.** MaxSim sums one max per query token, so a 12-token query scores around 6–12 where cosine scores around 0.5. `score_kind` exists precisely so a display does not present those as the same quantity.

**The benchmark reranks 50 down to `k`, not to 5.** The spec's "top 50 to top 5" is the CLI default (`top_k=5`). The benchmark measures Recall@20, so it reranks into 20 — otherwise every reranked row would be capped at 5 results and the comparison against the table above would be meaningless.

## File Structure

```
rag/
  embedding.py          MODIFY: encode_tokens()
  late_interaction.py   NEW: maxsim(), rerank()
  config.py             MODIFY: rerank_depth
  pipeline.py           MODIFY: ask(..., rerank=False)
  __main__.py           MODIFY: --rerank
evaluation/benchmark.py MODIFY: --rerank
tests/                  test_late_interaction.py (new); others modified
README.md               MODIFY
```

---

### Task 1: Token embeddings without pooling

**Files:**
- Modify: `rag/embedding.py`
- Test: `tests/test_embedding.py`

**Interfaces:**
- Produces: `Embedder.encode_tokens(texts: list[str], batch_size: int = 16) -> list[np.ndarray]` — one `(n_tokens, dim)` float32 array per text, each row L2-normalised, padding and special tokens removed.

`encode` throws away every position but the mean. Late interaction needs them all.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_embedding.py`:

```python
# --- token embeddings for late interaction -----------------------------------

@pytest.mark.slow
def test_encode_tokens_returns_one_matrix_per_text():
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2", max_length=64)
    out = embedder.encode_tokens(["hello world", "a longer sentence here"])
    assert len(out) == 2
    assert all(m.ndim == 2 for m in out)


@pytest.mark.slow
def test_encode_tokens_keeps_every_position():
    # The whole point: encode() collapses the sequence to one vector, this
    # must not. A longer text must yield strictly more rows.
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2", max_length=64)
    short, long = embedder.encode_tokens(["cat", "the cat sat on the mat today"])
    assert long.shape[0] > short.shape[0]


@pytest.mark.slow
def test_encode_tokens_rows_are_unit_length():
    # MaxSim is a dot product standing in for cosine, which is only valid
    # when every row is already unit length.
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2", max_length=64)
    matrix = embedder.encode_tokens(["hello world"])[0]
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


@pytest.mark.slow
def test_encode_tokens_width_is_the_model_dim():
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2", max_length=64)
    assert embedder.encode_tokens(["hello"])[0].shape[1] == embedder.dim


@pytest.mark.slow
def test_encode_tokens_excludes_padding():
    # Batched together, the short text must not inherit the long one's
    # padding: a padded row is a real vector and would win a max.
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2", max_length=64)
    together = embedder.encode_tokens(["hi", "a considerably longer piece of text"])
    alone = embedder.encode_tokens(["hi"])
    assert together[0].shape == alone[0].shape


@pytest.mark.slow
def test_encode_tokens_excludes_special_tokens():
    # [CLS] and [SEP] appear in every sequence, so they match across every
    # pair at near-1.0 and add near-constant noise to every score.
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2", max_length=64)
    ids = embedder.tokenizer("hello world")["input_ids"]
    assert embedder.encode_tokens(["hello world"])[0].shape[0] == len(ids) - 2


@pytest.mark.slow
def test_encode_tokens_respects_max_length():
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2", max_length=16)
    assert embedder.encode_tokens(["word " * 200])[0].shape[0] <= 16


@pytest.mark.slow
def test_encode_tokens_of_nothing_is_an_empty_list():
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2", max_length=64)
    assert embedder.encode_tokens([]) == []


@pytest.mark.slow
def test_encode_tokens_is_deterministic():
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2", max_length=64)
    a = embedder.encode_tokens(["repeatable text"])[0]
    b = embedder.encode_tokens(["repeatable text"])[0]
    assert np.array_equal(a, b)
```

Also add `encode_tokens` to `tests/conftest.py`'s `FakeEmbedder`, so the offline tests in later tasks have it:

```python
    def encode_tokens(self, texts: list[str], batch_size: int = 16) -> list:
        """One row per whitespace token, CRC-seeded like `encode`.

        Deterministic and unit-length, so MaxSim over it is meaningful
        without loading a real model.
        """
        out = []
        for text in texts:
            words = text.split() or [text]
            rows = np.zeros((len(words), self.dim), dtype=np.float32)
            for i, word in enumerate(words):
                seed = zlib.crc32(word.encode("utf-8"))
                rows[i] = np.random.default_rng(seed).normal(size=self.dim)
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out.append((rows / norms).astype(np.float32))
        return out
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_embedding.py -q -m slow`
Expected: `AttributeError: 'Embedder' object has no attribute 'encode_tokens'`.

- [ ] **Step 3: Implement `encode_tokens` in `rag/embedding.py`**

Add to the `Embedder` class, after `encode`:

```python
    def encode_tokens(
        self, texts: list[str], batch_size: int = 16
    ) -> list[np.ndarray]:
        """Per-token embeddings, one (n_tokens, dim) matrix per text.

        `encode` averages the sequence away; late interaction needs every
        position, because its whole claim is that a query term should be
        able to match one specific term in a passage rather than the
        passage's average meaning.

        Padding is dropped via the attention mask, and `[CLS]`/`[SEP]` via
        the special-tokens mask. Special tokens matter more than they look:
        they appear in every sequence, so a query's `[CLS]` matches every
        document's `[CLS]` at near-1.0 and contributes a near-constant term
        to every score. Near-constant is not constant, so it is noise on the
        ranking rather than a harmless offset.

        A smaller default batch than `encode` because the output here is
        (batch x seq x dim) rather than (batch x dim) -- keeping all
        positions is exactly what makes it heavy.
        """
        if not texts:
            return []

        out: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            special = encoded.pop("special_tokens_mask").cpu().numpy()
            encoded = encoded.to(self.device)
            with self._torch.no_grad():
                hidden = self.model(**encoded).last_hidden_state.cpu().numpy()
            attention = encoded["attention_mask"].cpu().numpy()
            keep = (attention == 1) & (special == 0)
            for row in range(hidden.shape[0]):
                out.append(l2_normalize(hidden[row][keep[row]]))
        return out
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_embedding.py -q -m slow`
Then the offline suite: `python -m pytest -q` (637 passed, 6 deselected, unchanged).

- [ ] **Step 5: Commit**

```bash
git add rag/embedding.py tests/
git commit -m "feat: per-token embeddings for late interaction

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: MaxSim

**Files:**
- Create: `rag/late_interaction.py`
- Test: `tests/test_late_interaction.py`

**Interfaces:**
- Produces: `maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float`

The scoring function from the PDF: sum over query tokens of the maximum similarity against any document token.

- [ ] **Step 1: Write the failing test**

Create `tests/test_late_interaction.py`:

```python
import numpy as np
import pytest

from rag.late_interaction import maxsim


def _unit(rows):
    m = np.asarray(rows, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_identical_single_tokens_score_one():
    v = _unit([[1.0, 0.0]])
    assert maxsim(v, v) == pytest.approx(1.0)


def test_orthogonal_single_tokens_score_zero():
    q = _unit([[1.0, 0.0]])
    d = _unit([[0.0, 1.0]])
    assert maxsim(q, d) == pytest.approx(0.0)


def test_the_score_sums_over_query_tokens():
    # Two query tokens each matching perfectly sum to 2.0, not average to
    # 1.0. That sum is why a maxsim score is not comparable with a cosine.
    q = _unit([[1.0, 0.0], [0.0, 1.0]])
    d = _unit([[1.0, 0.0], [0.0, 1.0]])
    assert maxsim(q, d) == pytest.approx(2.0)


def test_only_the_best_document_token_counts_per_query_token():
    # One query token, three document tokens: the two poor matches must not
    # dilute the good one. That is the "max" in MaxSim.
    q = _unit([[1.0, 0.0]])
    d = _unit([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert maxsim(q, d) == pytest.approx(1.0)


def test_one_document_token_may_serve_several_query_tokens():
    # No assignment constraint: both query tokens can max against the same
    # document token.
    q = _unit([[1.0, 0.0], [1.0, 0.0]])
    d = _unit([[1.0, 0.0]])
    assert maxsim(q, d) == pytest.approx(2.0)


def test_extra_irrelevant_document_tokens_do_not_lower_the_score():
    q = _unit([[1.0, 0.0]])
    short = _unit([[1.0, 0.0]])
    padded = _unit([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 1.0]])
    assert maxsim(q, padded) >= maxsim(q, short) - 1e-6


def test_a_longer_query_scores_higher_all_else_equal():
    # A consequence worth knowing: maxsim is not length-normalised, so it
    # compares documents for one query, never queries with each other.
    d = _unit([[1.0, 0.0], [0.0, 1.0]])
    assert maxsim(_unit([[1.0, 0.0], [0.0, 1.0]]), d) > maxsim(_unit([[1.0, 0.0]]), d)


def test_an_empty_document_scores_zero():
    assert maxsim(_unit([[1.0, 0.0]]), np.zeros((0, 2), dtype=np.float32)) == 0.0


def test_an_empty_query_scores_zero():
    assert maxsim(np.zeros((0, 2), dtype=np.float32), _unit([[1.0, 0.0]])) == 0.0


def test_mismatched_dimensions_are_an_error():
    with pytest.raises(ValueError, match="dimension"):
        maxsim(_unit([[1.0, 0.0]]), _unit([[1.0, 0.0, 0.0]]))


def test_the_result_is_a_plain_float():
    # It travels into RetrievedChunk.score and then into JSON via the trace;
    # a np.float32 there serialises badly.
    assert type(maxsim(_unit([[1.0, 0.0]]), _unit([[1.0, 0.0]]))) is float
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_late_interaction.py -q`
Expected: `ModuleNotFoundError: No module named 'rag.late_interaction'`.

- [ ] **Step 3: Implement `maxsim` in `rag/late_interaction.py`**

```python
"""ColBERT-style late interaction, as a reranker.

Dense retrieval compares one vector per query against one vector per chunk,
so a passage that answers a question in one clause is represented by the
average of every clause it contains. Late interaction keeps every token on
both sides and asks a narrower question: for each query term, how well does
the best-matching term in this passage match it?

Two limitations, kept in the README rather than glossed:

- This reranks a shortlist rather than indexing every token of the corpus.
  Full late-interaction indexing is roughly 100x the storage, and reranking
  is what is deployed in practice.
- These are MiniLM token vectors, not trained ColBERT weights. MiniLM was
  trained for pooled sentence similarity, so the gain may be small or
  negative. The benchmark reports whichever it is.
"""

from __future__ import annotations

import numpy as np


def maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """Sum over query tokens of the best similarity against any doc token.

    Both matrices are (n_tokens, dim) with unit-length rows, so the matrix
    product is a matrix of cosines and no normalisation happens here.

    The score is a sum, not a mean, so it grows with query length and is
    only ever used to compare documents *for one query* -- never to compare
    one query's results with another's.
    """
    query_tokens = np.asarray(query_tokens, dtype=np.float32)
    doc_tokens = np.asarray(doc_tokens, dtype=np.float32)
    if query_tokens.size == 0 or doc_tokens.size == 0:
        return 0.0
    if query_tokens.shape[1] != doc_tokens.shape[1]:
        raise ValueError(
            f"dimension mismatch: query {query_tokens.shape[1]} vs "
            f"document {doc_tokens.shape[1]}"
        )
    return float((query_tokens @ doc_tokens.T).max(axis=1).sum())
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_late_interaction.py -q`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/late_interaction.py tests/test_late_interaction.py
git commit -m "feat: MaxSim scoring for late interaction

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: The reranker

**Files:**
- Modify: `rag/late_interaction.py`
- Test: `tests/test_late_interaction.py`

**Interfaces:**
- Consumes: `maxsim`, `Embedder.encode_tokens`, `RetrievedChunk`
- Produces: `rerank(question: str, retrieved: list[RetrievedChunk], embedder, k: int, trace=None) -> list[RetrievedChunk]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_late_interaction.py`:

```python
from rag.chunking import Chunk, RetrievedChunk
from rag.late_interaction import rerank
from rag.trace import Trace
from tests.conftest import FakeEmbedder


def _hit(i, text, score, rank):
    return RetrievedChunk(
        chunk=Chunk(f"d:{i}", "d", i, text, 0, 10, i * 50, i * 50 + 40),
        score=score,
        rank=rank,
    )


def _candidates():
    return [
        _hit(0, "alpha beta gamma", 0.90, 1),
        _hit(1, "delta epsilon zeta", 0.80, 2),
        _hit(2, "question words appear here", 0.70, 3),
    ]


def test_returns_at_most_k():
    out = rerank("question words", _candidates(), FakeEmbedder(), k=2)
    assert len(out) == 2


def test_ranks_are_renumbered_from_one():
    out = rerank("question words", _candidates(), FakeEmbedder(), k=3)
    assert [r.rank for r in out] == [1, 2, 3]


def test_scores_are_marked_maxsim():
    # A maxsim score sums one max per query token, so it lands around 5-20
    # where a cosine lands around 0.5. Presenting both as "score" would make
    # reranking look like a tenfold quality jump.
    out = rerank("question words", _candidates(), FakeEmbedder(), k=3)
    assert all(r.score_kind == "maxsim" for r in out)


def test_results_are_sorted_by_descending_score():
    out = rerank("question words", _candidates(), FakeEmbedder(), k=3)
    assert [r.score for r in out] == sorted((r.score for r in out), reverse=True)


def test_the_order_can_differ_from_the_dense_order():
    # If reranking never reordered anything it would be a no-op, and the
    # benchmark comparison would be measuring nothing.
    candidates = _candidates()
    out = rerank("question words appear here", candidates, FakeEmbedder(), k=3)
    assert [r.chunk.chunk_id for r in out] != [c.chunk.chunk_id for c in candidates]


def test_the_chunks_themselves_are_unchanged():
    out = rerank("question words", _candidates(), FakeEmbedder(), k=3)
    assert {r.chunk.chunk_id for r in out} <= {"d:0", "d:1", "d:2"}
    assert all(r.chunk.text for r in out)


def test_reranking_nothing_returns_nothing():
    assert rerank("q", [], FakeEmbedder(), k=5) == []


def test_k_larger_than_the_candidate_pool_returns_all_of_it():
    out = rerank("q", _candidates(), FakeEmbedder(), k=99)
    assert len(out) == 3


def test_it_records_what_it_did():
    trace = Trace(question="q")
    rerank("question words", _candidates(), FakeEmbedder(), k=2, trace=trace)
    assert any("rerank" in n for n in trace.notes)


def test_an_embedder_without_token_support_degrades_rather_than_crashing():
    # Recorded with the degraded sentinel, so the benchmark hard-fails
    # instead of silently reporting dense numbers as reranked ones.
    class NoTokens:
        def encode_tokens(self, texts, batch_size=16):
            raise RuntimeError("no token support")

    trace = Trace(question="q")
    out = rerank("q", _candidates(), NoTokens(), k=2, trace=trace)
    assert [r.chunk.chunk_id for r in out] == ["d:0", "d:1"]
    assert any("degraded" in n for n in trace.notes)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_late_interaction.py -q`
Expected: `ImportError: cannot import name 'rerank'`.

- [ ] **Step 3: Implement `rerank`**

Append to `rag/late_interaction.py`:

```python
from dataclasses import replace

from rag.chunking import RetrievedChunk


def rerank(
    question: str,
    retrieved: list[RetrievedChunk],
    embedder,
    k: int,
    trace=None,
) -> list[RetrievedChunk]:
    """Rescore `retrieved` with MaxSim and return the best `k`, renumbered.

    Falling back to the dense order on failure is recorded with the
    `degraded` sentinel: reranking that silently did nothing would be
    reported as a measured null result, which is a worse outcome than a
    loud failure.
    """
    if not retrieved:
        return []

    try:
        query_tokens = embedder.encode_tokens([question])[0]
        doc_tokens = embedder.encode_tokens([r.chunk.text for r in retrieved])
    except Exception as exc:                        # model or tokenizer failure
        if trace is not None:
            trace.degraded(f"reranking failed: {exc}", "dense order kept")
        return [
            replace(item, rank=rank)
            for rank, item in enumerate(retrieved[:k], start=1)
        ]

    scored = [
        (maxsim(query_tokens, tokens), item)
        for tokens, item in zip(doc_tokens, retrieved)
    ]
    # Stable on ties, so an exact score tie keeps the dense ordering rather
    # than depending on sort internals.
    scored.sort(key=lambda pair: pair[0], reverse=True)

    out = [
        replace(item, score=score, rank=rank, score_kind="maxsim")
        for rank, (score, item) in enumerate(scored[:k], start=1)
    ]
    if trace is not None:
        trace.note(f"rerank: {len(retrieved)} candidates scored by maxsim -> {len(out)}")
    return out
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest -q`

- [ ] **Step 5: Commit**

```bash
git add rag/late_interaction.py tests/test_late_interaction.py
git commit -m "feat: late-interaction reranker over a dense shortlist

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Wire reranking into the pipeline and CLI

**Files:**
- Modify: `rag/config.py`, `rag/pipeline.py`, `rag/__main__.py`
- Test: `tests/test_config.py`, `tests/test_pipeline.py`, `tests/test_cli.py`

**Interfaces:**
- Produces: `Config.rerank_depth: int = 50`; `ask(..., rerank: bool = False)`; `python -m rag ask "..." --rerank`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_rerank_depth_default():
    assert Config().rerank_depth == 50


def test_rerank_depth_must_be_at_least_top_k():
    import pytest
    with pytest.raises(ValueError, match="rerank_depth"):
        Config(rerank_depth=3, top_k=5)
```

Append to `tests/test_pipeline.py`:

```python
def test_rerank_marks_the_results_maxsim(tiny_corpus: Config):
    build_index(tiny_corpus, FakeEmbedder(), mode="flat")
    store = load_index(tiny_corpus)
    trace = ask("q", store, FakeEmbedder(), None, tiny_corpus,
                generate=False, rerank=True)
    assert trace.retrieved
    assert all(r.score_kind == "maxsim" for r in trace.retrieved)


def test_without_rerank_the_results_stay_cosine(tiny_corpus: Config):
    build_index(tiny_corpus, FakeEmbedder(), mode="flat")
    store = load_index(tiny_corpus)
    trace = ask("q", store, FakeEmbedder(), None, tiny_corpus, generate=False)
    assert all(r.score_kind == "cosine" for r in trace.retrieved)


def test_rerank_deepens_the_candidate_pool(tiny_corpus: Config):
    # Reranking a pool the same size as k can only reorder k items; the
    # technique is supposed to pull a better item up from deeper down.
    build_index(tiny_corpus, FakeEmbedder(), mode="flat")
    store = load_index(tiny_corpus)
    trace = ask("q", store, FakeEmbedder(), None, tiny_corpus,
                generate=False, rerank=True)
    assert any("rerank" in n for n in trace.notes)
    note = next(n for n in trace.notes if "rerank" in n)
    assert "candidates" in note


def test_rerank_returns_at_most_k(tiny_corpus: Config):
    build_index(tiny_corpus, FakeEmbedder(), mode="flat")
    store = load_index(tiny_corpus)
    trace = ask("q", store, FakeEmbedder(), None, tiny_corpus,
                generate=False, rerank=True, k=2)
    assert len(trace.retrieved) <= 2


def test_rerank_works_with_a_translation_strategy(tiny_corpus: Config):
    # Reranking must sit after the strategy, so it applies to all six
    # rather than only the default path.
    build_index(tiny_corpus, FakeEmbedder(), mode="flat")
    store = load_index(tiny_corpus)
    trace = ask("q", store, FakeEmbedder(), FakeLLM("a\nb"), tiny_corpus,
                generate=False, rerank=True, strategy="multi-query")
    assert all(r.score_kind == "maxsim" for r in trace.retrieved)
```

Append to `tests/test_cli.py`:

```python
def test_ask_accepts_rerank(tiny_corpus, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())
    assert main(["ask", "q", "--rerank", "--no-llm"], **_factories()) == 0
    assert "maxsim" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_config.py tests/test_pipeline.py tests/test_cli.py -q`
Expected: `TypeError: Config.__init__() got an unexpected keyword argument 'rerank_depth'`, and `ask() got an unexpected keyword argument 'rerank'`.

- [ ] **Step 3: Add the config field**

In `rag/config.py`, after `retrieval_depth`:

```python
    rerank_depth: int = 50
    """How many candidates late-interaction reranking rescores.

    Deeper than `retrieval_depth` on purpose: reranking can only promote
    something the dense pass already returned, so a shallow pool caps how
    much it could possibly help.
    """
```

and in `__post_init__`:

```python
        if self.rerank_depth < self.top_k:
            raise ValueError(
                f"rerank_depth ({self.rerank_depth}) must be at least "
                f"top_k ({self.top_k})"
            )
```

- [ ] **Step 4: Wire it into `ask`**

In `rag/pipeline.py`, add `rerank: bool = False` to `ask`'s signature (after `semantic_prompt`), import `from rag.late_interaction import rerank as rerank_results`, and deepen the pool when it is on:

```python
    # Reranking can only promote a chunk the dense pass already returned,
    # so the pool it draws from has to be deeper than the answer. Both
    # knobs have to move: every strategy ends with
    # `retrieved[: ctx.config.top_k]`, so deepening `retrieval_depth` alone
    # would still hand the reranker exactly the k items it was meant to
    # reorder into -- inert, and indistinguishable in the benchmark from
    # reranking that does not work.
    pool = max(config.rerank_depth, effective_k) if rerank else effective_k
    depth = config.rerank_depth if rerank else config.retrieval_depth
    ctx = StrategyContext(
        store=store,
        embedder=embedder,
        llm=llm,
        config=replace(
            config,
            top_k=pool,
            retrieval_depth=max(depth, pool),
        ),
        trace=trace,
        mask=mask,
    )
```

Then, immediately before `trace.retrieved = result.retrieved` and **after** the docstore expansion block (so a multirep hit is reranked on its full document text, not on its summary):

```python
    if rerank:
        with trace.stage("rerank"):
            result = replace(
                result,
                retrieved=rerank_results(
                    question, result.retrieved, embedder, effective_k, trace
                ),
            )
```

- [ ] **Step 5: Make the trace display name the score it is showing**

`rag/__main__.py:85` hardcodes the word `score`:

```python
                f"  [{item.rank}] {chunk.chunk_id}  score {item.score:.3f}  "
```

A maxsim score of 8.3 printed as `score 8.300` beside a cosine `score 0.692`
is exactly the confusion `score_kind` was introduced to prevent, and the
prompt builder (`rag/prompts.py:42`) already prints the kind. Change it to:

```python
                f"  [{item.rank}] {chunk.chunk_id}  "
                f"{item.score_kind} {item.score:.3f}  "
```

No existing test asserts on the literal word `score` here, but run the CLI
tests after the change and say so if one does.

- [ ] **Step 6: Add the CLI flag**

On the `ask` subparser in `rag/__main__.py` (the variable is `ask_parser`):

```python
    ask_parser.add_argument(
        "--rerank", action="store_true",
        help="rescore the top rerank_depth candidates with ColBERT-style "
             "late interaction before answering",
    )
```

and pass `rerank=args.rerank` into `ask(...)`.

- [ ] **Step 7: Run the tests and verify they pass**

Run: `python -m pytest -q`

Then against the real index, which is the first time this runs on a real model:

```
python -m rag ask "How does ColBERT score a document?" --rerank --no-llm --trace
```

Report the top 3 with their `maxsim` scores and the `rerank` stage timing.

- [ ] **Step 8: Commit**

```bash
git add rag/config.py rag/pipeline.py rag/__main__.py tests/
git commit -m "feat: --rerank wires late interaction into the pipeline

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Measure it, and write down what it actually did

**Files:**
- Modify: `evaluation/benchmark.py`, `README.md`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Produces: `python -m evaluation.benchmark --rerank`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_benchmark.py`:

```python
def test_benchmark_accepts_rerank():
    from evaluation.benchmark import build_parser

    assert build_parser().parse_args(["--rerank"]).rerank is True


def test_rerank_defaults_off():
    from evaluation.benchmark import build_parser

    assert build_parser().parse_args([]).rerank is False
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_benchmark.py -q`
Expected: `AttributeError: 'Namespace' object has no attribute 'rerank'`.

- [ ] **Step 3: Add the flag**

In `evaluation/benchmark.py`'s `build_parser`:

```python
    parser.add_argument(
        "--rerank", action="store_true",
        help="rescore each strategy's candidates with ColBERT-style late "
             "interaction before scoring. Produces the same columns, so the "
             "two tables are directly comparable -- that comparison is what "
             "says whether late interaction helps.",
    )
```

Pass `rerank=args.rerank` through to `ask(...)`, and print `rerank: on/off` above the table next to the index mode.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest -q`

- [ ] **Step 5: Measure**

```
python -m evaluation.benchmark --index flat > bench-flat.txt
python -m evaluation.benchmark --index flat --rerank > bench-flat-rerank.txt
```

The first must reproduce the six baseline rows in this plan exactly; if it does not, stop and say so, because something changed that should not have.

Report both tables and the per-strategy delta. **Report the latency honestly**: reranking runs a forward pass over 50 chunks per question, so `Mean ms (warm)` should rise substantially. That cost is part of the result.

- [ ] **Step 6: Check whether it reordered anything at all**

A reranker that changes no ranks is indistinguishable from one that is switched off, and both would produce identical numbers:

```
python -c "
from pathlib import Path
from rag.__main__ import load_config
from rag.embedding import Embedder
from rag.loader import load_documents
from rag.pipeline import ask, load_index
from evaluation.gold import load_gold
cfg = load_config()
docs = load_documents(cfg.corpus_dir, cfg.metadata_path)
gold = load_gold(Path('evaluation/gold.json'), docs)
emb = Embedder(cfg.embedding_model, max_length=cfg.max_seq_tokens)
store = load_index(cfg)
moved = same = 0
for q in gold:
    a = ask(q.question, store, emb, None, cfg, k=20, generate=False)
    b = ask(q.question, store, emb, None, cfg, k=20, generate=False, rerank=True)
    ids_a = [r.chunk.chunk_id for r in a.retrieved]
    ids_b = [r.chunk.chunk_id for r in b.retrieved]
    if ids_a == ids_b: same += 1
    else: moved += 1
print(f'{moved} of {moved+same} questions had their top-20 order changed')
"
```

Report the number. If it is 0, the reranker is inert and that is the finding.

- [ ] **Step 7: Write the README section**

Cover what late interaction is and why it might beat pooled cosine, the MaxSim formula, both tables, the latency cost, and how many orderings actually changed.

**The two limitations the spec requires, stated plainly rather than glossed:**
- it reranks a shortlist rather than indexing every token, because full late-interaction indexing is roughly 100x the storage and reranking is what is deployed;
- these are MiniLM token vectors, not trained ColBERT weights, so the gain may be small or negative.

If the result is negative, say so in the heading. A measured negative result with a mechanism is a stronger portfolio signal than an unmeasured technique, and this README already does that for routing. Keep the existing caveats: 10 questions, retrieval only, incomplete chunk-level gold set.

Do **not** tick Phase 6 in the roadmap — the dashboard is Phase 6b.

- [ ] **Step 8: Commit**

```bash
git add evaluation/benchmark.py README.md tests/
git commit -m "feat: benchmark late-interaction reranking and report the result

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 6a Definition of Done

- [ ] `python -m pytest` passes offline; `-m slow` passes with the real model
- [ ] `encode_tokens` keeps every position, drops padding and special tokens, rows unit-length
- [ ] `maxsim` is a sum of per-query-token maxima, in plain NumPy
- [ ] Reranked results carry `score_kind="maxsim"`, never mixed with cosine
- [ ] Reranking sits after the strategy, so all six benefit
- [ ] The un-reranked flat table still reproduces its six baseline rows exactly
- [ ] The number of reordered questions is measured and reported
- [ ] Both spec-required limitations are in the README
- [ ] Working tree clean

## What Phase 6b needs from this

- `score_kind="maxsim"` is what lets the dashboard's chunk table show dense and reranked scores without presenting them as the same quantity.
- The `rerank` stage timing feeds the dashboard's stage-timings panel.
