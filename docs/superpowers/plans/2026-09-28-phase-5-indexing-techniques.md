# Phase 5: Indexing Techniques Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the two indexing techniques from `RAG.pdf` — multi-representation indexing and RAPTOR — and measure whether either beats flat chunking.

**Architecture:** Both techniques change what goes *into* the index rather than how it is searched, so each builds its own index file and the existing flat index stays untouched. Multi-representation embeds one summary per document and returns the whole document on a hit. RAPTOR clusters chunk embeddings with hand-written k-means, summarises each cluster, embeds the summaries, and recurses — every level living in one store so a single search spans raw chunks and abstractions together.

**Tech Stack:** Python 3.14, NumPy (k-means written out, not scikit-learn), google-genai for summarisation, pytest. No new dependencies.

## Global Constraints

- **No RAG framework.** `langchain`, `llama_index`, `sentence_transformers`, `sklearn`, `bs4`, `requests`, `dotenv` are installed and MUST NOT be imported. `tests/test_no_frameworks.py` scans `rag/`, `scripts/`, `tests/`, `evaluation/` and `pyproject.toml`. **`sklearn` matters especially this phase** — k-means is the thing being demonstrated, and importing it would defeat the point.
- **Allowed third-party imports:** `numpy`, `torch`, `transformers`, `google.genai`, `pytest`.
- **Determinism.** LLM temperature 0, k-means seeded, stable sorts. Benchmark numbers must not drift.
- **Test-driven.** Write the test, run it, watch it fail for the expected reason, then implement.
- **Every unit test runs offline.** No API call, no model load, unless marked `@pytest.mark.slow` or `@pytest.mark.live`; both deselected by default.
- **Platform is Windows.** `pathlib`, explicit `encoding="utf-8"` on every file read and write.
- **Degrade, but record it.** Use `Trace.degraded(what, fallback)` — it writes the `degraded` sentinel the benchmark hard-fails on. `tests/test_degradation_notes.py` enforces this with an AST walk over every `.note()` call in `rag/` against an explicit allowlist; a bare `note()` for a fallback fails that test.
- **Commit after every task.** Messages end with a blank line then exactly:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## What Phases 1–4 provide

| Thing | Signature |
|---|---|
| `rag.chunking.Chunk` | frozen: `chunk_id`, `doc_id`, `index`, `text`, `token_start`, `token_end`, `char_start`, `char_end` |
| `rag.chunking.RetrievedChunk` | `chunk`, `score`, `rank`, `score_kind` |
| `rag.chunking.chunk_documents(docs, tokenizer, size, overlap)` | `-> list[Chunk]` |
| `rag.loader.Document` | frozen: `doc_id`, `text`, `title`, `source`, `publish_date`, `author`, `url`, `topic` |
| `rag.store.VectorStore` | `vectors`, `chunks`, `meta`, `doc_meta`; `.search(query_vectors, k, mask=None)`, `.save(path, meta=None)`, `.load(path, expect_meta=None)`; persists JSON blobs in an `.npz`, reading each with `if "<key>" in data.files` |
| `rag.pipeline` | `build_index(config, embedder)`, `load_index(config)`, `_index_meta(config, dim)`, `ask(question, store, embedder, llm, config, k=None, strategy="direct", strategy_options=None, generate=True, route=False, construct=False, semantic_prompt=False)` |
| `rag.embedding.Embedder` | `.tokenizer`, `.dim`, `.encode(texts, batch_size=32) -> np.ndarray` (L2-normalised float32) |
| `rag.llm.GeminiLLM` | `.generate(prompt)`, `.structured(prompt, schema)`, on-disk cache keyed by model+prompt+temperature; evicts an entry that fails validation |
| `rag.trace.Trace` | `stage(name)` (depth-aware), `note(msg)`, `degraded(what, fallback)`, `add_translation(kind, text)` |
| `rag.query_construction` | `MetadataFilter`, `compile_mask(filter_, chunks, doc_meta)` |
| `evaluation/` | `load_gold`, `relevant_chunk_ids`, `recall_at_k`, `reciprocal_rank`, `ndcg_at_k`, `doc_precision_at_k`, `score_strategy`, `format_table`, `check_not_degraded` |

Corpus: 38 documents, 5,116 chunks, five topics. Current suite: 543 passed, 6 deselected.

Benchmark baseline, flat index, k=20:

| Strategy | Recall@20 | MRR@20 | nDCG@20 | DocPrec@5 |
|---|---:|---:|---:|---:|
| hyde | 0.558 | 0.358 | 0.315 | 0.660 |
| rag-fusion | 0.392 | 0.140 | 0.184 | 0.560 |
| step-back | 0.383 | 0.136 | 0.170 | 0.400 |
| multi-query | 0.375 | 0.190 | 0.208 | 0.400 |
| decomposition | 0.333 | 0.194 | 0.187 | 0.600 |
| direct | 0.325 | 0.103 | 0.137 | 0.560 |

## The decision that shapes this phase

The spec says RAPTOR's levels live in one `VectorStore` "so a single search spans raw chunks and abstractions together". That is right, and it is kept — **within** a RAPTOR index.

But `evaluation/benchmark.py` scores gold spans against `store.chunks`. Adding summary nodes to the *default* index would put them in every strategy's candidate pool, moving all six baseline rows for reasons unrelated to the strategies, and destroying the comparison Phase 3 exists to provide.

So each technique builds its **own index file**, selected by `--index-mode`:

```
data/index.npz            flat        5,116 chunks            (unchanged)
data/index-multirep.npz   multi-rep   38 summaries + docstore
data/index-raptor.npz     raptor      5,116 chunks + summary nodes at levels 1..n
```

The benchmark gains an `--index` flag, so RAPTOR and multi-representation are measured as their own runs against the same gold set and the same baseline. This is a deviation from a literal reading of the spec's "same `VectorStore`", made to protect the measurement; the spec's actual intent — one search spanning levels — holds inside the RAPTOR index.

## Four blockers the Phase 4 review identified

Each would produce a silent wrong result rather than an error.

1. **`Chunk` has no `level`**, and `VectorStore.load` reconstructs with `Chunk(**record)`. A field without a default breaks every existing index file. Task 1 adds `level: int = 0`.
2. **Character offsets are meaningless for a synthetic summary node.** Task 1 fixes the convention at `-1` and proves gold scoring ignores such nodes.
3. **`compile_mask` tests `chunk.doc_id in allowed`.** A RAPTOR summary spanning documents has no single `doc_id`. Task 1 settles the semantics and tests them, rather than leaving summaries to silently vanish from filtered search.
4. **`build_index` takes no `llm`, and `_index_meta` records nothing about the index mode.** An index built flat would load clean against a config expecting RAPTOR — exactly the "plausible nonsense, no error" failure `_index_meta` exists to prevent. Task 6 extends both.

## File Structure

```
rag/
  chunking.py            MODIFY: Chunk.level, SYNTHETIC_SPAN, synthetic-node helpers
  clustering.py          NEW: kmeans, choose_k
  summarise.py           NEW: summarise(); DOC_SUMMARY_TEMPLATE, CLUSTER_SUMMARY_TEMPLATE
  indexing/
    __init__.py          NEW
    multi_representation.py  NEW: build_multi_representation
    raptor.py            NEW: build_raptor
  store.py               MODIFY: docstore field
  pipeline.py            MODIFY: build_index(config, embedder, llm=None, mode="flat"); index_path_for
  config.py              MODIFY: raptor_max_depth, raptor_cluster_size
  query_construction.py  MODIFY: document the synthetic-node filter rule
  __main__.py            MODIFY: --index-mode on index; --index on ask
evaluation/benchmark.py  MODIFY: --index
tests/                   test_clustering.py, test_summarise.py, test_multi_representation.py,
                         test_raptor.py (new); others modified
README.md                MODIFY
```

`rag/indexing/` imports from `rag/`; nothing in `rag/` imports from `rag/indexing/` except `pipeline.py`.

---

### Task 1: Synthetic node conventions

Foundation. Everything after this creates nodes that are not spans of a document, and three separate mechanisms need to agree on what that means.

**Files:**
- Modify: `rag/chunking.py`, `rag/query_construction.py`
- Test: `tests/test_chunking.py`, `tests/test_spans.py`, `tests/test_query_construction.py`

**Interfaces:**
- Produces:
  - `Chunk.level: int = 0` — 0 for a real span of a document, 1+ for a summary node
  - `SYNTHETIC_SPAN = -1` in `rag/chunking.py`
  - `make_summary_chunk(chunk_id: str, doc_id: str, index: int, text: str, level: int) -> Chunk`
  - `Chunk.is_synthetic` property — `self.level > 0`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chunking.py`:

```python
def test_a_chunk_defaults_to_level_zero():
    # Every index written before this field existed reconstructs with
    # Chunk(**record); a field without a default would break all of them.
    chunk = Chunk("d:0", "d", 0, "text", 0, 10, 0, 4)
    assert chunk.level == 0
    assert not chunk.is_synthetic


def test_a_summary_chunk_is_synthetic_and_has_no_span():
    from rag.chunking import SYNTHETIC_SPAN, make_summary_chunk

    node = make_summary_chunk("raptor:1:3", "raptor:1:3", 3, "a summary", level=1)
    assert node.level == 1
    assert node.is_synthetic
    assert node.char_start == SYNTHETIC_SPAN
    assert node.char_end == SYNTHETIC_SPAN
    assert node.token_start == SYNTHETIC_SPAN
    assert node.token_end == SYNTHETIC_SPAN


def test_a_summary_chunk_keeps_its_text():
    from rag.chunking import make_summary_chunk

    assert make_summary_chunk("s:0", "s:0", 0, "the summary", level=2).text == "the summary"
```

Append to `tests/test_spans.py`:

```python
def test_a_synthetic_node_never_satisfies_a_gold_span():
    # Gold spans are character ranges in a real document. A summary node is
    # not a span of anything, so it must never count as a correct retrieval
    # -- otherwise RAPTOR would score for returning its own summaries.
    from rag.chunking import make_summary_chunk

    chunks = _chunks() + [make_summary_chunk("raptor:1:0", "raptor:1:0", 0, "s", level=1)]
    relevant = chunks_overlapping("alpha", 0, 300, chunks)
    assert "raptor:1:0" not in relevant


def test_a_synthetic_node_with_the_same_doc_id_is_still_excluded():
    # Multi-representation summaries carry their document's real doc_id, so
    # exclusion cannot rely on the id being synthetic -- only on the span.
    from rag.chunking import make_summary_chunk

    node = make_summary_chunk("alpha:summary", "alpha", 0, "s", level=1)
    assert chunks_overlapping("alpha", 0, 300, _chunks() + [node]) == {
        "a:0", "a:1", "a:2"
    }
```

Append to `tests/test_query_construction.py`:

```python
def test_a_multi_representation_summary_inherits_its_documents_filter():
    # Its doc_id is the real document's, so a topic or date filter that keeps
    # the document keeps its summary too. That is what makes filtered search
    # work against a multi-representation index.
    from rag.chunking import make_summary_chunk

    chunks = _chunks() + [make_summary_chunk("a:summary", "a", 0, "s", level=1)]
    mask = compile_mask(
        MetadataFilter(topics=("retrieval-models",)), chunks, _doc_meta()
    )
    assert mask[-1]


def test_a_raptor_cluster_summary_is_excluded_by_any_filter():
    # A cluster summary spans several documents, so it has no single doc_id
    # and no metadata. Excluding it is deliberate: a summary of documents
    # that mostly fail the filter should not survive it. Documented because
    # it means a filtered RAPTOR search loses its abstraction levels.
    from rag.chunking import make_summary_chunk

    node = make_summary_chunk("raptor:1:0", "raptor:1:0", 0, "s", level=1)
    mask = compile_mask(
        MetadataFilter(topics=("retrieval-models",)), _chunks() + [node], _doc_meta()
    )
    assert not mask[-1]
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_chunking.py tests/test_spans.py tests/test_query_construction.py -q`

Expected: `AttributeError: 'Chunk' object has no attribute 'level'` and `ImportError` for `make_summary_chunk`.

- [ ] **Step 3: Implement in `rag/chunking.py`**

Add the constant near the top:

```python
SYNTHETIC_SPAN = -1
"""Character and token offsets for a node that is not a span of a document.

A RAPTOR cluster summary or a multi-representation document summary is
generated text, not an extract, so it has no position in any source. Using
-1 rather than 0 matters: `chunks_overlapping` tests
`chunk.char_start < span_end and span_start < chunk.char_end`, and a
(-1, -1) node fails the second half for every real span, which is exactly
what stops a summary counting as a correct retrieval for a gold question.
"""
```

Add to the `Chunk` dataclass, after `char_end`:

```python
    level: int = 0
    """0 for a real span of a document, 1 or more for a summary node.

    Defaulted so that an index written before this field existed still
    reconstructs through `Chunk(**record)`.
    """

    @property
    def is_synthetic(self) -> bool:
        return self.level > 0
```

And the constructor helper:

```python
def make_summary_chunk(
    chunk_id: str, doc_id: str, index: int, text: str, level: int
) -> Chunk:
    """A chunk holding generated text rather than an extract.

    `doc_id` is the real document for a multi-representation summary, so it
    inherits that document's metadata filters, and a synthetic id for a
    RAPTOR cluster summary, which belongs to no single document.
    """
    if level < 1:
        raise ValueError(f"a summary chunk needs level >= 1, got {level}")
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        index=index,
        text=text,
        token_start=SYNTHETIC_SPAN,
        token_end=SYNTHETIC_SPAN,
        char_start=SYNTHETIC_SPAN,
        char_end=SYNTHETIC_SPAN,
        level=level,
    )
```

- [ ] **Step 4: Document the filter rule in `rag/query_construction.py`**

No code change is needed — `compile_mask` already excludes a chunk whose `doc_id` is absent from `doc_meta`, which gives exactly the tested behaviour. Add to `compile_mask`'s docstring:

```
    Summary nodes follow from the existing rule rather than a special case. A
    multi-representation summary carries its document's real `doc_id`, so it
    inherits that document's filters. A RAPTOR cluster summary spans several
    documents, has a synthetic `doc_id` absent from `doc_meta`, and is
    therefore excluded by any active filter — which means a filtered search
    against a RAPTOR index loses its abstraction levels and falls back to
    raw chunks. That is the conservative choice: a summary of documents that
    mostly fail the filter should not survive it.
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `python -m pytest -q`
Expected: all pass, including the existing index round-trip tests — `Chunk(**record)` on an index without `level` must still work.

Then confirm against the real index:

```
python -c "
from rag.config import Config
from rag.pipeline import load_index
s = load_index(Config())
print(len(s), 'chunks;', sum(c.is_synthetic for c in s.chunks), 'synthetic')
print('levels:', sorted({c.level for c in s.chunks}))
"
```

Expect 5,116 chunks, 0 synthetic, levels `[0]` — the existing index loads unchanged.

- [ ] **Step 6: Commit**

```bash
git add rag/chunking.py rag/query_construction.py tests/
git commit -m "feat: level and synthetic-span conventions for summary nodes

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: K-means in NumPy

**Files:**
- Create: `rag/clustering.py`
- Test: `tests/test_clustering.py`

**Interfaces:**
- Produces:
  - `kmeans(vectors: np.ndarray, k: int, seed: int = 0, max_iter: int = 50) -> np.ndarray` — integer labels, shape `(n,)`
  - `choose_k(n_items: int, target_size: int) -> int`

Written out rather than imported, because clustering is the mechanic RAPTOR is demonstrating and `sklearn` is on the forbidden list.

- [ ] **Step 1: Write the failing test**

Create `tests/test_clustering.py`:

```python
import numpy as np
import pytest

from rag.clustering import choose_k, kmeans


def _two_clusters(n=30, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[5.0, 5.0], scale=0.3, size=(n, 2))
    b = rng.normal(loc=[-5.0, -5.0], scale=0.3, size=(n, 2))
    return np.vstack([a, b]).astype(np.float32), n


# --- correctness on separable data -------------------------------------------

def test_separates_two_well_separated_clusters():
    vectors, n = _two_clusters()
    labels = kmeans(vectors, k=2, seed=0)
    # Every point in each half shares a label, and the halves differ.
    assert len(set(labels[:n])) == 1
    assert len(set(labels[n:])) == 1
    assert labels[0] != labels[n]


def test_returns_one_label_per_vector():
    vectors, _ = _two_clusters()
    assert kmeans(vectors, k=2).shape == (len(vectors),)


def test_labels_are_in_range():
    vectors, _ = _two_clusters()
    labels = kmeans(vectors, k=3, seed=0)
    assert labels.min() >= 0
    assert labels.max() < 3


# --- determinism --------------------------------------------------------------

def test_the_same_seed_gives_the_same_labels():
    # The benchmark must not drift between runs, so clustering is seeded.
    vectors, _ = _two_clusters()
    assert np.array_equal(kmeans(vectors, k=3, seed=7), kmeans(vectors, k=3, seed=7))


def test_a_different_seed_may_give_different_labels():
    vectors = np.random.default_rng(1).normal(size=(40, 4)).astype(np.float32)
    a = kmeans(vectors, k=4, seed=0)
    b = kmeans(vectors, k=4, seed=1)
    assert a.shape == b.shape  # shape is stable even when assignment is not


# --- edge cases ---------------------------------------------------------------

def test_k_of_one_puts_everything_in_one_cluster():
    vectors, _ = _two_clusters()
    assert set(kmeans(vectors, k=1).tolist()) == {0}


def test_k_equal_to_n_gives_every_point_its_own_cluster():
    vectors = np.eye(5, dtype=np.float32)
    assert len(set(kmeans(vectors, k=5, seed=0).tolist())) == 5


def test_k_larger_than_n_is_clamped():
    vectors = np.eye(3, dtype=np.float32)
    labels = kmeans(vectors, k=10, seed=0)
    assert labels.max() < 3


def test_k_below_one_is_rejected():
    with pytest.raises(ValueError, match="k"):
        kmeans(np.eye(3, dtype=np.float32), k=0)


def test_no_vectors_gives_no_labels():
    assert kmeans(np.zeros((0, 4), dtype=np.float32), k=2).shape == (0,)


def test_identical_vectors_do_not_hang_or_crash():
    # An empty cluster can arise when every point lands on one centroid;
    # the loop must terminate rather than spin re-seeding.
    vectors = np.ones((10, 3), dtype=np.float32)
    labels = kmeans(vectors, k=3, seed=0)
    assert labels.shape == (10,)


def test_converges_before_the_iteration_cap():
    # On separable data the assignment should stop changing quickly; running
    # to the cap would mean the convergence check is broken.
    vectors, _ = _two_clusters()
    assert np.array_equal(
        kmeans(vectors, k=2, seed=0, max_iter=3),
        kmeans(vectors, k=2, seed=0, max_iter=50),
    )


# --- choose_k -----------------------------------------------------------------

def test_choose_k_targets_the_requested_cluster_size():
    assert choose_k(80, target_size=8) == 10


def test_choose_k_is_at_least_one():
    assert choose_k(3, target_size=8) == 1


def test_choose_k_never_exceeds_the_item_count():
    assert choose_k(2, target_size=1) == 2
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_clustering.py -q`
Expected: `ModuleNotFoundError: No module named 'rag.clustering'`.

- [ ] **Step 3: Implement `rag/clustering.py`**

```python
"""K-means, written out.

RAPTOR's whole mechanic is recursive clustering, so importing a clustering
library would hide the thing the phase exists to show. This is Lloyd's
algorithm in about thirty lines of NumPy: assign each point to its nearest
centroid, move each centroid to the mean of its members, repeat until the
assignment stops changing.

Seeded throughout, because the benchmark must not drift between runs.
"""

from __future__ import annotations

import numpy as np


def choose_k(n_items: int, target_size: int) -> int:
    """How many clusters to ask for, aiming at roughly `target_size` each.

    Clamped to at least one and at most `n_items`, so a level with three
    nodes does not ask for eight clusters.
    """
    if n_items <= 0:
        return 0
    return max(1, min(n_items, round(n_items / max(1, target_size))))


def kmeans(
    vectors: np.ndarray, k: int, seed: int = 0, max_iter: int = 50
) -> np.ndarray:
    """Cluster `vectors` into at most `k` groups, returning integer labels.

    Empty clusters are left empty rather than re-seeded. A cluster nobody
    joined simply contributes no summary node, and re-seeding would make the
    labelling unstable under small input changes, because the number of RNG
    draws would then depend on how many clusters happened to empty. (It would
    not break reproducibility — a re-seed from this same seeded generator is
    still bit-identical run to run — so stability, not determinism, is the
    reason.)
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    vectors = np.asarray(vectors, dtype=np.float32)
    n = vectors.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)

    k = min(k, n)
    rng = np.random.default_rng(seed)
    centroids = vectors[rng.choice(n, size=k, replace=False)].copy()

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        # (n, k) squared distances, via the expansion that avoids a big
        # intermediate: |x - c|^2 = |x|^2 - 2 x.c + |c|^2, and |x|^2 is
        # constant per row so it cannot change the argmin.
        distances = (centroids**2).sum(axis=1) - 2.0 * (vectors @ centroids.T)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = vectors[labels == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)
    return labels
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_clustering.py -q`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/clustering.py tests/test_clustering.py
git commit -m "feat: k-means in NumPy for RAPTOR clustering

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Summarisation

**Files:**
- Create: `rag/summarise.py`
- Test: `tests/test_summarise.py`

**Interfaces:**
- Produces:
  - `DOC_SUMMARY_TEMPLATE`, `CLUSTER_SUMMARY_TEMPLATE`
  - `summarise(text: str, llm, template: str, max_chars: int = 12000) -> str`
  - `SummaryError(Exception)`

Both index builders call this. Summaries are cached by the LLM's own prompt cache, so rebuilding an index costs nothing the second time.

- [ ] **Step 1: Write the failing test**

Create `tests/test_summarise.py`:

```python
import pytest

from rag.llm import LLMError
from rag.summarise import (
    CLUSTER_SUMMARY_TEMPLATE,
    DOC_SUMMARY_TEMPLATE,
    SummaryError,
    summarise,
)


class ReplyLLM:
    def __init__(self, reply="a summary"):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.reply


def test_returns_the_models_summary():
    assert summarise("some text", ReplyLLM("the summary"), DOC_SUMMARY_TEMPLATE) == "the summary"


def test_sends_the_text_to_the_model():
    llm = ReplyLLM()
    summarise("distinctive body text", llm, DOC_SUMMARY_TEMPLATE)
    assert "distinctive body text" in llm.prompts[0]


def test_long_text_is_truncated_before_sending():
    # A 200KB paper would blow the context window and cost a fortune; the
    # summary only needs enough to characterise the document.
    llm = ReplyLLM()
    summarise("x" * 50_000, llm, DOC_SUMMARY_TEMPLATE, max_chars=1000)
    assert len(llm.prompts[0]) < 5_000


def test_truncation_keeps_the_beginning():
    # A paper states its contribution in its opening; truncating from the
    # front would throw away the most summarisable part.
    llm = ReplyLLM()
    summarise("HEADLINE" + "x" * 50_000, llm, DOC_SUMMARY_TEMPLATE, max_chars=1000)
    assert "HEADLINE" in llm.prompts[0]


def test_short_text_is_not_truncated():
    llm = ReplyLLM()
    summarise("short", llm, DOC_SUMMARY_TEMPLATE, max_chars=1000)
    assert "short" in llm.prompts[0]


def test_an_empty_summary_is_an_error():
    with pytest.raises(SummaryError, match="empty"):
        summarise("text", ReplyLLM("   "), DOC_SUMMARY_TEMPLATE)


def test_an_llm_failure_becomes_a_summary_error():
    class Failing:
        def generate(self, prompt):
            raise LLMError("rate limited")

    with pytest.raises(SummaryError, match="rate limited"):
        summarise("text", Failing(), DOC_SUMMARY_TEMPLATE)


def test_both_templates_have_a_text_placeholder():
    for template in (DOC_SUMMARY_TEMPLATE, CLUSTER_SUMMARY_TEMPLATE):
        assert "{text}" in template


def test_the_cluster_template_asks_for_shared_themes():
    # A cluster summary that just concatenates its members is useless as an
    # abstraction layer; the prompt has to ask for what they have in common.
    lowered = CLUSTER_SUMMARY_TEMPLATE.lower()
    assert "common" in lowered or "shared" in lowered or "theme" in lowered
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_summarise.py -q`
Expected: `ModuleNotFoundError: No module named 'rag.summarise'`.

- [ ] **Step 3: Implement `rag/summarise.py`**

```python
"""Summaries for the two indexing techniques.

Both go through the LLM's on-disk cache, so rebuilding an index after the
first run costs nothing — which matters when a full RAPTOR tree over 5,116
chunks is several hundred calls.
"""

from __future__ import annotations

from rag.llm import LLMError


class SummaryError(Exception):
    """Raised when a summary could not be produced."""


DOC_SUMMARY_TEMPLATE = """Summarise the document below in three or four sentences.

Say what it proposes or reports, the problem it addresses, and the terms a
reader would search for to find it. Write plainly, as documentation rather
than as a review. Do not begin with "This document" or "This paper".

Document:
{text}

Summary:"""


CLUSTER_SUMMARY_TEMPLATE = """Summarise what the passages below have in common.

They were grouped because their embeddings are close, so they share a subject
even though they come from different places. Describe the shared theme and the
specific ideas that recur, in three or four sentences. Do not list the
passages separately or number them — a summary that just concatenates them is
useless as an abstraction.

Passages:
{text}

Shared summary:"""


def summarise(text: str, llm, template: str, max_chars: int = 12000) -> str:
    """Summarise `text` with `llm`, truncating over-long input.

    Truncation keeps the beginning: a paper states its contribution in its
    opening, so the front is the most summarisable part, and a 200,000
    character document would otherwise exceed the context window.
    """
    excerpt = text[:max_chars]
    try:
        summary = llm.generate(template.format(text=excerpt)).strip()
    except LLMError as exc:
        raise SummaryError(f"summarisation failed: {exc}") from exc
    if not summary:
        raise SummaryError("model returned an empty summary")
    return summary
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_summarise.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/summarise.py tests/test_summarise.py
git commit -m "feat: document and cluster summarisation

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Multi-representation indexing

**Files:**
- Create: `rag/indexing/__init__.py`, `rag/indexing/multi_representation.py`
- Modify: `rag/store.py`
- Test: `tests/test_multi_representation.py`, `tests/test_store.py`

**Interfaces:**
- Produces:
  - `VectorStore.docstore: dict[str, str]` — `doc_id` to full document text, persisted
  - `build_multi_representation(documents, embedder, llm, trace=None) -> VectorStore`
  - `expand_to_documents(retrieved: list[RetrievedChunk], docstore: dict) -> list[RetrievedChunk]`

`RAG.pdf`: embed a summary, retrieve on the summary, then hand the **whole document** to the LLM. The summary is a search key; the document is the context.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_store.py`:

```python
def test_docstore_round_trips(tmp_path):
    path = tmp_path / "index.npz"
    store = _store(2)
    store.docstore = {"d": "the full document text"}
    store.save(path)
    assert VectorStore.load(path).docstore == {"d": "the full document text"}


def test_an_index_written_without_a_docstore_still_loads(tmp_path):
    # Note: going through save() would NOT test this -- save always writes a
    # docstore key, so load never reaches its `else {}` branch. The legacy
    # path is only exercised by writing an npz that genuinely lacks the key,
    # which is what every index built before this field looks like, including
    # the real 5,116-chunk one. Model this on the existing
    # test_an_index_written_without_doc_meta_still_loads.
    import json
    from dataclasses import asdict

    path = tmp_path / "index.npz"
    store = _store(2)
    np.savez_compressed(
        path,
        vectors=store.vectors,
        chunks=np.array(json.dumps([asdict(c) for c in store.chunks])),
    )
    loaded = VectorStore.load(path)
    assert len(loaded) == 2
    assert loaded.docstore == {}
```

Create `tests/test_multi_representation.py`:

```python
import pytest

from rag.chunking import RetrievedChunk
from rag.indexing.multi_representation import (
    build_multi_representation,
    expand_to_documents,
)
from rag.loader import Document
from rag.summarise import SummaryError
from rag.trace import Trace
from tests.conftest import FakeEmbedder


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "fallback summary"


def _docs():
    return [
        Document("alpha", "Alpha body text about cosine similarity.", "Alpha", "arxiv", topic="t"),
        Document("beta", "Beta body text about rank fusion.", "Beta", "arxiv", topic="t"),
    ]


def test_one_node_per_document():
    store = build_multi_representation(_docs(), FakeEmbedder(), ScriptedLLM(["s1", "s2"]))
    assert len(store) == 2
    assert {c.doc_id for c in store.chunks} == {"alpha", "beta"}


def test_nodes_hold_the_summary_not_the_document():
    store = build_multi_representation(_docs(), FakeEmbedder(), ScriptedLLM(["SUMMARY A", "SUMMARY B"]))
    texts = {c.text for c in store.chunks}
    assert texts == {"SUMMARY A", "SUMMARY B"}


def test_nodes_are_synthetic_at_level_one():
    store = build_multi_representation(_docs(), FakeEmbedder(), ScriptedLLM(["s1", "s2"]))
    assert all(c.is_synthetic and c.level == 1 for c in store.chunks)


def test_nodes_keep_their_documents_id_so_filters_still_work():
    store = build_multi_representation(_docs(), FakeEmbedder(), ScriptedLLM(["s1", "s2"]))
    assert sorted(c.doc_id for c in store.chunks) == ["alpha", "beta"]


def test_the_docstore_holds_the_full_documents():
    store = build_multi_representation(_docs(), FakeEmbedder(), ScriptedLLM(["s1", "s2"]))
    assert store.docstore["alpha"] == "Alpha body text about cosine similarity."


def test_document_metadata_is_carried_over():
    store = build_multi_representation(_docs(), FakeEmbedder(), ScriptedLLM(["s1", "s2"]))
    assert store.doc_meta["alpha"]["title"] == "Alpha"


def test_a_document_whose_summary_fails_is_skipped_and_noted():
    class OneFails:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            if self.calls == 1:
                from rag.llm import LLMError

                raise LLMError("rate limited")
            return "s2"

    trace = Trace(question="build")
    store = build_multi_representation(_docs(), FakeEmbedder(), OneFails(), trace)
    assert len(store) == 1
    assert any("degraded" in n for n in trace.notes)


def test_every_summary_failing_raises_rather_than_returning_an_empty_index():
    class AllFail:
        def generate(self, prompt):
            from rag.llm import LLMError

            raise LLMError("down")

    with pytest.raises(SummaryError, match="no documents"):
        build_multi_representation(_docs(), FakeEmbedder(), AllFail(), Trace(question="b"))


# --- expansion ----------------------------------------------------------------

def _hit(doc_id, text):
    from rag.chunking import make_summary_chunk

    return RetrievedChunk(
        chunk=make_summary_chunk(f"{doc_id}:summary", doc_id, 0, text, level=1),
        score=0.5,
        rank=1,
    )


def test_expansion_replaces_the_summary_with_the_document():
    docstore = {"alpha": "the whole document"}
    out = expand_to_documents([_hit("alpha", "the summary")], docstore)
    assert out[0].chunk.text == "the whole document"


def test_expansion_preserves_score_and_rank():
    out = expand_to_documents([_hit("alpha", "s")], {"alpha": "full"})
    assert out[0].score == 0.5
    assert out[0].rank == 1


def test_expansion_leaves_a_hit_alone_when_the_document_is_missing():
    out = expand_to_documents([_hit("ghost", "s")], {})
    assert out[0].chunk.text == "s"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_multi_representation.py tests/test_store.py -q`
Expected: `ModuleNotFoundError: No module named 'rag.indexing'` and `AttributeError` for `docstore`.

- [ ] **Step 3: Add `docstore` to `VectorStore`**

In `rag/store.py`, add the field after `doc_meta`:

```python
    docstore: dict = field(default_factory=dict)
    """Full document text by doc_id, for multi-representation indexing.

    That technique embeds a summary and retrieves on it, then hands the whole
    document to the generator — so the document text has to travel with the
    index, not be re-read from the corpus at query time.
    """
```

Persist it in `save` alongside `doc_meta`, and read it back in `load` with the same `if "docstore" in data.files else {}` pattern.

- [ ] **Step 4: Implement `rag/indexing/multi_representation.py`**

Create an empty `rag/indexing/__init__.py`, then:

```python
"""Multi-representation indexing.

RAG.pdf's argument: a chunk is a poor search key for a document, because it
describes only its own paragraph. Summarising the whole document gives a key
that describes the whole document — and once a summary matches, the thing
worth handing to the generator is the document, not the summary.

So the index holds one node per document containing its summary, and a
separate docstore holds the full text. Retrieval matches summaries;
`expand_to_documents` swaps in the documents afterwards.
"""

from __future__ import annotations

from rag.chunking import RetrievedChunk, make_summary_chunk
from rag.store import VectorStore
from rag.summarise import DOC_SUMMARY_TEMPLATE, SummaryError, summarise


def build_multi_representation(documents, embedder, llm, trace=None) -> VectorStore:
    """One summary node per document, with the full text in the docstore.

    A document whose summary fails is skipped and recorded rather than
    aborting the build: losing one document from the index is much better
    than losing the index. If every document fails, that is not a degraded
    index but no index, and it raises.
    """
    nodes = []
    summaries = []
    docstore = {}
    doc_meta = {}

    for doc in documents:
        try:
            summary = summarise(doc.text, llm, DOC_SUMMARY_TEMPLATE)
        except SummaryError as exc:
            if trace is not None:
                trace.degraded(
                    f"summary failed for {doc.doc_id}: {exc}",
                    "skipping that document",
                )
            continue
        nodes.append(
            make_summary_chunk(
                chunk_id=f"{doc.doc_id}:summary",
                doc_id=doc.doc_id,
                index=0,
                text=summary,
                level=1,
            )
        )
        summaries.append(summary)
        docstore[doc.doc_id] = doc.text
        doc_meta[doc.doc_id] = {
            "title": doc.title,
            "source": doc.source,
            "topic": doc.topic,
            "publish_date": doc.publish_date,
            "author": doc.author,
            "url": doc.url,
        }

    if not nodes:
        raise SummaryError(
            "no documents could be summarised; refusing to build an empty index"
        )

    store = VectorStore(vectors=embedder.encode(summaries), chunks=nodes)
    store.docstore = docstore
    store.doc_meta = doc_meta
    return store


def expand_to_documents(
    retrieved: list[RetrievedChunk], docstore: dict
) -> list[RetrievedChunk]:
    """Swap each retrieved summary for the document it summarises.

    A hit whose document is missing from the docstore is left as the summary
    rather than dropped — a slightly worse context beats a hole in the
    results.
    """
    from dataclasses import replace

    expanded = []
    for item in retrieved:
        text = docstore.get(item.chunk.doc_id)
        if text is None:
            expanded.append(item)
            continue
        expanded.append(replace(item, chunk=replace(item.chunk, text=text)))
    return expanded
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `python -m pytest -q`

- [ ] **Step 6: Commit**

```bash
git add rag/indexing rag/store.py tests/
git commit -m "feat: multi-representation indexing with a docstore

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: RAPTOR tree building

**Files:**
- Create: `rag/indexing/raptor.py`
- Modify: `rag/config.py`
- Test: `tests/test_raptor.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `Config.raptor_max_depth: int = 3`, `Config.raptor_cluster_size: int = 8`
  - `build_raptor(chunks, vectors, embedder, llm, config, trace=None) -> tuple[list[Chunk], np.ndarray]`

`RAG.pdf`: a high-level question needs more context than top-k raw chunks can supply. Cluster the chunks, summarise each cluster, embed those summaries, cluster *those*, and keep going. Every level goes in one store, so a single search can return a raw chunk or an abstraction over hundreds of them.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_raptor_defaults():
    cfg = Config()
    assert cfg.raptor_max_depth == 3
    assert cfg.raptor_cluster_size == 8


def test_raptor_max_depth_must_be_positive():
    import pytest
    with pytest.raises(ValueError, match="raptor_max_depth"):
        Config(raptor_max_depth=0)
```

Create `tests/test_raptor.py`:

```python
import numpy as np
import pytest

from rag.chunking import Chunk
from rag.config import Config
from rag.indexing.raptor import build_raptor
from rag.trace import Trace
from tests.conftest import FakeEmbedder


class ScriptedLLM:
    def __init__(self, reply="a cluster summary"):
        self.reply = reply
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return f"{self.reply} {self.calls}"


def _chunks(n=40):
    return [Chunk(f"d:{i}", f"d{i % 4}", i, f"chunk text {i}", 0, 10, i * 50, i * 50 + 40)
            for i in range(n)]


def _vectors(n=40, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 8)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _config(**kw):
    return Config(raptor_max_depth=3, raptor_cluster_size=8, **kw)


def test_returns_the_original_chunks_plus_summaries():
    chunks, vectors = _chunks(), _vectors()
    out_chunks, out_vectors = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    assert len(out_chunks) > len(chunks)
    assert out_chunks[: len(chunks)] == chunks


def test_original_chunks_come_first_and_keep_their_order():
    # evaluation/spans.py resolves gold spans positionally against
    # store.chunks; appending summaries rather than interleaving them keeps
    # Phase 3's numbers comparable.
    chunks, vectors = _chunks(), _vectors()
    out_chunks, _ = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    assert [c.chunk_id for c in out_chunks[: len(chunks)]] == [c.chunk_id for c in chunks]


def test_vectors_and_chunks_stay_aligned():
    chunks, vectors = _chunks(), _vectors()
    out_chunks, out_vectors = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    assert out_vectors.shape[0] == len(out_chunks)


def test_summaries_are_synthetic_and_levelled():
    chunks, vectors = _chunks(), _vectors()
    out_chunks, _ = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    synthetic = [c for c in out_chunks if c.is_synthetic]
    assert synthetic
    assert all(c.level >= 1 for c in synthetic)


def test_levels_increase_as_the_tree_is_built():
    chunks, vectors = _chunks(80), _vectors(80)
    out_chunks, _ = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    levels = sorted({c.level for c in out_chunks})
    assert levels[0] == 0
    assert len(levels) >= 2


def test_recursion_stops_at_the_depth_cap():
    chunks, vectors = _chunks(200), _vectors(200)
    out_chunks, _ = build_raptor(
        chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config(raptor_max_depth=2)
    )
    assert max(c.level for c in out_chunks) <= 2


def test_recursion_stops_when_a_level_collapses_to_one_node():
    # Without this, a level of one would cluster into one cluster forever.
    chunks, vectors = _chunks(10), _vectors(10)
    out_chunks, _ = build_raptor(
        chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config(raptor_max_depth=9, raptor_cluster_size=8)
    )
    assert max(c.level for c in out_chunks) < 9


def test_summary_ids_are_unique():
    chunks, vectors = _chunks(80), _vectors(80)
    out_chunks, _ = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    ids = [c.chunk_id for c in out_chunks]
    assert len(ids) == len(set(ids))


def test_a_cluster_summary_has_a_synthetic_doc_id():
    # It spans several documents, so it belongs to none of them. That is what
    # makes compile_mask exclude it from filtered search.
    chunks, vectors = _chunks(), _vectors()
    out_chunks, _ = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    summary = next(c for c in out_chunks if c.is_synthetic)
    assert summary.doc_id not in {c.doc_id for c in chunks}


def test_building_is_deterministic():
    chunks, vectors = _chunks(), _vectors()
    a, _ = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    b, _ = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_a_failed_cluster_summary_is_skipped_and_noted():
    class OneFails:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            if self.calls == 1:
                from rag.llm import LLMError

                raise LLMError("rate limited")
            return f"summary {self.calls}"

    trace = Trace(question="build")
    chunks, vectors = _chunks(), _vectors()
    out_chunks, _ = build_raptor(chunks, vectors, FakeEmbedder(), OneFails(), _config(), trace)
    assert any("degraded" in n for n in trace.notes)
    assert len(out_chunks) > len(chunks)


def test_too_few_chunks_to_cluster_returns_them_unchanged():
    chunks, vectors = _chunks(1), _vectors(1)
    out_chunks, out_vectors = build_raptor(chunks, vectors, FakeEmbedder(), ScriptedLLM(), _config())
    assert out_chunks == chunks
    assert out_vectors.shape[0] == 1
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_raptor.py tests/test_config.py -q`
Expected: `ModuleNotFoundError: No module named 'rag.indexing.raptor'` and `TypeError` for the unexpected `raptor_max_depth`.

- [ ] **Step 3: Add the config fields**

In `rag/config.py`, after `retrieval_depth`:

```python
    raptor_max_depth: int = 3
    """How many summary levels RAPTOR builds above the raw chunks.

    Each level costs one LLM call per cluster, so depth is the main lever on
    build cost. Three levels over 5,116 chunks is roughly 640 + 80 + 10 calls.
    """

    raptor_cluster_size: int = 8
    """Target chunks per cluster, which sets how fast the tree narrows."""
```

and in `__post_init__`:

```python
        if self.raptor_max_depth < 1:
            raise ValueError(
                f"raptor_max_depth must be at least 1, got {self.raptor_max_depth}"
            )
        if self.raptor_cluster_size < 2:
            raise ValueError(
                f"raptor_cluster_size must be at least 2, got "
                f"{self.raptor_cluster_size}"
            )
```

- [ ] **Step 4: Implement `rag/indexing/raptor.py`**

```python
"""RAPTOR: recursive clustering and summarisation.

RAG.pdf's argument: top-k retrieval over raw chunks answers a specific
question well and a broad one badly, because the answer to "what do these
papers have in common" is not in any one chunk. RAPTOR builds abstractions —
cluster the chunks, summarise each cluster, embed the summaries, cluster
those — so a single search can return either a raw passage or a summary
standing for hundreds of them.

Two structural choices worth knowing:

- **Original chunks come first in the returned list, in their original
  order.** `evaluation/spans.py` resolves gold spans positionally against
  `store.chunks`, so appending rather than interleaving keeps Phase 3's
  numbers comparable.
- **A cluster summary gets a synthetic `doc_id`.** It spans several
  documents and belongs to none, which is what makes `compile_mask` exclude
  it from a filtered search.
"""

from __future__ import annotations

import numpy as np

from rag.chunking import Chunk, make_summary_chunk
from rag.clustering import choose_k, kmeans
from rag.config import Config
from rag.summarise import CLUSTER_SUMMARY_TEMPLATE, SummaryError, summarise

JOIN = "\n\n---\n\n"


def build_raptor(
    chunks: list[Chunk],
    vectors: np.ndarray,
    embedder,
    llm,
    config: Config,
    trace=None,
) -> tuple[list[Chunk], np.ndarray]:
    """Build the summary tree, returning every node and its vectors.

    Recursion stops at `config.raptor_max_depth`, or as soon as a level
    produces one node or fewer — without that second condition a level of one
    would cluster into one cluster forever.
    """
    all_chunks = list(chunks)
    all_vectors = [np.asarray(vectors, dtype=np.float32)]

    current_chunks = list(chunks)
    current_vectors = np.asarray(vectors, dtype=np.float32)

    for level in range(1, config.raptor_max_depth + 1):
        if len(current_chunks) <= 1:
            break

        k = choose_k(len(current_chunks), config.raptor_cluster_size)
        if k >= len(current_chunks):
            break
        labels = kmeans(current_vectors, k=k, seed=level)

        level_chunks: list[Chunk] = []
        level_texts: list[str] = []
        for cluster in range(k):
            members = [c for c, lab in zip(current_chunks, labels) if lab == cluster]
            if not members:
                continue
            joined = JOIN.join(m.text for m in members)
            try:
                summary = summarise(joined, llm, CLUSTER_SUMMARY_TEMPLATE)
            except SummaryError as exc:
                if trace is not None:
                    trace.degraded(
                        f"cluster summary failed at level {level}, cluster "
                        f"{cluster}: {exc}",
                        "skipping that cluster",
                    )
                continue
            level_chunks.append(
                make_summary_chunk(
                    chunk_id=f"raptor:{level}:{cluster}",
                    doc_id=f"raptor:{level}:{cluster}",
                    index=cluster,
                    text=summary,
                    level=level,
                )
            )
            level_texts.append(summary)

        if not level_chunks:
            break

        level_vectors = embedder.encode(level_texts)
        all_chunks.extend(level_chunks)
        all_vectors.append(level_vectors)

        current_chunks = level_chunks
        current_vectors = level_vectors

    return all_chunks, np.vstack(all_vectors)
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `python -m pytest -q`

- [ ] **Step 6: Commit**

```bash
git add rag/indexing/raptor.py rag/config.py tests/
git commit -m "feat: RAPTOR recursive clustering and summarisation

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Index modes

**Files:**
- Modify: `rag/pipeline.py`, `rag/__main__.py`
- Test: `tests/test_pipeline.py`, `tests/test_cli.py`

**Interfaces:**
- Produces:
  - `INDEX_MODES = ("flat", "multirep", "raptor")`
  - `index_path_for(config: Config, mode: str) -> Path`
  - `build_index(config, embedder, llm=None, mode="flat") -> VectorStore`
  - `load_index(config, mode="flat") -> VectorStore`
  - `_index_meta(config, dim, mode="flat")` records `index_mode`, and for raptor `raptor_max_depth` and `raptor_cluster_size`
  - `python -m rag index --index-mode {flat,multirep,raptor}`

Provenance matters here: without `index_mode` in the meta, an index built flat loads clean against a config expecting RAPTOR, and the only symptom is worse answers.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
def test_index_path_differs_per_mode(tiny_corpus: Config):
    from rag.pipeline import index_path_for

    flat = index_path_for(tiny_corpus, "flat")
    raptor = index_path_for(tiny_corpus, "raptor")
    assert flat != raptor
    assert flat == tiny_corpus.index_path
    assert "raptor" in raptor.name


def test_an_unknown_mode_is_rejected(tiny_corpus: Config):
    from rag.pipeline import index_path_for

    with pytest.raises(ValueError, match="nope"):
        index_path_for(tiny_corpus, "nope")


def test_index_meta_records_the_mode(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder(), mode="flat")
    assert store.meta["index_mode"] == "flat"


def test_loading_a_flat_index_as_raptor_is_refused(tiny_corpus: Config):
    from rag.pipeline import index_path_for

    build_index(tiny_corpus, FakeEmbedder(), mode="flat")
    # Point the raptor path at the flat file to simulate a stale build.
    index_path_for(tiny_corpus, "raptor").write_bytes(
        index_path_for(tiny_corpus, "flat").read_bytes()
    )
    with pytest.raises(ValueError, match="index_mode"):
        load_index(tiny_corpus, mode="raptor")


def test_a_legacy_index_without_a_mode_still_loads_as_flat(tiny_corpus: Config):
    # VectorStore.load tolerates a key that is absent from the stored meta
    # ("if key in meta and meta[key] != value"), which is what lets an index
    # written before index_mode existed still load. Do not tighten that: the
    # non-flat modes write their own files, so a file at the flat path with
    # no index_mode can only be a legacy flat index, and refusing it would
    # force a needless rebuild.
    import json

    import numpy as np

    from rag.pipeline import index_path_for

    build_index(tiny_corpus, FakeEmbedder(), mode="flat")
    path = index_path_for(tiny_corpus, "flat")
    with np.load(path, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}
    meta = json.loads(str(payload["meta"]))
    del meta["index_mode"]
    payload["meta"] = np.array(json.dumps(meta, sort_keys=True))
    np.savez_compressed(path, **payload)

    store = load_index(tiny_corpus, mode="flat")
    assert len(store) > 0


def test_building_multirep_needs_an_llm(tiny_corpus: Config):
    with pytest.raises(ValueError, match="llm"):
        build_index(tiny_corpus, FakeEmbedder(), llm=None, mode="multirep")


def test_building_raptor_needs_an_llm(tiny_corpus: Config):
    with pytest.raises(ValueError, match="llm"):
        build_index(tiny_corpus, FakeEmbedder(), llm=None, mode="raptor")


def test_multirep_index_has_one_node_per_document(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder(), llm=FakeLLM("a summary"), mode="multirep")
    assert len(store) == 2
    assert store.docstore


def test_raptor_index_is_larger_than_flat(tiny_corpus: Config):
    flat = build_index(tiny_corpus, FakeEmbedder(), mode="flat")
    raptor = build_index(
        tiny_corpus, FakeEmbedder(), llm=FakeLLM("a summary"), mode="raptor"
    )
    assert len(raptor) >= len(flat)
```

Append to `tests/test_cli.py`:

```python
def test_index_mode_flag_is_accepted(tiny_corpus, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    assert main(["index", "--index-mode", "flat"], **_factories()) == 0
    assert "chunks" in capsys.readouterr().out


def test_an_unknown_index_mode_is_rejected(tiny_corpus, monkeypatch):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    with pytest.raises(SystemExit):
        main(["index", "--index-mode", "nope"], **_factories())
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_pipeline.py tests/test_cli.py -q`
Expected: `ImportError` for `index_path_for`, `TypeError` for the unexpected `mode`.

- [ ] **Step 3: Implement the mode plumbing in `rag/pipeline.py`**

```python
INDEX_MODES = ("flat", "multirep", "raptor")


def index_path_for(config: Config, mode: str) -> Path:
    """Where the index for this mode lives.

    Each mode gets its own file so the flat index — which every benchmark
    number in the README rests on — is never overwritten by a RAPTOR build.
    """
    if mode not in INDEX_MODES:
        raise ValueError(
            f"unknown index mode: {mode} (choose one of {', '.join(INDEX_MODES)})"
        )
    if mode == "flat":
        return config.index_path
    return config.index_path.with_name(
        f"{config.index_path.stem}-{mode}{config.index_path.suffix}"
    )
```

Extend `_index_meta`:

```python
def _index_meta(config: Config, dim: int | None, mode: str = "flat") -> dict:
    meta = {
        "embedding_model": config.embedding_model,
        "chunk_tokens": config.chunk_tokens,
        "chunk_overlap": config.chunk_overlap,
        "dim": dim,
        "index_mode": mode,
    }
    if mode == "raptor":
        meta["raptor_max_depth"] = config.raptor_max_depth
        meta["raptor_cluster_size"] = config.raptor_cluster_size
    return meta
```

Rewrite `build_index` to dispatch, keeping the flat path byte-for-byte what it was:

```python
def build_index(config: Config, embedder, llm=None, mode: str = "flat") -> VectorStore:
    """Build and persist the index for `mode`.

    `multirep` and `raptor` both summarise with the LLM, so they need one;
    asking for them without it is a configuration error, not something to
    degrade around — an index silently built without its summaries would be
    a flat index wearing the wrong name.
    """
    path = index_path_for(config, mode)
    documents = load_documents(config.corpus_dir, config.metadata_path)
    if mode != "flat" and llm is None:
        raise ValueError(f"index mode {mode} needs an llm to summarise with")

    if mode == "multirep":
        store = build_multi_representation(documents, embedder, llm)
    else:
        chunks = chunk_documents(
            documents, embedder.tokenizer, config.chunk_tokens, config.chunk_overlap
        )
        vectors = embedder.encode([chunk.text for chunk in chunks])
        if mode == "raptor":
            chunks, vectors = build_raptor(chunks, vectors, embedder, llm, config)
        store = VectorStore(vectors=vectors, chunks=chunks)
        store.doc_meta = {
            doc.doc_id: {
                "title": doc.title,
                "source": doc.source,
                "topic": doc.topic,
                "publish_date": doc.publish_date,
                "author": doc.author,
                "url": doc.url,
            }
            for doc in documents
        }

    store.meta = _index_meta(config, store.dim, mode)
    store.save(path, meta=store.meta)
    return store


def load_index(config: Config, mode: str = "flat") -> VectorStore:
    expected = _index_meta(config, dim=None, mode=mode)
    expected.pop("dim")
    return VectorStore.load(index_path_for(config, mode), expect_meta=expected)
```

Add the imports for `build_multi_representation` and `build_raptor`.

- [ ] **Step 4: Add the CLI flag**

On the `index` subparser in `rag/__main__.py`:

```python
    index_parser.add_argument(
        "--index-mode", choices=list(INDEX_MODES), default="flat",
        help="flat chunks, one summary per document, or a RAPTOR tree",
    )
```

In `_run`'s index branch, build the LLM when the mode needs one and pass `mode=args.index_mode` through, reporting the mode and the node count in the printed summary.

- [ ] **Step 5: Run the tests and verify they pass**

Run: `python -m pytest -q`

- [ ] **Step 6: Commit**

```bash
git add rag/pipeline.py rag/__main__.py tests/
git commit -m "feat: index modes with per-mode files and provenance

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Retrieval against the new indexes

**Files:**
- Modify: `rag/pipeline.py`, `rag/__main__.py`
- Test: `tests/test_pipeline.py`, `tests/test_cli.py`

**Interfaces:**
- Produces: `ask(..., docstore: dict | None = None)` expanding summary hits to documents; `python -m rag ask --index {flat,multirep,raptor}`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
def test_a_multirep_hit_is_expanded_to_the_document(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder(), llm=FakeLLM("a summary"), mode="multirep")
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, generate=False)
    texts = {r.chunk.text for r in trace.retrieved}
    assert "a summary" not in texts
    assert any("Cosine similarity" in t or "Reciprocal rank" in t for t in texts)


def test_expansion_does_not_happen_without_a_docstore(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder(), mode="flat")
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, generate=False)
    assert trace.retrieved


def test_the_trace_records_which_index_was_used(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder(), llm=FakeLLM("s"), mode="multirep")
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, generate=False)
    assert any("multirep" in n for n in trace.notes)
```

Append to `tests/test_cli.py`:

```python
def test_ask_accepts_an_index_flag(tiny_corpus, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())
    assert main(["ask", "q", "--index", "flat", "--no-llm"], **_factories()) == 0


def test_asking_against_a_missing_index_reports_which_one(tiny_corpus, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    assert main(["ask", "q", "--index", "raptor", "--no-llm"], **_factories()) == 1
    err = capsys.readouterr().err
    assert "raptor" in err
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_pipeline.py tests/test_cli.py -q`

- [ ] **Step 3: Expand summary hits in `ask`**

`VectorStore` already carries its own `docstore`, so `ask` can decide without a new parameter. After the strategy returns and before the trace is populated:

```python
    if store.docstore:
        result = replace(
            result, retrieved=expand_to_documents(result.retrieved, store.docstore)
        )
        trace.note(
            f"index mode multirep: {len(result.retrieved)} summary hits expanded "
            "to full documents"
        )
```

Add `from rag.indexing.multi_representation import expand_to_documents`.

For a RAPTOR index, note the mode too so the trace says which index answered:

```python
    elif any(c.is_synthetic for c in store.chunks):
        trace.note("index mode raptor: search spans raw chunks and summaries")
```

- [ ] **Step 4: Add the CLI flag**

`--index {flat,multirep,raptor}` on the `ask` subparser, defaulting to `flat`, passed to `load_index(config, mode=args.index)`. The existing `FileNotFoundError` handler already names the path, which contains the mode.

- [ ] **Step 5: Run the tests and verify they pass**

Run: `python -m pytest -q`

- [ ] **Step 6: Commit**

```bash
git add rag/pipeline.py rag/__main__.py tests/
git commit -m "feat: retrieve against multirep and raptor indexes

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Build the real indexes, measure both, and write it up

The task that decides whether either technique is worth anything.

**Files:**
- Modify: `evaluation/benchmark.py`, `README.md`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Produces: `python -m evaluation.benchmark --index {flat,multirep,raptor}`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_benchmark.py`:

```python
def test_benchmark_accepts_an_index_flag():
    import argparse

    from evaluation.benchmark import build_parser

    args = build_parser().parse_args(["--index", "raptor"])
    assert args.index == "raptor"


def test_the_index_flag_defaults_to_flat():
    from evaluation.benchmark import build_parser

    assert build_parser().parse_args([]).index == "flat"


def test_an_unknown_index_is_rejected():
    from evaluation.benchmark import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--index", "nope"])
```

If `benchmark.py` builds its parser inline in `main`, extract it to `build_parser()` so it is testable without running a sweep.

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_benchmark.py -q`

- [ ] **Step 3: Add the flag**

`--index` with the same choices, passed to `load_index(config, mode=args.index)`. Print the mode above the table so a pasted result says which index produced it.

- [ ] **Step 4: Build the real indexes**

Both make many LLM calls on a cold cache — multi-representation is 38, RAPTOR roughly 640 + 80 + 10 at the default depth and cluster size. The cache makes re-runs free. Expect this to take a while.

```
python -m rag index --index-mode multirep
python -m rag index --index-mode raptor
```

Report the node counts. For RAPTOR, report the level histogram:

```
python -c "
import collections
from rag.config import Config
from rag.pipeline import load_index
s = load_index(Config(), mode='raptor')
print('nodes:', len(s))
print('levels:', dict(sorted(collections.Counter(c.level for c in s.chunks).items())))
"
```

Expect level 0 at 5,116 and each level above it roughly an eighth of the one below.

- [ ] **Step 5: Run all three benchmarks**

```
python -m evaluation.benchmark --index flat
python -m evaluation.benchmark --index multirep
python -m evaluation.benchmark --index raptor
```

**Two things to watch, and report honestly either way.**

First, **multi-representation will probably score badly on this gold set, and that is a measurement artifact, not only a result.** The gold set marks chunks that overlap an answering span. A multi-representation index contains no such chunks — it contains 38 summaries and returns whole documents. `relevant_chunk_ids` can therefore never match, and Recall@20 may be exactly 0. Check whether that is what happens. If it is, say so plainly: the metric cannot see this technique, `DocPrec@5` is the only column that means anything for it, and reporting a 0.000 as though the technique failed would be the dishonest reading.

Second, **RAPTOR's summary nodes can occupy top-k slots without ever being gold-relevant**, since synthetic nodes never satisfy a span. That is a real cost of the technique under this metric and should be reported as such — but note how many of the retrieved items were synthetic, so a reader can tell the difference between "RAPTOR retrieved worse chunks" and "RAPTOR spent its slots on summaries the metric cannot credit".

- [ ] **Step 6: Run the spec's demo**

The spec asks for a high-level question that flat top-k answers poorly and RAPTOR answers well. Try several and report what actually happens — do not pick only the flattering one:

```
python -m rag ask "What problem do most of these retrieval techniques share?" --index flat --trace
python -m rag ask "What problem do most of these retrieval techniques share?" --index raptor --trace
python -m rag ask "What are the main themes across these papers?" --index raptor --trace
```

For the RAPTOR runs, say whether any retrieved item was a summary node and at which level. If RAPTOR does not visibly beat flat here, report that — the spec asserts the motivation, and this phase is where it gets tested rather than repeated.

- [ ] **Step 7: Write the README section**

Cover both techniques, the per-mode index files and why they are separate, the node counts, all three benchmark tables, and the demo with its honest outcome. State the multi-representation metric mismatch explicitly rather than letting a 0.000 stand unexplained. Keep the existing caveats: 10 questions, retrieval only, incomplete chunk-level gold set, single-author bias. Check Phase 5 in the roadmap.

- [ ] **Step 8: Commit**

```bash
git add evaluation/benchmark.py README.md tests/
git commit -m "feat: benchmark the indexing techniques and report the results

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 5 Definition of Done

- [ ] `python -m pytest` passes with no network access; `-m slow` passes
- [ ] K-means is written in NumPy; `sklearn` appears nowhere
- [ ] `Chunk.level` defaults to 0, so pre-existing indexes still load
- [ ] A synthetic node can never satisfy a gold span
- [ ] The flat index and its six benchmark rows are unchanged
- [ ] `data/index-multirep.npz` and `data/index-raptor.npz` build, and loading one as the wrong mode is refused
- [ ] RAPTOR's levels are present, the recursion terminates, and node ids are unique
- [ ] All three benchmarks run and the README reports them, including the multi-representation metric mismatch
- [ ] The spec's high-level demo is run and its real outcome reported
- [ ] Working tree clean

## What Phase 6 needs from this

- `Chunk.level` and the synthetic-span convention are what let ColBERT's reranker tell a summary from a passage.
- Per-mode index files mean Phase 6 can rerank any of the three without rebuilding.
- `VectorStore.docstore` is the pattern Phase 6's dashboard will use to show a retrieved document rather than only its chunk.
