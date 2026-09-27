# Phase 3: Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the six retrieval strategies actually help, with Recall@k, MRR and nDCG over a span-tagged gold set, reported as a markdown table.

**Architecture:** A gold set of 10 hand-written questions, each tagged with a `doc_id` and a verbatim `quote` from the document that answers it. At load time the quote is resolved to a character span, and at scoring time the span is resolved to whichever chunks overlap it. Metrics are computed in plain NumPy over chunk ids. A benchmark runner sweeps every strategy over every question and emits a table.

**Tech Stack:** Python 3.14, NumPy, pytest. No new dependencies.

## Global Constraints

- **No RAG framework.** `langchain`, `llama_index`, `sentence_transformers`, `sklearn`, `bs4`, `requests`, `dotenv` are installed and MUST NOT be imported. `tests/test_no_frameworks.py` enforces this and scans `pyproject.toml`. Add `evaluation` to its `SOURCE_DIRS`.
- **Allowed third-party imports:** `numpy`, `torch`, `transformers`, `google.genai`, `pytest`.
- **Determinism.** LLM temperature 0, stable sorts, seeded anything random. A benchmark that drifts between runs measures nothing.
- **Test-driven.** Write the test, run it, watch it fail for the expected reason, then implement.
- **Every unit test runs offline.** No test may call the real API or load the real embedding model unless marked `@pytest.mark.slow` (model) or `@pytest.mark.live` (API); both are deselected by default.
- **Platform is Windows.** `pathlib`, never string path concatenation. `encoding="utf-8"` explicit on every file read and write.
- **Commit after every task.** Messages end with a blank line then exactly:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## What Phases 1–2 already provide

| Thing | Signature |
|---|---|
| `rag.chunking.Chunk` | `chunk_id`, `doc_id`, `index`, `text`, `token_start`, `token_end`, `char_start`, `char_end` (exclusive) |
| `rag.chunking.RetrievedChunk` | `chunk`, `score`, `rank` (1-based), `score_kind` (`"cosine"` or `"rrf"`) |
| `rag.loader.load_documents` | `(corpus_dir, metadata_path) -> list[Document]`, sorted by `doc_id`, raises with actionable messages |
| `rag.store.VectorStore` | `.search(query_vectors, k) -> list[list[RetrievedChunk]]`, `.save(path, meta=)`, `.load(path, expect_meta=)`, `.meta` |
| `rag.pipeline` | `build_index(config, embedder)`, `load_index(config)`, `ask(question, store, embedder, llm, config, k=None, strategy="direct", strategy_options=None) -> Trace` |
| `rag.trace.Trace` | `question`, `strategy`, `queries`, `translation`, `retrieved`, `prompt`, `answer`, `timings` (with `depth`), `notes`, `total_ms`, `to_dict()` |
| `rag.strategies` | `STRATEGY_NAMES == ('direct','multi-query','rag-fusion','step-back','hyde','decomposition')`, `get_strategy(name, **options)` |
| `rag.config.Config` | frozen; `embedding_model`, `llm_model`, `chunk_tokens=200`, `chunk_overlap=50`, `max_seq_tokens=256`, `top_k=5`, `corpus_dir`, `metadata_path`, `index_path`, `cache_dir`, `api_key` |
| `rag.llm.GeminiLLM` | `.generate(prompt)`, `.last_call_cached`, `.call_count`, on-disk cache keyed by model+prompt+temperature |

Corpus: 38 documents, 3,064,396 characters, 5,116 chunks. Current suite: 368 passed, 4 deselected.

## Three constraints carried from the Phase 2 review

These would each distort the benchmark, so they are designed in rather than discovered in the numbers.

1. **`top_k` currently doubles as the per-query retrieval depth and the final truncation.** RAG-Fusion therefore fuses five 5-item lists down to 5, which structurally compresses the difference between it and multi-query — the benchmark would report "no difference" for reasons of plumbing, not method. Task 1 gives the per-query depth its own knob.
2. **Stage timings are not comparable across strategies.** Four emit a `translate` stage that is LLM-only; decomposition emits one `decompose` stage that swallows retrieval *and* N generations. `total_ms` is comparable; per-stage timings are not. The benchmark reports `total_ms` only.
3. **A silent LLM outage would be measured as "no technique helps."** Every strategy degrades to direct retrieval and records a note. The benchmark must hard-fail on any such note rather than averaging the result in.

## A refinement to the spec's gold-set format

The spec says gold questions are tagged with "the character span of the text that answers them" (singular). Hand-authoring character offsets into a 200KB document is not workable, and an offset typed by hand is unverifiable by eye.

The gold set therefore stores a **verbatim quote** and resolves it to a span at load time, rather than a hand-typed offset. That part of the spec's intent survives unchanged: the span-based property that matters — re-chunking does not invalidate the gold set — is preserved, and loading validates that each quote occurs **exactly once** in its document, so a typo or an ambiguous quote fails loudly instead of silently scoring against the wrong passage.

What changed from the spec, and from the original version of this section, is cardinality: `GoldQuestion.quotes` is a **list** of quotes, each resolved to its own span, giving `GoldQuestion.spans: tuple[tuple[int, int], ...]` — one span per quote, not one span per question. A question usually has more than one passage that actually answers it: a paper states its contribution in the abstract and then explains it properly in the body. Accepting only the first as correct scores a strategy zero for retrieving the better explanation of the same thing, which measures the gold set's incompleteness rather than the strategy.

This was not a design preference, it was forced by calibration. Scoring plain retrieval against an early, single-quote version of this gold set produced a mean Recall@5 of **0.05** across the ten questions — there was no headroom left to tell any of the six strategies apart, since they were all pinned near zero for the same structural reason. Moving to a list of quotes per question fixed a real defect (a question with more than one answering passage was undercounted whenever a strategy found a passage the single quote didn't name), but it did not move Recall@5: `progress.md` records **0.050 both before and after** the multi-quote fix, because with 5,116 chunks and only 1-4 relevant per question, Recall@5 has a low ceiling regardless of gold-set quality. What actually gave the benchmark table headroom to show a difference between strategies was two separate changes: measuring recall/MRR/nDCG at k=20 rather than k=5 (see the benchmark's own docstring), and adding document-precision@5 alongside it.

## File Structure

```
evaluation/
  __init__.py
  gold.py          GoldQuestion dataclass; load_gold(); quote -> span resolution
  gold.json        10 questions: id, question, doc_id, quote, why
  spans.py         span -> overlapping chunk ids
  metrics.py       recall_at_k, reciprocal_rank, ndcg_at_k
  benchmark.py     sweep strategies x questions -> markdown table; CLI entry
tests/
  test_gold.py
  test_spans.py
  test_metrics.py
  test_benchmark.py
rag/
  config.py        MODIFY: retrieval_depth
  pipeline.py      MODIFY: ask(generate=True)
  strategies/*.py  MODIFY: search at retrieval_depth, truncate to top_k
tests/test_no_frameworks.py   MODIFY: scan evaluation/
README.md          MODIFY: benchmark table
```

`evaluation/` imports from `rag/`; nothing in `rag/` imports from `evaluation/`.

---

### Task 1: Separate retrieval depth from final k, and make generation optional

Two changes to Phases 1–2 that the benchmark needs. Done together because both touch `ask` and the strategies.

**Files:**
- Modify: `rag/config.py`, `rag/pipeline.py`, `rag/strategies/direct.py`, `multi_query.py`, `rag_fusion.py`, `step_back.py`, `hyde.py`, `decomposition.py`
- Test: `tests/test_config.py`, `tests/test_pipeline.py`, `tests/test_strategies.py`

**Interfaces:**
- Consumes: `Config`, `StrategyContext`, `ask`
- Produces:
  - `Config.retrieval_depth: int = 20` — how many chunks each individual query retrieves, before combination and final truncation to `top_k`
  - `ask(..., generate: bool = True)` — when `False`, retrieval runs and no answer is generated

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_retrieval_depth_defaults_above_top_k():
    # Each query retrieves deeper than the final answer needs, so fusion has
    # material to work with. If they were equal, RAG-Fusion would fuse five
    # 5-item lists down to 5 and could not differ much from a plain union.
    cfg = Config()
    assert cfg.retrieval_depth == 20
    assert cfg.retrieval_depth >= cfg.top_k


def test_retrieval_depth_below_top_k_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="retrieval_depth"):
        Config(retrieval_depth=3, top_k=5)
```

Append to `tests/test_pipeline.py`:

```python
def test_ask_can_skip_generation(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    llm = FakeLLM()
    trace = ask("q", store, FakeEmbedder(), llm, tiny_corpus, generate=False)
    assert trace.retrieved
    assert trace.answer is None
    assert llm.prompts == []
    assert [t.name for t in trace.timings] == ["embed", "search"]


def test_ask_without_generation_still_uses_the_llm_for_translation(
    tiny_corpus: Config,
):
    # generate=False must skip the *answer*, not the rewrite — otherwise the
    # benchmark would measure every strategy as plain retrieval.
    store = build_index(tiny_corpus, FakeEmbedder())
    llm = FakeLLM("1. first rewrite\n2. second rewrite")
    trace = ask(
        "q", store, FakeEmbedder(), llm, tiny_corpus,
        strategy="multi-query", generate=False,
    )
    assert len(llm.prompts) == 1
    assert trace.answer is None
    assert len(trace.queries) == 3
```

Append to `tests/test_strategies.py`:

```python
def test_strategies_retrieve_at_depth_then_truncate_to_top_k(tiny_corpus: Config):
    from dataclasses import replace

    deep = replace(tiny_corpus, retrieval_depth=6, top_k=2)
    ctx = build_context(deep, llm=FakeLLM(MULTI_QUERY_REPLY))
    result = get_strategy("multi-query").run("q", ctx)
    assert len(result.retrieved) <= 2


def test_direct_also_honours_retrieval_depth(tiny_corpus: Config):
    from dataclasses import replace

    deep = replace(tiny_corpus, retrieval_depth=6, top_k=2)
    ctx = build_context(deep, llm=None)
    assert len(get_strategy("direct").run("q", ctx).retrieved) <= 2
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_config.py tests/test_pipeline.py tests/test_strategies.py -q`
Expected: `TypeError` for the unexpected `retrieval_depth` and `generate` keywords.

- [ ] **Step 3: Add `retrieval_depth` to `Config`**

In `rag/config.py`, add the field after `top_k`:

```python
    retrieval_depth: int = 20
    """How many chunks each individual query retrieves, before combination.

    Distinct from `top_k`, which is how many survive into the answer. Keeping
    them separate matters for the multi-query strategies: if each of five
    rewrites retrieved only `top_k` chunks, fusion would have almost nothing
    to fuse and RAG-Fusion could not differ meaningfully from a plain union.
    """
```

And validate in `__post_init__`:

```python
        if self.retrieval_depth < self.top_k:
            raise ValueError(
                f"retrieval_depth ({self.retrieval_depth}) must be at least "
                f"top_k ({self.top_k}); each query retrieves at depth and the "
                "combined result is then truncated to top_k"
            )
```

- [ ] **Step 4: Point every strategy at `retrieval_depth`**

In each of `direct.py`, `multi_query.py`, `rag_fusion.py`, `step_back.py`, `hyde.py`, `decomposition.py`, and in `degrade_to_direct` in `base.py`, change every `ctx.search(..., ctx.config.top_k)` call to `ctx.search(..., ctx.config.retrieval_depth)`.

Leave the final `[: ctx.config.top_k]` truncations exactly as they are. `direct.py` currently returns the search result untruncated because depth and k were the same; it now needs an explicit `[: ctx.config.top_k]` like the others:

```python
class DirectStrategy:
    name = "direct"

    def run(self, question: str, ctx: StrategyContext) -> StrategyResult:
        ctx.trace.queries = [question]
        results = ctx.search([question], ctx.config.retrieval_depth)[0]
        return StrategyResult(retrieved=results[: ctx.config.top_k])
```

Apply the same two-line shape to `degrade_to_direct` in `rag/strategies/base.py`.

- [ ] **Step 5: Add `generate` to `ask`**

In `rag/pipeline.py`, add the parameter and guard the generation call:

```python
def ask(
    question: str,
    store: VectorStore,
    embedder,
    llm,
    config: Config,
    k: int | None = None,
    strategy: str = "direct",
    strategy_options: dict | None = None,
    generate: bool = True,
) -> Trace:
```

and replace the generation block:

```python
    if llm is None:
        trace.note("retrieval only: no LLM configured")
        return trace

    if not generate:
        # The strategy still used the LLM to translate the query; only the
        # final answer is skipped. The benchmark measures retrieval, and
        # generating an answer it never reads would cost a call per question.
        return trace

    generate_answer(
        llm, question, result.retrieved, trace, extra_context=result.extra_context
    )
    return trace
```

Note `k` still overrides `top_k` via the `replace(config, top_k=effective_k)` already in `ask`. When `k` exceeds `retrieval_depth` that copy would now fail validation, so also raise `retrieval_depth` to at least `k` in the same `replace` call:

```python
    ctx = StrategyContext(
        store=store,
        embedder=embedder,
        llm=llm,
        config=replace(
            config,
            top_k=effective_k,
            retrieval_depth=max(config.retrieval_depth, effective_k),
        ),
        trace=trace,
    )
```

- [ ] **Step 6: Run the tests and verify they pass**

Run: `python -m pytest -q`
Expected: all pass. Existing strategy tests that assumed `top_k` was the retrieval depth may need their expectations adjusted; check each failure is a changed expectation rather than a real regression, and do not weaken what any test asserts.

- [ ] **Step 7: Check the change is visible against the real index**

```
python -m rag ask "How does ColBERT score a document?" --strategy rag-fusion --trace
```

Expect 5 results as before, but now fused from 20-deep lists. Report whether the retrieved set changed compared to before this task — it plausibly should, and that is the point.

- [ ] **Step 8: Commit**

```bash
git add rag/ tests/
git commit -m "feat: separate per-query retrieval depth from final top_k

Each query now retrieves retrieval_depth chunks and the combined result
is truncated to top_k. With them equal, RAG-Fusion fused five 5-item
lists into 5 and could not differ meaningfully from a plain union --
the benchmark would have measured plumbing, not method.

Also adds ask(generate=False) so the benchmark can measure retrieval
without paying for an answer it never reads.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Gold set loading and quote resolution

**Files:**
- Create: `evaluation/__init__.py`, `evaluation/gold.py`
- Test: `tests/test_gold.py`

**Interfaces:**
- Consumes: `rag.loader.load_documents`
- Produces:
  - `GoldQuestion` — frozen dataclass: `id: str`, `question: str`, `doc_id: str`, `quote: str`, `why: str`, `char_start: int`, `char_end: int`
  - `load_gold(path: Path, documents: list[Document]) -> list[GoldQuestion]` — resolves each quote to a span, raising if a quote is missing, ambiguous, or names an unknown document

- [ ] **Step 1: Write the failing test**

Create `tests/test_gold.py`:

```python
import json

import pytest

from evaluation.gold import GoldQuestion, load_gold
from rag.loader import Document


def _docs():
    return [
        Document(
            doc_id="alpha",
            text="Cosine similarity ignores magnitude. It measures the angle only.",
            title="Alpha",
            source="test",
        ),
        Document(doc_id="beta", text="Rank fusion sums reciprocal ranks.", title="Beta", source="test"),
    ]


def _write(tmp_path, questions):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps({"questions": questions}), encoding="utf-8")
    return path


def test_loads_a_question_and_resolves_its_quote(tmp_path):
    path = _write(tmp_path, [{
        "id": "q1",
        "question": "Does cosine similarity care about length?",
        "doc_id": "alpha",
        "quote": "Cosine similarity ignores magnitude.",
        "why": "states the property directly",
    }])
    gold = load_gold(path, _docs())
    assert len(gold) == 1
    assert isinstance(gold[0], GoldQuestion)
    assert gold[0].char_start == 0
    assert gold[0].char_end == len("Cosine similarity ignores magnitude.")


def test_resolved_span_reproduces_the_quote(tmp_path):
    path = _write(tmp_path, [{
        "id": "q1", "question": "q", "doc_id": "alpha",
        "quote": "measures the angle", "why": "w",
    }])
    gold = load_gold(path, _docs())
    text = _docs()[0].text
    assert text[gold[0].char_start:gold[0].char_end] == "measures the angle"


def test_a_quote_that_does_not_appear_is_an_error(tmp_path):
    path = _write(tmp_path, [{
        "id": "q1", "question": "q", "doc_id": "alpha",
        "quote": "this text is not in the document", "why": "w",
    }])
    with pytest.raises(ValueError, match="q1"):
        load_gold(path, _docs())


def test_an_ambiguous_quote_is_an_error(tmp_path):
    # Two occurrences means the span is undetermined, and scoring would
    # silently use whichever came first.
    docs = [Document(doc_id="alpha", text="repeat. repeat.", title="A", source="t")]
    path = _write(tmp_path, [{
        "id": "q1", "question": "q", "doc_id": "alpha", "quote": "repeat.", "why": "w",
    }])
    with pytest.raises(ValueError, match="twice|2 times|ambiguous"):
        load_gold(path, docs)


def test_an_unknown_doc_id_is_an_error(tmp_path):
    path = _write(tmp_path, [{
        "id": "q1", "question": "q", "doc_id": "nope", "quote": "x", "why": "w",
    }])
    with pytest.raises(ValueError, match="nope"):
        load_gold(path, _docs())


def test_duplicate_question_ids_are_an_error(tmp_path):
    path = _write(tmp_path, [
        {"id": "q1", "question": "a", "doc_id": "alpha", "quote": "angle", "why": "w"},
        {"id": "q1", "question": "b", "doc_id": "beta", "quote": "fusion", "why": "w"},
    ])
    with pytest.raises(ValueError, match="q1"):
        load_gold(path, _docs())


def test_a_missing_required_field_is_an_error(tmp_path):
    path = _write(tmp_path, [{"id": "q1", "question": "q", "doc_id": "alpha"}])
    with pytest.raises(ValueError, match="quote"):
        load_gold(path, _docs())


def test_questions_load_in_file_order(tmp_path):
    path = _write(tmp_path, [
        {"id": "q2", "question": "b", "doc_id": "beta", "quote": "fusion", "why": "w"},
        {"id": "q1", "question": "a", "doc_id": "alpha", "quote": "angle", "why": "w"},
    ])
    assert [g.id for g in load_gold(path, _docs())] == ["q2", "q1"]
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_gold.py -q`
Expected: `ModuleNotFoundError: No module named 'evaluation'`.

- [ ] **Step 3: Implement**

Create an empty `evaluation/__init__.py`, then `evaluation/gold.py`:

```python
"""The gold set: questions paired with the text that answers them.

Each question names a document and quotes the passage that answers it,
verbatim. The quote is resolved to a character span when the gold set loads,
and spans are resolved to chunk ids at scoring time.

Storing a quote rather than raw offsets is what makes the file writable by
hand and checkable by eye — and loading validates that the quote occurs
exactly once, so a typo or an ambiguous phrase fails loudly instead of
silently scoring against the wrong passage.

The span survives re-chunking, which is the property that matters: chunk ids
shift whenever chunk size or overlap changes, so a chunk-id-based gold set
would break the first time retrieval is tuned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag.loader import Document

REQUIRED_FIELDS = ("id", "question", "doc_id", "quote", "why")


@dataclass(frozen=True)
class GoldQuestion:
    id: str
    question: str
    doc_id: str
    quote: str
    why: str
    char_start: int
    char_end: int


def load_gold(path: Path, documents: list[Document]) -> list[GoldQuestion]:
    """Load the gold set, resolving each quote to a character span."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw["questions"]
    by_id = {doc.doc_id: doc for doc in documents}

    gold: list[GoldQuestion] = []
    seen: set[str] = set()
    for entry in entries:
        missing = [f for f in REQUIRED_FIELDS if f not in entry]
        if missing:
            raise ValueError(
                f"gold question {entry.get('id', '<no id>')} in {path} is "
                f"missing required field(s): {', '.join(missing)}"
            )
        question_id = entry["id"]
        if question_id in seen:
            raise ValueError(f"duplicate gold question id in {path}: {question_id}")
        seen.add(question_id)

        doc = by_id.get(entry["doc_id"])
        if doc is None:
            raise ValueError(
                f"gold question {question_id} names unknown doc_id: {entry['doc_id']}"
            )

        quote = entry["quote"]
        occurrences = doc.text.count(quote)
        if occurrences == 0:
            raise ValueError(
                f"gold question {question_id}: quote not found in {doc.doc_id}. "
                "It must match the document text exactly, including whitespace."
            )
        if occurrences > 1:
            raise ValueError(
                f"gold question {question_id}: quote is ambiguous, it appears "
                f"{occurrences} times in {doc.doc_id}. Extend it until unique."
            )

        start = doc.text.index(quote)
        gold.append(
            GoldQuestion(
                id=question_id,
                question=entry["question"],
                doc_id=entry["doc_id"],
                quote=quote,
                why=entry["why"],
                char_start=start,
                char_end=start + len(quote),
            )
        )
    return gold
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_gold.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add evaluation/ tests/test_gold.py
git commit -m "feat: gold set loading with quote-to-span resolution

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Span to chunk resolution

**Files:**
- Create: `evaluation/spans.py`
- Test: `tests/test_spans.py`

**Interfaces:**
- Consumes: `rag.chunking.Chunk`, `evaluation.gold.GoldQuestion`
- Produces:
  - `chunks_overlapping(doc_id: str, char_start: int, char_end: int, chunks: list[Chunk]) -> set[str]`
  - `relevant_chunk_ids(question: GoldQuestion, chunks: list[Chunk]) -> set[str]`

A chunk counts as relevant when its character range overlaps the gold span at all. Partial overlap counts: chunks are 200 tokens with 50 overlap, so an answering sentence routinely straddles a boundary, and requiring containment would mark a chunk that holds most of the answer as a miss.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spans.py`:

```python
from evaluation.gold import GoldQuestion
from evaluation.spans import chunks_overlapping, relevant_chunk_ids
from rag.chunking import Chunk


def _chunk(chunk_id, doc_id, start, end):
    return Chunk(chunk_id, doc_id, 0, "text", 0, 10, start, end)


def _chunks():
    return [
        _chunk("a:0", "alpha", 0, 100),
        _chunk("a:1", "alpha", 80, 180),
        _chunk("a:2", "alpha", 160, 260),
        _chunk("b:0", "beta", 0, 100),
    ]


def test_a_span_inside_one_chunk_matches_only_it():
    assert chunks_overlapping("alpha", 10, 50, _chunks()) == {"a:0"}


def test_a_span_straddling_a_boundary_matches_both_chunks():
    # Chunks overlap by design, so an answering sentence routinely spans two.
    # Requiring containment would score a chunk holding most of the answer
    # as a miss.
    assert chunks_overlapping("alpha", 90, 120, _chunks()) == {"a:0", "a:1"}


def test_a_span_covering_three_chunks_matches_all_three():
    assert chunks_overlapping("alpha", 10, 250, _chunks()) == {"a:0", "a:1", "a:2"}


def test_chunks_from_other_documents_never_match():
    assert "b:0" not in chunks_overlapping("alpha", 0, 300, _chunks())


def test_a_span_touching_only_the_exclusive_end_does_not_match():
    # char_end is exclusive, so a span starting exactly at a chunk's end
    # shares no characters with it.
    assert chunks_overlapping("alpha", 100, 110, _chunks()) == {"a:1"}


def test_a_span_in_an_unindexed_document_matches_nothing():
    assert chunks_overlapping("gamma", 0, 50, _chunks()) == set()


def test_an_empty_span_matches_nothing():
    assert chunks_overlapping("alpha", 50, 50, _chunks()) == set()


def test_relevant_chunk_ids_uses_the_questions_span():
    question = GoldQuestion(
        id="q1", question="q", doc_id="alpha", quote="x", why="w",
        char_start=90, char_end=120,
    )
    assert relevant_chunk_ids(question, _chunks()) == {"a:0", "a:1"}
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_spans.py -q`
Expected: `ModuleNotFoundError: No module named 'evaluation.spans'`.

- [ ] **Step 3: Implement `evaluation/spans.py`**

```python
"""Resolve a gold span to the chunks that contain it.

Relevance is overlap, not containment. Chunks are 200 tokens with 50 tokens
of overlap, so the sentence answering a question frequently straddles a chunk
boundary. Requiring a chunk to contain the whole span would mark a chunk
holding most of the answer as a miss, and the metrics would understate every
strategy equally — which is worse than it sounds, because it would also
compress the differences between them.
"""

from __future__ import annotations

from evaluation.gold import GoldQuestion
from rag.chunking import Chunk


def chunks_overlapping(
    doc_id: str, char_start: int, char_end: int, chunks: list[Chunk]
) -> set[str]:
    """Ids of chunks in `doc_id` whose character range overlaps the span.

    Both the span and the chunk ranges are half-open, so a span beginning
    exactly where a chunk ends does not overlap it.
    """
    if char_end <= char_start:
        return set()
    return {
        chunk.chunk_id
        for chunk in chunks
        if chunk.doc_id == doc_id
        and chunk.char_start < char_end
        and char_start < chunk.char_end
    }


def relevant_chunk_ids(question: GoldQuestion, chunks: list[Chunk]) -> set[str]:
    """The chunks that count as a correct retrieval for this question."""
    return chunks_overlapping(
        question.doc_id, question.char_start, question.char_end, chunks
    )
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_spans.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add evaluation/spans.py tests/test_spans.py
git commit -m "feat: resolve gold spans to overlapping chunks

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Metrics

**Files:**
- Create: `evaluation/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float`
  - `reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float`
  - `ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float`

`retrieved_ids` is in rank order, best first. Relevance is binary.

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
import math

import pytest

from evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


# --- recall@k ----------------------------------------------------------------

def test_recall_is_the_fraction_of_relevant_chunks_found():
    assert recall_at_k(["a", "b", "c"], {"a", "d"}, k=3) == pytest.approx(0.5)


def test_recall_is_one_when_everything_relevant_is_retrieved():
    assert recall_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)


def test_recall_is_zero_when_nothing_relevant_is_retrieved():
    assert recall_at_k(["x", "y"], {"a"}, k=2) == pytest.approx(0.0)


def test_recall_ignores_hits_below_k():
    assert recall_at_k(["x", "a"], {"a"}, k=1) == pytest.approx(0.0)
    assert recall_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(1.0)


def test_recall_with_no_relevant_chunks_is_zero_not_a_crash():
    # A gold question whose span resolves to nothing is a broken question,
    # but it must not divide by zero mid-benchmark.
    assert recall_at_k(["a"], set(), k=1) == pytest.approx(0.0)


def test_recall_with_k_larger_than_the_result_list_uses_what_is_there():
    assert recall_at_k(["a"], {"a"}, k=10) == pytest.approx(1.0)


# --- reciprocal rank ---------------------------------------------------------

def test_reciprocal_rank_of_a_first_place_hit_is_one():
    assert reciprocal_rank(["a", "b"], {"a"}) == pytest.approx(1.0)


def test_reciprocal_rank_of_a_third_place_hit_is_one_third():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_the_first_relevant_hit_only():
    assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == pytest.approx(0.5)


def test_reciprocal_rank_with_no_hit_is_zero():
    assert reciprocal_rank(["x", "y"], {"a"}) == pytest.approx(0.0)


def test_reciprocal_rank_of_an_empty_result_is_zero():
    assert reciprocal_rank([], {"a"}) == pytest.approx(0.0)


# --- nDCG@k ------------------------------------------------------------------

def test_ndcg_is_one_when_the_only_relevant_chunk_ranks_first():
    assert ndcg_at_k(["a", "x", "y"], {"a"}, k=3) == pytest.approx(1.0)


def test_ndcg_matches_the_hand_computed_value_for_a_second_place_hit():
    # DCG  = 1/log2(2+1) = 0.630929...
    # IDCG = 1/log2(1+1) = 1.0
    assert ndcg_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(1 / math.log2(3))


def test_ndcg_matches_the_hand_computed_value_for_two_hits():
    # retrieved: x a b -> DCG  = 1/log2(3) + 1/log2(4) = 0.630929 + 0.5
    # ideal:     a b x -> IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.630929
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["x", "a", "b"], {"a", "b"}, k=3) == pytest.approx(expected)


def test_ndcg_rewards_ranking_a_hit_higher():
    assert ndcg_at_k(["a", "x"], {"a"}, k=2) > ndcg_at_k(["x", "a"], {"a"}, k=2)


def test_ndcg_with_no_relevant_chunks_is_zero():
    assert ndcg_at_k(["a"], set(), k=1) == pytest.approx(0.0)


def test_ndcg_with_no_hits_is_zero():
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == pytest.approx(0.0)


def test_ndcg_ideal_accounts_for_more_relevant_chunks_than_k():
    # Only k can be retrieved, so the ideal ranking is capped at k too;
    # otherwise a question with 10 relevant chunks could never score 1.0.
    assert ndcg_at_k(["a", "b"], {"a", "b", "c", "d"}, k=2) == pytest.approx(1.0)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: `ModuleNotFoundError: No module named 'evaluation.metrics'`.

- [ ] **Step 3: Implement `evaluation/metrics.py`**

```python
"""Retrieval metrics, written out.

Relevance is binary: a chunk either overlaps the gold span or it does not.
All three metrics take results in rank order, best first.

What each one tells you, since they disagree usefully:
- Recall@k: did we find the answer at all, within k?
- MRR: how near the top was the first correct chunk?
- nDCG@k: how well ordered is the whole result list?

A strategy can win on recall and lose on MRR by finding the answer but
burying it, which is worth knowing when the answer prompt only gets k chunks.
"""

from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the relevant chunks that appear in the top k."""
    if not relevant_ids:
        return 0.0
    found = set(retrieved_ids[:k]) & relevant_ids
    return len(found) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / rank of the first relevant chunk, or 0 if none was retrieved."""
    for position, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalised discounted cumulative gain over the top k.

    The ideal ranking is capped at k as well as at the number of relevant
    chunks: only k results can be returned, so a question with more relevant
    chunks than k must still be able to score 1.0.
    """
    if not relevant_ids:
        return 0.0

    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, chunk_id in enumerate(retrieved_ids[:k], start=1)
        if chunk_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: recall@k, MRR and nDCG@k

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Write the gold set

Manual data work, and the quality of everything downstream depends on it. Budget real time; a sloppy gold set produces confident numbers that mean nothing.

**Files:**
- Create: `evaluation/gold.json`
- Test: `tests/test_gold_set.py`

**Interfaces:**
- Consumes: `load_gold`, `relevant_chunk_ids`, the real corpus
- Produces: `evaluation/gold.json` with 10 questions

- [ ] **Step 1: Write the failing test**

Create `tests/test_gold_set.py`:

```python
"""Checks on the real gold set, as opposed to the loading machinery."""

from pathlib import Path

import pytest

from evaluation.gold import load_gold
from evaluation.spans import relevant_chunk_ids
from rag.chunking import chunk_documents
from rag.config import Config
from rag.loader import load_documents

GOLD_PATH = Path("evaluation/gold.json")


@pytest.fixture(scope="module")
def corpus():
    cfg = Config()
    return load_documents(cfg.corpus_dir, cfg.metadata_path)


def test_the_gold_set_loads(corpus):
    assert load_gold(GOLD_PATH, corpus)


def test_the_gold_set_has_ten_questions(corpus):
    assert len(load_gold(GOLD_PATH, corpus)) == 10


def test_every_question_has_a_reason_recorded(corpus):
    for question in load_gold(GOLD_PATH, corpus):
        assert question.why.strip(), f"{question.id} has no 'why'"


def test_questions_are_spread_across_documents(corpus):
    # A gold set concentrated on one paper measures that paper, not retrieval.
    gold = load_gold(GOLD_PATH, corpus)
    assert len({g.doc_id for g in gold}) >= 6


def test_quotes_are_substantial(corpus):
    # A three-word quote resolves to a span so small it may sit inside a
    # single chunk by luck rather than because that chunk answers anything.
    for question in load_gold(GOLD_PATH, corpus):
        assert len(question.quote) >= 40, f"{question.id}: quote too short"


@pytest.mark.slow
def test_every_question_resolves_to_at_least_one_chunk(corpus):
    from transformers import AutoTokenizer

    cfg = Config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.embedding_model, use_fast=True)
    chunks = chunk_documents(corpus, tokenizer, cfg.chunk_tokens, cfg.chunk_overlap)
    for question in load_gold(GOLD_PATH, corpus):
        relevant = relevant_chunk_ids(question, chunks)
        assert relevant, f"{question.id} resolves to no chunks"
        assert len(relevant) <= 6, (
            f"{question.id} resolves to {len(relevant)} chunks; the quote is "
            "probably too long to be a precise target"
        )
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_gold_set.py -q`
Expected: failure loading `evaluation/gold.json` — it does not exist.

- [ ] **Step 3: Explore the corpus and choose questions**

The corpus is 38 documents in `data/corpus/`. Read across them — `colbert`, `colbertv2`, `raptor`, `hyde`, `dpr`, `rag`, `self_rag`, `step_back`, `ircot`, `crag`, `contriever`, `splade`, `rag_survey` are all substantial.

Use commands like these to find candidate passages rather than reading whole files:

```bash
grep -n -i "we propose\|our approach\|the key idea\|in contrast" data/corpus/colbert.txt | head -20
python -c "
from pathlib import Path
t = Path('data/corpus/hyde.txt').read_text(encoding='utf-8')
i = t.lower().index('zero-shot')
print(repr(t[i-200:i+400]))
"
```

**Aim for a mix that can discriminate between strategies**, since a gold set where every question is a keyword lookup will show all six strategies as equal:

- 3–4 questions whose wording closely matches the document (direct retrieval should do well)
- 2–3 whose natural phrasing differs from the paper's vocabulary (multi-query and HyDE should have an edge)
- 2 that need two facts from different parts of a document, or from different documents (decomposition's case)
- 1–2 that ask about a general concept a specific paper explains in passing (step-back's case)

**Record `why` honestly** — it is the note explaining what the question is testing, and it is what makes the gold set reviewable rather than a black box.

- [ ] **Step 4: Verify each quote before writing it down**

For every candidate, confirm the quote appears exactly once:

```bash
python -c "
from pathlib import Path
quote = 'PASTE THE EXACT QUOTE HERE'
t = Path('data/corpus/DOC_ID.txt').read_text(encoding='utf-8')
print('occurrences:', t.count(quote))
"
```

It must print `1`. If it prints `0` the whitespace differs — the corpus is normalised text, so a quote copied from a PDF will not match. If it prints more than 1, extend the quote until unique.

- [ ] **Step 5: Write `evaluation/gold.json`**

Format. The example below is real: that quote occurs exactly once in
`data/corpus/colbert.txt` and is 117 characters. Keep it and add nine more:

```json
{
  "questions": [
    {
      "id": "colbert-scoring",
      "question": "How does ColBERT score a document against a query?",
      "doc_id": "colbert",
      "quote": "every query embedding interacts with all document embeddings via a MaxSim operator, which computes maximum similarity",
      "why": "Direct vocabulary match; baseline case plain retrieval should handle."
    }
  ]
}
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_gold_set.py -q`
Then: `python -m pytest tests/test_gold_set.py -q -m slow`

Expected: all pass, including that every question resolves to between 1 and 6 chunks. If a question resolves to 0, the quote spans a gap in chunking — extend or move it. If it resolves to more than 6, the quote is too long to be a precise target.

- [ ] **Step 7: Sanity-check that the gold set is answerable at all**

```
python -c "
from pathlib import Path
from transformers import AutoTokenizer
from rag.config import Config
from rag.loader import load_documents
from rag.chunking import chunk_documents
from rag.embedding import Embedder
from rag.pipeline import load_index
from evaluation.gold import load_gold
from evaluation.spans import relevant_chunk_ids
from evaluation.metrics import recall_at_k

cfg = Config.from_env(env_file=Path('.env'))
e = Embedder(cfg.embedding_model, max_length=cfg.max_seq_tokens)
docs = load_documents(cfg.corpus_dir, cfg.metadata_path)
chunks = chunk_documents(docs, e.tokenizer, cfg.chunk_tokens, cfg.chunk_overlap)
store = load_index(cfg)
gold = load_gold(Path('evaluation/gold.json'), docs)
total = 0.0
for g in gold:
    hits = [r.chunk.chunk_id for r in store.search(e.encode([g.question]), 5)[0]]
    r = recall_at_k(hits, relevant_chunk_ids(g, chunks), 5)
    total += r
    print(f'{r:.2f}  {g.id}')
print(f'mean Recall@5 (direct): {total/len(gold):.3f}')
"
```

Report the per-question numbers. A mean Recall@5 around 0.4–0.8 is a healthy gold set. **If it is 1.0, the questions are too easy** and no strategy can beat direct — go back and make some harder. **If it is near 0.0, something is wrong** with the spans rather than with retrieval; investigate before proceeding.

- [ ] **Step 8: Commit**

```bash
git add evaluation/gold.json tests/test_gold_set.py
git commit -m "feat: ten gold questions with verified answer spans

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Benchmark runner, CLI and README

**Files:**
- Create: `evaluation/benchmark.py`
- Modify: `tests/test_no_frameworks.py`, `README.md`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: everything above, `rag.pipeline.ask`, `rag.strategies.STRATEGY_NAMES`
- Produces:
  - `StrategyScore` — dataclass: `strategy: str`, `recall_at_k: float`, `mrr: float`, `ndcg_at_k: float`, `mean_ms: float`, `questions: int`
  - `score_strategy(...) -> StrategyScore`
  - `format_table(scores: list[StrategyScore], k: int) -> str`
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmark.py`:

```python
import pytest

from evaluation.benchmark import StrategyScore, format_table


def _scores():
    return [
        StrategyScore("direct", 0.62, 0.55, 0.58, 41.6, 10),
        StrategyScore("rag-fusion", 0.74, 0.61, 0.66, 1502.3, 10),
    ]


def test_table_has_a_row_per_strategy():
    table = format_table(_scores(), k=5)
    assert "direct" in table
    assert "rag-fusion" in table


def test_table_names_the_k_it_measured():
    assert "Recall@5" in format_table(_scores(), k=5)


def test_table_is_markdown():
    lines = format_table(_scores(), k=5).splitlines()
    assert lines[0].startswith("|")
    assert set(lines[1].replace("|", "").replace(" ", "")) <= {"-", ":"}


def test_table_reports_the_question_count():
    # A number computed over 10 questions must never be read as if it were
    # computed over 30.
    assert "10" in format_table(_scores(), k=5)


def test_table_rows_are_ordered_by_recall_descending():
    rows = [l for l in format_table(_scores(), k=5).splitlines() if l.startswith("|")]
    assert "rag-fusion" in rows[2]
    assert "direct" in rows[3]


def test_empty_scores_produce_a_table_with_no_rows():
    table = format_table([], k=5)
    assert table.splitlines()[0].startswith("|")


def test_a_degraded_trace_is_fatal():
    from evaluation.benchmark import check_not_degraded
    from rag.trace import Trace

    trace = Trace(question="q")
    trace.strategy = "hyde"
    trace.note("hyde needs an LLM; degraded to direct retrieval")
    with pytest.raises(RuntimeError, match="degraded"):
        check_not_degraded(trace)


def test_an_undegraded_trace_passes_the_check():
    from evaluation.benchmark import check_not_degraded
    from rag.trace import Trace

    trace = Trace(question="q")
    trace.note("generation served from cache")
    check_not_degraded(trace)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_benchmark.py -q`
Expected: `ModuleNotFoundError: No module named 'evaluation.benchmark'`.

- [ ] **Step 3: Implement `evaluation/benchmark.py`**

```python
"""Run every strategy over the gold set and report how they compare.

Three deliberate choices, each one a thing that would otherwise make the
numbers lie:

- **Only `total_ms` is reported, never per-stage timings.** Four strategies
  emit a `translate` stage that is LLM-only; decomposition emits one
  `decompose` stage that swallows retrieval and N generations. The stage names
  are not comparable across strategies even though they look like they are.

- **A degraded trace is fatal.** Every strategy falls back to plain retrieval
  when the LLM is unavailable, recording a note. Averaging those in would
  report "no technique helps" when the real finding is "the API key expired".

- **Generation is skipped.** These are retrieval metrics; the answer is never
  read, and generating one would cost a call per question per strategy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from evaluation.gold import GoldQuestion, load_gold
from evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from evaluation.spans import relevant_chunk_ids
from rag.chunking import Chunk, chunk_documents
from rag.config import Config
from rag.embedding import Embedder
from rag.llm import GeminiLLM
from rag.loader import load_documents
from rag.pipeline import ask, load_index
from rag.store import VectorStore
from rag.strategies import STRATEGY_NAMES
from rag.trace import Trace

DEGRADED = "degraded to direct retrieval"


@dataclass
class StrategyScore:
    strategy: str
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    mean_ms: float
    questions: int


def check_not_degraded(trace: Trace) -> None:
    """Fail loudly if a strategy silently fell back to plain retrieval."""
    for note in trace.notes:
        if DEGRADED in note:
            raise RuntimeError(
                f"strategy {trace.strategy!r} degraded on question "
                f"{trace.question!r}: {note}. Benchmarking a degraded run "
                "would measure plain retrieval and report it as the strategy."
            )


def score_strategy(
    strategy: str,
    gold: list[GoldQuestion],
    chunks: list[Chunk],
    store: VectorStore,
    embedder,
    llm,
    config: Config,
) -> StrategyScore:
    """Run one strategy over every gold question and average the metrics."""
    recalls: list[float] = []
    rrs: list[float] = []
    ndcgs: list[float] = []
    times: list[float] = []

    for question in gold:
        trace = ask(
            question.question,
            store,
            embedder,
            llm,
            config,
            strategy=strategy,
            generate=False,
        )
        check_not_degraded(trace)

        retrieved_ids = [r.chunk.chunk_id for r in trace.retrieved]
        relevant = relevant_chunk_ids(question, chunks)
        recalls.append(recall_at_k(retrieved_ids, relevant, config.top_k))
        rrs.append(reciprocal_rank(retrieved_ids, relevant))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant, config.top_k))
        times.append(trace.total_ms)

    n = len(gold)
    mean = lambda values: sum(values) / n if n else 0.0  # noqa: E731
    return StrategyScore(
        strategy=strategy,
        recall_at_k=mean(recalls),
        mrr=mean(rrs),
        ndcg_at_k=mean(ndcgs),
        mean_ms=mean(times),
        questions=n,
    )


def format_table(scores: list[StrategyScore], k: int) -> str:
    """Render scores as a markdown table, best recall first."""
    header = (
        f"| Strategy | Recall@{k} | MRR | nDCG@{k} | Mean ms | Questions |\n"
        "|---|---:|---:|---:|---:|---:|"
    )
    rows = [
        f"| {s.strategy} | {s.recall_at_k:.3f} | {s.mrr:.3f} | "
        f"{s.ndcg_at_k:.3f} | {s.mean_ms:.0f} | {s.questions} |"
        for s in sorted(scores, key=lambda s: (-s.recall_at_k, s.strategy))
    ]
    return "\n".join([header, *rows])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evaluation.benchmark",
        description="Measure every retrieval strategy against the gold set.",
    )
    parser.add_argument("--gold", type=Path, default=Path("evaluation/gold.json"))
    parser.add_argument("--k", type=int, help="final top-k (default: config)")
    parser.add_argument(
        "--strategy",
        action="append",
        choices=sorted(STRATEGY_NAMES),
        help="measure only these strategies (repeatable)",
    )
    parser.add_argument("--out", type=Path, help="also write the table here")
    args = parser.parse_args(argv)

    config = Config.from_env(env_file=Path(".env"))
    if args.k is not None:
        from dataclasses import replace

        config = replace(
            config,
            top_k=args.k,
            retrieval_depth=max(config.retrieval_depth, args.k),
        )

    embedder = Embedder(config.embedding_model, max_length=config.max_seq_tokens)
    documents = load_documents(config.corpus_dir, config.metadata_path)
    chunks = chunk_documents(
        documents, embedder.tokenizer, config.chunk_tokens, config.chunk_overlap
    )
    store = load_index(config)
    gold = load_gold(args.gold, documents)
    llm = GeminiLLM(
        model=config.llm_model, api_key=config.api_key, cache_dir=config.cache_dir
    )

    selected = args.strategy or list(STRATEGY_NAMES)
    scores = []
    for strategy in selected:
        print(f"running {strategy}...", flush=True)
        scores.append(
            score_strategy(
                strategy, gold, chunks, store, embedder, llm, config
            )
        )

    table = format_table(scores, k=config.top_k)
    print()
    print(table)
    if args.out:
        args.out.write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_benchmark.py -q`
Expected: 8 passed.

- [ ] **Step 5: Extend the no-frameworks guard to `evaluation/`**

In `tests/test_no_frameworks.py`, add `"evaluation"` to `SOURCE_DIRS`.

Run: `python -m pytest tests/test_no_frameworks.py -q` — still passing.

- [ ] **Step 6: Run the real benchmark**

This makes real API calls — one rewrite per question for each translation strategy, plus decomposition's per-sub-question calls. With a cold cache expect a few minutes; the cache makes re-runs nearly free.

```
python -m evaluation.benchmark
```

Report the full table. Then run it a second time and confirm the numbers are **identical** — if they drift, something is non-deterministic and the benchmark cannot be trusted.

If it raises `RuntimeError` about a degraded strategy, that is the check working: the LLM was unavailable. Fix the cause rather than removing the check.

- [ ] **Step 7: Put the table in the README, honestly**

Add a Benchmark section with the real table and, alongside it, the things that stop it being over-read:

- It is **10 questions**, not 30. Differences of a few points are noise at this sample size; Phase 7 expands the set.
- Recall, MRR and nDCG measure **retrieval only**. Answer quality is not measured anywhere yet.
- The gold set was written by the same person who built the system, which is a real bias worth naming.
- Say what the numbers actually show. If a technique does not help on this corpus, **say so** — a measured negative result, explained, is a stronger portfolio signal than an unmeasured claim.

- [ ] **Step 8: Commit**

```bash
git add evaluation/benchmark.py tests/test_benchmark.py tests/test_no_frameworks.py README.md
git commit -m "feat: benchmark runner and the first measured results

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 3 Definition of Done

- [ ] `python -m pytest` passes with no network access
- [ ] `python -m pytest -m slow` passes
- [ ] `python -m evaluation.benchmark` produces a table for all six strategies
- [ ] Running it twice produces identical numbers
- [ ] A degraded strategy raises rather than being averaged in
- [ ] The gold set is 10 questions across at least 6 documents, each quote verified unique, each resolving to 1–6 chunks
- [ ] `tests/test_no_frameworks.py` covers `evaluation/`
- [ ] The README states the sample size, that only retrieval is measured, and the gold set's authorship bias
- [ ] Working tree clean

## What later phases need from this one

- `load_gold` and `relevant_chunk_ids` are reused unchanged when Phase 7 grows the set to 30.
- `score_strategy` takes a strategy name, so Phases 4–6 are measured by adding a name to `STRATEGY_NAMES` and nothing else.
- `Config.retrieval_depth` is the knob Phase 6's ColBERT reranker needs: it reranks a deep candidate list down to `top_k`.
- The benchmark's degradation check is what stops a Phase 4 routing bug from being silently measured as "routing does not help".
