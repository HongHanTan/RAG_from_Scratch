# Phase 2: Query Translation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the five query-translation strategies from `RAG.pdf` Stage 1 — multi-query, RAG-Fusion, decomposition (both variants), step-back and HyDE — selectable with `--strategy`.

**Architecture:** A `Strategy` protocol takes a question and a `StrategyContext` (store, embedder, llm, config, trace) and returns a `StrategyResult` (retrieved chunks, plus optional extra context for the answer prompt). `pipeline.ask` calls exactly one strategy and is otherwise unchanged; strategies never touch the store directly, they call `ctx.search()`, which owns the embed and search timing. Every strategy that needs the LLM degrades to plain retrieval when it is unavailable or fails, and records that it degraded.

**Tech Stack:** Python 3.14, NumPy, google-genai, pytest. No new dependencies.

## Global Constraints

Copied from Phase 1; they still bind.

- **No RAG framework.** `langchain`, `llama_index`, `sentence_transformers`, `sklearn`, `bs4`, `requests`, `dotenv` are installed and MUST NOT be imported. `tests/test_no_frameworks.py` enforces this and also scans `pyproject.toml` dependencies.
- **Allowed third-party imports:** `numpy`, `torch`, `transformers`, `google.genai`, `pytest`.
- **Determinism:** LLM temperature 0, no unseeded randomness. Benchmark numbers in Phase 3 must be reproducible.
- **Test-driven.** Write the test, run it, watch it fail for the expected reason, then implement.
- **Every test runs offline.** No test may call the real API or load the real embedding model unless marked `@pytest.mark.slow` (model) or `@pytest.mark.live` (API); both are deselected by default. `tests/conftest.py` provides `tiny_corpus`, `FakeEmbedder`, `FakeTokenizer`, `FakeLLM`.
- **Platform is Windows.** `pathlib`, never string path concatenation. `encoding="utf-8"` explicit on every file read and write.
- **Degrade, but record it.** A strategy whose LLM call fails falls back to direct retrieval and calls `trace.note(...)`. Silent degradation is worse than a crash.
- **Commit after every task.** Messages end with a blank line then exactly:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## What Phase 1 already provides

Read these before starting; the plan builds directly on them.

| Thing | Signature |
|---|---|
| `rag.chunking.RetrievedChunk` | dataclass: `chunk: Chunk`, `score: float`, `rank: int` |
| `rag.store.VectorStore.search` | `(query_vectors: np.ndarray, k: int) -> list[list[RetrievedChunk]]`, one list per query row, ranked from 1 |
| `rag.similarity.top_k` | `(scores, k) -> (indices, values)`, stable, ties to lower index |
| `rag.trace.Trace` | `question`, `queries: list[str]`, `retrieved: list[RetrievedChunk]`, `prompt`, `answer`, `timings: list[StageTiming]`, `notes: list[str]`, `stage(name)` context manager, `note(msg)`, `total_ms`, `to_dict()` |
| `rag.prompts` | `NO_CONTEXT`, `ANSWER_TEMPLATE`, `format_context(list[RetrievedChunk]) -> str`, `build_answer_prompt(question, retrieved) -> str` |
| `rag.generation.generate_answer` | `(llm, question, retrieved, trace) -> str \| None`; returns `None` and notes on `LLMError` |
| `rag.llm.GeminiLLM` | `.generate(prompt) -> str`, raises `LLMError`; `.last_call_cached: bool`; on-disk cache keyed by model+prompt+temperature |
| `rag.pipeline.ask` | `(question, store, embedder, llm, config, k=None) -> Trace` |
| `rag.config.Config` | includes `top_k`, `llm_model`, `cache_dir`, `api_key` |

Current suite: 246 passed, 4 deselected.

## Known issues this phase must fix

Three came out of Phase 1's final review and are scheduled here because Phase 2 is what makes them bite:

1. **`Trace.total_ms` double-counts nested stages.** It sums every timing. Decomposition nests (an outer translate stage wrapping N retrieval and LLM calls), so wall time would be reported as roughly double. Task 1 adds depth.
2. **`format_context` labels every score `score`.** After fusion the number is an RRF score (~0.03), not a cosine similarity (~0.55), and reads as a catastrophic drop. Task 1 makes the score name itself.
3. **`format_trace` never prints `trace.queries`.** Rewritten queries are Phase 2's entire visible output. Task 9 renders them.

## File Structure

```
rag/
  chunking.py            MODIFY: RetrievedChunk gains score_kind
  trace.py               MODIFY: stage depth, strategy, TranslationStep
  prompts.py             MODIFY: score label; five new templates
  similarity.py          MODIFY: reciprocal_rank_fusion, merge_best_score
  pipeline.py            MODIFY: ask() dispatches to a strategy
  __main__.py            MODIFY: --strategy, --decomposition-mode, rendering
  strategies/
    __init__.py          registry: STRATEGIES, get_strategy(name)
    base.py              Strategy protocol, StrategyContext, StrategyResult, degrade_to_direct
    direct.py            DirectStrategy
    multi_query.py       MultiQueryStrategy
    rag_fusion.py        RagFusionStrategy
    step_back.py         StepBackStrategy
    hyde.py              HydeStrategy
    decomposition.py     DecompositionStrategy (recursive | independent)
tests/
  test_trace.py          MODIFY: depth, strategy, translation
  test_similarity.py     MODIFY: fusion and merge
  test_prompts.py        MODIFY: score labels, new templates
  test_strategies.py     NEW: all six strategies, LLM mocked
  test_pipeline.py       MODIFY: strategy dispatch
  test_cli.py            MODIFY: --strategy plumbing
README.md                MODIFY: strategies section
```

Dependency direction stays one-way: `strategies/` import from `chunking`, `similarity`, `prompts`, `trace`, `llm`; `pipeline` imports `strategies`; nothing imports `pipeline` except `__main__`.

---

### Task 1: Trace depth, score labels, and translation steps

Foundation for everything else. Three small changes to shared types, done together because later tasks assume all three.

**Files:**
- Modify: `rag/trace.py`, `rag/chunking.py`, `rag/prompts.py`
- Test: `tests/test_trace.py`, `tests/test_prompts.py`

**Interfaces:**
- Consumes: `rag.chunking.RetrievedChunk`, `rag.trace.Trace`
- Produces:
  - `RetrievedChunk.score_kind: str = "cosine"` (new field, defaulted so existing construction sites keep working)
  - `StageTiming.depth: int = 0`
  - `Trace.strategy: str = "direct"`
  - `TranslationStep` — frozen dataclass: `kind: str`, `text: str`
  - `Trace.translation: list[TranslationStep]`
  - `Trace.add_translation(kind: str, text: str) -> None`
  - `Trace.total_ms` sums only depth-0 timings

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trace.py`:

```python
def test_nested_stages_record_their_depth():
    trace = Trace(question="q")
    with trace.stage("translate"):
        with trace.stage("embed"):
            pass
    assert [(t.name, t.depth) for t in trace.timings] == [("embed", 1), ("translate", 0)]


def test_total_ms_counts_only_top_level_stages():
    # An outer stage already includes its children's time; summing both
    # would report roughly double the real wall time.
    trace = Trace(question="q")
    trace.timings = [
        StageTiming("embed", 10.0, depth=1),
        StageTiming("search", 5.0, depth=1),
        StageTiming("translate", 20.0, depth=0),
    ]
    assert trace.total_ms == 20.0


def test_depth_unwinds_after_a_nested_stage_raises():
    trace = Trace(question="q")
    with trace.stage("outer"):
        try:
            with trace.stage("inner"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        with trace.stage("after"):
            pass
    assert [(t.name, t.depth) for t in trace.timings] == [
        ("inner", 1),
        ("after", 1),
        ("outer", 0),
    ]


def test_strategy_defaults_to_direct():
    assert Trace(question="q").strategy == "direct"


def test_translation_steps_are_recorded_in_order():
    trace = Trace(question="q")
    trace.add_translation("query", "first rewrite")
    trace.add_translation("hypothetical", "a fake document")
    assert [(s.kind, s.text) for s in trace.translation] == [
        ("query", "first rewrite"),
        ("hypothetical", "a fake document"),
    ]


def test_to_dict_includes_strategy_and_translation():
    import json

    trace = Trace(question="q")
    trace.strategy = "hyde"
    trace.add_translation("hypothetical", "text")
    with trace.stage("outer"):
        with trace.stage("inner"):
            pass
    decoded = json.loads(json.dumps(trace.to_dict()))
    assert decoded["strategy"] == "hyde"
    assert decoded["translation"] == [{"kind": "hypothetical", "text": "text"}]
    assert decoded["timings"][0]["depth"] == 1
```

Append to `tests/test_prompts.py`:

```python
def test_context_labels_a_cosine_score_as_cosine():
    chunk = Chunk("a:0", "a", 0, "text", 0, 5, 0, 4)
    context = format_context([RetrievedChunk(chunk=chunk, score=0.552, rank=1)])
    assert "cosine 0.552" in context


def test_context_labels_a_fused_score_as_rrf():
    # After fusion the number is ~0.03, not a similarity. Labelling both
    # "score" makes fusion look like a catastrophic quality drop.
    chunk = Chunk("a:0", "a", 0, "text", 0, 5, 0, 4)
    retrieved = [
        RetrievedChunk(chunk=chunk, score=0.0328, rank=1, score_kind="rrf")
    ]
    context = format_context(retrieved)
    assert "rrf 0.033" in context
    assert "cosine" not in context
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_trace.py tests/test_prompts.py -q`
Expected: failures — `TypeError` for the unexpected `depth` / `score_kind` keyword, `AttributeError` for `add_translation` and `strategy`.

- [ ] **Step 3: Add `score_kind` to `RetrievedChunk`**

In `rag/chunking.py`, add the field to the `RetrievedChunk` dataclass, after `rank`:

```python
    score_kind: str = "cosine"
    """What `score` means: "cosine" from a plain search, "rrf" after fusion.

    Fused scores are around 0.03 where cosine scores are around 0.5, so a
    display that calls both "score" makes fusion look like a collapse in
    quality. The name travels with the number.
    """
```

- [ ] **Step 4: Add depth, strategy and translation to `Trace`**

In `rag/trace.py`:

```python
@dataclass
class StageTiming:
    name: str
    ms: float
    depth: int = 0


@dataclass(frozen=True)
class TranslationStep:
    """One artifact a translation strategy produced, for display.

    `kind` is one of "query", "step_back", "hypothetical", "sub_question",
    "sub_answer". Keeping this generic means a new strategy does not need a
    new Trace field.
    """

    kind: str
    text: str
```

Add to `__all__`: `"TranslationStep"`.

Add the two fields to `Trace`, after `queries`:

```python
    strategy: str = "direct"
    translation: list[TranslationStep] = field(default_factory=list)
```

Track depth with a private counter and use it in `stage`:

```python
    _depth: int = field(default=0, repr=False, compare=False)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block and record it, whether or not the block raises.

        Nested stages record their depth so total_ms can count only the
        outermost ones — an outer stage already contains its children's time.
        """
        started = time.perf_counter()
        depth = self._depth
        self._depth = depth + 1
        try:
            yield
        finally:
            self._depth = depth
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.timings.append(StageTiming(name=name, ms=elapsed_ms, depth=depth))

    def add_translation(self, kind: str, text: str) -> None:
        """Record something a translation strategy produced."""
        self.translation.append(TranslationStep(kind=kind, text=text))
```

Change `total_ms`:

```python
    @property
    def total_ms(self) -> float:
        """Wall time, counting each top-level stage once.

        Nested stages are excluded because their time is already inside the
        stage that contains them.
        """
        return sum(t.ms for t in self.timings if t.depth == 0)
```

Extend `to_dict`:

```python
            "strategy": self.strategy,
            "translation": [
                {"kind": s.kind, "text": s.text} for s in self.translation
            ],
            "timings": [
                {"name": t.name, "ms": t.ms, "depth": t.depth} for t in self.timings
            ],
```

Also add `"score_kind": r.score_kind` to each entry in the `retrieved` list.

- [ ] **Step 5: Label the score in `format_context`**

In `rag/prompts.py`, change the block header line inside `format_context` so it uses the score's own name:

```python
        blocks.append(
            f"[{number}] source: {item.chunk.doc_id}, chunk {item.chunk.index}, "
            f"{item.score_kind} {item.score:.3f}\n{item.chunk.text}"
        )
```

- [ ] **Step 6: Run the tests and verify they pass**

Run: `python -m pytest -q`
Expected: all pass. Existing tests that assert `"score 0.910"` in the context will now fail — update those assertions to `"cosine 0.910"`. That is a deliberate display change, not a regression.

- [ ] **Step 7: Commit**

```bash
git add rag/trace.py rag/chunking.py rag/prompts.py tests/test_trace.py tests/test_prompts.py
git commit -m "feat: stage depth, score labels, and translation steps on Trace

Nested stages would double-count in total_ms, which decomposition makes
real. Fused scores read as a quality collapse when labelled the same as
cosine scores. Both fixed before the strategies land.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Reciprocal rank fusion and best-score merging

**Files:**
- Modify: `rag/similarity.py`
- Test: `tests/test_similarity.py`

**Interfaces:**
- Consumes: `rag.chunking.RetrievedChunk`
- Produces:
  - `reciprocal_rank_fusion(result_lists: list[list[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]` — fused, re-ranked from 1, `score_kind="rrf"`
  - `merge_best_score(result_lists: list[list[RetrievedChunk]]) -> list[RetrievedChunk]` — dedupe by `chunk_id` keeping the highest cosine score, re-ranked from 1, `score_kind="cosine"`

The `k=60` constant is from the original RRF paper (Cormack et al., 2009). It damps the difference between top ranks so one list cannot dominate.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_similarity.py`:

```python
from rag.chunking import Chunk, RetrievedChunk
from rag.similarity import merge_best_score, reciprocal_rank_fusion


def _rc(chunk_id: str, score: float, rank: int) -> RetrievedChunk:
    doc, _, index = chunk_id.partition(":")
    chunk = Chunk(chunk_id, doc, int(index), f"text {chunk_id}", 0, 5, 0, 6)
    return RetrievedChunk(chunk=chunk, score=score, rank=rank)


# --- reciprocal rank fusion --------------------------------------------------

def test_rrf_scores_match_the_hand_computed_formula():
    # a is rank 1 in list one and rank 2 in list two:
    #   1/(60+1) + 1/(60+2) = 0.016393... + 0.016129... = 0.032522...
    # b is rank 2 in list one only: 1/(60+2) = 0.016129...
    lists = [[_rc("d:0", 0.9, 1), _rc("d:1", 0.8, 2)], [_rc("d:2", 0.7, 1), _rc("d:0", 0.6, 2)]]
    fused = reciprocal_rank_fusion(lists)
    by_id = {r.chunk.chunk_id: r.score for r in fused}
    assert by_id["d:0"] == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["d:1"] == pytest.approx(1 / 62)
    assert by_id["d:2"] == pytest.approx(1 / 61)


def test_rrf_ranks_a_chunk_found_by_two_queries_above_one_found_by_one():
    lists = [[_rc("d:0", 0.5, 1), _rc("d:1", 0.4, 2)], [_rc("d:0", 0.3, 2), _rc("d:2", 0.9, 1)]]
    fused = reciprocal_rank_fusion(lists)
    assert fused[0].chunk.chunk_id == "d:0"


def test_rrf_ignores_the_original_similarity_scores():
    # Rank is all that matters: a chunk with a poor cosine score that ranks
    # first in two lists must beat one with a great score in a single list.
    lists = [[_rc("d:0", 0.01, 1)], [_rc("d:0", 0.01, 1)], [_rc("d:9", 0.99, 1)]]
    fused = reciprocal_rank_fusion(lists)
    assert fused[0].chunk.chunk_id == "d:0"


def test_rrf_reranks_from_one_and_is_descending():
    lists = [[_rc("d:0", 0.9, 1), _rc("d:1", 0.8, 2), _rc("d:2", 0.7, 3)]]
    fused = reciprocal_rank_fusion(lists)
    assert [r.rank for r in fused] == [1, 2, 3]
    assert [r.score for r in fused] == sorted((r.score for r in fused), reverse=True)


def test_rrf_marks_its_scores_as_rrf():
    fused = reciprocal_rank_fusion([[_rc("d:0", 0.9, 1)]])
    assert fused[0].score_kind == "rrf"


def test_rrf_deduplicates_by_chunk_id():
    lists = [[_rc("d:0", 0.9, 1)], [_rc("d:0", 0.8, 1)], [_rc("d:0", 0.7, 1)]]
    assert len(reciprocal_rank_fusion(lists)) == 1


def test_rrf_of_no_lists_is_empty():
    assert reciprocal_rank_fusion([]) == []


def test_rrf_of_empty_lists_is_empty():
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_k_constant_is_configurable():
    lists = [[_rc("d:0", 0.9, 1)]]
    assert reciprocal_rank_fusion(lists, k=0)[0].score == pytest.approx(1.0)


def test_rrf_ties_break_deterministically_by_chunk_id():
    # Two chunks at identical rank in identical lists must come back in the
    # same order on every run, or Phase 3's benchmark drifts for no reason.
    lists = [[_rc("d:1", 0.5, 1)], [_rc("d:0", 0.5, 1)]]
    assert [r.chunk.chunk_id for r in reciprocal_rank_fusion(lists)] == ["d:0", "d:1"]


# --- best-score merge --------------------------------------------------------

def test_merge_keeps_the_highest_score_for_a_repeated_chunk():
    lists = [[_rc("d:0", 0.4, 1)], [_rc("d:0", 0.9, 1)]]
    merged = merge_best_score(lists)
    assert len(merged) == 1
    assert merged[0].score == pytest.approx(0.9)


def test_merge_sorts_by_score_and_reranks_from_one():
    lists = [[_rc("d:0", 0.4, 1), _rc("d:1", 0.9, 2)]]
    merged = merge_best_score(lists)
    assert [r.chunk.chunk_id for r in merged] == ["d:1", "d:0"]
    assert [r.rank for r in merged] == [1, 2]


def test_merge_keeps_scores_labelled_cosine():
    assert merge_best_score([[_rc("d:0", 0.4, 1)]])[0].score_kind == "cosine"


def test_merge_of_nothing_is_empty():
    assert merge_best_score([]) == []


def test_merge_ties_break_deterministically_by_chunk_id():
    lists = [[_rc("d:1", 0.5, 1), _rc("d:0", 0.5, 2)]]
    assert [r.chunk.chunk_id for r in merge_best_score(lists)] == ["d:0", "d:1"]
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_similarity.py -q`
Expected: `ImportError: cannot import name 'reciprocal_rank_fusion'`.

- [ ] **Step 3: Implement both functions**

Append to `rag/similarity.py`:

```python
from dataclasses import replace

from rag.chunking import RetrievedChunk

RRF_K = 60
"""Damping constant from the original reciprocal rank fusion paper.

It flattens the gap between top ranks, so a chunk that several queries rank
highly beats one that a single query ranks first. Lower k sharpens the
advantage of rank 1; higher k makes agreement across lists matter more.
"""


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]], k: int = RRF_K
) -> list[RetrievedChunk]:
    """Fuse ranked lists into one, scoring each chunk by sum of 1/(k + rank).

    Only ranks are used; the original similarity scores are discarded. That is
    the point of the method — it combines lists that are not on a common scale.
    Results are re-ranked from 1 and marked `score_kind="rrf"`.
    """
    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for item in results:
            chunk_id = item.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + item.rank)
            best.setdefault(chunk_id, item)

    # Sort by descending score, then by chunk_id so ties are reproducible.
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return [
        replace(best[chunk_id], score=score, rank=rank, score_kind="rrf")
        for rank, (chunk_id, score) in enumerate(ordered, start=1)
    ]


def merge_best_score(
    result_lists: list[list[RetrievedChunk]],
) -> list[RetrievedChunk]:
    """Union several result lists, keeping each chunk's best cosine score.

    Used where the lists come from the same embedding space and the scores are
    therefore comparable — unlike fusion, which deliberately ignores them.
    """
    best: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for item in results:
            chunk_id = item.chunk.chunk_id
            if chunk_id not in best or item.score > best[chunk_id].score:
                best[chunk_id] = item

    ordered = sorted(best.values(), key=lambda r: (-r.score, r.chunk.chunk_id))
    return [
        replace(item, rank=rank, score_kind="cosine")
        for rank, item in enumerate(ordered, start=1)
    ]
```

`dataclasses.replace` is used to copy a result with a new rank and score. `RetrievedChunk` is a plain (not frozen) dataclass, which `replace` handles fine; copying rather than mutating keeps the input lists untouched, which matters because fusion reads the same lists more than once.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_similarity.py -q`
Expected: 16 original plus 15 new pass.

- [ ] **Step 5: Commit**

```bash
git add rag/similarity.py tests/test_similarity.py
git commit -m "feat: reciprocal rank fusion and best-score merging

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Strategy protocol, context, and the direct strategy

**Files:**
- Create: `rag/strategies/__init__.py`, `rag/strategies/base.py`, `rag/strategies/direct.py`
- Modify: `rag/pipeline.py`
- Test: `tests/test_strategies.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `VectorStore.search`, `Trace`, `Config`, `rag.llm.LLMError`
- Produces:
  - `StrategyContext` — dataclass: `store`, `embedder`, `llm`, `config`, `trace`; method `search(queries: list[str], k: int) -> list[list[RetrievedChunk]]`
  - `StrategyResult` — dataclass: `retrieved: list[RetrievedChunk]`, `extra_context: str | None = None`
  - `Strategy` — Protocol with `name: str` and `run(question: str, ctx: StrategyContext) -> StrategyResult`
  - `degrade_to_direct(question: str, ctx: StrategyContext, reason: str) -> StrategyResult`
  - `DirectStrategy`
  - `rag.strategies.get_strategy(name: str, **options) -> Strategy` and `STRATEGY_NAMES: tuple[str, ...]`
  - `pipeline.ask(..., strategy: str = "direct", strategy_options: dict | None = None)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_strategies.py`:

```python
import pytest

from rag.chunking import chunk_documents
from rag.config import Config
from rag.llm import LLMError
from rag.loader import load_documents
from rag.store import VectorStore
from rag.strategies import STRATEGY_NAMES, get_strategy
from rag.strategies.base import StrategyContext, StrategyResult, degrade_to_direct
from rag.trace import Trace
from tests.conftest import FakeEmbedder, FakeLLM


def build_context(tiny_corpus: Config, llm=None) -> StrategyContext:
    embedder = FakeEmbedder()
    documents = load_documents(tiny_corpus.corpus_dir, tiny_corpus.metadata_path)
    chunks = chunk_documents(
        documents, embedder.tokenizer, tiny_corpus.chunk_tokens, tiny_corpus.chunk_overlap
    )
    store = VectorStore(
        vectors=embedder.encode([c.text for c in chunks]), chunks=chunks
    )
    return StrategyContext(
        store=store,
        embedder=embedder,
        llm=llm,
        config=tiny_corpus,
        trace=Trace(question="q"),
    )


class FailingLLM:
    def generate(self, prompt: str) -> str:
        raise LLMError("rate limited")


# --- context -----------------------------------------------------------------

def test_context_search_returns_one_list_per_query(tiny_corpus: Config):
    ctx = build_context(tiny_corpus)
    results = ctx.search(["alpha", "beta"], k=2)
    assert len(results) == 2
    assert all(len(r) == 2 for r in results)


def test_context_search_times_embed_and_search(tiny_corpus: Config):
    ctx = build_context(tiny_corpus)
    ctx.search(["alpha"], k=1)
    assert [t.name for t in ctx.trace.timings] == ["embed", "search"]


def test_context_search_of_no_queries_returns_nothing(tiny_corpus: Config):
    assert build_context(tiny_corpus).search([], k=3) == []


# --- direct strategy ---------------------------------------------------------

def test_direct_retrieves_top_k(tiny_corpus: Config):
    ctx = build_context(tiny_corpus)
    result = get_strategy("direct").run("what is cosine?", ctx)
    assert len(result.retrieved) == tiny_corpus.top_k
    assert result.extra_context is None


def test_direct_records_the_question_as_the_only_query(tiny_corpus: Config):
    ctx = build_context(tiny_corpus)
    get_strategy("direct").run("q", ctx)
    assert ctx.trace.queries == ["q"]


def test_direct_needs_no_llm(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=None)
    assert get_strategy("direct").run("q", ctx).retrieved


# --- degradation helper ------------------------------------------------------

def test_degrade_falls_back_to_direct_retrieval(tiny_corpus: Config):
    ctx = build_context(tiny_corpus)
    result = degrade_to_direct("q", ctx, "multi-query rewrite failed")
    assert len(result.retrieved) == tiny_corpus.top_k


def test_degrade_records_the_reason_on_the_trace(tiny_corpus: Config):
    ctx = build_context(tiny_corpus)
    degrade_to_direct("q", ctx, "multi-query rewrite failed")
    assert any("multi-query rewrite failed" in n for n in ctx.trace.notes)
    assert any("direct retrieval" in n for n in ctx.trace.notes)


# --- registry ----------------------------------------------------------------

def test_registry_exposes_every_strategy_name():
    assert "direct" in STRATEGY_NAMES


def test_unknown_strategy_is_rejected_by_name():
    with pytest.raises(ValueError, match="nope"):
        get_strategy("nope")
```

Append to `tests/test_pipeline.py`:

```python
def test_ask_defaults_to_the_direct_strategy(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus)
    assert trace.strategy == "direct"


def test_ask_records_the_strategy_it_used(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    trace = ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, strategy="direct")
    assert trace.strategy == "direct"


def test_ask_rejects_an_unknown_strategy(tiny_corpus: Config):
    store = build_index(tiny_corpus, FakeEmbedder())
    with pytest.raises(ValueError, match="nope"):
        ask("q", store, FakeEmbedder(), FakeLLM(), tiny_corpus, strategy="nope")
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_strategies.py -q`
Expected: `ModuleNotFoundError: No module named 'rag.strategies'`.

- [ ] **Step 3: Implement `rag/strategies/base.py`**

```python
"""The Strategy protocol and the context strategies work through.

A strategy turns one question into retrieved chunks. It never touches the
store or the embedder directly — it calls `ctx.search()`, which owns the embed
and search timing, so every strategy's trace is shaped the same way and the
CLI and benchmark can read them uniformly.

Strategies that need the LLM must degrade rather than fail: retrieval usually
still works without the rewrite, and an answer from plain retrieval beats no
answer. `degrade_to_direct` records why, because a silent downgrade would make
a strategy look like it ran when it did not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rag.chunking import RetrievedChunk
from rag.config import Config
from rag.store import VectorStore
from rag.trace import Trace


@dataclass
class StrategyContext:
    store: VectorStore
    embedder: object
    llm: object | None
    config: Config
    trace: Trace

    def search(self, queries: list[str], k: int) -> list[list[RetrievedChunk]]:
        """Embed queries and retrieve k chunks for each, timing both stages."""
        if not queries:
            return []
        with self.trace.stage("embed"):
            vectors = self.embedder.encode(queries)
        with self.trace.stage("search"):
            return self.store.search(vectors, k)


@dataclass
class StrategyResult:
    retrieved: list[RetrievedChunk]
    extra_context: str | None = None
    """Text the strategy produced that belongs in the answer prompt.

    Only decomposition uses it, to carry sub-question/sub-answer pairs. It is
    kept separate from `retrieved` because it is generated text, not a
    retrieved excerpt, and must not be presented to the model as a source.
    """


class Strategy(Protocol):
    name: str

    def run(self, question: str, ctx: StrategyContext) -> StrategyResult: ...


def degrade_to_direct(
    question: str, ctx: StrategyContext, reason: str
) -> StrategyResult:
    """Fall back to plain retrieval, recording why on the trace."""
    ctx.trace.note(f"{reason}; degraded to direct retrieval")
    ctx.trace.queries = [question]
    return StrategyResult(retrieved=ctx.search([question], ctx.config.top_k)[0])
```

- [ ] **Step 4: Implement `rag/strategies/direct.py`**

```python
"""Plain top-k retrieval: the baseline every other strategy is measured against."""

from __future__ import annotations

from rag.strategies.base import StrategyContext, StrategyResult


class DirectStrategy:
    name = "direct"

    def run(self, question: str, ctx: StrategyContext) -> StrategyResult:
        ctx.trace.queries = [question]
        return StrategyResult(
            retrieved=ctx.search([question], ctx.config.top_k)[0]
        )
```

- [ ] **Step 5: Implement the registry in `rag/strategies/__init__.py`**

```python
"""Strategy registry.

One place that maps a CLI name to a strategy, so adding a strategy does not
mean touching the pipeline or the CLI.
"""

from __future__ import annotations

from rag.strategies.base import (
    Strategy,
    StrategyContext,
    StrategyResult,
    degrade_to_direct,
)
from rag.strategies.direct import DirectStrategy

__all__ = [
    "Strategy",
    "StrategyContext",
    "StrategyResult",
    "degrade_to_direct",
    "STRATEGY_NAMES",
    "get_strategy",
]

_REGISTRY: dict[str, type] = {
    DirectStrategy.name: DirectStrategy,
}

STRATEGY_NAMES: tuple[str, ...] = tuple(_REGISTRY)


def get_strategy(name: str, **options) -> Strategy:
    """Build a strategy by name. Unknown names raise rather than defaulting."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown strategy: {name} (choose one of {', '.join(sorted(_REGISTRY))})"
        ) from None
    return cls(**options)
```

- [ ] **Step 6: Route `pipeline.ask` through a strategy**

Replace the body of `ask` in `rag/pipeline.py`:

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
) -> Trace:
    """Answer one question. Pass llm=None to retrieve without generating.

    The named strategy decides what to retrieve; everything after that is the
    same for all of them.
    """
    trace = Trace(question=question)
    trace.strategy = strategy
    effective_k = config.top_k if k is None else k

    chosen = get_strategy(strategy, **(strategy_options or {}))
    ctx = StrategyContext(
        store=store,
        embedder=embedder,
        llm=llm,
        config=replace(config, top_k=effective_k),
        trace=trace,
    )
    result = chosen.run(question, ctx)
    trace.retrieved = result.retrieved

    if llm is None:
        trace.note("retrieval only: no LLM configured")
        return trace

    generate_answer(
        llm, question, result.retrieved, trace, extra_context=result.extra_context
    )
    return trace
```

Add the imports `from dataclasses import replace`, `from rag.strategies import get_strategy`, `from rag.strategies.base import StrategyContext`.

Note the `replace(config, top_k=effective_k)` — strategies read `ctx.config.top_k` rather than taking `k` separately, so `--k` has to reach them through the config. `Config` is frozen, so `replace` is the way.

- [ ] **Step 7: Extend `generate_answer` to accept extra context**

In `rag/generation.py`, add the parameter and pass it through:

```python
def generate_answer(
    llm,
    question: str,
    retrieved: list[RetrievedChunk],
    trace: Trace,
    extra_context: str | None = None,
) -> str | None:
```

and change the prompt call to `build_answer_prompt(question, retrieved, extra_context=extra_context)`.

In `rag/prompts.py`, extend `build_answer_prompt`:

```python
def build_answer_prompt(
    question: str,
    retrieved: list[RetrievedChunk],
    extra_context: str | None = None,
) -> str:
    """Fill the answer template.

    `extra_context` holds text the strategy derived (decomposition's
    sub-answers). It is labelled separately from the retrieved excerpts so the
    model does not cite generated text as though it were a source.
    """
    context = format_context(retrieved)
    if extra_context:
        context = (
            f"{context}\n\n"
            f"Working notes (derived from the excerpts above, not a source — "
            f"do not cite these):\n{extra_context}"
        )
    return ANSWER_TEMPLATE.format(context=context, question=question)
```

- [ ] **Step 8: Run the tests and verify they pass**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add rag/strategies tests/test_strategies.py rag/pipeline.py rag/generation.py rag/prompts.py tests/test_pipeline.py
git commit -m "feat: strategy protocol, context, and direct strategy

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Multi-query

**Files:**
- Create: `rag/strategies/multi_query.py`
- Modify: `rag/prompts.py`, `rag/strategies/__init__.py`
- Test: `tests/test_strategies.py`

**Interfaces:**
- Consumes: `StrategyContext`, `StrategyResult`, `degrade_to_direct`, `merge_best_score`, `LLMError`
- Produces:
  - `MULTI_QUERY_TEMPLATE` in `rag/prompts.py`
  - `parse_query_list(raw: str) -> list[str]` in `rag/prompts.py`
  - `MultiQueryStrategy(n: int = 5)` registered as `"multi-query"`

`RAG.pdf`: rewrite the question from several perspectives, so at least one rewrite lands nearer the right documents than the original did.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_strategies.py`:

```python
MULTI_QUERY_REPLY = """1. How does cosine similarity work?
2. What does the angle between vectors measure?
3. Why is magnitude ignored in cosine similarity?"""


def test_multi_query_sends_the_question_to_the_llm(tiny_corpus: Config):
    llm = FakeLLM(MULTI_QUERY_REPLY)
    ctx = build_context(tiny_corpus, llm=llm)
    get_strategy("multi-query").run("what is cosine?", ctx)
    assert "what is cosine?" in llm.prompts[0]


def test_multi_query_records_the_original_plus_the_rewrites(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    get_strategy("multi-query").run("what is cosine?", ctx)
    assert ctx.trace.queries[0] == "what is cosine?"
    assert "How does cosine similarity work?" in ctx.trace.queries
    assert len(ctx.trace.queries) == 4


def test_multi_query_records_each_rewrite_as_a_translation_step(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    get_strategy("multi-query").run("q", ctx)
    kinds = [s.kind for s in ctx.trace.translation]
    assert kinds == ["query", "query", "query"]


def test_multi_query_returns_at_most_top_k(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    result = get_strategy("multi-query").run("q", ctx)
    assert len(result.retrieved) <= tiny_corpus.top_k


def test_multi_query_deduplicates_chunks_found_by_several_queries(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    result = get_strategy("multi-query").run("q", ctx)
    ids = [r.chunk.chunk_id for r in result.retrieved]
    assert len(ids) == len(set(ids))


def test_multi_query_results_are_ranked_from_one(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    result = get_strategy("multi-query").run("q", ctx)
    assert [r.rank for r in result.retrieved] == list(range(1, len(result.retrieved) + 1))


def test_multi_query_degrades_when_the_llm_fails(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FailingLLM())
    result = get_strategy("multi-query").run("q", ctx)
    assert result.retrieved
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)


def test_multi_query_degrades_when_there_is_no_llm(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=None)
    result = get_strategy("multi-query").run("q", ctx)
    assert result.retrieved
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)


def test_multi_query_degrades_when_the_llm_returns_nothing_usable(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM("   "))
    result = get_strategy("multi-query").run("q", ctx)
    assert result.retrieved
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)
```

Append to `tests/test_prompts.py`:

```python
def test_parse_query_list_strips_numbering():
    from rag.prompts import parse_query_list

    raw = "1. first question\n2. second question\n3. third question"
    assert parse_query_list(raw) == [
        "first question",
        "second question",
        "third question",
    ]


def test_parse_query_list_handles_bullets_and_blank_lines():
    from rag.prompts import parse_query_list

    raw = "- first\n\n* second\n\n\n  third  \n"
    assert parse_query_list(raw) == ["first", "second", "third"]


def test_parse_query_list_drops_duplicates_preserving_order():
    from rag.prompts import parse_query_list

    assert parse_query_list("1. same\n2. same\n3. other") == ["same", "other"]


def test_parse_query_list_of_blank_input_is_empty():
    from rag.prompts import parse_query_list

    assert parse_query_list("   \n\n ") == []
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_strategies.py tests/test_prompts.py -q`
Expected: `ValueError: unknown strategy: multi-query`, and `ImportError` for `parse_query_list`.

- [ ] **Step 3: Add the template and parser to `rag/prompts.py`**

```python
MULTI_QUERY_TEMPLATE = """You are helping a search system find relevant documents.

Rewrite the question below into {n} alternative search queries. Each should
approach the same information need from a different angle — different
vocabulary, a broader or narrower framing, or an underlying concept the
question implies. The goal is that at least one rewrite matches wording the
documents actually use.

Reply with one query per line, numbered. No other text.

Question: {question}"""


def parse_query_list(raw: str) -> list[str]:
    """Extract one query per line from a numbered or bulleted model reply.

    Models drift between "1." and "-" and occasionally repeat themselves, so
    the parsing is forgiving and deduplicates while preserving order.
    """
    queries: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        text = line.strip()
        if not text:
            continue
        text = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", text).strip()
        if text and text not in seen:
            seen.add(text)
            queries.append(text)
    return queries
```

Add `import re` at the top of `rag/prompts.py`.

- [ ] **Step 4: Implement `rag/strategies/multi_query.py`**

```python
"""Multi-query: ask the same thing several ways, union the results.

A question and the documents that answer it often use different words. Several
rewrites give several chances for one of them to land near the right chunks.
Results are merged on best cosine score, since every rewrite is embedded in
the same space and the scores are directly comparable.
"""

from __future__ import annotations

from rag.llm import LLMError
from rag.prompts import MULTI_QUERY_TEMPLATE, parse_query_list
from rag.similarity import merge_best_score
from rag.strategies.base import StrategyContext, StrategyResult, degrade_to_direct


class MultiQueryStrategy:
    name = "multi-query"

    def __init__(self, n: int = 5) -> None:
        self.n = n

    def run(self, question: str, ctx: StrategyContext) -> StrategyResult:
        if ctx.llm is None:
            return degrade_to_direct(question, ctx, "multi-query needs an LLM")

        try:
            with ctx.trace.stage("translate"):
                raw = ctx.llm.generate(
                    MULTI_QUERY_TEMPLATE.format(question=question, n=self.n)
                )
        except LLMError as exc:
            return degrade_to_direct(question, ctx, f"multi-query rewrite failed: {exc}")

        rewrites = [q for q in parse_query_list(raw) if q != question]
        if not rewrites:
            return degrade_to_direct(
                question, ctx, "multi-query rewrite produced no usable queries"
            )

        for rewrite in rewrites:
            ctx.trace.add_translation("query", rewrite)

        queries = [question, *rewrites]
        ctx.trace.queries = queries
        lists = ctx.search(queries, ctx.config.top_k)
        return StrategyResult(retrieved=merge_best_score(lists)[: ctx.config.top_k])
```

- [ ] **Step 5: Register it**

In `rag/strategies/__init__.py`, import `MultiQueryStrategy` and add it to `_REGISTRY`:

```python
from rag.strategies.multi_query import MultiQueryStrategy

_REGISTRY: dict[str, type] = {
    DirectStrategy.name: DirectStrategy,
    MultiQueryStrategy.name: MultiQueryStrategy,
}
```

- [ ] **Step 6: Run the tests and verify they pass**

Run: `python -m pytest -q`

- [ ] **Step 7: Commit**

```bash
git add rag/strategies tests/test_strategies.py rag/prompts.py tests/test_prompts.py
git commit -m "feat: multi-query strategy

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: RAG-Fusion

**Files:**
- Create: `rag/strategies/rag_fusion.py`
- Modify: `rag/strategies/__init__.py`
- Test: `tests/test_strategies.py`

**Interfaces:**
- Consumes: the same rewrite machinery as Task 4, plus `reciprocal_rank_fusion`
- Produces: `RagFusionStrategy(n: int = 5)` registered as `"rag-fusion"`

`RAG.pdf`: identical to multi-query except results are fused by reciprocal rank rather than unioned by score. The comparison between the two is the interesting part, which is why they are separate strategies rather than a flag.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_strategies.py`:

```python
def test_rag_fusion_scores_are_labelled_rrf(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    result = get_strategy("rag-fusion").run("q", ctx)
    assert all(r.score_kind == "rrf" for r in result.retrieved)


def test_rag_fusion_scores_differ_from_multi_query_scores(tiny_corpus: Config):
    # Same rewrites, same corpus: if fusion returned cosine scores it would be
    # multi-query wearing a different name.
    fusion_ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    fusion = get_strategy("rag-fusion").run("q", fusion_ctx)
    merge_ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    merged = get_strategy("multi-query").run("q", merge_ctx)
    assert fusion.retrieved[0].score != merged.retrieved[0].score


def test_rag_fusion_returns_at_most_top_k(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    assert len(get_strategy("rag-fusion").run("q", ctx).retrieved) <= tiny_corpus.top_k


def test_rag_fusion_reranks_from_one(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(MULTI_QUERY_REPLY))
    result = get_strategy("rag-fusion").run("q", ctx)
    assert [r.rank for r in result.retrieved] == list(range(1, len(result.retrieved) + 1))


def test_rag_fusion_degrades_when_the_llm_fails(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FailingLLM())
    result = get_strategy("rag-fusion").run("q", ctx)
    assert result.retrieved
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)


def test_rag_fusion_degrades_when_there_is_no_llm(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=None)
    result = get_strategy("rag-fusion").run("q", ctx)
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_strategies.py -q`
Expected: `ValueError: unknown strategy: rag-fusion`.

- [ ] **Step 3: Implement `rag/strategies/rag_fusion.py`**

```python
"""RAG-Fusion: multi-query, then fuse the ranked lists by reciprocal rank.

The difference from multi-query is the combination step, and it matters
because the lists are not on a shared scale in any meaningful sense — a 0.6
from one rewrite is not the same evidence as a 0.6 from another. Reciprocal
rank fusion throws the scores away and rewards agreement on rank instead.
"""

from __future__ import annotations

from rag.llm import LLMError
from rag.prompts import MULTI_QUERY_TEMPLATE, parse_query_list
from rag.similarity import reciprocal_rank_fusion
from rag.strategies.base import StrategyContext, StrategyResult, degrade_to_direct


class RagFusionStrategy:
    name = "rag-fusion"

    def __init__(self, n: int = 5) -> None:
        self.n = n

    def run(self, question: str, ctx: StrategyContext) -> StrategyResult:
        if ctx.llm is None:
            return degrade_to_direct(question, ctx, "rag-fusion needs an LLM")

        try:
            with ctx.trace.stage("translate"):
                raw = ctx.llm.generate(
                    MULTI_QUERY_TEMPLATE.format(question=question, n=self.n)
                )
        except LLMError as exc:
            return degrade_to_direct(question, ctx, f"rag-fusion rewrite failed: {exc}")

        rewrites = [q for q in parse_query_list(raw) if q != question]
        if not rewrites:
            return degrade_to_direct(
                question, ctx, "rag-fusion rewrite produced no usable queries"
            )

        for rewrite in rewrites:
            ctx.trace.add_translation("query", rewrite)

        queries = [question, *rewrites]
        ctx.trace.queries = queries
        lists = ctx.search(queries, ctx.config.top_k)
        return StrategyResult(
            retrieved=reciprocal_rank_fusion(lists)[: ctx.config.top_k]
        )
```

The rewrite block is duplicated from `multi_query.py` rather than shared. That is deliberate at this size: the two strategies are meant to be read side by side and compared, and a shared helper would hide the single line that differs. If a third strategy needs the same block, extract it then.

- [ ] **Step 4: Register it**

Add `RagFusionStrategy` to the imports and `_REGISTRY` in `rag/strategies/__init__.py`.

- [ ] **Step 5: Run the tests and verify they pass**

Run: `python -m pytest -q`

- [ ] **Step 6: Commit**

```bash
git add rag/strategies tests/test_strategies.py
git commit -m "feat: RAG-Fusion strategy

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Step-back

**Files:**
- Create: `rag/strategies/step_back.py`
- Modify: `rag/prompts.py`, `rag/strategies/__init__.py`
- Test: `tests/test_strategies.py`

**Interfaces:**
- Produces: `STEP_BACK_TEMPLATE` in `rag/prompts.py`; `StepBackStrategy` registered as `"step-back"`

`RAG.pdf`: the opposite of decomposition. Generate one *more general* question, retrieve for both it and the original, and combine. Background context often lives in passages a specific question does not match.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_strategies.py`:

```python
def test_step_back_asks_for_a_more_general_question(tiny_corpus: Config):
    llm = FakeLLM("What is vector similarity?")
    ctx = build_context(tiny_corpus, llm=llm)
    get_strategy("step-back").run("how does cosine handle magnitude?", ctx)
    assert "how does cosine handle magnitude?" in llm.prompts[0]


def test_step_back_searches_both_questions(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM("What is vector similarity?"))
    get_strategy("step-back").run("specific question", ctx)
    assert ctx.trace.queries == ["specific question", "What is vector similarity?"]


def test_step_back_records_the_general_question_as_a_translation_step(
    tiny_corpus: Config,
):
    ctx = build_context(tiny_corpus, llm=FakeLLM("What is vector similarity?"))
    get_strategy("step-back").run("q", ctx)
    assert [(s.kind, s.text) for s in ctx.trace.translation] == [
        ("step_back", "What is vector similarity?")
    ]


def test_step_back_returns_at_most_top_k_deduplicated(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM("general question"))
    result = get_strategy("step-back").run("q", ctx)
    ids = [r.chunk.chunk_id for r in result.retrieved]
    assert len(ids) == len(set(ids))
    assert len(ids) <= tiny_corpus.top_k


def test_step_back_degrades_when_the_llm_fails(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FailingLLM())
    result = get_strategy("step-back").run("q", ctx)
    assert result.retrieved
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)


def test_step_back_degrades_when_the_general_question_is_blank(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM("   "))
    result = get_strategy("step-back").run("q", ctx)
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_strategies.py -q`
Expected: `ValueError: unknown strategy: step-back`.

- [ ] **Step 3: Add the template to `rag/prompts.py`**

```python
STEP_BACK_TEMPLATE = """Given a specific question, write one more general
question about the underlying concept or principle it depends on.

The general question should be broad enough that a document explaining the
background would answer it, while staying on the same subject. Do not answer
either question.

Reply with the general question only, on one line.

Specific question: {question}"""
```

- [ ] **Step 4: Implement `rag/strategies/step_back.py`**

```python
"""Step-back: also retrieve for a more general form of the question.

A narrow question matches narrow passages. The background a good answer needs
is often in a passage that explains the concept rather than the specific case,
and that passage does not match the narrow wording. Asking the broader
question too pulls it in.

Results are merged on best cosine score: both queries are embedded in the same
space, so the scores are comparable.
"""

from __future__ import annotations

from rag.llm import LLMError
from rag.prompts import STEP_BACK_TEMPLATE
from rag.similarity import merge_best_score
from rag.strategies.base import StrategyContext, StrategyResult, degrade_to_direct


class StepBackStrategy:
    name = "step-back"

    def run(self, question: str, ctx: StrategyContext) -> StrategyResult:
        if ctx.llm is None:
            return degrade_to_direct(question, ctx, "step-back needs an LLM")

        try:
            with ctx.trace.stage("translate"):
                raw = ctx.llm.generate(STEP_BACK_TEMPLATE.format(question=question))
        except LLMError as exc:
            return degrade_to_direct(question, ctx, f"step-back failed: {exc}")

        general = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        if not general:
            return degrade_to_direct(
                question, ctx, "step-back produced no general question"
            )

        ctx.trace.add_translation("step_back", general)
        queries = [question, general]
        ctx.trace.queries = queries
        lists = ctx.search(queries, ctx.config.top_k)
        return StrategyResult(retrieved=merge_best_score(lists)[: ctx.config.top_k])
```

- [ ] **Step 5: Register it, run the tests, commit**

Add `StepBackStrategy` to `rag/strategies/__init__.py`.

Run: `python -m pytest -q`

```bash
git add rag/strategies tests/test_strategies.py rag/prompts.py
git commit -m "feat: step-back strategy

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: HyDE

**Files:**
- Create: `rag/strategies/hyde.py`
- Modify: `rag/prompts.py`, `rag/strategies/__init__.py`
- Test: `tests/test_strategies.py`

**Interfaces:**
- Produces: `HYDE_TEMPLATE` in `rag/prompts.py`; `HydeStrategy(include_question: bool = True)` registered as `"hyde"`

`RAG.pdf`: a question is short and a document is dense, so they sit apart in embedding space even when the document answers the question. HyDE has the model write a *hypothetical* answer document and embeds that instead — document-shaped text lands nearer real documents.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_strategies.py`:

```python
HYDE_REPLY = (
    "Cosine similarity measures the angle between two vectors in an inner "
    "product space. Because it normalises for magnitude, two documents on the "
    "same topic score highly even when one is far longer than the other."
)


def test_hyde_generates_a_hypothetical_document(tiny_corpus: Config):
    llm = FakeLLM(HYDE_REPLY)
    ctx = build_context(tiny_corpus, llm=llm)
    get_strategy("hyde").run("what is cosine similarity?", ctx)
    assert "what is cosine similarity?" in llm.prompts[0]


def test_hyde_records_the_hypothetical_document(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(HYDE_REPLY))
    get_strategy("hyde").run("q", ctx)
    assert [(s.kind, s.text) for s in ctx.trace.translation] == [
        ("hypothetical", HYDE_REPLY)
    ]


def test_hyde_searches_on_the_hypothetical_document(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(HYDE_REPLY))
    get_strategy("hyde").run("q", ctx)
    assert HYDE_REPLY in ctx.trace.queries


def test_hyde_also_searches_the_original_question_by_default(tiny_corpus: Config):
    # The hypothetical document can be wrong. Keeping the original question in
    # the mix means a bad hallucination degrades the result rather than
    # replacing it.
    ctx = build_context(tiny_corpus, llm=FakeLLM(HYDE_REPLY))
    get_strategy("hyde").run("q", ctx)
    assert "q" in ctx.trace.queries


def test_hyde_can_search_the_hypothetical_document_alone(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(HYDE_REPLY))
    get_strategy("hyde", include_question=False).run("q", ctx)
    assert ctx.trace.queries == [HYDE_REPLY]


def test_hyde_returns_at_most_top_k(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM(HYDE_REPLY))
    assert len(get_strategy("hyde").run("q", ctx).retrieved) <= tiny_corpus.top_k


def test_hyde_degrades_when_the_llm_fails(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FailingLLM())
    result = get_strategy("hyde").run("q", ctx)
    assert result.retrieved
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)


def test_hyde_degrades_when_the_document_is_blank(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM("  \n "))
    result = get_strategy("hyde").run("q", ctx)
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_strategies.py -q`
Expected: `ValueError: unknown strategy: hyde`.

- [ ] **Step 3: Add the template to `rag/prompts.py`**

```python
HYDE_TEMPLATE = """Write a short passage that answers the question below, as
it might appear in a technical paper or reference document.

Write it as documentation, not as a reply: no preamble, no "the answer is",
just the passage. Being factually wrong is acceptable — this text is used to
search with, not to show anyone. Match the vocabulary and register a real
document on this subject would use.

Keep it under 120 words.

Question: {question}"""
```

The "being wrong is acceptable" instruction is deliberate and is the part people find surprising. The passage is never shown to the user and never reaches the answer prompt; only its embedding is used. What matters is that it *looks* like the documents being searched.

- [ ] **Step 4: Implement `rag/strategies/hyde.py`**

```python
"""HyDE: search with a hypothetical answer document instead of the question.

A question is short and interrogative; a document is long and declarative.
Embedded in the same space they land apart even when one answers the other.
HyDE closes that gap by having the model write a passage in the shape of a
document and searching with that.

The generated text is used only as a search key. It never enters the answer
prompt, so a hallucinated passage costs retrieval quality but cannot put false
statements in front of the user. By default the original question is searched
alongside it, so a bad hypothetical degrades the result rather than replacing
it.
"""

from __future__ import annotations

from rag.llm import LLMError
from rag.prompts import HYDE_TEMPLATE
from rag.similarity import merge_best_score
from rag.strategies.base import StrategyContext, StrategyResult, degrade_to_direct


class HydeStrategy:
    name = "hyde"

    def __init__(self, include_question: bool = True) -> None:
        self.include_question = include_question

    def run(self, question: str, ctx: StrategyContext) -> StrategyResult:
        if ctx.llm is None:
            return degrade_to_direct(question, ctx, "hyde needs an LLM")

        try:
            with ctx.trace.stage("translate"):
                raw = ctx.llm.generate(HYDE_TEMPLATE.format(question=question))
        except LLMError as exc:
            return degrade_to_direct(question, ctx, f"hyde generation failed: {exc}")

        document = raw.strip()
        if not document:
            return degrade_to_direct(
                question, ctx, "hyde produced an empty hypothetical document"
            )

        ctx.trace.add_translation("hypothetical", document)
        queries = [question, document] if self.include_question else [document]
        ctx.trace.queries = queries
        lists = ctx.search(queries, ctx.config.top_k)
        return StrategyResult(retrieved=merge_best_score(lists)[: ctx.config.top_k])
```

- [ ] **Step 5: Register it, run the tests, commit**

Add `HydeStrategy` to `rag/strategies/__init__.py`.

Run: `python -m pytest -q`

```bash
git add rag/strategies tests/test_strategies.py rag/prompts.py
git commit -m "feat: HyDE strategy

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Decomposition, recursive and independent

The largest task. `RAG.pdf` describes two forms, and both are implemented behind one `mode` flag because they share everything except how sub-answers combine.

**Files:**
- Create: `rag/strategies/decomposition.py`
- Modify: `rag/prompts.py`, `rag/strategies/__init__.py`
- Test: `tests/test_strategies.py`

**Interfaces:**
- Produces:
  - `DECOMPOSE_TEMPLATE`, `SUB_ANSWER_TEMPLATE` in `rag/prompts.py`
  - `DecompositionStrategy(mode: str = "recursive", max_sub_questions: int = 3)` registered as `"decomposition"`; `mode` is `"recursive"` (IR-CoT — each sub-answer feeds the next sub-question) or `"independent"` (all sub-questions answered separately, then concatenated)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_strategies.py`:

```python
class ScriptedLLM:
    """Returns canned replies in order, recording the prompts it received."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "fallback answer"


DECOMPOSE_REPLY = "1. What is a vector?\n2. How is similarity measured?"


def _scripted():
    return ScriptedLLM([DECOMPOSE_REPLY, "A vector is a list of numbers.",
                        "Similarity is the angle between them."])


def test_decomposition_records_each_sub_question(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=_scripted())
    get_strategy("decomposition").run("q", ctx)
    kinds = [s.kind for s in ctx.trace.translation]
    assert kinds.count("sub_question") == 2


def test_decomposition_records_each_sub_answer(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=_scripted())
    get_strategy("decomposition").run("q", ctx)
    kinds = [s.kind for s in ctx.trace.translation]
    assert kinds.count("sub_answer") == 2


def test_decomposition_returns_sub_answers_as_extra_context(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=_scripted())
    result = get_strategy("decomposition").run("q", ctx)
    assert "A vector is a list of numbers." in result.extra_context
    assert "What is a vector?" in result.extra_context


def test_recursive_mode_feeds_earlier_answers_into_later_sub_questions(
    tiny_corpus: Config,
):
    llm = _scripted()
    ctx = build_context(tiny_corpus, llm=llm)
    get_strategy("decomposition", mode="recursive").run("q", ctx)
    # prompts: [decompose, sub-answer 1, sub-answer 2]
    assert "A vector is a list of numbers." in llm.prompts[2]


def test_independent_mode_does_not_feed_earlier_answers_forward(
    tiny_corpus: Config,
):
    llm = _scripted()
    ctx = build_context(tiny_corpus, llm=llm)
    get_strategy("decomposition", mode="independent").run("q", ctx)
    assert "A vector is a list of numbers." not in llm.prompts[2]


def test_decomposition_retrieves_for_each_sub_question(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=_scripted())
    get_strategy("decomposition").run("q", ctx)
    assert [s.text for s in ctx.trace.translation if s.kind == "sub_question"] == [
        "What is a vector?",
        "How is similarity measured?",
    ]


def test_decomposition_nests_its_stages_so_total_ms_is_not_doubled(
    tiny_corpus: Config,
):
    ctx = build_context(tiny_corpus, llm=_scripted())
    get_strategy("decomposition").run("q", ctx)
    top_level = [t for t in ctx.trace.timings if t.depth == 0]
    assert [t.name for t in top_level] == ["decompose"]
    assert any(t.depth > 0 for t in ctx.trace.timings)


def test_decomposition_caps_the_number_of_sub_questions(tiny_corpus: Config):
    many = "\n".join(f"{i}. question {i}" for i in range(1, 9))
    llm = ScriptedLLM([many] + [f"answer {i}" for i in range(8)])
    ctx = build_context(tiny_corpus, llm=llm)
    get_strategy("decomposition", max_sub_questions=3).run("q", ctx)
    kinds = [s.kind for s in ctx.trace.translation]
    assert kinds.count("sub_question") == 3


def test_decomposition_returns_at_most_top_k(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=_scripted())
    result = get_strategy("decomposition").run("q", ctx)
    assert len(result.retrieved) <= tiny_corpus.top_k


def test_decomposition_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="sideways"):
        get_strategy("decomposition", mode="sideways")


def test_decomposition_degrades_when_the_llm_fails(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FailingLLM())
    result = get_strategy("decomposition").run("q", ctx)
    assert result.retrieved
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)


def test_decomposition_degrades_when_no_sub_questions_come_back(tiny_corpus: Config):
    ctx = build_context(tiny_corpus, llm=FakeLLM("   "))
    result = get_strategy("decomposition").run("q", ctx)
    assert any("degraded to direct retrieval" in n for n in ctx.trace.notes)


def test_a_failed_sub_answer_does_not_abort_the_whole_strategy(tiny_corpus: Config):
    class PartlyFailingLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                return DECOMPOSE_REPLY
            if self.calls == 2:
                raise LLMError("rate limited")
            return "second answer"

    ctx = build_context(tiny_corpus, llm=PartlyFailingLLM())
    result = get_strategy("decomposition").run("q", ctx)
    assert result.retrieved
    assert any("sub-question" in n for n in ctx.trace.notes)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_strategies.py -q`
Expected: `ValueError: unknown strategy: decomposition`.

- [ ] **Step 3: Add the templates to `rag/prompts.py`**

```python
DECOMPOSE_TEMPLATE = """Break the question below into at most {n} simpler
sub-questions that can each be looked up on their own.

Each sub-question must be answerable from documents independently of the
others, and answering all of them should be enough to answer the original.
If the question is already simple, reply with just the original question.

Reply with one sub-question per line, numbered. No other text.

Question: {question}"""


SUB_ANSWER_TEMPLATE = """Answer the sub-question using only the context below.
If the context does not answer it, say so in one sentence.

Be brief: two sentences at most. This answer is working material for a larger
question, not a final response.
{prior}
Context:
{context}

Sub-question: {question}

Answer:"""


PRIOR_ANSWERS_HEADER = """
Already established:
{prior}
"""
```

- [ ] **Step 4: Implement `rag/strategies/decomposition.py`**

```python
"""Decomposition: split the question, answer the parts, then answer the whole.

Two forms, as described in RAG.pdf:

- "recursive" (IR-CoT): each sub-answer is carried into the next sub-question's
  prompt, so later steps can build on earlier ones. Necessary when the parts
  are genuinely dependent — you cannot answer the second without the first.
- "independent": every sub-question is answered on its own and the pairs are
  concatenated at the end. Cheaper to reason about, and correct when the parts
  do not depend on each other.

Both return the sub-answers as `extra_context` rather than as retrieved chunks,
because they are generated text. The answer prompt labels them as working
notes so the model does not cite them as sources.
"""

from __future__ import annotations

from rag.llm import LLMError
from rag.prompts import (
    DECOMPOSE_TEMPLATE,
    PRIOR_ANSWERS_HEADER,
    SUB_ANSWER_TEMPLATE,
    format_context,
    parse_query_list,
)
from rag.similarity import merge_best_score
from rag.strategies.base import StrategyContext, StrategyResult, degrade_to_direct

MODES = ("recursive", "independent")


class DecompositionStrategy:
    name = "decomposition"

    def __init__(self, mode: str = "recursive", max_sub_questions: int = 3) -> None:
        if mode not in MODES:
            raise ValueError(
                f"unknown decomposition mode: {mode} (choose {' or '.join(MODES)})"
            )
        self.mode = mode
        self.max_sub_questions = max_sub_questions

    def run(self, question: str, ctx: StrategyContext) -> StrategyResult:
        if ctx.llm is None:
            return degrade_to_direct(question, ctx, "decomposition needs an LLM")

        # One outer stage wrapping everything, so total_ms counts this work
        # once rather than adding up each nested retrieval and LLM call.
        with ctx.trace.stage("decompose"):
            try:
                raw = ctx.llm.generate(
                    DECOMPOSE_TEMPLATE.format(
                        question=question, n=self.max_sub_questions
                    )
                )
            except LLMError as exc:
                return degrade_to_direct(
                    question, ctx, f"decomposition failed: {exc}"
                )

            sub_questions = parse_query_list(raw)[: self.max_sub_questions]
            if not sub_questions:
                return degrade_to_direct(
                    question, ctx, "decomposition produced no sub-questions"
                )

            pairs: list[tuple[str, str]] = []
            all_results = []
            for sub_question in sub_questions:
                ctx.trace.add_translation("sub_question", sub_question)
                results = ctx.search([sub_question], ctx.config.top_k)[0]
                all_results.append(results)

                prior = ""
                if self.mode == "recursive" and pairs:
                    prior = PRIOR_ANSWERS_HEADER.format(
                        prior="\n".join(f"Q: {q}\nA: {a}" for q, a in pairs)
                    )

                try:
                    answer = ctx.llm.generate(
                        SUB_ANSWER_TEMPLATE.format(
                            question=sub_question,
                            context=format_context(results),
                            prior=prior,
                        )
                    ).strip()
                except LLMError as exc:
                    # One failed sub-question should not lose the others.
                    ctx.trace.note(
                        f"sub-question failed, continuing without it: {exc}"
                    )
                    continue

                ctx.trace.add_translation("sub_answer", answer)
                pairs.append((sub_question, answer))

            ctx.trace.queries = [question, *sub_questions]
            retrieved = merge_best_score(all_results)[: ctx.config.top_k]

        extra = (
            "\n\n".join(f"Q: {q}\nA: {a}" for q, a in pairs) if pairs else None
        )
        return StrategyResult(retrieved=retrieved, extra_context=extra)
```

- [ ] **Step 5: Register it, run the tests, commit**

Add `DecompositionStrategy` to `rag/strategies/__init__.py`.

Run: `python -m pytest -q`

```bash
git add rag/strategies tests/test_strategies.py rag/prompts.py
git commit -m "feat: decomposition strategy, recursive and independent

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: CLI, trace rendering, and README

**Files:**
- Modify: `rag/__main__.py`, `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `rag.strategies.STRATEGY_NAMES`, `pipeline.ask(..., strategy=, strategy_options=)`
- Produces: `--strategy`, `--decomposition-mode`, `--queries`; `format_trace` renders translation steps

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_strategy_flag_reaches_the_pipeline(tiny_corpus, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())
    seen = {}

    real_ask = __import__("rag.__main__", fromlist=["ask"]).ask

    def spy(question, store, embedder, llm, config, k=None, strategy="direct",
            strategy_options=None):
        seen["strategy"] = strategy
        seen["options"] = strategy_options
        return real_ask(question, store, embedder, llm, config, k=k,
                        strategy=strategy, strategy_options=strategy_options)

    monkeypatch.setattr("rag.__main__.ask", spy)
    assert main(["ask", "q", "--strategy", "direct"], **_factories()) == 0
    assert seen["strategy"] == "direct"


def test_decomposition_mode_is_passed_as_a_strategy_option(
    tiny_corpus, monkeypatch, capsys
):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())
    seen = {}

    def spy(question, store, embedder, llm, config, k=None, strategy="direct",
            strategy_options=None):
        seen["options"] = strategy_options
        from rag.trace import Trace

        return Trace(question=question)

    monkeypatch.setattr("rag.__main__.ask", spy)
    main(
        ["ask", "q", "--strategy", "decomposition", "--decomposition-mode",
         "independent"],
        **_factories(),
    )
    assert seen["options"] == {"mode": "independent"}


def test_an_llm_strategy_with_no_llm_is_rejected(tiny_corpus, monkeypatch, capsys):
    # --no-llm means no rewrites are possible, so every strategy would silently
    # degrade to direct. Saying so beats pretending the flag did something.
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())
    assert main(["ask", "q", "--strategy", "hyde", "--no-llm"], **_factories()) == 1
    err = capsys.readouterr().err
    assert "--no-llm" in err
    assert "hyde" in err


def test_direct_strategy_with_no_llm_is_allowed(tiny_corpus, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())
    assert main(["ask", "q", "--strategy", "direct", "--no-llm"], **_factories()) == 0


def test_unknown_strategy_is_rejected_by_argparse(tiny_corpus, monkeypatch):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    with pytest.raises(SystemExit):
        main(["ask", "q", "--strategy", "nope"], **_factories())


def test_queries_flag_prints_the_rewritten_queries():
    from rag.trace import Trace

    trace = Trace(question="original")
    trace.strategy = "multi-query"
    trace.queries = ["original", "rewrite one", "rewrite two"]
    trace.add_translation("query", "rewrite one")
    trace.add_translation("query", "rewrite two")
    output = format_trace(trace, verbose=False, show_queries=True)
    assert "rewrite one" in output
    assert "rewrite two" in output


def test_trace_output_names_the_strategy():
    from rag.trace import Trace

    trace = Trace(question="q")
    trace.strategy = "rag-fusion"
    assert "rag-fusion" in format_trace(trace, verbose=True)


def test_verbose_output_shows_translation_steps():
    from rag.trace import Trace

    trace = Trace(question="q")
    trace.strategy = "hyde"
    trace.add_translation("hypothetical", "a hypothetical passage")
    assert "a hypothetical passage" in format_trace(trace, verbose=True)


def test_nested_timings_are_indented_in_verbose_output():
    from rag.trace import StageTiming, Trace

    trace = Trace(question="q")
    trace.timings = [
        StageTiming("embed", 1.0, depth=1),
        StageTiming("decompose", 5.0, depth=0),
    ]
    output = format_trace(trace, verbose=True)
    assert "  embed" in output
    assert "5.0" in output
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_cli.py -q`
Expected: `unrecognized arguments: --strategy`, and `TypeError` for `show_queries`.

- [ ] **Step 3: Add the flags to the parser**

In `_build_parser` in `rag/__main__.py`, on the `ask` subparser:

```python
    ask_parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGY_NAMES),
        default="direct",
        help="query translation strategy (default: direct)",
    )
    ask_parser.add_argument(
        "--decomposition-mode",
        choices=("recursive", "independent"),
        default="recursive",
        help="how decomposition combines sub-answers (default: recursive)",
    )
    ask_parser.add_argument(
        "--queries",
        action="store_true",
        help="show the queries the strategy generated",
    )
```

Add `from rag.strategies import STRATEGY_NAMES` to the imports.

- [ ] **Step 4: Reject an LLM strategy combined with `--no-llm`**

In `_run`, after the `--k` check:

```python
    if args.command == "ask" and args.no_llm and args.strategy != "direct":
        raise ValueError(
            f"--no-llm cannot be combined with --strategy {args.strategy}: "
            f"{args.strategy} needs the LLM to rewrite the question, so it "
            "would silently fall back to direct retrieval. Use --strategy "
            "direct, or drop --no-llm."
        )
```

- [ ] **Step 5: Pass the strategy through**

In `_run`, replace the `ask(...)` call:

```python
    strategy_options: dict = {}
    if args.strategy == "decomposition":
        strategy_options["mode"] = args.decomposition_mode

    trace = ask(
        args.question,
        store,
        embedder,
        llm,
        config,
        k=args.k,
        strategy=args.strategy,
        strategy_options=strategy_options,
    )
    print(format_trace(trace, verbose=args.trace, show_queries=args.queries))
```

- [ ] **Step 6: Render strategy, queries and nested timings**

Change `format_trace`'s signature to `format_trace(trace: Trace, verbose: bool, show_queries: bool = False) -> str` and add these blocks.

After the answer, before `Sources:`:

```python
    if (show_queries or verbose) and trace.translation:
        lines.append(f"Strategy: {trace.strategy}")
        for step in trace.translation:
            label = step.kind.replace("_", " ")
            text = step.text if verbose else step.text[:120]
            lines.append(f"  {label}: {text}")
        lines.append("")
```

In the verbose timings block, indent by depth and label the total:

```python
        lines.append("Timings:")
        for timing in trace.timings:
            indent = "  " * (timing.depth + 1)
            lines.append(f"{indent}{timing.name:<10} {timing.ms:8.1f} ms")
        lines.append(f"  {'total':<10} {trace.total_ms:8.1f} ms")
```

Also make the strategy visible in verbose output even with no translation steps, by adding near the top of the verbose block:

```python
        lines.append(f"Strategy: {trace.strategy}")
```

- [ ] **Step 7: Run the tests and verify they pass**

Run: `python -m pytest -q`

- [ ] **Step 8: Verify against the real corpus**

Run each and include the output in your report. Do **not** set `PYTHONIOENCODING`; that variable masks a Windows console bug this project has hit before.

```
python -m rag ask "How does ColBERT score a document?" --strategy multi-query --queries
python -m rag ask "How does ColBERT score a document?" --strategy rag-fusion --trace
python -m rag ask "What problem does HyDE solve?" --strategy hyde --queries
python -m rag ask "Why does RAPTOR cluster documents?" --strategy step-back --queries
python -m rag ask "How does ColBERT differ from dense retrieval?" --strategy decomposition --trace
python -m rag ask "q" --strategy hyde --no-llm
```

Expected: each strategy prints its generated queries; RAG-Fusion shows `rrf` scores rather than `cosine`; decomposition shows sub-questions with sub-answers and a single top-level `decompose` timing whose total is not double-counted; the last command exits 1 with the conflict message.

The interesting comparison to report: does any strategy retrieve documents that `--strategy direct` misses for the same question? Report honestly if they do not — Phase 3 exists to measure this properly, and an unmeasured claim is worth nothing.

- [ ] **Step 9: Update the README**

Add a Strategies section documenting the six names, the two decomposition modes, what each does in one sentence, and a worked example showing generated queries. State plainly that Phase 1 measured nothing about whether they help, and that Phase 3's benchmark is what will answer that.

Update the roadmap to check Phase 2.

- [ ] **Step 10: Commit**

```bash
git add rag/__main__.py tests/test_cli.py README.md
git commit -m "feat: --strategy flag, query rendering, and README

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 2 Definition of Done

- [ ] `python -m pytest` passes with no network access
- [ ] `python -m pytest -m slow` passes
- [ ] All six strategies run against the real corpus and print their generated queries
- [ ] RAG-Fusion results are labelled `rrf`, plain retrieval `cosine`
- [ ] Decomposition reports one top-level `decompose` timing, with nested stages not double-counted in the total
- [ ] Every strategy degrades to direct retrieval on LLM failure, with a note on the trace
- [ ] `--strategy hyde --no-llm` is rejected with an explanation
- [ ] `tests/test_no_frameworks.py` still passes
- [ ] Working tree clean

## What Phase 3 will need from this phase

- `Trace.strategy` names the strategy, so the benchmark can group results by it.
- `Trace.retrieved` carries `chunk_id` and `rank` for every result, which is what Recall@k, MRR and nDCG are computed from.
- `Trace.to_dict()` is JSON-serialisable including `strategy` and `translation`.
- `STRATEGY_NAMES` enumerates what to benchmark, so the benchmark does not hardcode a list.
- Strategies are constructed by name through `get_strategy`, so the benchmark can sweep them without importing each module.
