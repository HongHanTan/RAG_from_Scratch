import math

import numpy as np
import pytest

from evaluation.benchmark import StrategyScore, format_table
from rag.config import Config
from rag.pipeline import ask, build_index
from tests.conftest import FakeEmbedder, FakeLLM


def _scores():
    return [
        StrategyScore("direct", 0.62, 0.55, 0.58, 0.56, 0.0, 41.6, 10),
        StrategyScore("rag-fusion", 0.74, 0.61, 0.66, 0.60, 1.0, 1502.3, 10),
    ]


def test_table_has_a_row_per_strategy():
    table = format_table(_scores(), k=20)
    assert "direct" in table
    assert "rag-fusion" in table


def test_table_names_the_k_it_measured():
    assert "Recall@20" in format_table(_scores(), k=20)
    assert "nDCG@20" in format_table(_scores(), k=20)


def test_table_reports_llm_calls_and_warm_ms_not_raw_mean_ms():
    # The old "Mean ms" column measured cache ordering, not strategy cost:
    # two strategies that build an identical prompt share an LLM cache key,
    # so whichever strategy STRATEGY_NAMES puts first pays for every rewrite
    # and the other one is all cache hits. LLM calls and a warm-cache timing
    # are both order-independent.
    table = format_table(_scores(), k=20)
    assert "LLM calls" in table
    assert "Mean ms (warm)" in table
    assert "| Mean ms |" not in table


def test_table_always_reports_doc_precision_at_5_regardless_of_k():
    # DocPrec is pinned to the answer prompt's real top_k (5), not the chunk
    # metrics' cutoff, so it must not move when --k changes.
    assert "DocPrec@5" in format_table(_scores(), k=20)
    assert "DocPrec@5" in format_table(_scores(), k=5)


def test_table_is_markdown():
    lines = format_table(_scores(), k=20).splitlines()
    assert lines[0].startswith("|")
    assert set(lines[1].replace("|", "").replace(" ", "")) <= {"-", ":"}


def test_table_reports_the_question_count():
    # A number computed over 10 questions must never be read as if it were
    # computed over 30.
    assert "10" in format_table(_scores(), k=20)


def test_table_rows_are_ordered_by_recall_descending():
    rows = [l for l in format_table(_scores(), k=20).splitlines() if l.startswith("|")]
    assert "rag-fusion" in rows[2]
    assert "direct" in rows[3]


def test_empty_scores_produce_a_table_with_no_rows():
    table = format_table([], k=20)
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


# --- _CallCountingLLM --------------------------------------------------------
#
# The mean-ms column measured cache ordering rather than strategy cost:
# multi-query and rag-fusion build an identical rewrite prompt for the same
# question, so they share one LLM cache key, and whichever name
# STRATEGY_NAMES lists first pays for every rewrite while the other is all
# cache hits. `LLM calls` counts real cost instead -- how many times a
# strategy *asks* the model for something -- and that count cannot be
# perturbed by cache state or run order.


def test_counting_llm_counts_every_call_including_repeats():
    from evaluation.benchmark import _CallCountingLLM

    llm = _CallCountingLLM(FakeLLM("same answer"))
    llm.generate("prompt a")
    llm.generate("prompt a")  # a real cache hit would still be a logical call
    llm.generate("prompt b")
    assert llm.logical_call_count == 3


def test_counting_llm_starts_at_zero():
    from evaluation.benchmark import _CallCountingLLM

    llm = _CallCountingLLM(FakeLLM())
    assert llm.logical_call_count == 0


def test_counting_llm_does_not_alter_the_returned_text():
    from evaluation.benchmark import _CallCountingLLM

    llm = _CallCountingLLM(FakeLLM("exact text"))
    assert llm.generate("anything") == "exact text"


def test_direct_strategy_never_calls_the_llm(tiny_corpus: Config):
    # `direct` never touches ctx.llm at all, so wrapping it must not
    # manufacture calls that did not happen.
    from evaluation.benchmark import _CallCountingLLM

    store = build_index(tiny_corpus, FakeEmbedder())
    counting_llm = _CallCountingLLM(FakeLLM())
    ask(
        "what is cosine similarity?",
        store,
        FakeEmbedder(),
        counting_llm,
        tiny_corpus,
        strategy="direct",
        generate=False,
    )
    assert counting_llm.logical_call_count == 0


def test_counting_llm_counts_a_strategy_that_does_call_the_llm(tiny_corpus: Config):
    from evaluation.benchmark import _CallCountingLLM

    store = build_index(tiny_corpus, FakeEmbedder())
    counting_llm = _CallCountingLLM(FakeLLM("query one\nquery two"))
    ask(
        "what is cosine similarity?",
        store,
        FakeEmbedder(),
        counting_llm,
        tiny_corpus,
        strategy="multi-query",
        generate=False,
    )
    assert counting_llm.logical_call_count == 1


# --- score_strategy end to end -------------------------------------------------
#
# Nothing previously exercised score_strategy itself -- the function that
# produces every published number. Swapping retrieved_doc_ids for
# retrieved_ids inside it, for instance, would still pass the rest of the
# suite. This builds a small, fully deterministic world (fixed embeddings, an
# in-memory store, a scripted LLM rewrite) so every StrategyScore field can be
# checked against a value computed by hand from the known retrieval order.


class _FixedVectorEmbedder:
    """Maps known strings to known vectors. No randomness, no model."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.array([self._vectors[t] for t in texts], dtype=np.float32)


def _hand_scored_world():
    """Three orthogonal chunks, two questions, one scripted rewrite.

    doc1 has two chunks (doc1:0, doc1:1); doc2 has one (doc2:0). Query and
    rewrite vectors are axis-aligned unit vectors, so cosine similarity is
    either exactly 1.0 or exactly 0.0 and the merged order is fully
    determined by `merge_best_score`'s documented tie-break (score, then
    chunk_id).
    """
    from evaluation.gold import GoldQuestion
    from rag.chunking import Chunk
    from rag.store import VectorStore

    chunks = [
        Chunk("doc1:0", "doc1", 0, "a", 0, 1, 0, 10),
        Chunk("doc1:1", "doc1", 1, "b", 0, 1, 10, 20),
        Chunk("doc2:0", "doc2", 0, "c", 0, 1, 0, 10),
    ]
    vectors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    store = VectorStore(vectors=vectors, chunks=chunks)

    embedder = _FixedVectorEmbedder(
        {
            "q1": [1.0, 0.0, 0.0],
            "q2": [0.0, 0.0, 1.0],
            "rewrite one": [0.0, 1.0, 0.0],
        }
    )
    llm = FakeLLM("1. rewrite one")

    gold = [
        GoldQuestion(
            id="g1", question="q1", doc_id="doc1", quotes=("x",), why="w",
            spans=((2, 5),),
        ),
        GoldQuestion(
            id="g2", question="q2", doc_id="doc2", quotes=("y",), why="w",
            spans=((2, 5),),
        ),
    ]
    config = Config(top_k=3, retrieval_depth=3)
    return gold, store, embedder, llm, config


def test_score_strategy_matches_metrics_computed_by_hand():
    from evaluation.benchmark import score_strategy

    gold, store, embedder, llm, config = _hand_scored_world()
    score = score_strategy("multi-query", gold, store, embedder, llm, config, k=3)

    # q1's merged order is [doc1:0, doc1:1, doc2:0]; the gold span sits in
    # doc1:0 only, so the first hit is at rank 1.
    # q2's merged order is [doc1:1, doc2:0, doc1:0]; the gold span sits in
    # doc2:0, so the first hit is at rank 2.
    assert score.strategy == "multi-query"
    assert score.questions == 2
    assert score.recall_at_k == pytest.approx(1.0)
    assert score.mrr == pytest.approx((1.0 + 0.5) / 2)
    assert score.ndcg_at_k == pytest.approx((1.0 + 1 / math.log2(3)) / 2)
    assert score.doc_precision == pytest.approx((2 / 3 + 1 / 3) / 2)
    # multi-query makes exactly one rewrite call per question.
    assert score.llm_calls == pytest.approx(1.0)
    assert score.mean_ms_warm >= 0.0


def test_score_strategy_resolves_gold_spans_against_the_stores_own_chunks():
    # If the benchmark ever went back to resolving spans against a separately
    # recomputed chunk list, this world would still "work" (ids and offsets
    # happen to agree here) -- so this test pins the *source* of the chunks
    # used for scoring, not just the numbers that come out. relevant_chunk_ids
    # must be called with store.chunks, not any other chunk list, so a chunk
    # that exists only in the store is still resolvable.
    import evaluation.benchmark as benchmark_module

    gold, store, embedder, llm, config = _hand_scored_world()
    seen_chunk_lists = []
    original = benchmark_module.relevant_chunk_ids

    def spy(question, chunks):
        seen_chunk_lists.append(chunks)
        return original(question, chunks)

    benchmark_module.relevant_chunk_ids = spy
    try:
        benchmark_module.score_strategy(
            "multi-query", gold, store, embedder, llm, config, k=3
        )
    finally:
        benchmark_module.relevant_chunk_ids = original

    assert seen_chunk_lists
    assert all(chunks is store.chunks for chunks in seen_chunk_lists)
