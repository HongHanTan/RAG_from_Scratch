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
