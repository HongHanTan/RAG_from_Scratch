import numpy as np
import pytest

from rag.similarity import cosine_similarity, top_k


# --- cosine -----------------------------------------------------------------

def test_identical_vectors_score_one():
    v = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(cosine_similarity(v, v), [[1.0]], atol=1e-6)


def test_orthogonal_vectors_score_zero():
    q = np.array([[1.0, 0.0]])
    m = np.array([[0.0, 1.0]])
    np.testing.assert_allclose(cosine_similarity(q, m), [[0.0]], atol=1e-6)


def test_opposite_vectors_score_minus_one():
    q = np.array([[1.0, 0.0]])
    m = np.array([[-1.0, 0.0]])
    np.testing.assert_allclose(cosine_similarity(q, m), [[-1.0]], atol=1e-6)


def test_magnitude_is_ignored():
    q = np.array([[1.0, 0.0]])
    m = np.array([[100.0, 0.0]])
    np.testing.assert_allclose(cosine_similarity(q, m), [[1.0]], atol=1e-6)


def test_matches_the_textbook_formula():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(3, 5))
    m = rng.normal(size=(7, 5))
    expected = np.array([
        [
            float(qi @ mj / (np.linalg.norm(qi) * np.linalg.norm(mj)))
            for mj in m
        ]
        for qi in q
    ])
    np.testing.assert_allclose(cosine_similarity(q, m), expected, atol=1e-5)


def test_output_shape_is_queries_by_items():
    assert cosine_similarity(np.ones((3, 4)), np.ones((7, 4))).shape == (3, 7)


def test_a_one_dimensional_query_is_treated_as_a_single_query():
    assert cosine_similarity(np.ones(4), np.ones((7, 4))).shape == (1, 7)


def test_zero_vector_scores_zero_rather_than_nan():
    result = cosine_similarity(np.zeros((1, 3)), np.ones((2, 3)))
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, [[0.0, 0.0]])


def test_dimension_mismatch_is_rejected():
    with pytest.raises(ValueError, match="dimension"):
        cosine_similarity(np.ones((1, 4)), np.ones((2, 5)))


# --- top-k ------------------------------------------------------------------

def test_returns_highest_scores_in_descending_order():
    scores = np.array([[0.1, 0.9, 0.5, 0.7]])
    indices, values = top_k(scores, 2)
    np.testing.assert_array_equal(indices, [[1, 3]])
    np.testing.assert_allclose(values, [[0.9, 0.7]])


def test_ties_are_broken_by_lower_index_for_determinism():
    scores = np.array([[0.5, 0.9, 0.9, 0.1]])
    indices, _ = top_k(scores, 2)
    np.testing.assert_array_equal(indices, [[1, 2]])


def test_k_larger_than_the_corpus_returns_everything():
    scores = np.array([[0.1, 0.9]])
    indices, values = top_k(scores, 10)
    assert indices.shape == (1, 2)
    np.testing.assert_array_equal(indices, [[1, 0]])


def test_k_of_zero_returns_empty():
    indices, values = top_k(np.array([[0.1, 0.9]]), 0)
    assert indices.shape == (1, 0)
    assert values.shape == (1, 0)


def test_negative_k_is_rejected():
    with pytest.raises(ValueError, match="k"):
        top_k(np.array([[0.1]]), -1)


def test_each_query_row_is_ranked_independently():
    scores = np.array([[0.1, 0.9], [0.9, 0.1]])
    indices, _ = top_k(scores, 1)
    np.testing.assert_array_equal(indices, [[1], [0]])


def test_empty_corpus_yields_empty_results():
    indices, values = top_k(np.zeros((2, 0)), 5)
    assert indices.shape == (2, 0)
    assert values.shape == (2, 0)


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
