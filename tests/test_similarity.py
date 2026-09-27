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
