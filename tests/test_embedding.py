import numpy as np
import pytest

from rag.embedding import Embedder, l2_normalize, mean_pool


# --- mean pooling -----------------------------------------------------------

def test_mean_pool_averages_only_unmasked_positions():
    hidden = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    mask = np.array([[1, 1, 0]])
    # (1+3)/2 = 2, (2+4)/2 = 3 — the third position is padding and must not count.
    np.testing.assert_allclose(mean_pool(hidden, mask), [[2.0, 3.0]])


def test_mean_pool_with_all_positions_unmasked():
    hidden = np.array([[[1.0, 1.0], [3.0, 5.0]]])
    mask = np.array([[1, 1]])
    np.testing.assert_allclose(mean_pool(hidden, mask), [[2.0, 3.0]])


def test_mean_pool_handles_a_fully_masked_row_without_dividing_by_zero():
    hidden = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    mask = np.array([[0, 0]])
    result = mean_pool(hidden, mask)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, [[0.0, 0.0]])


def test_mean_pool_is_independent_across_batch_rows():
    hidden = np.array([
        [[1.0, 1.0], [3.0, 3.0]],
        [[10.0, 0.0], [0.0, 0.0]],
    ])
    mask = np.array([[1, 1], [1, 0]])
    np.testing.assert_allclose(mean_pool(hidden, mask), [[2.0, 2.0], [10.0, 0.0]])


def test_mean_pool_returns_float32():
    hidden = np.ones((1, 2, 2), dtype=np.float64)
    mask = np.ones((1, 2), dtype=np.int64)
    assert mean_pool(hidden, mask).dtype == np.float32


# --- L2 normalisation -------------------------------------------------------

def test_l2_normalize_produces_unit_vectors():
    np.testing.assert_allclose(l2_normalize(np.array([[3.0, 4.0]])), [[0.6, 0.8]])


def test_l2_normalize_leaves_a_zero_vector_at_zero():
    result = l2_normalize(np.array([[0.0, 0.0]]))
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, [[0.0, 0.0]])


def test_l2_normalize_acts_per_row():
    result = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    np.testing.assert_allclose(result, [[0.6, 0.8], [0.0, 1.0]])


def test_l2_normalize_is_idempotent():
    once = l2_normalize(np.array([[3.0, 4.0], [1.0, 1.0]]))
    np.testing.assert_allclose(l2_normalize(once), once, atol=1e-6)


# --- the real model ---------------------------------------------------------

@pytest.mark.slow
def test_real_embedder_shape_and_normalisation():
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    vectors = embedder.encode(["cosine similarity", "rank fusion"])
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0], atol=1e-5)


@pytest.mark.slow
def test_real_embedder_places_related_text_closer_than_unrelated():
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    v = embedder.encode([
        "cosine similarity between two vectors",
        "measuring the angle between vectors",
        "a recipe for banana bread",
    ])
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


@pytest.mark.slow
def test_real_embedder_batching_matches_single_pass():
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    texts = [f"sentence number {i}" for i in range(5)]
    np.testing.assert_allclose(
        embedder.encode(texts, batch_size=2),
        embedder.encode(texts, batch_size=32),
        atol=1e-5,
    )


@pytest.mark.slow
def test_real_embedder_returns_empty_matrix_for_empty_input():
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    assert embedder.encode([]).shape == (0, 384)
