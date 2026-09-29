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
