import numpy as np
import pytest

from rag.dashboard.projection import project_2d


def _unit(rows):
    m = np.asarray(rows, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def _blobs(n=12, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=2.0, scale=0.2, size=(n, dim))
    b = rng.normal(loc=-2.0, scale=0.2, size=(n, dim))
    return _unit(np.vstack([a, b])), n


def test_returns_two_dimensional_coordinates():
    vectors, _ = _blobs()
    coords, query = project_2d(vectors, _unit([[1.0] * 8])[0])
    assert coords.shape == (len(vectors), 2)
    assert query.shape == (2,)


def test_the_query_is_projected_into_the_same_space():
    # The query marker is only meaningful if it went through the same
    # transform as the chunks; projecting it separately would place it
    # somewhere arbitrary.
    vectors, _ = _blobs()
    q = vectors[0].copy()
    coords, query = project_2d(vectors, q)
    assert np.allclose(coords[0], query, atol=1e-4)


def test_separable_clusters_stay_separated_in_2d():
    # A projection that collapses obviously distinct groups is not showing
    # anything; this is the property the panel depends on.
    vectors, n = _blobs()
    coords, _ = project_2d(vectors, vectors.mean(axis=0))
    first, second = coords[:n].mean(axis=0), coords[n:].mean(axis=0)
    spread = np.linalg.norm(coords - coords.mean(axis=0), axis=1).mean()
    assert np.linalg.norm(first - second) > spread


def test_it_is_deterministic():
    vectors, _ = _blobs()
    q = _unit([[1.0] * 8])[0]
    a, aq = project_2d(vectors, q)
    b, bq = project_2d(vectors, q)
    assert np.array_equal(a, b)
    assert np.array_equal(aq, bq)


def test_output_is_finite():
    vectors, _ = _blobs()
    coords, query = project_2d(vectors, _unit([[1.0] * 8])[0])
    assert np.isfinite(coords).all()
    assert np.isfinite(query).all()


def test_a_single_vector_still_projects():
    # Top-k of 1 is a legitimate request and must not raise.
    coords, query = project_2d(_unit([[1.0, 0.0, 0.0]]), _unit([[0.0, 1.0, 0.0]])[0])
    assert coords.shape == (1, 2)
    assert np.isfinite(coords).all()


def test_two_vectors_still_project():
    coords, _ = project_2d(_unit([[1.0, 0.0], [0.0, 1.0]]), _unit([[1.0, 1.0]])[0])
    assert coords.shape == (2, 2)
    assert np.isfinite(coords).all()


def test_no_vectors_returns_empty_coordinates():
    coords, query = project_2d(np.zeros((0, 4), dtype=np.float32),
                               _unit([[1.0, 0.0, 0.0, 0.0]])[0])
    assert coords.shape == (0, 2)
    assert query.shape == (2,)


def test_identical_vectors_do_not_produce_nan():
    # Zero variance means a degenerate SVD; the panel must still render.
    coords, query = project_2d(np.ones((5, 4), dtype=np.float32),
                               np.ones(4, dtype=np.float32))
    assert np.isfinite(coords).all()
    assert np.isfinite(query).all()


def test_a_mismatched_query_dimension_is_an_error():
    with pytest.raises(ValueError, match="dimension"):
        project_2d(_unit([[1.0, 0.0]]), np.ones(3, dtype=np.float32))
