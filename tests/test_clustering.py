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
