"""K-means, written out.

RAPTOR's whole mechanic is recursive clustering, so importing a clustering
library would hide the thing the phase exists to show. This is Lloyd's
algorithm in about thirty lines of NumPy: assign each point to its nearest
centroid, move each centroid to the mean of its members, repeat until the
assignment stops changing.

Seeded throughout, because the benchmark must not drift between runs.
"""

from __future__ import annotations

import numpy as np


def choose_k(n_items: int, target_size: int) -> int:
    """How many clusters to ask for, aiming at roughly `target_size` each.

    Clamped to at least one and at most `n_items`, so a level with three
    nodes does not ask for eight clusters.
    """
    if n_items <= 0:
        return 0
    return max(1, min(n_items, round(n_items / max(1, target_size))))


def kmeans(
    vectors: np.ndarray, k: int, seed: int = 0, max_iter: int = 50
) -> np.ndarray:
    """Cluster `vectors` into at most `k` groups, returning integer labels.

    Empty clusters are left empty rather than re-seeded: re-seeding makes the
    result depend on iteration order in ways that defeat the seeding, and a
    cluster nobody joined simply contributes no summary node.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    vectors = np.asarray(vectors, dtype=np.float32)
    n = vectors.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)

    k = min(k, n)
    rng = np.random.default_rng(seed)
    centroids = vectors[rng.choice(n, size=k, replace=False)].copy()

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        # (n, k) squared distances, via the expansion that avoids a big
        # intermediate: |x - c|^2 = |x|^2 - 2 x.c + |c|^2, and |x|^2 is
        # constant per row so it cannot change the argmin.
        distances = (centroids**2).sum(axis=1) - 2.0 * (vectors @ centroids.T)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = vectors[labels == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)
    return labels
