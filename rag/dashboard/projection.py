"""Chunk embeddings reduced to two dimensions, for the dashboard scatter.

This is PCA done with `np.linalg.svd`: centre the points, take the two
directions of greatest variance, project onto them. Those two directions
are chosen to preserve as much spread as any plane can, which is a much
weaker promise than it looks -- 384 dimensions do not fit in 2, and most of
the geometry is discarded. The panel is labelled a projection of 384
dimensions rather than the space itself for exactly that reason.

The query is projected with the same transform as the chunks, not its own,
because a marker placed by a different transform would sit somewhere
arbitrary relative to the points it is meant to be compared against.
"""

from __future__ import annotations

import numpy as np


def project_2d(
    vectors: np.ndarray, query_vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project chunk vectors and the query into a shared 2D plane."""
    vectors = np.asarray(vectors, dtype=np.float32)
    query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)

    if vectors.size and vectors.shape[1] != query_vector.shape[0]:
        raise ValueError(
            f"dimension mismatch: chunks {vectors.shape[1]} vs "
            f"query {query_vector.shape[0]}"
        )
    if vectors.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)

    # The query is stacked in so it is centred and rotated with everything
    # else, then split back out at the end.
    stacked = np.vstack([vectors, query_vector[None, :]])
    centred = stacked - stacked.mean(axis=0, keepdims=True)

    # full_matrices=False keeps this (n, min(n, dim)) rather than (dim, dim).
    _, _, vt = np.linalg.svd(centred, full_matrices=False)

    # Fewer than two components exist when there are fewer than three points
    # or the points are identical; pad so the caller always gets 2 columns
    # rather than a ragged array it has to special-case.
    axes = vt[:2]
    if axes.shape[0] < 2:
        axes = np.vstack([axes, np.zeros((2 - axes.shape[0], axes.shape[1]))])

    coords = (centred @ axes.T).astype(np.float32)
    return coords[:-1], coords[-1]
