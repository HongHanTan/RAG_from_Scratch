"""Similarity search: exact, brute-force, and written out.

This is the whole of "the vector database". Scoring every query against every
chunk is one matrix multiply, which at a few thousand chunks is faster than the
overhead of an approximate index would be. It is O(n) per query, and past
roughly a million vectors an ANN index (FAISS, HNSW) becomes the right answer.

Inputs are normalised defensively even though Embedder already returns unit
vectors, so these functions are correct in isolation.
"""

from __future__ import annotations

import numpy as np


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def cosine_similarity(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of every query against every row of matrix.

    Returns an (n_queries, n_items) array. All-zero rows score 0, not NaN.
    """
    queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float32))
    if queries.shape[1] != matrix.shape[1]:
        raise ValueError(
            f"dimension mismatch: queries are {queries.shape[1]}-d, "
            f"items are {matrix.shape[1]}-d"
        )
    return _unit_rows(queries) @ _unit_rows(matrix).T


def top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and values of the k highest scores per row, descending.

    Ties break toward the lower index: argsort is stable, so equal scores keep
    their original order. Without that, two runs could rank identical chunks
    differently and the benchmark would drift for no reason.
    """
    if k < 0:
        raise ValueError(f"k must not be negative, got {k}")
    scores = np.atleast_2d(np.asarray(scores, dtype=np.float32))
    k = min(k, scores.shape[1])
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(scores, order, axis=1)
