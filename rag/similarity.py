"""Similarity search: exact, brute-force, and written out.

This is the whole of "the vector database". Scoring every query against every
chunk is one matrix multiply, which at a few thousand chunks is faster than the
overhead of an approximate index would be. It is O(n) per query, and past
roughly a million vectors an ANN index (FAISS, HNSW) becomes the right answer.

Inputs are normalised defensively even though Embedder already returns unit
vectors, so these functions are correct in isolation.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from rag.chunking import RetrievedChunk


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


RRF_K = 60
"""Damping constant from the original reciprocal rank fusion paper.

It flattens the gap between top ranks, so a chunk that several queries rank
highly beats one that a single query ranks first. Lower k sharpens the
advantage of rank 1; higher k makes agreement across lists matter more.
"""


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]], k: int = RRF_K
) -> list[RetrievedChunk]:
    """Fuse ranked lists into one, scoring each chunk by sum of 1/(k + rank).

    Only ranks are used; the original similarity scores are discarded. That is
    the point of the method — it combines lists that are not on a common scale.
    Results are re-ranked from 1 and marked `score_kind="rrf"`.
    """
    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for item in results:
            chunk_id = item.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + item.rank)
            best.setdefault(chunk_id, item)

    # Sort by descending score, then by chunk_id so ties are reproducible.
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return [
        replace(best[chunk_id], score=score, rank=rank, score_kind="rrf")
        for rank, (chunk_id, score) in enumerate(ordered, start=1)
    ]


def merge_best_score(
    result_lists: list[list[RetrievedChunk]],
) -> list[RetrievedChunk]:
    """Union several result lists, keeping each chunk's best cosine score.

    Used where the lists come from the same embedding space and the scores are
    therefore comparable — unlike fusion, which deliberately ignores them.
    """
    best: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for item in results:
            chunk_id = item.chunk.chunk_id
            if chunk_id not in best or item.score > best[chunk_id].score:
                best[chunk_id] = item

    ordered = sorted(best.values(), key=lambda r: (-r.score, r.chunk.chunk_id))
    return [
        replace(item, rank=rank, score_kind="cosine")
        for rank, item in enumerate(ordered, start=1)
    ]
