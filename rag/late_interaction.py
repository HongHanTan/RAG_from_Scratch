"""ColBERT-style late interaction, as a reranker.

Dense retrieval compares one vector per query against one vector per chunk,
so a passage that answers a question in one clause is represented by the
average of every clause it contains. Late interaction keeps every token on
both sides and asks a narrower question: for each query term, how well does
the best-matching term in this passage match it?

Two limitations, kept in the README rather than glossed:

- This reranks a shortlist rather than indexing every token of the corpus.
  Full late-interaction indexing is roughly 100x the storage, and reranking
  is what is deployed in practice.
- These are MiniLM token vectors, not trained ColBERT weights. MiniLM was
  trained for pooled sentence similarity, so the gain may be small or
  negative. The benchmark reports whichever it is.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from rag.chunking import RetrievedChunk


def maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """Sum over query tokens of the best similarity against any doc token.

    Both matrices are (n_tokens, dim) with unit-length rows, so the matrix
    product is a matrix of cosines and no normalisation happens here.

    The score is a sum, not a mean, so it grows with query length and is
    only ever used to compare documents *for one query* -- never to compare
    one query's results with another's.
    """
    query_tokens = np.asarray(query_tokens, dtype=np.float32)
    doc_tokens = np.asarray(doc_tokens, dtype=np.float32)
    if query_tokens.size == 0 or doc_tokens.size == 0:
        return 0.0
    if query_tokens.shape[1] != doc_tokens.shape[1]:
        raise ValueError(
            f"dimension mismatch: query {query_tokens.shape[1]} vs "
            f"document {doc_tokens.shape[1]}"
        )
    return float((query_tokens @ doc_tokens.T).max(axis=1).sum())


def rerank(
    question: str,
    retrieved: list[RetrievedChunk],
    embedder,
    k: int,
    trace=None,
) -> list[RetrievedChunk]:
    """Rescore `retrieved` with MaxSim and return the best `k`, renumbered.

    Falling back to the dense order on failure is recorded with the
    `degraded` sentinel: reranking that silently did nothing would be
    reported as a measured null result, which is a worse outcome than a
    loud failure.
    """
    if not retrieved:
        return []

    try:
        query_tokens = embedder.encode_tokens([question])[0]
        doc_tokens = embedder.encode_tokens([r.chunk.text for r in retrieved])
    except Exception as exc:                        # model or tokenizer failure
        if trace is not None:
            trace.degraded(f"reranking failed: {exc}", "dense order kept")
        return [
            replace(item, rank=rank)
            for rank, item in enumerate(retrieved[:k], start=1)
        ]

    scored = [
        (maxsim(query_tokens, tokens), item)
        for tokens, item in zip(doc_tokens, retrieved)
    ]
    # Stable on ties, so an exact score tie keeps the dense ordering rather
    # than depending on sort internals.
    scored.sort(key=lambda pair: pair[0], reverse=True)

    out = [
        replace(item, score=score, rank=rank, score_kind="maxsim")
        for rank, (score, item) in enumerate(scored[:k], start=1)
    ]
    if trace is not None:
        trace.note(f"rerank: {len(retrieved)} candidates scored by maxsim -> {len(out)}")
    return out
