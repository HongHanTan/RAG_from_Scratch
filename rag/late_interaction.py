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

import numpy as np


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
