"""Retrieval metrics, written out.

Relevance is binary: a chunk either overlaps the gold span or it does not.
All three metrics take results in rank order, best first.

What each one tells you, since they disagree usefully:
- Recall@k: did we find the answer at all, within k?
- MRR: how near the top was the first correct chunk?
- nDCG@k: how well ordered is the whole result list?

A strategy can win on recall and lose on MRR by finding the answer but
burying it, which is worth knowing when the answer prompt only gets k chunks.
"""

from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the relevant chunks that appear in the top k."""
    if not relevant_ids:
        return 0.0
    found = set(retrieved_ids[:k]) & relevant_ids
    return len(found) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / rank of the first relevant chunk, or 0 if none was retrieved."""
    for position, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalised discounted cumulative gain over the top k.

    The ideal ranking is capped at k as well as at the number of relevant
    chunks: only k results can be returned, so a question with more relevant
    chunks than k must still be able to score 1.0.
    """
    if not relevant_ids:
        return 0.0

    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, chunk_id in enumerate(retrieved_ids[:k], start=1)
        if chunk_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
