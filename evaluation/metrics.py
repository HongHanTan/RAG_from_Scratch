"""Retrieval metrics, written out.

Relevance is binary: a chunk either overlaps the gold span or it does not.
All four metrics take results in rank order, best first.

What each one tells you, since they disagree usefully:
- Recall@k: did we find the answer at all, within k?
- MRR: how near the top was the first correct chunk?
- nDCG@k: how well ordered is the whole result list?
- DocPrec@k: of the top k results, how many come from the right document at
  all? It does not depend on the chunk-level gold set, only on doc_id, so it
  stays meaningful even though that gold set is necessarily incomplete.

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


def doc_precision_at_k(
    retrieved_doc_ids: list[str], gold_doc_id: str, k: int
) -> float:
    """Fraction of the top k results that come from the document holding the answer.

    Robust to an incomplete chunk-level gold set: it does not matter which
    passage of the right paper was retrieved, only that the paper was found.
    Measured at 0.560 for plain retrieval with a 0.0-1.0 spread across
    questions, so it discriminates rather than saturating.

    The denominator is `len(top)`, i.e. `min(k, len(retrieved_doc_ids))`, not
    a fixed `k`. If fewer than `k` results were retrieved at all -- a strategy
    that returned fewer chunks than asked for, or the benchmark's own
    `--k` set below `DOC_PRECISION_K` -- the missing slots are not results;
    scoring them as misses would measure coverage, not the precision of what
    was actually returned. This is inert whenever `retrieved_doc_ids` has at
    least `k` entries (true throughout the benchmark's default `--k 20`,
    since DocPrec is always measured at `k=5`), and only changes the number
    if it does not.
    """
    top = retrieved_doc_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for doc_id in top if doc_id == gold_doc_id)
    return hits / len(top)
