"""RAPTOR: recursive clustering and summarisation.

RAG.pdf's argument: top-k retrieval over raw chunks answers a specific
question well and a broad one badly, because the answer to "what do these
papers have in common" is not in any one chunk. RAPTOR builds abstractions —
cluster the chunks, summarise each cluster, embed the summaries, cluster
those — so a single search can return either a raw passage or a summary
standing for hundreds of them.

Two structural choices worth knowing:

- **Original chunks come first in the returned list, in their original
  order.** `evaluation/spans.py` resolves gold spans positionally against
  `store.chunks`, so appending rather than interleaving keeps Phase 3's
  numbers comparable.
- **A cluster summary gets a synthetic `doc_id`.** It spans several
  documents and belongs to none, which is what makes `compile_mask` exclude
  it from a filtered search.
"""

from __future__ import annotations

import numpy as np

from rag.chunking import Chunk, make_summary_chunk
from rag.clustering import choose_k, kmeans
from rag.config import Config
from rag.summarise import CLUSTER_SUMMARY_TEMPLATE, SummaryError, summarise

JOIN = "\n\n---\n\n"


def build_raptor(
    chunks: list[Chunk],
    vectors: np.ndarray,
    embedder,
    llm,
    config: Config,
    trace=None,
) -> tuple[list[Chunk], np.ndarray]:
    """Build the summary tree, returning every node and its vectors.

    Recursion stops at `config.raptor_max_depth`, or as soon as a level
    produces one node or fewer — without that second condition a level of one
    would cluster into one cluster forever.
    """
    all_chunks = list(chunks)
    all_vectors = [np.asarray(vectors, dtype=np.float32)]

    current_chunks = list(chunks)
    current_vectors = np.asarray(vectors, dtype=np.float32)

    for level in range(1, config.raptor_max_depth + 1):
        if len(current_chunks) <= 1:
            break

        k = choose_k(len(current_chunks), config.raptor_cluster_size)
        if k >= len(current_chunks):
            break
        labels = kmeans(current_vectors, k=k, seed=level)

        level_chunks: list[Chunk] = []
        level_texts: list[str] = []
        for cluster in range(k):
            members = [c for c, lab in zip(current_chunks, labels) if lab == cluster]
            if not members:
                continue
            joined = JOIN.join(m.text for m in members)
            try:
                summary = summarise(joined, llm, CLUSTER_SUMMARY_TEMPLATE)
            except SummaryError as exc:
                if trace is not None:
                    trace.degraded(
                        f"cluster summary failed at level {level}, cluster "
                        f"{cluster}: {exc}",
                        "skipping that cluster",
                    )
                continue
            level_chunks.append(
                make_summary_chunk(
                    chunk_id=f"raptor:{level}:{cluster}",
                    doc_id=f"raptor:{level}:{cluster}",
                    index=cluster,
                    text=summary,
                    level=level,
                )
            )
            level_texts.append(summary)

        if not level_chunks:
            break

        level_vectors = embedder.encode(level_texts)
        all_chunks.extend(level_chunks)
        all_vectors.append(level_vectors)

        current_chunks = level_chunks
        current_vectors = level_vectors

    return all_chunks, np.vstack(all_vectors)
