"""Resolve a gold span to the chunks that contain it.

Relevance is overlap, not containment. Chunks are 200 tokens with 50 tokens
of overlap, so the sentence answering a question frequently straddles a chunk
boundary. Requiring a chunk to contain the whole span would mark a chunk
holding most of the answer as a miss, and the metrics would understate every
strategy equally — which is worse than it sounds, because it would also
compress the differences between them.
"""

from __future__ import annotations

from evaluation.gold import GoldQuestion
from rag.chunking import Chunk


def chunks_overlapping(
    doc_id: str, char_start: int, char_end: int, chunks: list[Chunk]
) -> set[str]:
    """Ids of chunks in `doc_id` whose character range overlaps the span.

    Both the span and the chunk ranges are half-open, so a span beginning
    exactly where a chunk ends does not overlap it.
    """
    if char_end <= char_start:
        return set()
    return {
        chunk.chunk_id
        for chunk in chunks
        if chunk.doc_id == doc_id
        and chunk.char_start < char_end
        and char_start < chunk.char_end
    }


def relevant_chunk_ids(question: GoldQuestion, chunks: list[Chunk]) -> set[str]:
    """The chunks that count as a correct retrieval for this question.

    A question may have several answering passages, and any of them counts.
    Requiring one specific passage would score a strategy zero for finding a
    better explanation of the same thing than the one the gold set happened
    to quote.
    """
    relevant: set[str] = set()
    for char_start, char_end in question.spans:
        relevant |= chunks_overlapping(
            question.doc_id, char_start, char_end, chunks
        )
    return relevant
