"""Token-aware sliding-window chunking.

Character-based splitting is the standard beginner bug here: MiniLM truncates
at 256 tokens, so a 2000-character chunk loses its tail with no error raised.
Sliding over token ids instead means every chunk fits the model by construction.

Character offsets come from the tokenizer's offset mapping, so chunk text is a
slice of the original document rather than a detokenised approximation. Phase 3
needs those exact offsets to score span-tagged gold questions.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.loader import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    text: str
    token_start: int
    token_end: int   # exclusive
    char_start: int
    char_end: int    # exclusive


@dataclass
class RetrievedChunk:
    """One chunk as it came back from a search, with its rank in that result.

    This is the single representation of a retrieval result used across the
    pipeline: `VectorStore.search`, `prompts.format_context`,
    `generation.generate_answer` and `Trace.retrieved` all speak this type,
    rather than `search` handing back plain `(Chunk, score)` tuples that get
    converted to `RetrievedChunk` (with its `rank`) only on the way into the
    trace. Phase 2's rank fusion needs `rank` on every result, not just the
    ones that happen to reach the trace.
    """

    chunk: Chunk
    score: float
    rank: int   # 1-based


def window_bounds(n_tokens: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Half-open [start, end) windows covering n_tokens.

    Stops as soon as a window reaches the end, so no trailing stub window is
    emitted whose content is already wholly contained in its predecessor.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, got {overlap}")
    if overlap >= size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than size ({size}), "
            "or the window never advances"
        )
    if n_tokens <= 0:
        return []

    step = size - overlap
    bounds: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + size, n_tokens)
        bounds.append((start, end))
        if end >= n_tokens:
            return bounds
        start += step


def chunk_document(doc: Document, tokenizer, size: int, overlap: int) -> list[Chunk]:
    """Split one document into overlapping token windows."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "chunking needs a fast tokenizer for offset_mapping; "
            "load it with AutoTokenizer.from_pretrained(..., use_fast=True)"
        )

    encoded = tokenizer(
        doc.text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]

    chunks: list[Chunk] = []
    for index, (token_start, token_end) in enumerate(
        window_bounds(len(offsets), size, overlap)
    ):
        char_start = offsets[token_start][0]
        char_end = offsets[token_end - 1][1]
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{index}",
                doc_id=doc.doc_id,
                index=index,
                text=doc.text[char_start:char_end],
                token_start=token_start,
                token_end=token_end,
                char_start=char_start,
                char_end=char_end,
            )
        )
    return chunks


def chunk_documents(
    docs: list[Document], tokenizer, size: int, overlap: int
) -> list[Chunk]:
    """Chunk every document, preserving document order."""
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, tokenizer, size, overlap))
    return chunks
