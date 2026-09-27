"""Every prompt template in the project, in one file.

Prompts are the interface to the model and they change often; keeping them
together means a change is one diff in one place rather than a hunt through the
strategies. Later phases append their templates here.
"""

from __future__ import annotations

from rag.chunking import RetrievedChunk

NO_CONTEXT = "(no documents retrieved)"

ANSWER_TEMPLATE = """You are answering a question using retrieved excerpts.

Rules:
- Use only the context below. Do not use prior knowledge.
- If the context does not answer the question, say so plainly and stop.
- If no documents were retrieved at all, reply that no documents were retrieved
  and that you therefore cannot answer. Do not repeat this instruction back.
- Cite the excerpts you used with their bracketed numbers, like [1] or [2].
- Be concise.

Context:
{context}

Question: {question}

Answer:"""


def format_context(retrieved: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered, citable block."""
    if not retrieved:
        return NO_CONTEXT
    blocks = []
    for item in retrieved:
        blocks.append(
            f"[{item.rank}] source: {item.chunk.doc_id}, chunk {item.chunk.index}, "
            f"{item.score_kind} {item.score:.3f}\n{item.chunk.text}"
        )
    return "\n\n".join(blocks)


def build_answer_prompt(
    question: str, retrieved: list[RetrievedChunk]
) -> str:
    """Fill the answer template. Chunk text may contain braces; it is not
    re-formatted, so `{}` in a document cannot break or inject anything."""
    return ANSWER_TEMPLATE.format(
        context=format_context(retrieved), question=question
    )
