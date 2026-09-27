"""Every prompt template in the project, in one file.

Prompts are the interface to the model and they change often; keeping them
together means a change is one diff in one place rather than a hunt through the
strategies. Later phases append their templates here.
"""

from __future__ import annotations

from rag.chunking import Chunk

NO_CONTEXT = "(no documents retrieved)"

ANSWER_TEMPLATE = """You are answering a question using retrieved excerpts.

Rules:
- Use only the context below. Do not use prior knowledge.
- If the context does not answer the question, say so plainly and stop.
- Cite the excerpts you used with their bracketed numbers, like [1] or [2].
- Be concise.

Context:
{context}

Question: {question}

Answer:"""


def format_context(retrieved: list[tuple[Chunk, float]]) -> str:
    """Render retrieved chunks as a numbered, citable block."""
    if not retrieved:
        return NO_CONTEXT
    blocks = []
    for number, (chunk, score) in enumerate(retrieved, start=1):
        blocks.append(
            f"[{number}] source: {chunk.doc_id}, chunk {chunk.index}, "
            f"score {score:.3f}\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_answer_prompt(
    question: str, retrieved: list[tuple[Chunk, float]]
) -> str:
    """Fill the answer template. Chunk text may contain braces; it is not
    re-formatted, so `{}` in a document cannot break or inject anything."""
    return ANSWER_TEMPLATE.format(
        context=format_context(retrieved), question=question
    )
