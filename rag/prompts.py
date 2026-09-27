"""Every prompt template in the project, in one file.

Prompts are the interface to the model and they change often; keeping them
together means a change is one diff in one place rather than a hunt through the
strategies. Later phases append their templates here.
"""

from __future__ import annotations

import re

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
    question: str,
    retrieved: list[RetrievedChunk],
    extra_context: str | None = None,
) -> str:
    """Fill the answer template.

    `extra_context` holds text the strategy derived (decomposition's
    sub-answers). It is labelled separately from the retrieved excerpts so the
    model does not cite generated text as though it were a source.
    """
    context = format_context(retrieved)
    if extra_context:
        context = (
            f"{context}\n\n"
            f"Working notes (derived from the excerpts above, not a source — "
            f"do not cite these):\n{extra_context}"
        )
    return ANSWER_TEMPLATE.format(context=context, question=question)


MULTI_QUERY_TEMPLATE = """You are helping a search system find relevant documents.

Rewrite the question below into {n} alternative search queries. Each should
approach the same information need from a different angle — different
vocabulary, a broader or narrower framing, or an underlying concept the
question implies. The goal is that at least one rewrite matches wording the
documents actually use.

Reply with one query per line, numbered. No other text.

Question: {question}"""


STEP_BACK_TEMPLATE = """Given a specific question, write one more general
question about the underlying concept or principle it depends on.

The general question should be broad enough that a document explaining the
background would answer it, while staying on the same subject. Do not answer
either question.

Reply with the general question only, on one line.

Specific question: {question}"""


_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


def parse_query_list(raw: str) -> list[str]:
    """Extract one query per line from a numbered or bulleted model reply.

    Models drift between "1." and "-" and occasionally repeat themselves, so
    the parsing is forgiving and deduplicates while preserving order.

    Models also tend to wrap the list in a preamble ("Here are five
    alternatives:") and a sign-off ("I hope these help!"). When at least one
    line in the reply carries a list marker, only marked lines are kept, which
    drops that surrounding chatter. When no line is marked, every non-blank
    line is kept, since the model may have legitimately replied with a bare
    list.
    """
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    any_marked = any(_LIST_MARKER.match(line) for line in lines)

    queries: list[str] = []
    seen: set[str] = set()
    for line in lines:
        marked = _LIST_MARKER.match(line)
        if any_marked and not marked:
            continue
        text = _LIST_MARKER.sub("", line).strip()
        if text and text not in seen:
            seen.add(text)
            queries.append(text)
    return queries
