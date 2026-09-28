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
    prompt_name: str | None = None,
) -> str:
    """Fill the answer template.

    `extra_context` holds text the strategy derived (decomposition's
    sub-answers). It is labelled separately from the retrieved excerpts so the
    model does not cite generated text as though it were a source.

    `prompt_name` selects a variant from `PROMPT_VARIANTS` (set by semantic
    routing); `None` or a name not in `PROMPT_VARIANTS` falls back to the
    default `ANSWER_TEMPLATE`.

    Chunk text may contain braces; it is not re-formatted, only inserted after
    `str.format` has already scanned the template, so `{}` in a document
    cannot break or inject anything.
    """
    context = format_context(retrieved)
    if extra_context:
        context = (
            f"{context}\n\n"
            f"Working notes (derived from the excerpts above, not a source — "
            f"do not cite these):\n{extra_context}"
        )
    template = PROMPT_VARIANTS.get(prompt_name, ANSWER_TEMPLATE)
    return template.format(context=context, question=question)


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


HYDE_TEMPLATE = """Write a short passage that answers the question below, as
it might appear in a technical paper or reference document.

Write it as documentation, not as a reply: no preamble, no "the answer is",
just the passage. Being factually wrong is acceptable — this text is used to
search with, not to show anyone. Match the vocabulary and register a real
document on this subject would use.

Keep it under 120 words.

Question: {question}"""


DECOMPOSE_TEMPLATE = """Break the question below into at most {n} simpler
sub-questions that can each be looked up on their own.

Each sub-question must be answerable from documents independently of the
others, and answering all of them should be enough to answer the original.
If the question is already simple, reply with just the original question.

Reply with one sub-question per line, numbered. No other text.

Question: {question}"""


SUB_ANSWER_TEMPLATE = """Answer the sub-question using only the context below.
If the context does not answer it, say so in one sentence.

Be brief: two sentences at most. This answer is working material for a larger
question, not a final response.
{prior}
Context:
{context}

Sub-question: {question}

Answer:"""


PRIOR_ANSWERS_HEADER = """
Already established:
{prior}
"""


FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {"type": "array", "items": {"type": "string"}},
        "authors": {"type": "array", "items": {"type": "string"}},
        "published_before": {"type": "string"},
        "published_after": {"type": "string"},
    },
    "required": [],
}

FILTER_TEMPLATE = """Extract any metadata constraints from the question below.

The documents are research papers with these fields:
- topic, one of: {topics}
- author, a surname or "Surname et al."
- publish_date, an ISO date

Reply with a JSON object containing only the constraints the question
actually states. Omit any field the question does not constrain. Do not
invent a topic that is not in the list above.

Dates must be full ISO dates. "before 2024" means
{{"published_before": "2024-01-01"}}.

Most questions state no constraint at all; for those, reply with {{}}.

Question: {question}

JSON:"""


ROUTE_SCHEMA = {
    "type": "object",
    "properties": {"topics": {"type": "array", "items": {"type": "string"}}},
    "required": ["topics"],
}

ROUTE_TEMPLATE = """Decide which collections of papers to search for this question.

Available collections:
{descriptions}

Choose every collection that might hold the answer, and no more. Choosing too
few loses the answer; choosing all of them is the same as not routing.

Reply with a JSON object: {{"topics": ["name", ...]}}.

Question: {question}

JSON:"""

TOPIC_DESCRIPTIONS = {
    "foundations": "transformer and language-model architecture papers",
    "retrieval-models": "dense, sparse and late-interaction retrieval models and embeddings",
    "rag-systems": "retrieval-augmented generation systems and their architectures",
    "prompting-reasoning": "prompting, chain-of-thought reasoning and query transformation",
    "evaluation-benchmarks": "benchmarks, datasets, metrics and evaluation studies",
}

PROMPT_VARIANTS = {
    "definition": ANSWER_TEMPLATE,
    "mechanism": """You are explaining how something works, using retrieved excerpts.

Rules:
- Use only the context below. Do not use prior knowledge.
- If the context does not answer the question, say so plainly and stop.
- If no documents were retrieved at all, reply that no documents were retrieved
  and that you therefore cannot answer. Do not repeat this instruction back.
- Describe the mechanism in order, step by step.
- Cite the excerpts you used with their bracketed numbers, like [1] or [2].

Context:
{context}

Question: {question}

Answer:""",
    "comparison": """You are comparing approaches, using retrieved excerpts.

Rules:
- Use only the context below. Do not use prior knowledge.
- If the context does not answer the question, say so plainly and stop.
- If no documents were retrieved at all, reply that no documents were retrieved
  and that you therefore cannot answer. Do not repeat this instruction back.
- State what each approach does, then what distinguishes them.
- Cite the excerpts you used with their bracketed numbers, like [1] or [2].

Context:
{context}

Question: {question}

Answer:""",
}

PROMPT_EXEMPLARS = {
    "definition": "What is X? What does X mean? Define X.",
    "mechanism": "How does X work? How does X do Y? What are the steps in X?",
    "comparison": "How does X differ from Y? Compare X and Y. What sets X apart from Y?",
}


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
