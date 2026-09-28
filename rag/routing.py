"""PDF Stage 2: decide where to search, and how to answer.

Two different mechanisms, deliberately kept apart:

- **Logical routing** asks the model which collections could hold the answer.
  It needs reasoning about what each collection contains, which is what an
  LLM is for, and it uses structured output so the answer is a field rather
  than prose to be parsed.
- **Semantic routing** picks which answer prompt to use by embedding the
  question and comparing it to embedded exemplar questions for each prompt. No
  LLM call — if it needed one it would be logical routing with extra steps.
  The exemplars are questions, not descriptions of question types: matching a
  question against a *description of an intent* asks the embedder to encode
  intent, and a model like MiniLM encodes topic instead. Matching a question
  against other questions compares like with like. This is the same insight
  HyDE rests on — embed something shaped like what you are searching for —
  applied to routing instead of retrieval.

Both degrade to their unrestricted default and record it: routing narrows
things, so a routing failure should widen the search back rather than
narrowing it wrongly.
"""

from __future__ import annotations

import numpy as np

from rag.llm import LLMError
from rag.prompts import ROUTE_SCHEMA, ROUTE_TEMPLATE, TOPIC_DESCRIPTIONS
from rag.similarity import cosine_similarity


def logical_route(
    question: str, llm, topics: tuple[str, ...], trace
) -> tuple[str, ...]:
    """Which collections to search. An empty tuple means all of them."""
    if llm is None:
        trace.degraded("logical routing needs an LLM", "searching everything")
        return ()

    described = "\n".join(
        f"- {name}: {TOPIC_DESCRIPTIONS.get(name, name)}" for name in topics
    )
    prompt = ROUTE_TEMPLATE.format(question=question, descriptions=described)
    try:
        with trace.stage("route"):
            parsed = llm.structured(prompt, ROUTE_SCHEMA)
    except LLMError as exc:
        trace.degraded(f"logical routing failed: {exc}", "searching everything")
        return ()

    chosen = tuple(t for t in parsed.get("topics", []) if t in topics)
    if not chosen:
        trace.degraded("logical routing chose nothing valid", "searching everything")
        return ()
    trace.add_translation("route", f"search {', '.join(chosen)}")
    return chosen


class SemanticRouter:
    """Picks an answer prompt by cosine similarity, with no LLM call.

    Compares the question against embedded *exemplar questions* for each
    prompt, not against prose descriptions of what each prompt is for — a
    description asks the embedder to encode intent, while an exemplar is
    question-shaped like the thing it is being matched against.

    Exemplars are embedded once at construction; each question costs one
    embedding and one matmul against a handful of vectors.
    """

    def __init__(self, embedder, exemplars: dict[str, str]) -> None:
        if not exemplars:
            raise ValueError("a semantic router needs at least one exemplar")
        self.names = tuple(exemplars)
        self._embedder = embedder
        self._vectors = embedder.encode([exemplars[n] for n in self.names])

    def route(self, question: str, trace) -> str:
        """The name of the best-matching prompt."""
        query = self._embedder.encode([question])
        scores = cosine_similarity(query, self._vectors)[0]
        chosen = self.names[int(np.argmax(scores))]
        trace.add_translation(
            "route", f"prompt {chosen} (cosine {float(scores.max()):.3f})"
        )
        return chosen
