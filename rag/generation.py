"""Turn retrieved chunks into an answer."""

from __future__ import annotations

from rag.chunking import RetrievedChunk
from rag.llm import LLMError
from rag.prompts import build_answer_prompt
from rag.trace import Trace


def generate_answer(
    llm,
    question: str,
    retrieved: list[RetrievedChunk],
    trace: Trace,
    extra_context: str | None = None,
) -> str | None:
    """Generate an answer, recording prompt, answer and timing on the trace.

    Returns None and records a note if the model call fails. Retrieval already
    succeeded at this point, so the trace is still worth showing — failing hard
    would throw away the useful half of the result.
    """
    prompt = build_answer_prompt(question, retrieved, extra_context=extra_context)
    trace.prompt = prompt
    with trace.stage("generate"):
        try:
            answer = llm.generate(prompt)
        except LLMError as exc:
            trace.note(f"generation failed: {exc}")
            return None
    if getattr(llm, "last_call_cached", False):
        trace.note("generation served from cache")
    trace.answer = answer
    return answer
