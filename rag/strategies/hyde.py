"""HyDE: search with a hypothetical answer document instead of the question.

A question is short and interrogative; a document is long and declarative.
Embedded in the same space they land apart even when one answers the other.
HyDE closes that gap by having the model write a passage in the shape of a
document and searching with that.

The generated text is used only as a search key. It never enters the answer
prompt, so a hallucinated passage costs retrieval quality but cannot put false
statements in front of the user. By default the original question is searched
alongside it, so a bad hypothetical degrades the result rather than replacing
it.
"""

from __future__ import annotations

from rag.llm import LLMError
from rag.prompts import HYDE_TEMPLATE
from rag.similarity import merge_best_score
from rag.strategies.base import StrategyContext, StrategyResult, degrade_to_direct


class HydeStrategy:
    name = "hyde"

    def __init__(self, include_question: bool = True) -> None:
        self.include_question = include_question

    def run(self, question: str, ctx: StrategyContext) -> StrategyResult:
        if ctx.llm is None:
            return degrade_to_direct(question, ctx, "hyde needs an LLM")

        try:
            with ctx.trace.stage("translate"):
                raw = ctx.llm.generate(HYDE_TEMPLATE.format(question=question))
        except LLMError as exc:
            return degrade_to_direct(question, ctx, f"hyde generation failed: {exc}")

        document = raw.strip()
        if not document:
            return degrade_to_direct(
                question, ctx, "hyde produced an empty hypothetical document"
            )

        ctx.trace.add_translation("hypothetical", document)
        queries = [question, document] if self.include_question else [document]
        ctx.trace.queries = queries
        lists = ctx.search(queries, ctx.config.top_k)
        return StrategyResult(retrieved=merge_best_score(lists)[: ctx.config.top_k])
