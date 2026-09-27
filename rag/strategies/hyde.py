"""HyDE: search with a hypothetical answer document instead of the question.

A question is short and interrogative; a document is long and declarative.
Embedded in the same space they land apart even when one answers the other.
HyDE closes that gap by having the model write a passage in the shape of a
document and searching with that.

The generated text is used only as a search key. It never enters the answer
prompt, so a hallucinated passage costs retrieval quality but cannot put false
statements in front of the user. By default the original question is searched
alongside it, and the two lists are combined by reciprocal rank fusion rather
than best cosine score. This is not a stylistic choice: a hypothetical
document is a long declarative passage and the question is short and
interrogative, so their cosine scores are systematically on different scales
(measured against the real index, hypothetical-document scores land around
0.7-0.8 against the question's 0.3-0.4). A best-score merge would then always
prefer the hypothetical document's results, silently making
`include_question` inert -- the question's hits would never surface even when
they alone found something relevant. Fusion decides by rank instead, so the
question's results genuinely compete rather than being outscored on a scale
that was never comparable to begin with. When `include_question` is false
there is only one list, so best-score merge is used and results stay labelled
`cosine`.
"""

from __future__ import annotations

from rag.llm import LLMError
from rag.prompts import HYDE_TEMPLATE
from rag.similarity import merge_best_score, reciprocal_rank_fusion
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
        if self.include_question:
            merged = reciprocal_rank_fusion(lists)
        else:
            merged = merge_best_score(lists)
        return StrategyResult(retrieved=merged[: ctx.config.top_k])
