"""RAG-Fusion: multi-query, then fuse the ranked lists by reciprocal rank.

The difference from multi-query is the combination step, and it matters
because the lists are not on a shared scale in any meaningful sense — a 0.6
from one rewrite is not the same evidence as a 0.6 from another. Reciprocal
rank fusion throws the scores away and rewards agreement on rank instead.
"""

from __future__ import annotations

from rag.llm import LLMError
from rag.prompts import MULTI_QUERY_TEMPLATE, parse_query_list
from rag.similarity import reciprocal_rank_fusion
from rag.strategies.base import StrategyContext, StrategyResult, degrade_to_direct


class RagFusionStrategy:
    name = "rag-fusion"

    def __init__(self, n: int = 5) -> None:
        self.n = n

    def run(self, question: str, ctx: StrategyContext) -> StrategyResult:
        if ctx.llm is None:
            return degrade_to_direct(question, ctx, "rag-fusion needs an LLM")

        try:
            with ctx.trace.stage("translate"):
                raw = ctx.llm.generate(
                    MULTI_QUERY_TEMPLATE.format(question=question, n=self.n)
                )
        except LLMError as exc:
            return degrade_to_direct(question, ctx, f"rag-fusion rewrite failed: {exc}")

        rewrites = [q for q in parse_query_list(raw) if q != question]
        if not rewrites:
            return degrade_to_direct(
                question, ctx, "rag-fusion rewrite produced no usable queries"
            )

        for rewrite in rewrites:
            ctx.trace.add_translation("query", rewrite)

        queries = [question, *rewrites]
        ctx.trace.queries = queries
        lists = ctx.search(queries, ctx.config.top_k)
        return StrategyResult(
            retrieved=reciprocal_rank_fusion(lists)[: ctx.config.top_k]
        )
