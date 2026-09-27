"""Multi-query: ask the same thing several ways, union the results.

A question and the documents that answer it often use different words. Several
rewrites give several chances for one of them to land near the right chunks.
Results are merged on best cosine score, since every rewrite is embedded in
the same space and the scores are directly comparable.
"""

from __future__ import annotations

from rag.llm import LLMError
from rag.prompts import MULTI_QUERY_TEMPLATE, parse_query_list
from rag.similarity import merge_best_score
from rag.strategies.base import StrategyContext, StrategyResult, degrade_to_direct


class MultiQueryStrategy:
    name = "multi-query"

    def __init__(self, n: int = 5) -> None:
        self.n = n

    def run(self, question: str, ctx: StrategyContext) -> StrategyResult:
        if ctx.llm is None:
            return degrade_to_direct(question, ctx, "multi-query needs an LLM")

        try:
            with ctx.trace.stage("translate"):
                raw = ctx.llm.generate(
                    MULTI_QUERY_TEMPLATE.format(question=question, n=self.n)
                )
        except LLMError as exc:
            return degrade_to_direct(question, ctx, f"multi-query rewrite failed: {exc}")

        rewrites = [q for q in parse_query_list(raw) if q != question]
        if not rewrites:
            return degrade_to_direct(
                question, ctx, "multi-query rewrite produced no usable queries"
            )

        for rewrite in rewrites:
            ctx.trace.add_translation("query", rewrite)

        queries = [question, *rewrites]
        ctx.trace.queries = queries
        lists = ctx.search(queries, ctx.config.top_k)
        return StrategyResult(retrieved=merge_best_score(lists)[: ctx.config.top_k])
