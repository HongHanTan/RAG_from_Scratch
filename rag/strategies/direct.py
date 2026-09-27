"""Plain top-k retrieval: the baseline every other strategy is measured against."""

from __future__ import annotations

from rag.strategies.base import StrategyContext, StrategyResult


class DirectStrategy:
    name = "direct"

    def run(self, question: str, ctx: StrategyContext) -> StrategyResult:
        ctx.trace.queries = [question]
        results = ctx.search([question], ctx.config.retrieval_depth)[0]
        return StrategyResult(retrieved=results[: ctx.config.top_k])
