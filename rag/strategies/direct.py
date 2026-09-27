"""Plain top-k retrieval: the baseline every other strategy is measured against."""

from __future__ import annotations

from rag.strategies.base import StrategyContext, StrategyResult


class DirectStrategy:
    name = "direct"

    def run(self, question: str, ctx: StrategyContext) -> StrategyResult:
        ctx.trace.queries = [question]
        return StrategyResult(
            retrieved=ctx.search([question], ctx.config.top_k)[0]
        )
