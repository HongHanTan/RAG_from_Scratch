"""Step-back: also retrieve for a more general form of the question.

A narrow question matches narrow passages. The background a good answer needs
is often in a passage that explains the concept rather than the specific case,
and that passage does not match the narrow wording. Asking the broader
question too pulls it in.

Results are merged on best cosine score: both queries are embedded in the same
space, so the scores are comparable.

The general question is obtained via structured output rather than parsed out
of prose. Phase 2's version took four attempts to get this reliably out of a
model reply -- first line, last candidate, shape-based, then chatter-filtering
-- and recorded that a structured field was the principled fix. Phase 4
introduces `GeminiLLM.structured` for logical routing, and step-back adopts it
here instead of growing more prose heuristics.
"""

from __future__ import annotations

from rag.llm import LLMError
from rag.prompts import STEP_BACK_JSON_TEMPLATE, STEP_BACK_SCHEMA
from rag.similarity import merge_best_score
from rag.strategies.base import StrategyContext, StrategyResult, degrade_to_direct


class StepBackStrategy:
    name = "step-back"

    def run(self, question: str, ctx: StrategyContext) -> StrategyResult:
        if ctx.llm is None:
            return degrade_to_direct(question, ctx, "step-back needs an LLM")

        try:
            with ctx.trace.stage("translate"):
                parsed = ctx.llm.structured(
                    STEP_BACK_JSON_TEMPLATE.format(question=question),
                    STEP_BACK_SCHEMA,
                )
        except LLMError as exc:
            return degrade_to_direct(question, ctx, f"step-back failed: {exc}")

        general = parsed["question"].strip()
        if not general:
            return degrade_to_direct(
                question, ctx, "step-back produced no general question"
            )

        ctx.trace.add_translation("step_back", general)
        queries = [question, general]
        ctx.trace.queries = queries
        lists = ctx.search(queries, ctx.config.retrieval_depth)
        with ctx.trace.stage("merge"):
            merged = merge_best_score(lists)
        return StrategyResult(retrieved=merged[: ctx.config.top_k])
