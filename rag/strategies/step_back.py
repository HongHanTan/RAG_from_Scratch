"""Step-back: also retrieve for a more general form of the question.

A narrow question matches narrow passages. The background a good answer needs
is often in a passage that explains the concept rather than the specific case,
and that passage does not match the narrow wording. Asking the broader
question too pulls it in.

Results are merged on best cosine score: both queries are embedded in the same
space, so the scores are comparable.
"""

from __future__ import annotations

from rag.llm import LLMError
from rag.prompts import STEP_BACK_TEMPLATE, parse_query_list
from rag.similarity import merge_best_score
from rag.strategies.base import StrategyContext, StrategyResult, degrade_to_direct


class StepBackStrategy:
    name = "step-back"

    def run(self, question: str, ctx: StrategyContext) -> StrategyResult:
        if ctx.llm is None:
            return degrade_to_direct(question, ctx, "step-back needs an LLM")

        try:
            with ctx.trace.stage("translate"):
                raw = ctx.llm.generate(STEP_BACK_TEMPLATE.format(question=question))
        except LLMError as exc:
            return degrade_to_direct(question, ctx, f"step-back failed: {exc}")

        # parse_query_list only drops unmarked chatter when some other line
        # IS marked (see its docstring). The model here is asked for a single
        # unmarked line, so a preamble it adds anyway ("Sure, here's a more
        # general question:") is not filtered out and survives as its own
        # candidate alongside the real question. Preambles precede the
        # payload and the template asks for no sign-off, so the last
        # candidate is the question; a bare reply or a single numbered line
        # both leave exactly one candidate, so this is a no-op for them.
        candidates = parse_query_list(raw)
        general = candidates[-1] if candidates else ""
        if not general:
            return degrade_to_direct(
                question, ctx, "step-back produced no general question"
            )

        ctx.trace.add_translation("step_back", general)
        queries = [question, general]
        ctx.trace.queries = queries
        lists = ctx.search(queries, ctx.config.top_k)
        return StrategyResult(retrieved=merge_best_score(lists)[: ctx.config.top_k])
