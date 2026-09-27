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


def _pick_question(raw: str) -> str:
    """Find the general question in a reply that may be wrapped in chatter.

    The model is asked for one bare line and routinely ignores that, adding a
    preamble ("Sure, here's a more general question:"), a sign-off ("Hope that
    helps!"), or both. Neither positional rule survives that: taking the first
    line picks the preamble, taking the last picks the sign-off.

    So the question is identified by looking like one — a line ending in "?".
    Where several do, the first wins; where none does, the longest non-blank
    candidate is the best remaining guess, which handles a model that drops the
    question mark. Surrounding markdown emphasis is stripped, since models
    bold a single-line answer surprisingly often.
    """
    candidates = [c.strip().strip("*_").strip() for c in parse_query_list(raw)]
    candidates = [c for c in candidates if c]
    if not candidates:
        return ""
    questions = [c for c in candidates if c.endswith("?")]
    if questions:
        return questions[0]
    return max(candidates, key=len)


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

        general = _pick_question(raw)
        if not general:
            return degrade_to_direct(
                question, ctx, "step-back produced no general question"
            )

        ctx.trace.add_translation("step_back", general)
        queries = [question, general]
        ctx.trace.queries = queries
        lists = ctx.search(queries, ctx.config.top_k)
        return StrategyResult(retrieved=merge_best_score(lists)[: ctx.config.top_k])
