"""Decomposition: split the question, answer the parts, then answer the whole.

Two forms, as described in RAG.pdf:

- "recursive" (IR-CoT): each sub-answer is carried into the next sub-question's
  prompt, so later steps can build on earlier ones. Necessary when the parts
  are genuinely dependent — you cannot answer the second without the first.
- "independent": every sub-question is answered on its own and the pairs are
  concatenated at the end. Cheaper to reason about, and correct when the parts
  do not depend on each other.

Both return the sub-answers as `extra_context` rather than as retrieved chunks,
because they are generated text. The answer prompt labels them as working
notes so the model does not cite them as sources.
"""

from __future__ import annotations

from rag.llm import LLMError
from rag.prompts import (
    DECOMPOSE_TEMPLATE,
    PRIOR_ANSWERS_HEADER,
    SUB_ANSWER_TEMPLATE,
    format_context,
    parse_query_list,
)
from rag.similarity import merge_best_score
from rag.strategies.base import StrategyContext, StrategyResult, degrade_to_direct

MODES = ("recursive", "independent")


class DecompositionStrategy:
    name = "decomposition"

    def __init__(self, mode: str = "recursive", max_sub_questions: int = 3) -> None:
        if mode not in MODES:
            raise ValueError(
                f"unknown decomposition mode: {mode} (choose {' or '.join(MODES)})"
            )
        self.mode = mode
        self.max_sub_questions = max_sub_questions

    def run(self, question: str, ctx: StrategyContext) -> StrategyResult:
        if ctx.llm is None:
            return degrade_to_direct(question, ctx, "decomposition needs an LLM")

        # One outer stage wrapping everything, so total_ms counts this work
        # once rather than adding up each nested retrieval and LLM call.
        with ctx.trace.stage("decompose"):
            try:
                raw = ctx.llm.generate(
                    DECOMPOSE_TEMPLATE.format(
                        question=question, n=self.max_sub_questions
                    )
                )
            except LLMError as exc:
                return degrade_to_direct(
                    question, ctx, f"decomposition failed: {exc}"
                )

            sub_questions = parse_query_list(raw)[: self.max_sub_questions]
            if not sub_questions:
                return degrade_to_direct(
                    question, ctx, "decomposition produced no sub-questions"
                )

            pairs: list[tuple[str, str]] = []
            all_results = []
            for sub_question in sub_questions:
                ctx.trace.add_translation("sub_question", sub_question)
                results = ctx.search([sub_question], ctx.config.retrieval_depth)[0]
                all_results.append(results)

                prior = ""
                if self.mode == "recursive" and pairs:
                    prior = PRIOR_ANSWERS_HEADER.format(
                        prior="\n".join(f"Q: {q}\nA: {a}" for q, a in pairs)
                    )

                try:
                    answer = ctx.llm.generate(
                        SUB_ANSWER_TEMPLATE.format(
                            question=sub_question,
                            context=format_context(results),
                            prior=prior,
                        )
                    ).strip()
                except LLMError as exc:
                    # One failed sub-question should not lose the others, but
                    # the strategy is now running on fewer sub-answers than
                    # the method specifies -- a genuine partial degradation.
                    ctx.trace.degraded(
                        f"sub-question failed: {exc}", "continuing without it"
                    )
                    continue

                ctx.trace.add_translation("sub_answer", answer)
                pairs.append((sub_question, answer))

            ctx.trace.queries = sub_questions
            retrieved = merge_best_score(all_results)[: ctx.config.top_k]

        extra = (
            "\n\n".join(f"Q: {q}\nA: {a}" for q, a in pairs) if pairs else None
        )
        return StrategyResult(retrieved=retrieved, extra_context=extra)
