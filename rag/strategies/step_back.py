"""Step-back: also retrieve for a more general form of the question.

A narrow question matches narrow passages. The background a good answer needs
is often in a passage that explains the concept rather than the specific case,
and that passage does not match the narrow wording. Asking the broader
question too pulls it in.

Results are merged on best cosine score: both queries are embedded in the same
space, so the scores are comparable.
"""

from __future__ import annotations

import re

from rag.llm import LLMError
from rag.prompts import STEP_BACK_TEMPLATE, parse_query_list
from rag.similarity import merge_best_score
from rag.strategies.base import StrategyContext, StrategyResult, degrade_to_direct


CHATTER = re.compile(
    r"\b(?:you|your|me|my|let me know|hope|anything else|here you go)\b",
    re.IGNORECASE,
)
"""Marks a line that talks to the reader rather than about the subject.

The general question is about the subject matter; a preamble or sign-off is
about the interaction. Second and first person is the cheapest signal that
separates them.

The bare word "i" is deliberately not one of the alternatives. Matched
case-insensitively it also matches the "I" in "I/O", "AI", or similar
abbreviations, which are plausible substance in an IR corpus rather than the
model addressing the reader ("What is I/O batching in dense retrieval?" is a
legitimate step-back question, not chatter). The remaining markers still
catch the common preambles and sign-offs.
"""


def _pick_question(raw: str) -> str:
    """Find the general question in a reply that may be wrapped in chatter.

    The model is asked for one bare line and routinely ignores that, adding a
    preamble, a sign-off, or both. Position does not identify the question:
    taking the first line picks the preamble, taking the last picks the
    sign-off, and a preamble phrased as a question ("Are you asking about
    this in general?") defeats "first line ending in ?" too.

    So candidates that address the reader are dropped first, then the
    remainder is filtered by shape. If dropping would leave nothing, the
    unfiltered candidates are used — a legitimate question can contain "you"
    ("How do you measure similarity?"), and losing it entirely would be worse
    than occasionally keeping a preamble.

    This is a heuristic and it has a ceiling. The principled fix is to make
    the model return a structured field rather than prose, which is the
    mechanism Phase 4 introduces for logical routing; step-back should adopt
    it then rather than growing more rules here.
    """
    candidates = [c.strip().strip("*_").strip() for c in parse_query_list(raw)]
    candidates = [c for c in candidates if c]
    if not candidates:
        return ""

    substantive = [c for c in candidates if not CHATTER.search(c)] or candidates
    questions = [c for c in substantive if c.endswith("?")]
    if questions:
        return questions[0]
    return max(substantive, key=len)


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
