import pytest

from rag.llm import LLMError
from rag.summarise import (
    CLUSTER_SUMMARY_TEMPLATE,
    DOC_SUMMARY_TEMPLATE,
    SummaryError,
    summarise,
)


class ReplyLLM:
    def __init__(self, reply="a summary"):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.reply


def test_returns_the_models_summary():
    assert summarise("some text", ReplyLLM("the summary"), DOC_SUMMARY_TEMPLATE) == "the summary"


def test_sends_the_text_to_the_model():
    llm = ReplyLLM()
    summarise("distinctive body text", llm, DOC_SUMMARY_TEMPLATE)
    assert "distinctive body text" in llm.prompts[0]


def test_long_text_is_truncated_before_sending():
    # A 200KB paper would blow the context window and cost a fortune; the
    # summary only needs enough to characterise the document.
    llm = ReplyLLM()
    summarise("x" * 50_000, llm, DOC_SUMMARY_TEMPLATE, max_chars=1000)
    assert len(llm.prompts[0]) < 5_000


def test_truncation_keeps_the_beginning():
    # A paper states its contribution in its opening; truncating from the
    # front would throw away the most summarisable part.
    llm = ReplyLLM()
    summarise("HEADLINE" + "x" * 50_000, llm, DOC_SUMMARY_TEMPLATE, max_chars=1000)
    assert "HEADLINE" in llm.prompts[0]


def test_short_text_is_not_truncated():
    llm = ReplyLLM()
    summarise("short", llm, DOC_SUMMARY_TEMPLATE, max_chars=1000)
    assert "short" in llm.prompts[0]


def test_an_empty_summary_is_an_error():
    with pytest.raises(SummaryError, match="empty"):
        summarise("text", ReplyLLM("   "), DOC_SUMMARY_TEMPLATE)


def test_an_llm_failure_becomes_a_summary_error():
    class Failing:
        def generate(self, prompt):
            raise LLMError("rate limited")

    with pytest.raises(SummaryError, match="rate limited"):
        summarise("text", Failing(), DOC_SUMMARY_TEMPLATE)


def test_both_templates_have_a_text_placeholder():
    for template in (DOC_SUMMARY_TEMPLATE, CLUSTER_SUMMARY_TEMPLATE):
        assert "{text}" in template


def test_the_cluster_template_asks_for_shared_themes():
    # A cluster summary that just concatenates its members is useless as an
    # abstraction layer; the prompt has to ask for what they have in common.
    lowered = CLUSTER_SUMMARY_TEMPLATE.lower()
    assert "common" in lowered or "shared" in lowered or "theme" in lowered
