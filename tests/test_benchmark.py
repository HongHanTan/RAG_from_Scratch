import pytest

from evaluation.benchmark import StrategyScore, format_table


def _scores():
    return [
        StrategyScore("direct", 0.62, 0.55, 0.58, 0.56, 41.6, 10),
        StrategyScore("rag-fusion", 0.74, 0.61, 0.66, 0.60, 1502.3, 10),
    ]


def test_table_has_a_row_per_strategy():
    table = format_table(_scores(), k=20)
    assert "direct" in table
    assert "rag-fusion" in table


def test_table_names_the_k_it_measured():
    assert "Recall@20" in format_table(_scores(), k=20)
    assert "nDCG@20" in format_table(_scores(), k=20)


def test_table_always_reports_doc_precision_at_5_regardless_of_k():
    # DocPrec is pinned to the answer prompt's real top_k (5), not the chunk
    # metrics' cutoff, so it must not move when --k changes.
    assert "DocPrec@5" in format_table(_scores(), k=20)
    assert "DocPrec@5" in format_table(_scores(), k=5)


def test_table_is_markdown():
    lines = format_table(_scores(), k=20).splitlines()
    assert lines[0].startswith("|")
    assert set(lines[1].replace("|", "").replace(" ", "")) <= {"-", ":"}


def test_table_reports_the_question_count():
    # A number computed over 10 questions must never be read as if it were
    # computed over 30.
    assert "10" in format_table(_scores(), k=20)


def test_table_rows_are_ordered_by_recall_descending():
    rows = [l for l in format_table(_scores(), k=20).splitlines() if l.startswith("|")]
    assert "rag-fusion" in rows[2]
    assert "direct" in rows[3]


def test_empty_scores_produce_a_table_with_no_rows():
    table = format_table([], k=20)
    assert table.splitlines()[0].startswith("|")


def test_a_degraded_trace_is_fatal():
    from evaluation.benchmark import check_not_degraded
    from rag.trace import Trace

    trace = Trace(question="q")
    trace.strategy = "hyde"
    trace.note("hyde needs an LLM; degraded to direct retrieval")
    with pytest.raises(RuntimeError, match="degraded"):
        check_not_degraded(trace)


def test_an_undegraded_trace_passes_the_check():
    from evaluation.benchmark import check_not_degraded
    from rag.trace import Trace

    trace = Trace(question="q")
    trace.note("generation served from cache")
    check_not_degraded(trace)
