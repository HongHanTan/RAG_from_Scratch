import sys

import pytest

from rag.__main__ import format_trace, main
from rag.chunking import Chunk
from rag.config import Config
from rag.llm import LLMError
from rag.trace import RetrievedChunk, StageTiming, Trace
from tests.conftest import FakeEmbedder, FakeLLM


def _trace() -> Trace:
    trace = Trace(question="what is RRF?")
    trace.queries = ["what is RRF?"]
    trace.retrieved = [
        RetrievedChunk(
            chunk=Chunk("beta:1", "beta", 1, "RRF sums reciprocal ranks.", 0, 5, 0, 26),
            score=0.83,
            rank=1,
        )
    ]
    trace.answer = "It fuses ranked lists. [1]"
    trace.timings = [StageTiming("embed", 12.5), StageTiming("search", 0.4)]
    return trace


# --- output formatting ------------------------------------------------------

def test_output_includes_the_answer():
    assert "It fuses ranked lists. [1]" in format_trace(_trace(), verbose=False)


def test_output_lists_sources_even_without_verbose():
    # Sources are the point of RAG; hiding them behind a flag defeats it.
    assert "beta" in format_trace(_trace(), verbose=False)


def test_verbose_output_includes_scores():
    assert "0.83" in format_trace(_trace(), verbose=True)


def test_verbose_output_includes_timings():
    output = format_trace(_trace(), verbose=True)
    assert "embed" in output
    assert "12.5" in output


def test_verbose_output_includes_the_prompt():
    trace = _trace()
    trace.prompt = "THE PROMPT TEXT"
    assert "THE PROMPT TEXT" in format_trace(trace, verbose=True)


def test_non_verbose_output_omits_the_prompt():
    trace = _trace()
    trace.prompt = "THE PROMPT TEXT"
    assert "THE PROMPT TEXT" not in format_trace(trace, verbose=False)


def test_notes_are_always_shown():
    trace = _trace()
    trace.note("generation failed: rate limited")
    assert "rate limited" in format_trace(trace, verbose=False)


def test_missing_answer_is_reported_not_printed_as_none():
    trace = _trace()
    trace.answer = None
    output = format_trace(trace, verbose=False)
    assert "None" not in output


# --- command dispatch -------------------------------------------------------

def _factories():
    return {
        "embedder_factory": lambda config: FakeEmbedder(),
        "llm_factory": lambda config: FakeLLM("stub answer"),
    }


def test_index_command_builds_the_index(tiny_corpus: Config, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    assert main(["index"], **_factories()) == 0
    assert tiny_corpus.index_path.is_file()
    assert "chunks" in capsys.readouterr().out


def test_ask_command_prints_an_answer(tiny_corpus: Config, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())
    assert main(["ask", "what is cosine?"], **_factories()) == 0
    assert "stub answer" in capsys.readouterr().out


def test_ask_without_an_index_fails_with_guidance(
    tiny_corpus: Config, monkeypatch, capsys
):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    assert main(["ask", "q"], **_factories()) == 1
    assert "rag index" in capsys.readouterr().err


def test_no_llm_flag_skips_generation(tiny_corpus: Config, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())

    def exploding_llm_factory(config):
        raise AssertionError("--no-llm must not construct an LLM")

    assert main(
        ["ask", "q", "--no-llm"],
        embedder_factory=lambda config: FakeEmbedder(),
        llm_factory=exploding_llm_factory,
    ) == 0
    assert "retrieval only" in capsys.readouterr().out


def test_no_subcommand_prints_usage_and_fails():
    with pytest.raises(SystemExit):
        main([])


# --- Windows console encoding (Finding 1) -----------------------------------
#
# The real bug only reproduces when stdout/stderr are backed by an actual
# cp1252 console (see the subprocess repro in the task report). A capsys-based
# test can't see that, because pytest swaps in its own stream. What we CAN
# assert at unit level: main() attempts to reconfigure stdout/stderr to UTF-8
# with errors="replace", and it tolerates a stream object that has no
# `reconfigure` attribute at all (as some substitutes, including capsys's,
# may not).


class _RecordingStream:
    """A stream that records reconfigure() calls and collects writes."""

    def __init__(self):
        self.reconfigure_calls = []
        self.written = []

    def reconfigure(self, **kwargs):
        self.reconfigure_calls.append(kwargs)

    def write(self, text):
        self.written.append(text)

    def flush(self):
        pass


class _NoReconfigureStream:
    """A stream lacking `reconfigure`, like some substitutes (e.g. capsys)."""

    def __init__(self):
        self.written = []

    def write(self, text):
        self.written.append(text)

    def flush(self):
        pass


def test_main_reconfigures_stdout_and_stderr_to_utf8_with_replace(
    tiny_corpus: Config, monkeypatch
):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())

    fake_out = _RecordingStream()
    fake_err = _RecordingStream()
    monkeypatch.setattr(sys, "stdout", fake_out)
    monkeypatch.setattr(sys, "stderr", fake_err)

    assert main(["ask", "q", "--no-llm"], **_factories()) == 0
    assert fake_out.reconfigure_calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert fake_err.reconfigure_calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_main_tolerates_stdout_without_reconfigure(tiny_corpus: Config, monkeypatch):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())

    fake_out = _NoReconfigureStream()
    fake_err = _NoReconfigureStream()
    monkeypatch.setattr(sys, "stdout", fake_out)
    monkeypatch.setattr(sys, "stderr", fake_err)

    # Must not raise even though these streams have no `reconfigure` at all.
    assert main(["ask", "q", "--no-llm"], **_factories()) == 0
    assert any("retrieval only" in chunk for chunk in fake_out.written)


def test_importing_main_module_does_not_touch_stdout(monkeypatch):
    # Reconfiguration must happen inside main(), not at import time.
    fake_out = _RecordingStream()
    monkeypatch.setattr(sys, "stdout", fake_out)
    import importlib

    import rag.__main__ as main_module

    importlib.reload(main_module)
    assert fake_out.reconfigure_calls == []


# --- negative --k (Finding 2) ------------------------------------------------


def test_negative_k_exits_with_clear_message_and_no_traceback(
    tiny_corpus: Config, monkeypatch, capsys
):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())

    assert main(["ask", "q", "--k", "-1"], **_factories()) == 1
    err = capsys.readouterr().err
    assert "--k" in err
    assert "-1" in err
    assert "Traceback" not in err


def test_k_zero_still_behaves_sanely(tiny_corpus: Config, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())

    assert main(["ask", "q", "--k", "0"], **_factories()) == 0
    err = capsys.readouterr().err
    assert "Traceback" not in err


# --- uniform error handling --------------------------------------------------


def _factories_that_should_not_run():
    def boom(config):
        raise AssertionError("factories must not run when config is invalid")

    return {"embedder_factory": boom, "llm_factory": boom}


def test_invalid_chunk_tokens_reports_cleanly(capsys):
    assert main(["index", "--chunk-tokens", "0"], **_factories_that_should_not_run()) == 1
    err = capsys.readouterr().err
    assert "chunk_tokens" in err
    assert "Traceback" not in err


def test_chunk_overlap_larger_than_chunk_tokens_reports_cleanly(capsys):
    assert (
        main(["index", "--chunk-overlap", "500"], **_factories_that_should_not_run())
        == 1
    )
    err = capsys.readouterr().err
    assert "chunk_overlap" in err
    assert "Traceback" not in err


def test_negative_k_still_names_the_flag(capsys):
    assert main(["ask", "q", "--k", "-1"], **_factories_that_should_not_run()) == 1
    assert "--k" in capsys.readouterr().err


def test_malformed_metadata_reports_cleanly(tiny_corpus, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    tiny_corpus.metadata_path.write_text("{not json", encoding="utf-8")

    from tests.conftest import FakeEmbedder

    assert main(["index"], embedder_factory=lambda c: FakeEmbedder()) == 1
    err = capsys.readouterr().err
    assert "metadata" in err.lower()
    assert "Traceback" not in err


# --- Task 9: --strategy, trace rendering ------------------------------------


def test_strategy_flag_reaches_the_pipeline(tiny_corpus, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())
    seen = {}

    real_ask = __import__("rag.__main__", fromlist=["ask"]).ask

    def spy(question, store, embedder, llm, config, k=None, strategy="direct",
            strategy_options=None):
        seen["strategy"] = strategy
        seen["options"] = strategy_options
        return real_ask(question, store, embedder, llm, config, k=k,
                        strategy=strategy, strategy_options=strategy_options)

    monkeypatch.setattr("rag.__main__.ask", spy)
    assert main(["ask", "q", "--strategy", "direct"], **_factories()) == 0
    assert seen["strategy"] == "direct"


def test_decomposition_mode_is_passed_as_a_strategy_option(
    tiny_corpus, monkeypatch, capsys
):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())
    seen = {}

    def spy(question, store, embedder, llm, config, k=None, strategy="direct",
            strategy_options=None):
        seen["options"] = strategy_options
        from rag.trace import Trace

        return Trace(question=question)

    monkeypatch.setattr("rag.__main__.ask", spy)
    main(
        ["ask", "q", "--strategy", "decomposition", "--decomposition-mode",
         "independent"],
        **_factories(),
    )
    assert seen["options"] == {"mode": "independent"}


def test_decomposition_mode_with_a_different_strategy_is_rejected(
    tiny_corpus, monkeypatch, capsys
):
    # --decomposition-mode only means something for --strategy decomposition.
    # Silently ignoring it on another strategy would make the user think it
    # took effect when it did nothing at all.
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())
    assert (
        main(
            ["ask", "q", "--strategy", "hyde", "--decomposition-mode", "independent"],
            **_factories(),
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "--decomposition-mode" in err
    assert "hyde" in err


def test_decomposition_mode_defaults_to_recursive_when_not_passed(
    tiny_corpus, monkeypatch, capsys
):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())
    seen = {}

    def spy(question, store, embedder, llm, config, k=None, strategy="direct",
            strategy_options=None):
        seen["options"] = strategy_options
        from rag.trace import Trace

        return Trace(question=question)

    monkeypatch.setattr("rag.__main__.ask", spy)
    main(["ask", "q", "--strategy", "decomposition"], **_factories())
    assert seen["options"] == {"mode": "recursive"}


def test_an_llm_strategy_with_no_llm_is_rejected(tiny_corpus, monkeypatch, capsys):
    # --no-llm means no rewrites are possible, so every strategy would silently
    # degrade to direct. Saying so beats pretending the flag did something.
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())
    assert main(["ask", "q", "--strategy", "hyde", "--no-llm"], **_factories()) == 1
    err = capsys.readouterr().err
    assert "--no-llm" in err
    assert "hyde" in err


def test_direct_strategy_with_no_llm_is_allowed(tiny_corpus, monkeypatch, capsys):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())
    assert main(["ask", "q", "--strategy", "direct", "--no-llm"], **_factories()) == 0


def test_llm_build_failure_is_fatal_for_a_non_direct_strategy(
    tiny_corpus, monkeypatch, capsys
):
    # A missing key, bad auth, or exhausted quota must not silently degrade
    # every strategy to direct retrieval and exit 0 -- that would make a
    # benchmark loop over several strategies report identical rows for all of
    # them with no indication anything went wrong.
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())

    def failing_llm_factory(config):
        raise LLMError("no API key: set GOOGLE_API_KEY in the environment or in .env")

    assert (
        main(
            ["ask", "q", "--strategy", "hyde"],
            embedder_factory=lambda config: FakeEmbedder(),
            llm_factory=failing_llm_factory,
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "hyde" in err
    assert "GOOGLE_API_KEY" in err
    assert "Traceback" not in err


def test_llm_build_failure_still_degrades_for_the_direct_strategy(
    tiny_corpus, monkeypatch, capsys
):
    # direct retrieval never needed the LLM in the first place, so a missing
    # key should still fall back to retrieval-only rather than fail the run.
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    main(["index"], **_factories())

    def failing_llm_factory(config):
        raise LLMError("no API key: set GOOGLE_API_KEY in the environment or in .env")

    assert (
        main(
            ["ask", "q", "--strategy", "direct"],
            embedder_factory=lambda config: FakeEmbedder(),
            llm_factory=failing_llm_factory,
        )
        == 0
    )
    assert "retrieving without generation" in capsys.readouterr().err


def test_unknown_strategy_is_rejected_by_argparse(tiny_corpus, monkeypatch):
    monkeypatch.setattr("rag.__main__.load_config", lambda **kw: tiny_corpus)
    with pytest.raises(SystemExit):
        main(["ask", "q", "--strategy", "nope"], **_factories())


def test_queries_flag_prints_the_rewritten_queries():
    from rag.trace import Trace

    trace = Trace(question="original")
    trace.strategy = "multi-query"
    trace.queries = ["original", "rewrite one", "rewrite two"]
    trace.add_translation("query", "rewrite one")
    trace.add_translation("query", "rewrite two")
    output = format_trace(trace, verbose=False, show_queries=True)
    assert "rewrite one" in output
    assert "rewrite two" in output


def test_trace_output_names_the_strategy():
    from rag.trace import Trace

    trace = Trace(question="q")
    trace.strategy = "rag-fusion"
    assert "rag-fusion" in format_trace(trace, verbose=True)


def test_verbose_output_shows_translation_steps():
    from rag.trace import Trace

    trace = Trace(question="q")
    trace.strategy = "hyde"
    trace.add_translation("hypothetical", "a hypothetical passage")
    assert "a hypothetical passage" in format_trace(trace, verbose=True)


def test_strategy_line_is_not_duplicated_in_verbose_output_with_translation():
    from rag.trace import Trace

    trace = Trace(question="q")
    trace.strategy = "hyde"
    trace.add_translation("hypothetical", "a hypothetical passage")
    output = format_trace(trace, verbose=True)
    assert output.count("Strategy: hyde") == 1


def test_truncated_translation_text_gets_an_ellipsis():
    from rag.trace import Trace

    trace = Trace(question="q")
    trace.strategy = "hyde"
    trace.add_translation("hypothetical", "x" * 200)
    output = format_trace(trace, verbose=False, show_queries=True)
    assert "…" in output


def test_untruncated_translation_text_gets_no_ellipsis():
    from rag.trace import Trace

    trace = Trace(question="q")
    trace.strategy = "hyde"
    trace.add_translation("hypothetical", "short passage")
    output = format_trace(trace, verbose=False, show_queries=True)
    assert "…" not in output


def test_nested_timings_are_indented_in_verbose_output():
    from rag.trace import StageTiming, Trace

    trace = Trace(question="q")
    trace.timings = [
        StageTiming("embed", 1.0, depth=1),
        StageTiming("decompose", 5.0, depth=0),
    ]
    output = format_trace(trace, verbose=True)
    assert "  embed" in output
    assert "5.0" in output
