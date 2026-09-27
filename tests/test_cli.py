import pytest

from rag.__main__ import format_trace, main
from rag.chunking import Chunk
from rag.config import Config
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
