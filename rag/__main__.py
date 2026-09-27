"""Command line entry point.

    python -m rag index
    python -m rag ask "what is reciprocal rank fusion?"
    python -m rag ask "..." --k 8 --trace
    python -m rag ask "..." --no-llm       # retrieval only, no API key needed

The embedder and LLM are built by injected factories so the CLI can be tested
without loading a model or holding a key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import Config
from rag.embedding import Embedder
from rag.llm import GeminiLLM, LLMError
from rag.pipeline import ask, build_index, load_index
from rag.trace import Trace


def load_config(**overrides) -> Config:
    return Config.from_env(env_file=Path(".env"), **overrides)


def default_embedder(config: Config) -> Embedder:
    return Embedder(config.embedding_model, max_length=config.max_seq_tokens)


def default_llm(config: Config) -> GeminiLLM:
    return GeminiLLM(
        model=config.llm_model,
        api_key=config.api_key,
        cache_dir=config.cache_dir,
    )


def format_trace(trace: Trace, verbose: bool) -> str:
    lines: list[str] = []

    if trace.answer:
        lines.append(trace.answer)
    else:
        lines.append("(no answer generated)")
    lines.append("")

    lines.append("Sources:")
    for item in trace.retrieved:
        chunk = item.chunk
        if verbose:
            lines.append(
                f"  [{item.rank}] {chunk.chunk_id}  score {item.score:.3f}  "
                f"chars {chunk.char_start}-{chunk.char_end}"
            )
            lines.append(f"      {chunk.text[:160]}")
        else:
            lines.append(f"  [{item.rank}] {chunk.doc_id} (chunk {chunk.index})")

    for note in trace.notes:
        lines.append(f"note: {note}")

    if verbose:
        lines.append("")
        lines.append("Timings:")
        for timing in trace.timings:
            lines.append(f"  {timing.name:<10} {timing.ms:8.1f} ms")
        lines.append(f"  {'total':<10} {trace.total_ms:8.1f} ms")
        if trace.prompt:
            lines.append("")
            lines.append("Prompt sent:")
            lines.append(trace.prompt)

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="RAG from scratch.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="build the vector index")
    index_parser.add_argument("--chunk-tokens", type=int)
    index_parser.add_argument("--chunk-overlap", type=int)

    ask_parser = subparsers.add_parser("ask", help="answer a question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--k", type=int, help="number of chunks to retrieve")
    ask_parser.add_argument("--trace", action="store_true", help="show scores, timings, prompt")
    ask_parser.add_argument("--no-llm", action="store_true", help="retrieve only")

    return parser


def _reconfigure_streams_for_utf8() -> None:
    """Make stdout/stderr tolerate non-ASCII output on any console.

    On Windows, stdout/stderr default to the console's legacy code page
    (often cp1252) whenever they are not attached to a real console --
    piped to a file, `| tee`, CI, etc. The corpus and generated answers can
    contain characters outside that code page, which raises
    UnicodeEncodeError deep inside print() and kills the CLI with a
    traceback instead of an answer. Reconfiguring to UTF-8 with
    errors="replace" means an unrepresentable character degrades to a
    replacement glyph instead of crashing the process.

    This is deliberately best-effort: some stream objects (tests substitute
    their own) don't have `reconfigure` at all, and a console-configuration
    problem must never prevent the CLI from answering.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main(
    argv: list[str] | None = None,
    embedder_factory=default_embedder,
    llm_factory=default_llm,
) -> int:
    _reconfigure_streams_for_utf8()

    args = _build_parser().parse_args(argv)

    if args.command == "ask" and args.k is not None and args.k < 0:
        print(f"--k must not be negative, got {args.k}", file=sys.stderr)
        return 1

    overrides = {}
    for field in ("chunk_tokens", "chunk_overlap"):
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value
    config = load_config(**overrides)

    if args.command == "index":
        embedder = embedder_factory(config)
        store = build_index(config, embedder)
        documents = len({c.doc_id for c in store.chunks})
        print(
            f"indexed {documents} documents into {len(store)} chunks "
            f"({store.dim}-d) -> {config.index_path}"
        )
        return 0

    try:
        store = load_index(config)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    embedder = embedder_factory(config)
    llm = None
    if not args.no_llm:
        try:
            llm = llm_factory(config)
        except LLMError as exc:
            print(f"{exc}\nretrieving without generation", file=sys.stderr)

    trace = ask(args.question, store, embedder, llm, config, k=args.k)
    print(format_trace(trace, verbose=args.trace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
