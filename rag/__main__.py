"""Command line entry point.

    python -m rag index
    python -m rag ask "what is reciprocal rank fusion?"
    python -m rag ask "..." --k 8 --trace
    python -m rag ask "..." --no-llm       # retrieval only, no API key needed
    python -m rag ask "..." --strategy hyde --queries
    python -m rag ask "..." --strategy decomposition --decomposition-mode independent

The embedder and LLM are built by injected factories so the CLI can be tested
without loading a model or holding a key. `--strategy` other than `direct`
needs an LLM, so it is rejected together with `--no-llm`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import Config
from rag.embedding import Embedder
from rag.llm import GeminiLLM, LLMError
from rag.pipeline import ask, build_index, load_index
from rag.strategies import STRATEGY_NAMES
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


def format_trace(trace: Trace, verbose: bool, show_queries: bool = False) -> str:
    lines: list[str] = []

    if trace.answer:
        lines.append(trace.answer)
    else:
        lines.append("(no answer generated)")
    lines.append("")

    show_strategy = show_queries or verbose
    if show_strategy:
        lines.append(f"Strategy: {trace.strategy}")
    if show_strategy and trace.translation:
        for step in trace.translation:
            label = step.kind.replace("_", " ")
            if verbose:
                text = step.text
            else:
                text = step.text[:120]
                if len(step.text) > 120:
                    text += "…"
            lines.append(f"  {label}: {text}")
    if show_strategy:
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
            indent = "  " * (timing.depth + 1)
            lines.append(f"{indent}{timing.name:<10} {timing.ms:8.1f} ms")
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
    ask_parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGY_NAMES),
        default="direct",
        help="query translation strategy (default: direct)",
    )
    ask_parser.add_argument(
        "--decomposition-mode",
        choices=("recursive", "independent"),
        default=None,
        help="how decomposition combines sub-answers (default: recursive)",
    )
    ask_parser.add_argument(
        "--queries",
        action="store_true",
        help="show the queries the strategy generated",
    )

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


def _run(args, embedder_factory, llm_factory) -> int:
    """Execute one parsed command.

    Raises ValueError or FileNotFoundError for anything the user can fix;
    main() turns those into an exit code and a message.
    """
    if args.command == "ask" and args.k is not None and args.k < 0:
        raise ValueError(f"--k must not be negative, got {args.k}")

    if args.command == "ask" and args.no_llm and args.strategy != "direct":
        raise ValueError(
            f"--no-llm cannot be combined with --strategy {args.strategy}: "
            f"{args.strategy} needs the LLM to rewrite the question, so it "
            "would silently fall back to direct retrieval. Use --strategy "
            "direct, or drop --no-llm."
        )

    if (
        args.command == "ask"
        and args.decomposition_mode is not None
        and args.strategy != "decomposition"
    ):
        raise ValueError(
            f"--decomposition-mode cannot be combined with --strategy "
            f"{args.strategy}: it only affects --strategy decomposition and "
            "would otherwise be silently ignored. Use --strategy "
            "decomposition, or drop --decomposition-mode."
        )

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

    store = load_index(config)

    embedder = embedder_factory(config)
    llm = None
    if not args.no_llm:
        try:
            llm = llm_factory(config)
        except LLMError as exc:
            if args.strategy != "direct":
                raise ValueError(
                    f"--strategy {args.strategy} needs an LLM to rewrite the "
                    f"question, but one could not be built: {exc}"
                ) from exc
            print(f"{exc}\nretrieving without generation", file=sys.stderr)

    strategy_options: dict = {}
    if args.strategy == "decomposition":
        strategy_options["mode"] = args.decomposition_mode or "recursive"

    trace = ask(
        args.question,
        store,
        embedder,
        llm,
        config,
        k=args.k,
        strategy=args.strategy,
        strategy_options=strategy_options,
    )
    print(format_trace(trace, verbose=args.trace, show_queries=args.queries))
    return 0


def main(
    argv: list[str] | None = None,
    embedder_factory=default_embedder,
    llm_factory=default_llm,
) -> int:
    """Entry point.

    Every failure the user can act on -- a bad flag, an invalid config, a
    malformed metadata.json, a missing index -- exits 1 with the message and
    no traceback. Bugs still raise, so they stay visible.
    """
    _reconfigure_streams_for_utf8()
    args = _build_parser().parse_args(argv)
    try:
        return _run(args, embedder_factory, llm_factory)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
