"""Run every strategy over the gold set and report how they compare.

Four deliberate choices, each one a thing that would otherwise make the
numbers lie:

- **Chunk metrics are measured at k=20, not the answer prompt's k=5.**
  Calibrating this gold set against plain retrieval showed Recall@5 pinned
  near zero for every strategy (0.050 on plain retrieval) — there is no
  headroom to distinguish strategies at that depth. k=20 gives the metrics
  something to discriminate on (0.325 for plain retrieval).

- **DocPrec@5 is always reported at 5, never at the chunk k.** It answers a
  different question — "of the chunks that would actually reach the answer
  prompt, how many came from the right paper" — and the answer prompt gets
  `top_k=5` regardless of what the benchmark's chunk metrics use. It is also
  robust to the chunk-level gold set being necessarily incomplete: it does
  not matter which passage of the right paper was retrieved, only that the
  paper was found.

- **Only `total_ms` is reported, never per-stage timings.** Four strategies
  emit a `translate` stage that is LLM-only; decomposition emits one
  `decompose` stage that swallows retrieval and N generations. The stage names
  are not comparable across strategies even though they look like they are.

- **A degraded trace is fatal.** Every strategy falls back to plain retrieval
  when the LLM is unavailable, recording a note. Averaging those in would
  report "no technique helps" when the real finding is "the API key expired".

Generation is skipped throughout (`generate=False`): these are retrieval
metrics, the answer is never read, and generating one would cost a call per
question per strategy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from evaluation.gold import GoldQuestion, load_gold
from evaluation.metrics import doc_precision_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from evaluation.spans import relevant_chunk_ids
from rag.chunking import Chunk, chunk_documents
from rag.config import Config
from rag.embedding import Embedder
from rag.llm import GeminiLLM
from rag.loader import load_documents
from rag.pipeline import ask, load_index
from rag.store import VectorStore
from rag.strategies import STRATEGY_NAMES
from rag.trace import Trace

DEGRADED = "degraded to direct retrieval"

# The answer prompt's real top_k, independent of the chunk metrics' cutoff.
DOC_PRECISION_K = 5


@dataclass
class StrategyScore:
    strategy: str
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    doc_precision: float
    mean_ms: float
    questions: int


def check_not_degraded(trace: Trace) -> None:
    """Fail loudly if a strategy silently fell back to plain retrieval."""
    for note in trace.notes:
        if DEGRADED in note:
            raise RuntimeError(
                f"strategy {trace.strategy!r} degraded on question "
                f"{trace.question!r}: {note}. Benchmarking a degraded run "
                "would measure plain retrieval and report it as the strategy."
            )


def score_strategy(
    strategy: str,
    gold: list[GoldQuestion],
    chunks: list[Chunk],
    store: VectorStore,
    embedder,
    llm,
    config: Config,
    k: int,
) -> StrategyScore:
    """Run one strategy over every gold question and average the metrics.

    Retrieval runs once per question, at depth `k`. DocPrec@5 is read off the
    same ranked list's first 5 entries rather than issuing a second call at
    top_k=5: truncating a ranked list further does not change the order of
    what survives, so the first 5 of the k=20 list are exactly what a k=5 run
    would have returned.
    """
    recalls: list[float] = []
    rrs: list[float] = []
    ndcgs: list[float] = []
    doc_precisions: list[float] = []
    times: list[float] = []

    for question in gold:
        trace = ask(
            question.question,
            store,
            embedder,
            llm,
            config,
            k=k,
            strategy=strategy,
            generate=False,
        )
        check_not_degraded(trace)

        retrieved_ids = [r.chunk.chunk_id for r in trace.retrieved]
        retrieved_doc_ids = [r.chunk.doc_id for r in trace.retrieved]
        relevant = relevant_chunk_ids(question, chunks)
        recalls.append(recall_at_k(retrieved_ids, relevant, k))
        rrs.append(reciprocal_rank(retrieved_ids, relevant))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant, k))
        doc_precisions.append(
            doc_precision_at_k(retrieved_doc_ids, question.doc_id, DOC_PRECISION_K)
        )
        times.append(trace.total_ms)

    n = len(gold)
    mean = lambda values: sum(values) / n if n else 0.0  # noqa: E731
    return StrategyScore(
        strategy=strategy,
        recall_at_k=mean(recalls),
        mrr=mean(rrs),
        ndcg_at_k=mean(ndcgs),
        doc_precision=mean(doc_precisions),
        mean_ms=mean(times),
        questions=n,
    )


def format_table(scores: list[StrategyScore], k: int) -> str:
    """Render scores as a markdown table, best recall first."""
    header = (
        f"| Strategy | Recall@{k} | MRR | nDCG@{k} | DocPrec@{DOC_PRECISION_K} "
        "| Mean ms | Questions |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    rows = [
        f"| {s.strategy} | {s.recall_at_k:.3f} | {s.mrr:.3f} | "
        f"{s.ndcg_at_k:.3f} | {s.doc_precision:.3f} | {s.mean_ms:.0f} | "
        f"{s.questions} |"
        for s in sorted(scores, key=lambda s: (-s.recall_at_k, s.strategy))
    ]
    return "\n".join([header, *rows])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evaluation.benchmark",
        description="Measure every retrieval strategy against the gold set.",
    )
    parser.add_argument("--gold", type=Path, default=Path("evaluation/gold.json"))
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help=(
            "cutoff for Recall@k and nDCG@k (default: 20 -- Recall@5 saturates "
            "near zero for every strategy on this gold set and cannot "
            "distinguish them; DocPrec is always reported at 5 regardless)"
        ),
    )
    parser.add_argument(
        "--strategy",
        action="append",
        choices=sorted(STRATEGY_NAMES),
        help="measure only these strategies (repeatable)",
    )
    parser.add_argument("--out", type=Path, help="also write the table here")
    args = parser.parse_args(argv)

    config = Config.from_env(env_file=Path(".env"))

    embedder = Embedder(config.embedding_model, max_length=config.max_seq_tokens)
    documents = load_documents(config.corpus_dir, config.metadata_path)
    chunks = chunk_documents(
        documents, embedder.tokenizer, config.chunk_tokens, config.chunk_overlap
    )
    store = load_index(config)
    gold = load_gold(args.gold, documents)
    llm = GeminiLLM(
        model=config.llm_model, api_key=config.api_key, cache_dir=config.cache_dir
    )

    selected = args.strategy or list(STRATEGY_NAMES)
    scores = []
    for strategy in selected:
        print(f"running {strategy}...", flush=True)
        scores.append(
            score_strategy(strategy, gold, chunks, store, embedder, llm, config, args.k)
        )

    table = format_table(scores, k=args.k)
    print()
    print(table)
    if args.out:
        args.out.write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
