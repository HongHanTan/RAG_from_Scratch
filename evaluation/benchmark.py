"""Run every strategy over the gold set and report how they compare.

Six deliberate choices, each one a thing that would otherwise make the
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

- **`LLM calls` counts logical calls, not cache hits or misses.** Raw
  wall-clock time is not comparable across strategies here: multi-query and
  rag-fusion build the *same* rewrite prompt (`MULTI_QUERY_TEMPLATE`, same
  question, same `n`) and therefore hit the *same* cache key in
  `rag/llm.py`'s on-disk cache. Whichever strategy `STRATEGY_NAMES` happens to
  list first pays for every rewrite; the other gets them all as free cache
  hits. Reordering the tuple would swap which one "looks fast". `LLM calls` is
  the count of `.generate()` calls a strategy makes per question regardless of
  whether the cache served it, via a small counting wrapper
  (`_CallCountingLLM`) around the LLM. It cannot be perturbed by cache state
  or strategy order, and it is the real cost driver.

- **`Mean ms (warm)` is measured with every strategy's own cache already
  warm.** Each strategy runs over the gold set twice; only the second pass is
  scored. This still cannot be compared to a cold call's latency -- it
  measures retrieval and orchestration cost (embedding, vector search,
  merging/fusing lists, and a cache read), not the cost of an actual LLM
  round trip. Read it as "how much does this strategy cost beyond the LLM
  call", not as an end-to-end latency figure.

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
    llm_calls: float
    mean_ms_warm: float
    questions: int


class _CallCountingLLM:
    """Wraps an LLM to count logical `.generate()` calls, hit or miss.

    `rag.llm.GeminiLLM.call_count` deliberately counts only real API
    attempts -- a cache hit is free and does not increment it, which is right
    for "how many times did this actually reach Google". The benchmark wants
    the opposite question: how many times a strategy *asks* the model for
    something. That is the real cost driver, and unlike wall-clock time it
    cannot be perturbed by cache state or by which strategy happened to run
    (and therefore cache its prompt) first.

    Everything but `.generate()` is left alone -- there is no need to
    intercept it, since strategies only ever call `.generate(prompt)` on the
    LLM they are given. `__getattr__` forwards anything else to the wrapped
    LLM, so an attribute access that would work on the real LLM (a diagnostic
    such as `.call_count`, say) still works through this wrapper instead of
    failing only when the benchmark is what's asking.
    """

    def __init__(self, llm) -> None:
        self._llm = llm
        self.logical_call_count = 0

    def generate(self, prompt: str) -> str:
        self.logical_call_count += 1
        return self._llm.generate(prompt)

    def __getattr__(self, name: str):
        return getattr(self._llm, name)


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
    store: VectorStore,
    embedder,
    llm,
    config: Config,
    k: int,
) -> StrategyScore:
    """Run one strategy over every gold question and average the metrics.

    Gold spans are resolved against `store.chunks` -- the exact chunks that
    were embedded into the index and searched -- rather than a freshly
    recomputed chunk list. Chunk ids are positional (`f"{doc_id}:{index}"`),
    so if the corpus or the text extractor ever changed after the index was
    built, a separately recomputed chunk list would share ids with the
    index's chunks while covering different text, and scores would compare
    two disagreeing chunk lists under the same ids with no error anywhere.
    Scoring against `store.chunks` makes that mismatch structurally
    impossible: there is only one chunk list, and it is the one that was
    actually searched.

    Retrieval runs once per question, at depth `k`. DocPrec@5 is read off the
    same ranked list's first 5 entries rather than issuing a second call at
    top_k=5: truncating a ranked list further does not change the order of
    what survives, so the first 5 of the k=20 list are exactly what a k=5 run
    would have returned.

    The gold set is run twice. The first pass warms this strategy's own
    entries in the shared on-disk LLM cache; only the second pass is scored.
    That is what makes `mean_ms_warm` order-independent -- without it,
    whichever strategy happens to run first for a given rewrite prompt pays
    for the cache miss and every other strategy sharing that prompt gets it
    for free. Retrieval is deterministic given a warm cache, so the two
    passes agree on recall/MRR/nDCG/DocPrec; only the timings and the LLM-call
    count are taken from the (second, warm) pass that is kept.

    `llm` is wrapped in `_CallCountingLLM` so `llm_calls` counts every logical
    `.generate()` call, cache hit or miss.
    """
    counting_llm = _CallCountingLLM(llm) if llm is not None else None
    effective_llm = counting_llm if counting_llm is not None else llm

    def run_once() -> list[tuple[Trace, int]]:
        results = []
        for question in gold:
            if counting_llm is not None:
                counting_llm.logical_call_count = 0
            trace = ask(
                question.question,
                store,
                embedder,
                effective_llm,
                config,
                k=k,
                strategy=strategy,
                generate=False,
            )
            check_not_degraded(trace)
            calls = counting_llm.logical_call_count if counting_llm is not None else 0
            results.append((trace, calls))
        return results

    run_once()  # warm-up: not scored
    traced = run_once()

    recalls: list[float] = []
    rrs: list[float] = []
    ndcgs: list[float] = []
    doc_precisions: list[float] = []
    times: list[float] = []
    llm_calls: list[float] = []

    for question, (trace, calls) in zip(gold, traced):
        retrieved_ids = [r.chunk.chunk_id for r in trace.retrieved]
        retrieved_doc_ids = [r.chunk.doc_id for r in trace.retrieved]
        relevant = relevant_chunk_ids(question, store.chunks)
        recalls.append(recall_at_k(retrieved_ids, relevant, k))
        rrs.append(reciprocal_rank(retrieved_ids, relevant))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant, k))
        doc_precisions.append(
            doc_precision_at_k(retrieved_doc_ids, question.doc_id, DOC_PRECISION_K)
        )
        times.append(trace.total_ms)
        llm_calls.append(calls)

    n = len(gold)
    mean = lambda values: sum(values) / n if n else 0.0  # noqa: E731
    return StrategyScore(
        strategy=strategy,
        recall_at_k=mean(recalls),
        mrr=mean(rrs),
        ndcg_at_k=mean(ndcgs),
        doc_precision=mean(doc_precisions),
        llm_calls=mean(llm_calls),
        mean_ms_warm=mean(times),
        questions=n,
    )


def format_table(scores: list[StrategyScore], k: int) -> str:
    """Render scores as a markdown table, best recall first.

    `LLM calls` and `Mean ms (warm)` replace the single `Mean ms` column that
    used to be reported: raw wall-clock time was measuring which strategy's
    LLM cache happened to be warm, not its actual cost (see the module
    docstring). Neither replacement column can be perturbed by cache state or
    strategy run order.

    The MRR column is labelled `MRR@k`, matching `Recall@k` and `nDCG@k`, even
    though `reciprocal_rank` takes no cutoff of its own: `trace.retrieved` is
    already truncated to `k` by the time `score_strategy` reads it, so the
    values it sees are the same as if the cutoff were applied here too. The
    label makes that true scope explicit rather than implying an unbounded
    reciprocal rank over the whole corpus.
    """
    header = (
        f"| Strategy | Recall@{k} | MRR@{k} | nDCG@{k} | DocPrec@{DOC_PRECISION_K} "
        "| LLM calls | Mean ms (warm) | Questions |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [
        f"| {s.strategy} | {s.recall_at_k:.3f} | {s.mrr:.3f} | "
        f"{s.ndcg_at_k:.3f} | {s.doc_precision:.3f} | {s.llm_calls:.1f} | "
        f"{s.mean_ms_warm:.0f} | {s.questions} |"
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
            score_strategy(strategy, gold, store, embedder, llm, config, args.k)
        )

    table = format_table(scores, k=args.k)
    print()
    print(table)
    if args.out:
        args.out.write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
