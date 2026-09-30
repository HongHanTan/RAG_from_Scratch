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
  the count of `.generate()` and `.structured()` calls a strategy makes per
  question regardless of whether the cache served it, via a small counting
  wrapper (`_CallCountingLLM`) around the LLM. It cannot be perturbed by cache
  state or strategy order, and it is the real cost driver.

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
from dataclasses import dataclass, field
from pathlib import Path

from evaluation.gold import GoldQuestion, load_gold
from evaluation.metrics import doc_precision_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from evaluation.spans import relevant_chunk_ids
from rag.config import Config
from rag.embedding import Embedder
from rag.llm import GeminiLLM
from rag.loader import load_documents
from rag.pipeline import INDEX_MODES, ask, load_index
from rag.store import VectorStore
from rag.strategies import STRATEGY_NAMES
from rag.trace import DEGRADED, Trace

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
    recall_by_question: dict[str, float] = field(default_factory=dict)
    """Recall@k per gold question id, kept so `--predictions` can ask which
    strategy actually won each question. The averaged columns cannot answer
    that: a strategy can lead the table while losing most individual
    questions.
    """


class _CallCountingLLM:
    """Wraps an LLM to count logical `.generate()` calls, hit or miss.

    `rag.llm.GeminiLLM.call_count` deliberately counts only real API
    attempts -- a cache hit is free and does not increment it, which is right
    for "how many times did this actually reach Google". The benchmark wants
    the opposite question: how many times a strategy *asks* the model for
    something. That is the real cost driver, and unlike wall-clock time it
    cannot be perturbed by cache state or by which strategy happened to run
    (and therefore cache its prompt) first.

    `.generate()` and `.structured()` are both intercepted -- Phase 4's
    logical routing and step-back's structured output both ask the model for
    something via `.structured()` rather than `.generate()`, and that call is
    exactly as real a cost as a `.generate()` call. Leaving it to
    `__getattr__` would forward it straight to the wrapped LLM's own
    `.structured()`, which calls that LLM's own `.generate()` internally and
    bypasses this class's override entirely -- silently under-reporting cost
    for any strategy that switches to structured output. Everything else is
    left alone: `__getattr__` forwards anything else to the wrapped LLM, so an
    attribute access that would work on the real LLM (a diagnostic such as
    `.call_count`, say) still works through this wrapper instead of failing
    only when the benchmark is what's asking.
    """

    def __init__(self, llm) -> None:
        self._llm = llm
        self.logical_call_count = 0

    def generate(self, prompt: str) -> str:
        self.logical_call_count += 1
        return self._llm.generate(prompt)

    def structured(self, prompt: str, schema: dict) -> dict:
        self.logical_call_count += 1
        return self._llm.structured(prompt, schema)

    def __getattr__(self, name: str):
        return getattr(self._llm, name)


def check_not_degraded(trace: Trace) -> None:
    """Fail loudly if any stage silently fell back to a safe default."""
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
    route: bool = False,
    rerank: bool = False,
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

    `rerank=True` reranks each strategy's candidates with ColBERT-style late
    interaction before the metrics are read off them (`ask(..., rerank=True)`),
    so the reranked table has the same columns as the un-reranked one and the
    two are directly comparable. `check_not_degraded` still applies: a
    reranker that silently fell back to the dense order would report dense
    numbers as reranked ones, which is the one failure mode this measurement
    cannot tolerate.

    `route=True` runs every strategy with logical routing switched on ahead
    of it (`ask(..., route=True)`), narrowing the candidate set to the
    topics the router chooses before the strategy searches. `llm_calls` then
    includes the routing call itself (one `.structured()` call per question,
    via `_CallCountingLLM`), and `check_not_degraded` still applies: if the
    router ever falls back to searching everything for a gold question, that
    is exactly as fatal here as an LLM failure inside the strategy itself --
    both would otherwise let a degraded run be measured and reported as
    "routing".
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
                route=route,
                rerank=rerank,
            )
            check_not_degraded(trace)
            calls = counting_llm.logical_call_count if counting_llm is not None else 0
            results.append((trace, calls))
        return results

    run_once()  # warm-up: not scored
    traced = run_once()

    recall_by_question: dict[str, float] = {}
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
        recall = recall_at_k(retrieved_ids, relevant, k)
        recalls.append(recall)
        recall_by_question[question.id] = recall
        rrs.append(reciprocal_rank(retrieved_ids, relevant))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant, k))
        doc_precisions.append(
            doc_precision_at_k(
                retrieved_doc_ids, set(question.sources), DOC_PRECISION_K
            )
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
        recall_by_question=recall_by_question,
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


def score_predictions(
    per_question: dict[str, dict[str, float]], expects: dict[str, str]
) -> tuple[int, int]:
    """How often the pre-registered prediction named the winning strategy.

    A question whose strategies all tie is counted as a miss, not a hit: the
    prediction distinguished nothing there, and crediting it would inflate
    the scorecard exactly where it is least informative.
    """
    hits = total = 0
    for qid, scores in per_question.items():
        predicted = expects.get(qid, "")
        if not predicted or not scores:
            continue
        total += 1
        best = max(scores.values())
        if best > min(scores.values()) and scores.get(predicted, -1.0) == best:
            hits += 1
    return hits, total


def format_prediction_report(
    per_question: dict[str, dict[str, float]], expects: dict[str, str]
) -> str:
    """Render the scorecard: one row per predicted question, then the total.

    Reported whatever it says. The gold set was written by someone who knows
    what each technique does, which is what lets it separate them and also
    what would let it be fitted to a flattering answer; the prediction was
    recorded before any of it was measured, and printing how often it held is
    what makes that risk visible instead of hidden. A low score is a finding
    about the author's model of these techniques, not a defect in the report.
    """
    lines = [
        "prediction scorecard (pre-registered `expects`, "
        "scored on Recall@k per question):",
        "",
        "| Question | Predicted | Actual winner | Recall (predicted) "
        "| Recall (best) | Hit |",
        "|---|---|---|---:|---:|---|",
    ]
    for qid in sorted(per_question):
        scores = per_question[qid]
        predicted = expects.get(qid, "")
        if not predicted or not scores:
            continue
        best = max(scores.values())
        worst = min(scores.values())
        winners = sorted(name for name, v in scores.items() if v == best)
        decisive = best > worst
        hit = decisive and scores.get(predicted, -1.0) == best
        actual = ", ".join(winners) if decisive else "tie (all equal)"
        lines.append(
            f"| {qid} | {predicted} | {actual} | "
            f"{scores.get(predicted, 0.0):.3f} | {best:.3f} | "
            f"{'yes' if hit else 'no'} |"
        )
    hits, total = score_predictions(per_question, expects)
    share = f"{hits / total:.1%}" if total else "n/a"
    lines += ["", f"predictions correct: {hits}/{total} ({share})"]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Separate from `main` so the flags can be tested without running a sweep:
    every strategy over the whole gold set is minutes of work and a pile of
    LLM calls, which is far too much to pay to find out whether `--index`
    accepts `raptor`.
    """
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
    parser.add_argument(
        "--route",
        action="store_true",
        help=(
            "run every strategy with logical routing switched on first "
            "(ask(..., route=True)), narrowing the candidate set to the "
            "router's chosen topics before the strategy searches. Produces "
            "the same columns, so the two tables are directly comparable -- "
            "that comparison is what says whether routing helps or hurts "
            "retrieval, not the routing-only numbers in "
            "evaluation/routing_eval.py."
        ),
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "rescore each strategy's candidates with ColBERT-style late "
            "interaction before scoring. Produces the same columns, so the "
            "two tables are directly comparable -- that comparison is what "
            "says whether late interaction helps."
        ),
    )
    parser.add_argument(
        "--predictions",
        action="store_true",
        help=(
            "also report how often each question's pre-registered `expects` "
            "prediction named the strategy that actually scored best on it. "
            "Only questions carrying a prediction are counted; the original "
            "ten predate the practice and are skipped."
        ),
    )
    parser.add_argument(
        "--index",
        choices=list(INDEX_MODES),
        default="flat",
        help=(
            "which index to measure (default: flat). The non-flat modes must "
            "have been built first (rag index --index-mode ...); they are "
            "separate files, so this never touches the flat index."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = Config.from_env(env_file=Path(".env"))

    embedder = Embedder(config.embedding_model, max_length=config.max_seq_tokens)
    documents = load_documents(config.corpus_dir, config.metadata_path)
    store = load_index(config, mode=args.index)
    gold = load_gold(args.gold, documents)
    llm = GeminiLLM(
        model=config.llm_model, api_key=config.api_key, cache_dir=config.cache_dir
    )

    selected = args.strategy or list(STRATEGY_NAMES)
    scores = []
    for strategy in selected:
        suffix = "".join(
            [" (routed)" if args.route else "", " (reranked)" if args.rerank else ""]
        )
        print(f"running {strategy}{suffix}...", flush=True)
        scores.append(
            score_strategy(
                strategy,
                gold,
                store,
                embedder,
                llm,
                config,
                args.k,
                route=args.route,
                rerank=args.rerank,
            )
        )

    table = format_table(scores, k=args.k)
    print()
    # Above the table, not only in the log line: a pasted result should say
    # which index produced it, otherwise three tables of numbers are
    # indistinguishable once they leave the terminal.
    print(f"index: {args.index}")
    print(f"rerank: {'on' if args.rerank else 'off'}")
    if args.route:
        print("routed (--route):")
    print(table)
    if args.predictions:
        per_question: dict[str, dict[str, float]] = {}
        for score in scores:
            for qid, recall in score.recall_by_question.items():
                per_question.setdefault(qid, {})[score.strategy] = recall
        expects = {q.id: q.expects for q in gold}
        report = format_prediction_report(per_question, expects)
        print()
        print(report)
    if args.out:
        args.out.write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
