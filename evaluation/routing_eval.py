"""Measure logical routing against the gold set, at zero labelling cost.

Every gold question already names a `doc_id`, and every document in
`VectorStore.doc_meta` already carries a `topic` (`rag/loader.py`,
`rag/pipeline.py:build_index`). That means routing accuracy is computable
without adding a single new label: did `logical_route` return a topic set
containing the topic of the document that actually holds the answer?

Three numbers, not one, because any one of them alone can be misread:

- **Routing recall** answers "did the router keep the right door open".
  Returning `()` means "search everything", which by construction contains
  the gold topic -- so an abstention always counts as a hit. Counting it any
  other way would make abstaining look like a routing failure when it is
  actually routing declining to narrow, which is the safe, explicit fallback
  `rag/routing.py` is built around.
- **Abstention rate** exists because recall alone cannot tell a router that
  narrows correctly apart from one that gives up and searches everything
  every time -- the second scores 1.0 on recall and does nothing. Reported
  next to recall so a reader cannot see one without the other.
- **Mean topics chosen** is what makes recall interpretable at all. An
  abstention is scored as if it chose every topic (not zero), because that is
  what it actually causes the pipeline to search -- scoring it as zero would
  make a router that mostly gives up look like it is aggressively narrowing.
  A mean close to the number of available topics means recall is measuring a
  router that barely restricts anything; a mean close to 1 means recall is
  measuring real narrowing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from evaluation.gold import GoldQuestion, load_gold
from rag.config import Config
from rag.llm import GeminiLLM
from rag.loader import load_documents
from rag.pipeline import load_index
from rag.routing import logical_route
from rag.store import VectorStore
from rag.trace import Trace


@dataclass
class RoutingScore:
    recall: float
    """Fraction of questions where the chosen topic set contains the gold
    document's topic. Abstentions (`()`) always count as hits -- see module
    docstring."""
    abstention_rate: float
    """Fraction of questions where the router returned `()`."""
    mean_topics_chosen: float
    """Mean number of topics the router left in play. An abstention counts
    as choosing every available topic, since that is the set it actually
    causes `ask()` to search."""
    topics_available: int
    questions: int


def _topics_of(store: VectorStore) -> tuple[str, ...]:
    return tuple(
        sorted({r.get("topic") for r in store.doc_meta.values() if r.get("topic")})
    )


def score_routing(
    gold: list[GoldQuestion],
    store: VectorStore,
    llm,
    trace_factory: Callable[[str], Trace] = Trace,
) -> RoutingScore:
    """Run `logical_route` over every gold question and score it.

    `trace_factory` builds a fresh `Trace` per question -- `logical_route`
    writes into it (a `route` translation step on success, a `degraded` note
    on abstention), and a shared `Trace` across questions would let one
    question's routing decision bleed into the notes read for another. A
    caller that wants to inspect those notes (a test, say) passes its own
    factory to keep every question's trace around; `main` below does not need
    to and lets the default create-and-discard one apply.

    Each gold question must name documents that all have a `topic` in
    `store.doc_meta` -- every document in this corpus's index has one, so a
    missing topic means the gold set and the index disagree about which
    corpus they describe, and that is worth failing loudly on rather than
    silently scoring as a miss.

    A question may name several documents, and those documents need not
    share a topic. The router is scored correct when it picks **any** of the
    question's gold topics: it filters the search to the topics it names, so
    naming one topic that holds an answer leaves that answer reachable, and
    demanding every topic would mark a router that found a usable filter as
    wrong. For a single-source question this is exactly the previous rule.
    """
    topics = _topics_of(store)
    n = len(gold)
    hits = 0
    abstentions = 0
    topic_counts: list[int] = []

    for question in gold:
        gold_topics: set[str] = set()
        for source in question.sources:
            record = store.doc_meta.get(source)
            topic = record.get("topic") if record else None
            if not topic:
                raise ValueError(
                    f"gold question {question.id!r} names doc_id "
                    f"{source!r}, which has no topic in store.doc_meta"
                )
            gold_topics.add(topic)

        trace = trace_factory(question.question)
        chosen = logical_route(question.question, llm, topics, trace)

        if not chosen:
            abstentions += 1
            hits += 1  # () means "search everything" -- contains it by construction
            topic_counts.append(len(topics))
        else:
            if gold_topics & set(chosen):
                hits += 1
            topic_counts.append(len(chosen))

    return RoutingScore(
        recall=hits / n if n else 0.0,
        abstention_rate=abstentions / n if n else 0.0,
        mean_topics_chosen=sum(topic_counts) / n if n else 0.0,
        topics_available=len(topics),
        questions=n,
    )


def format_routing_table(score: RoutingScore) -> str:
    """Render a `RoutingScore` as a one-row markdown table.

    Every column that could be misread on its own sits next to the column
    that corrects the misreading: recall next to abstention rate (a router
    that always gives up scores 1.0 on recall alone), and mean topics chosen
    next to the number available (a mean near that number means recall is
    not measuring narrowing at all).
    """
    header = (
        "| Recall | Abstention rate | Mean topics chosen | Topics available "
        "| Questions |\n"
        "|---:|---:|---:|---:|---:|"
    )
    row = (
        f"| {score.recall:.3f} | {score.abstention_rate:.3f} | "
        f"{score.mean_topics_chosen:.2f} | {score.topics_available} | "
        f"{score.questions} |"
    )
    return "\n".join([header, row])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evaluation.routing_eval",
        description=(
            "Measure logical routing's recall, abstention rate and mean "
            "topics chosen against the gold set, using each gold question's "
            "own doc_id/topic -- no new labelling needed."
        ),
    )
    parser.add_argument("--gold", type=Path, default=Path("evaluation/gold.json"))
    args = parser.parse_args(argv)

    config = Config.from_env(env_file=Path(".env"))
    documents = load_documents(config.corpus_dir, config.metadata_path)
    store = load_index(config)
    gold = load_gold(args.gold, documents)
    llm = GeminiLLM(
        model=config.llm_model, api_key=config.api_key, cache_dir=config.cache_dir
    )

    score = score_routing(gold, store, llm)
    print(format_routing_table(score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
