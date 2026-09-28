"""Checks on the real corpus's topic assignment.

Logical routing needs something to route between. Every document in this
corpus has source "arxiv", so `topic` is the field that partitions it.
"""

from pathlib import Path

import pytest

from rag.config import Config
from rag.loader import load_documents

EXPECTED_TOPICS = {
    "foundations",
    "retrieval-models",
    "rag-systems",
    "prompting-reasoning",
    "evaluation-benchmarks",
}


@pytest.fixture(scope="module")
def corpus():
    cfg = Config()
    return load_documents(cfg.corpus_dir, cfg.metadata_path)


def test_every_document_has_a_topic(corpus):
    missing = [d.doc_id for d in corpus if not d.topic]
    assert not missing, f"documents with no topic: {missing}"


def test_topics_come_from_the_known_set(corpus):
    unknown = {d.topic for d in corpus} - EXPECTED_TOPICS
    assert not unknown, f"unexpected topics: {unknown}"


def test_every_topic_is_used(corpus):
    # A topic nobody is in is a routing option that can only ever be wrong.
    unused = EXPECTED_TOPICS - {d.topic for d in corpus}
    assert not unused, f"topics with no documents: {unused}"


def test_no_topic_holds_more_than_half_the_corpus(corpus):
    # A partition where one bucket is most of the corpus cannot route
    # usefully: picking it is barely different from searching everything.
    from collections import Counter

    counts = Counter(d.topic for d in corpus)
    biggest, n = counts.most_common(1)[0]
    assert n <= len(corpus) // 2, f"{biggest} holds {n} of {len(corpus)}"
