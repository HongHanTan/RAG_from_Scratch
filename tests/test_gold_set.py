"""Checks on the real gold set, as opposed to the loading machinery."""

from pathlib import Path

import pytest

from evaluation.gold import load_gold
from evaluation.spans import relevant_chunk_ids
from rag.chunking import chunk_documents
from rag.config import Config
from rag.loader import load_documents

GOLD_PATH = Path("evaluation/gold.json")


@pytest.fixture(scope="module")
def corpus():
    cfg = Config()
    return load_documents(cfg.corpus_dir, cfg.metadata_path)


def test_the_gold_set_loads(corpus):
    assert load_gold(GOLD_PATH, corpus)


def test_the_gold_set_size_is_pinned(corpus):
    # Pinned so a question cannot be lost, or quietly added, without a test
    # changing alongside it. Phase 7 grew the set to its final size: ten
    # original questions, ten cross-document, and ten technique-targeted
    # single-document ones.
    assert len(load_gold(GOLD_PATH, corpus)) == 30


def test_every_question_has_a_reason_recorded(corpus):
    for question in load_gold(GOLD_PATH, corpus):
        assert question.why.strip(), f"{question.id} has no 'why'"


def test_questions_are_spread_across_documents(corpus):
    # A gold set concentrated on one paper measures that paper, not retrieval.
    gold = load_gold(GOLD_PATH, corpus)
    assert len({g.doc_id for g in gold}) >= 6


def test_quotes_are_substantial(corpus):
    # A three-word quote resolves to a span so small it may sit inside a
    # single chunk by luck rather than because that chunk answers anything.
    for question in load_gold(GOLD_PATH, corpus):
        for quote in question.quotes:
            assert len(quote) >= 40, f"{question.id}: quote too short"


@pytest.mark.slow
def test_every_question_resolves_to_at_least_one_chunk(corpus):
    from transformers import AutoTokenizer

    cfg = Config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.embedding_model, use_fast=True)
    chunks = chunk_documents(corpus, tokenizer, cfg.chunk_tokens, cfg.chunk_overlap)
    for question in load_gold(GOLD_PATH, corpus):
        relevant = relevant_chunk_ids(question, chunks)
        assert relevant, f"{question.id} resolves to no chunks"
        assert len(relevant) <= 8, (
            f"{question.id} resolves to {len(relevant)} chunks; the quote is "
            "probably too long to be a precise target"
        )
