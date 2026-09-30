import math

import pytest

from evaluation.metrics import doc_precision_at_k, ndcg_at_k, recall_at_k, reciprocal_rank


# --- recall@k ----------------------------------------------------------------

def test_recall_is_the_fraction_of_relevant_chunks_found():
    assert recall_at_k(["a", "b", "c"], {"a", "d"}, k=3) == pytest.approx(0.5)


def test_recall_is_one_when_everything_relevant_is_retrieved():
    assert recall_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)


def test_recall_is_zero_when_nothing_relevant_is_retrieved():
    assert recall_at_k(["x", "y"], {"a"}, k=2) == pytest.approx(0.0)


def test_recall_ignores_hits_below_k():
    assert recall_at_k(["x", "a"], {"a"}, k=1) == pytest.approx(0.0)
    assert recall_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(1.0)


def test_recall_with_no_relevant_chunks_is_zero_not_a_crash():
    # A gold question whose span resolves to nothing is a broken question,
    # but it must not divide by zero mid-benchmark.
    assert recall_at_k(["a"], set(), k=1) == pytest.approx(0.0)


def test_recall_with_k_larger_than_the_result_list_uses_what_is_there():
    assert recall_at_k(["a"], {"a"}, k=10) == pytest.approx(1.0)


# --- reciprocal rank ---------------------------------------------------------

def test_reciprocal_rank_of_a_first_place_hit_is_one():
    assert reciprocal_rank(["a", "b"], {"a"}) == pytest.approx(1.0)


def test_reciprocal_rank_of_a_third_place_hit_is_one_third():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_the_first_relevant_hit_only():
    assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == pytest.approx(0.5)


def test_reciprocal_rank_with_no_hit_is_zero():
    assert reciprocal_rank(["x", "y"], {"a"}) == pytest.approx(0.0)


def test_reciprocal_rank_of_an_empty_result_is_zero():
    assert reciprocal_rank([], {"a"}) == pytest.approx(0.0)


# --- nDCG@k ------------------------------------------------------------------

def test_ndcg_is_one_when_the_only_relevant_chunk_ranks_first():
    assert ndcg_at_k(["a", "x", "y"], {"a"}, k=3) == pytest.approx(1.0)


def test_ndcg_matches_the_hand_computed_value_for_a_second_place_hit():
    # DCG  = 1/log2(2+1) = 0.630929...
    # IDCG = 1/log2(1+1) = 1.0
    assert ndcg_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(1 / math.log2(3))


def test_ndcg_matches_the_hand_computed_value_for_two_hits():
    # retrieved: x a b -> DCG  = 1/log2(3) + 1/log2(4) = 0.630929 + 0.5
    # ideal:     a b x -> IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.630929
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["x", "a", "b"], {"a", "b"}, k=3) == pytest.approx(expected)


def test_ndcg_rewards_ranking_a_hit_higher():
    assert ndcg_at_k(["a", "x"], {"a"}, k=2) > ndcg_at_k(["x", "a"], {"a"}, k=2)


def test_ndcg_with_no_relevant_chunks_is_zero():
    assert ndcg_at_k(["a"], set(), k=1) == pytest.approx(0.0)


def test_ndcg_with_no_hits_is_zero():
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == pytest.approx(0.0)


def test_ndcg_ideal_accounts_for_more_relevant_chunks_than_k():
    # Only k can be retrieved, so the ideal ranking is capped at k too;
    # otherwise a question with 10 relevant chunks could never score 1.0.
    assert ndcg_at_k(["a", "b"], {"a", "b", "c", "d"}, k=2) == pytest.approx(1.0)


# --- doc precision@k ----------------------------------------------------------

def test_doc_precision_is_one_when_every_result_is_from_the_gold_doc():
    assert doc_precision_at_k(["a", "a", "a"], "a", k=3) == pytest.approx(1.0)


def test_doc_precision_is_zero_when_no_result_is_from_the_gold_doc():
    assert doc_precision_at_k(["b", "c", "d"], "a", k=3) == pytest.approx(0.0)


def test_doc_precision_is_the_fraction_from_the_gold_doc():
    assert doc_precision_at_k(["a", "b", "a", "c", "a"], "a", k=5) == pytest.approx(0.6)


def test_doc_precision_with_k_larger_than_the_result_list_uses_what_is_there():
    # Only 2 results exist even though k asks for 5; the denominator must be
    # the number actually retrieved, not k, or a short list would be
    # penalised as if the missing slots were misses.
    assert doc_precision_at_k(["a", "a"], "a", k=5) == pytest.approx(1.0)
    assert doc_precision_at_k(["a", "b"], "a", k=5) == pytest.approx(0.5)


def test_doc_precision_of_an_empty_result_list_is_zero():
    assert doc_precision_at_k([], "a", k=5) == pytest.approx(0.0)


def test_doc_precision_ignores_results_beyond_k():
    assert doc_precision_at_k(["a", "b", "b", "b"], "a", k=1) == pytest.approx(1.0)


def test_doc_precision_counts_any_gold_document():
    # A cross-document question has several right answers; crediting only
    # one of them would score a correct retrieval as a miss.
    assert doc_precision_at_k(
        ["alpha", "beta", "gamma", "alpha"], {"alpha", "beta"}, 4
    ) == pytest.approx(0.75)


def test_doc_precision_still_accepts_a_bare_string():
    # Keeps every existing single-document call site working unchanged.
    assert doc_precision_at_k(["alpha", "beta"], "alpha", 2) == pytest.approx(0.5)
