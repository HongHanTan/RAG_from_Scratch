import numpy as np
import pytest

from rag.late_interaction import maxsim


def _unit(rows):
    m = np.asarray(rows, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_identical_single_tokens_score_one():
    v = _unit([[1.0, 0.0]])
    assert maxsim(v, v) == pytest.approx(1.0)


def test_orthogonal_single_tokens_score_zero():
    q = _unit([[1.0, 0.0]])
    d = _unit([[0.0, 1.0]])
    assert maxsim(q, d) == pytest.approx(0.0)


def test_the_score_sums_over_query_tokens():
    # Two query tokens each matching perfectly sum to 2.0, not average to
    # 1.0. That sum is why a maxsim score is not comparable with a cosine.
    q = _unit([[1.0, 0.0], [0.0, 1.0]])
    d = _unit([[1.0, 0.0], [0.0, 1.0]])
    assert maxsim(q, d) == pytest.approx(2.0)


def test_only_the_best_document_token_counts_per_query_token():
    # One query token, three document tokens: the two poor matches must not
    # dilute the good one. That is the "max" in MaxSim.
    q = _unit([[1.0, 0.0]])
    d = _unit([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert maxsim(q, d) == pytest.approx(1.0)


def test_one_document_token_may_serve_several_query_tokens():
    # No assignment constraint: both query tokens can max against the same
    # document token.
    q = _unit([[1.0, 0.0], [1.0, 0.0]])
    d = _unit([[1.0, 0.0]])
    assert maxsim(q, d) == pytest.approx(2.0)


def test_extra_irrelevant_document_tokens_do_not_lower_the_score():
    q = _unit([[1.0, 0.0]])
    short = _unit([[1.0, 0.0]])
    padded = _unit([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 1.0]])
    assert maxsim(q, padded) >= maxsim(q, short) - 1e-6


def test_a_longer_query_scores_higher_all_else_equal():
    # A consequence worth knowing: maxsim is not length-normalised, so it
    # compares documents for one query, never queries with each other.
    d = _unit([[1.0, 0.0], [0.0, 1.0]])
    assert maxsim(_unit([[1.0, 0.0], [0.0, 1.0]]), d) > maxsim(_unit([[1.0, 0.0]]), d)


def test_an_empty_document_scores_zero():
    assert maxsim(_unit([[1.0, 0.0]]), np.zeros((0, 2), dtype=np.float32)) == 0.0


def test_an_empty_query_scores_zero():
    assert maxsim(np.zeros((0, 2), dtype=np.float32), _unit([[1.0, 0.0]])) == 0.0


def test_mismatched_dimensions_are_an_error():
    with pytest.raises(ValueError, match="dimension"):
        maxsim(_unit([[1.0, 0.0]]), _unit([[1.0, 0.0, 0.0]]))


def test_the_result_is_a_plain_float():
    # It travels into RetrievedChunk.score and then into JSON via the trace;
    # a np.float32 there serialises badly.
    assert type(maxsim(_unit([[1.0, 0.0]]), _unit([[1.0, 0.0]]))) is float
