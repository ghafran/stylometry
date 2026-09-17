"""Repetition measures, and the length-robustness that is their whole point.

The project already had raw type-token ratio, which falls with length by construction: a longer text
runs out of new words. These indices are worth adding only insofar as they do not do that, so that is
what is pinned here, along with the edge cases that make naive implementations divide by zero.
"""
from __future__ import annotations

import pytest

from stylometry import repetition as rep

REPETITIVE = ("a b a b a b " * 60).split()
VARIED = [f"w{i}" for i in range(360)]


def test_repetition_measures_order_two_obvious_extremes():
    assert rep.yule_k(REPETITIVE) > rep.yule_k(VARIED)
    assert rep.simpson_d(REPETITIVE) > rep.simpson_d(VARIED)
    assert rep.repeat_rate(REPETITIVE) == 1.0 and rep.repeat_rate(VARIED) == 0.0
    assert rep.hapax_ratio(VARIED) == 1.0 and rep.hapax_ratio(REPETITIVE) == 0.0


def test_yule_k_barely_moves_when_the_same_text_is_doubled():
    """Yule built K to be a rate, not a total; doubling a text must not double its repetitiveness."""
    once = rep.yule_k(REPETITIVE)
    twice = rep.yule_k(REPETITIVE * 2)
    assert abs(twice - once) / once < 0.05


def test_moving_average_ttr_resists_the_length_bias_that_plain_ttr_has():
    short, long = VARIED[:100], VARIED * 4
    plain_short = len(set(short)) / len(short)
    plain_long = len(set(long)) / len(long)
    assert plain_long < plain_short * 0.5, "plain TTR collapses with length, which is the problem"
    assert rep.mattr(long, 100) == pytest.approx(rep.mattr(short, 100), abs=0.05)


@pytest.mark.parametrize("tokens", [[], ["only"], ["a", "a"]])
def test_degenerate_inputs_produce_finite_numbers_rather_than_exceptions(tokens):
    values = rep.features(tokens)
    assert len(values) == len(rep.NAMES)
    assert all(v == v and abs(v) != float("inf") for v in values)


def test_honore_r_survives_a_text_that_is_entirely_hapax():
    """R divides by (1 - V1/V), which is zero when every word occurs once."""
    value = rep.honore_r(VARIED)
    assert value == value and value < float("inf")


def test_a_window_smaller_than_two_tokens_is_refused():
    with pytest.raises(ValueError, match="at least two tokens"):
        rep.mattr(VARIED, 1)


def test_top_word_share_measures_how_far_a_text_leans_on_a_few_words():
    assert rep.top_word_share(REPETITIVE, top=10) == pytest.approx(1.0)
    assert rep.top_word_share(VARIED, top=10) == pytest.approx(10 / 360)


def test_matrix_has_one_row_per_document_and_one_column_per_named_measure():
    M = rep.matrix([REPETITIVE, VARIED])
    assert M.shape == (2, len(rep.NAMES))
