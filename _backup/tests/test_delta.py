"""Burrows's Delta: the standard baseline, and the controls that keep it honest.

Delta is simple enough that a broken implementation still produces plausible-looking numbers, so the
properties that matter are pinned here: the vocabulary is the corpus's, not each document's; a word
that never varies contributes nothing; and a document is never attributed by itself or by anything
held out with it.
"""
from __future__ import annotations

import numpy as np
import pytest

from stylometry.delta import (attribute, delta_distances, delta_matrix, frequencies,
                              most_frequent_words, zscores)


def doc(pattern: str, repeat: int = 40) -> list[str]:
    return (pattern + " ") .join([""] * (repeat + 1)).split()


def test_most_frequent_words_ranks_on_the_whole_corpus_not_per_document():
    docs = [["a"] * 10 + ["rare"], ["b"] * 8, ["b"] * 8]
    assert most_frequent_words(docs, 2) == ["b", "a"], "b totals 16 across the corpus, a totals 10"


def test_ties_break_alphabetically_so_the_vocabulary_is_deterministic():
    assert most_frequent_words([["b", "a", "c"]], 3) == ["a", "b", "c"]


def test_frequencies_are_relative_so_document_length_does_not_decide_similarity():
    short, long = ["a", "b"], ["a"] * 50 + ["b"] * 50
    f = frequencies([short, long], ["a", "b"])
    np.testing.assert_allclose(f[0], f[1])


def test_a_word_that_never_varies_contributes_nothing_rather_than_dividing_by_zero():
    f = np.array([[0.5, 0.1], [0.5, 0.3]])
    z = zscores(f)
    assert np.isfinite(z).all()
    np.testing.assert_allclose(z[:, 0], [0.0, 0.0]), "a constant column carries no information"
    assert z[0, 1] != z[1, 1]


@pytest.mark.parametrize("metric", ["classic", "cosine"])
def test_a_document_is_at_zero_distance_from_its_own_profile(metric):
    d, _ = delta_matrix([doc("a b c"), doc("a b c"), doc("x y z")], 10, metric)
    assert d[0, 0] == pytest.approx(0.0)
    assert d[0, 1] == pytest.approx(0.0), "identical documents are indistinguishable"
    assert d[0, 2] > d[0, 1]


@pytest.mark.parametrize("metric", ["classic", "cosine"])
def test_distances_are_symmetric(metric):
    d, _ = delta_matrix([doc("a b"), doc("b c"), doc("c a")], 10, metric)
    np.testing.assert_allclose(d, d.T, atol=1e-6)


def test_an_unknown_metric_is_refused_rather_than_silently_defaulted():
    with pytest.raises(ValueError, match="unknown Delta metric"):
        delta_distances(np.zeros((2, 2)), metric="euclidean")


def test_attribution_never_uses_a_document_held_out_with_the_one_being_judged():
    """The mistake that makes stylometry look far better than it is: judging a passage by its own
    neighbours from the same work. Here every near-identical twin is held out together, so the only
    available answer is the wrong one, and Delta must give it."""
    docs = [doc("a b c"), doc("a b c"), doc("x y z"), doc("x y z")]
    labels = ["A", "A", "B", "B"]
    assert attribute(docs, labels, groups=["w1", "w1", "w2", "w2"]) == ["B", "B", "A", "A"]
    assert attribute(docs, labels, groups=["w1", "w2", "w3", "w4"]) == ["A", "A", "B", "B"]


def test_attribution_validates_its_inputs():
    with pytest.raises(ValueError, match="every document needs a label"):
        attribute([doc("a")], [])
    with pytest.raises(ValueError, match="at least two documents"):
        attribute([doc("a")], ["A"])
    with pytest.raises(ValueError, match="holdout group"):
        attribute([doc("a"), doc("b")], ["A", "B"], groups=["w1"])
    with pytest.raises(ValueError, match="n_words must be positive"):
        most_frequent_words([["a"]], 0)
    with pytest.raises(ValueError, match="nonempty document"):
        most_frequent_words([[]], 5)


def test_delta_separates_two_habits_that_differ_only_in_common_words():
    """Delta's premise: authors are told apart by words they cannot avoid, not by subject matter."""
    a = [("ο δε και ο δε και " * 30).split() for _ in range(3)]
    b = [("η τε γαρ η τε γαρ " * 30).split() for _ in range(3)]
    docs, labels = a + b, ["A"] * 3 + ["B"] * 3
    groups = [f"w{i}" for i in range(6)]
    assert attribute(docs, labels, groups=groups) == labels
