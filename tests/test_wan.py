"""Word adjacency networks: style as arrangement rather than rate.

The graph is easy to build wrongly in ways that still yield plausible numbers, so what is pinned here
is the behaviour that defines the method: only markers enter, direction is kept, distance is
discounted, content words are stepped over rather than breaking a link, and an unseen transition is
improbable rather than impossible.
"""
from __future__ import annotations

import numpy as np
import pytest

from stylometry import wan

M = ["a", "b", "c"]


def test_only_marker_words_enter_the_graph():
    W = wan.adjacency("a zzz b".split(), M, window=10)
    assert W.shape == (3, 3)
    assert W[0, 1] > 0, "a is followed by b across an intervening content word"
    assert W.sum() == pytest.approx(W[0, 1]), "the content word contributes nothing of its own"


def test_direction_is_kept_because_word_order_is_the_point():
    W = wan.adjacency(["a", "b"], M, window=5)
    assert W[0, 1] > 0 and W[1, 0] == 0


def test_distance_is_discounted_so_neighbours_count_for_more():
    near = wan.adjacency(["a", "b"], M, window=5)
    far = wan.adjacency(["a", "x", "x", "b"], M, window=5)
    assert near[0, 1] > far[0, 1] > 0
    flat = wan.adjacency(["a", "x", "x", "b"], M, window=5, decay="uniform")
    assert flat[0, 1] == pytest.approx(1.0), "uniform decay counts every position in the window alike"


def test_nothing_is_linked_beyond_the_window():
    assert wan.adjacency(["a"] + ["x"] * 20 + ["b"], M, window=5)[0, 1] == 0


def test_rows_are_probabilities_and_an_unseen_transition_stays_possible():
    P = wan.transition_matrix(wan.adjacency(["a", "b"], M, window=5))
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    assert (P > 0).all(), "a zero would make the divergence infinite on one unseen bigram"
    assert P[0, 1] > P[0, 2]


def test_a_marker_the_text_never_uses_gives_an_uninformative_row_rather_than_a_crash():
    P = wan.transition_matrix(wan.adjacency(["a", "b"], M, window=5))
    np.testing.assert_allclose(P[2], np.full(3, 1 / 3))


def test_divergence_is_zero_against_itself_and_symmetric():
    p = wan.profile(("a b c " * 20).split(), M)
    q = wan.profile(("c b a " * 20).split(), M)
    assert wan.divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert wan.divergence(p, q) == pytest.approx(wan.divergence(q, p))
    assert wan.divergence(p, q) > 0


def test_two_habits_that_differ_only_in_ordering_are_told_apart():
    """The claim the method rests on: same words, same rates, different arrangement."""
    forward = [("a b c " * 40).split() for _ in range(3)]
    reverse = [("c b a " * 40).split() for _ in range(3)]
    docs, labels = forward + reverse, ["F"] * 3 + ["R"] * 3
    groups = [f"w{i}" for i in range(6)]
    got = wan.attribute(docs, labels, groups=groups, n_markers=None)
    # markers_for reads the language list, so drive the graph directly for this synthetic vocabulary
    profiles = [wan.profile(d, M) for d in docs]
    D = wan.distances(profiles)
    assert D[0, 1] < D[0, 3], "same arrangement is closer than reversed arrangement"
    assert len(got) == 6


@pytest.mark.parametrize("bad,match", [
    ({"window": 0}, "window must be positive"),
    ({"decay": "linear"}, "unknown decay"),
])
def test_invalid_settings_are_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        wan.adjacency(["a"], M, **bad)


def test_an_empty_marker_list_is_refused_rather_than_silently_empty():
    with pytest.raises(ValueError, match="at least one marker"):
        wan.adjacency(["a"], [])


def test_networks_built_on_different_vocabularies_are_not_compared():
    with pytest.raises(ValueError, match="share a marker vocabulary"):
        wan.divergence(np.ones((2, 2)) / 2, np.ones((3, 3)) / 3)


def test_pooled_attribution_excludes_everything_held_out_with_the_document():
    docs = [("a b " * 40).split(), ("a b " * 40).split(), ("b a " * 40).split(), ("b a " * 40).split()]
    labels = ["A", "A", "B", "B"]
    markers = ["a", "b"]
    import stylometry.wan as module
    original = module.markers_for
    module.markers_for = lambda language, limit=None: markers
    try:
        together = module.attribute_pooled(docs, labels, groups=["w1", "w1", "w2", "w2"])
        assert together == ["B", "B", "A", "A"], "with its own work held out only the wrong pool remains"
    finally:
        module.markers_for = original
