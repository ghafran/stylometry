"""Combining rate, placement and repetition into one representation.

The risk in stacking feature families is that the biggest block wins on column count rather than on
evidence: placement contributes hundreds of transition columns against repetition's nine. Variance
equalisation is what prevents that, so it is what is tested here.
"""
from __future__ import annotations

import numpy as np
import pytest

from stylometry import combined

A = ("ο δε και ο δε και " * 40).split()
B = ("η τε γαρ η τε γαρ " * 40).split()


def test_a_block_with_zero_weight_contributes_no_columns():
    X, names = combined.build([A, B, A], weights={"rate": 1, "placement": 0, "repetition": 0})
    assert all(n.startswith("rate:") for n in names)
    assert X.shape[1] == len(names)


def test_every_requested_block_is_present_and_labelled():
    _, names = combined.build([A, B, A])
    prefixes = {n.split(":")[0] for n in names}
    assert prefixes == {"rate", "place", "rep"}


def test_no_block_dominates_by_having_more_columns():
    """Equal weights must mean equal influence, not influence proportional to column count.

    The documents must differ on all three blocks: a block whose values never vary carries no
    variance to equalise, and would look like a failure here while behaving correctly.
    """
    varied = (" ".join(f"ο δε λογοσ{i} και τε{i}" for i in range(60))).split()
    docs = [A, B, varied, A + B]
    X, names = combined.build(docs)
    by_block = {}
    for prefix in ("rate", "place", "rep"):
        cols = [i for i, n in enumerate(names) if n.startswith(prefix + ":")]
        by_block[prefix] = float(np.var(X[:, cols], axis=0).sum())
    counts = {p: sum(n.startswith(p + ":") for n in names) for p in by_block}
    assert max(counts.values()) >= 2 * min(counts.values()), f"the blocks must differ in width: {counts}"
    assert max(by_block.values()) == pytest.approx(min(by_block.values()), rel=0.05), by_block


def test_weights_scale_a_block_as_asked():
    docs = [A, B, A, B]
    single, names = combined.build(docs, weights={"rate": 1, "placement": 0, "repetition": 0})
    double, _ = combined.build(docs, weights={"rate": 2, "placement": 0, "repetition": 0})
    assert float(np.var(double, axis=0).sum()) == pytest.approx(4 * float(np.var(single, axis=0).sum()), rel=1e-6)


@pytest.mark.parametrize("weights,match", [
    ({"rate": 0, "placement": 0, "repetition": 0}, "at least one block"),
    ({"rate": -1}, "finite and nonnegative"),
])
def test_useless_or_invalid_weights_are_refused(weights, match):
    with pytest.raises(ValueError, match=match):
        combined.build([A, B], weights=weights)


def test_attribution_never_uses_anything_held_out_with_the_document():
    docs = [A, A, B, B]
    labels = ["A", "A", "B", "B"]
    assert combined.attribute(docs, labels, groups=["w1", "w1", "w2", "w2"]) == ["B", "B", "A", "A"]
    assert combined.attribute(docs, labels, groups=["w1", "w2", "w3", "w4"]) == ["A", "A", "B", "B"]


def test_attribution_validates_its_inputs():
    with pytest.raises(ValueError, match="every document needs a label"):
        combined.attribute([A], [])
    with pytest.raises(ValueError, match="at least two documents"):
        combined.attribute([A], ["x"])
    with pytest.raises(ValueError, match="holdout group"):
        combined.attribute([A, B], ["x", "y"], groups=["w1"])


def test_placement_block_keeps_only_transitions_that_actually_occur():
    _, names = combined.placement_block([A, B], n_markers=50, keep=10_000)
    assert names, "some transitions do occur"
    assert len(names) < 50 * 50, "empty transitions are dropped rather than carried as zeros"
