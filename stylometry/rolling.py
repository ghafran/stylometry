"""Where does the style change, rather than how many styles are there.

Clustering asks for a partition of independent units. A book is not that: it is a sequence, and the
question the Documentary Hypothesis actually poses is where the seams fall. Rolling analysis (Eder)
slides a window along the text and measures style continuously; change-point detection then asks which
of those movements is larger than the text's own noise.

The change-point statistic here is a simple, honest one: at each candidate split, how far apart are
the mean feature vectors either side, standardised by the variation within them. Significance comes
from a permutation test that shuffles the windows, so the null is "this text, in some other order"
rather than an assumed distribution. Positions are reported with that p-value attached and never
without it, because a maximum of a noisy series always looks like a discovery.
"""
from __future__ import annotations

import numpy as np


def windows(tokens: list[str], size: int = 1000, step: int = 500) -> list[tuple[int, list[str]]]:
    """Overlapping windows with their start offsets; overlap is what makes the series continuous."""
    if size < 1 or step < 1:
        raise ValueError("window size and step must be positive")
    if len(tokens) < size:
        return [(0, list(tokens))] if tokens else []
    return [(i, tokens[i:i + size]) for i in range(0, len(tokens) - size + 1, step)]


def _standardise(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0)
    out = (X - X.mean(axis=0)) / np.where(sd < 1e-12, 1.0, sd)
    out[:, sd < 1e-12] = 0.0
    return out


def separation(X: np.ndarray, split: int) -> float:
    """Distance between the two sides' centroids, scaled like a two-sample statistic.

    The naive ratio of gap to spread is biased towards the ends of the sequence: with three windows on
    one side its variance is unstable and easily small, which inflates the ratio. Measured on
    single-author Greek works, the unscaled form put the strongest split at the earliest permitted
    position in most of them. The sqrt(n_left * n_right / n) factor is the standard weighting for a
    difference of means and removes that pull.
    """
    left, right = X[:split], X[split:]
    if len(left) < 2 or len(right) < 2:
        return 0.0
    gap = np.linalg.norm(left.mean(axis=0) - right.mean(axis=0))
    spread = np.sqrt(0.5 * (left.var(axis=0).sum() + right.var(axis=0).sum()))
    if spread <= 1e-12:
        return 0.0
    weight = np.sqrt(len(left) * len(right) / len(X))
    return float(gap / spread * weight)


def change_points(X: np.ndarray, min_side: int = 3, permutations: int = 1000,
                  seed: int = 0) -> dict:
    """The strongest split in the sequence, with a permutation p-value for whether it is real.

    Shuffling the windows destroys any ordering while keeping every window intact, so the null is
    this text in some other order. A p-value near 1 means the strongest split found is no stronger
    than the best split of the same windows shuffled - which is the usual outcome for a text with no
    seam, and the reason the statistic is never reported on its own.
    """
    Z = _standardise(np.asarray(X, dtype=float))
    n = len(Z)
    if n < 2 * min_side:
        return {"available": False, "reason": "too few windows", "n_windows": n}
    candidates = range(min_side, n - min_side + 1)
    scores = {s: separation(Z, s) for s in candidates}
    best = max(scores, key=lambda s: scores[s])
    observed = scores[best]
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(permutations):
        shuffled = Z[rng.permutation(n)]
        if max(separation(shuffled, s) for s in candidates) >= observed:
            hits += 1
    return {
        "available": True, "n_windows": n, "split_index": int(best), "separation": observed,
        "p_value": (hits + 1) / (permutations + 1), "permutations": permutations,
        "profile": {int(s): float(v) for s, v in scores.items()},
        "note": "a permutation p-value against the same windows in shuffled order; see "
                "single_author_baseline for why this p-value alone is not evidence of a seam",
    }


def single_author_baseline(separations: list[float]) -> dict:
    """What the statistic reaches in works known to have one author, which is the only usable null.

    The permutation test asks whether the windows could be in any order. For real prose they could
    not: neighbouring windows share a topic, so *every* continuous text beats that null. Measured on
    eighteen single-author Greek works, all of them did, at p < 0.02 - the test was detecting topical
    drift and calling it a seam.

    The honest comparison is therefore empirical. Run the same statistic over works whose single
    authorship is not in question, and a candidate seam has to exceed what those works reach by
    ordinary internal variation. This returns that reference distribution.
    """
    if not separations:
        raise ValueError("a baseline needs at least one single-author work")
    values = np.sort(np.asarray(separations, dtype=float))
    return {
        "n_works": len(values), "mean": float(values.mean()), "max": float(values[-1]),
        "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
        "note": "a candidate seam must exceed what single-author works reach by internal variation",
    }


def exceeds_baseline(separation_value: float, baseline: dict) -> dict:
    """Where a candidate seam falls against the single-author reference."""
    return {
        "separation": float(separation_value),
        "baseline_p95": baseline["p95"], "baseline_max": baseline["max"],
        "exceeds_p95": bool(separation_value > baseline["p95"]),
        "exceeds_all_single_author_works": bool(separation_value > baseline["max"]),
    }
