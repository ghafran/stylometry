"""Development-only rejection calibrated on authors absent from model fitting.

The calibration false-acceptance rate is an empirical constraint, not a bound
on the risk for a new author. Callers must keep all test authors and test works
out of both fitting and calibration, and keep impostor authors out of fitting.
"""
from __future__ import annotations

import numpy as np


def _candidates(authors):
    authors = list(authors)
    if (len(authors) < 2 or len(set(authors)) != len(authors)
            or any(not isinstance(a, str) or not a.strip() for a in authors)):
        raise ValueError("at least two distinct, nonempty candidate authors are required")
    return authors


def _scores(values, n_authors, n_rows=None):
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != n_authors:
        raise ValueError("scores must have one column per candidate author")
    if n_rows is not None and scores.shape[0] != n_rows:
        raise ValueError("score rows must match calibration rows")
    if not np.isfinite(scores).all():
        raise ValueError("scores must all be finite")
    return scores


def _row_metadata(rows):
    works = {}
    for row in rows:
        if any(not isinstance(row.get(key), str) or not row[key].strip()
               for key in ("author", "work_id")):
            raise ValueError("each calibration row needs a nonempty author and work_id")
        if row["work_id"] in works and works[row["work_id"]] != row["author"]:
            raise ValueError("one canonical work cannot have multiple author labels")
        works[row["work_id"]] = row["author"]
    return {"authors": len(set(works.values())), "works": len(works), "samples": len(rows)}


def _weights(rows):
    """Equal authors, then equal works per author, then equal chunks per work."""
    by_author = {}
    for row in rows:
        works = by_author.setdefault(row["author"], {})
        works[row["work_id"]] = works.get(row["work_id"], 0) + 1
    return np.array([1 / len(by_author) / len(by_author[row["author"]])
                     / by_author[row["author"]][row["work_id"]] for row in rows])


def _decisions(scores, threshold):
    top = scores.max(axis=1)
    unambiguous = (scores == top[:, None]).sum(axis=1) == 1
    return unambiguous & (top >= threshold)


def _rate(weights, accepted):
    # Normalize explicitly so an all-accepted batch is exactly one even when
    # fractional per-chunk weights accumulate with floating-point roundoff.
    return float(weights[accepted].sum() / weights.sum())


def calibrate_rejection(known_rows, known_scores, impostor_rows, impostor_scores,
                        authors, *, max_false_acceptance=.05,
                        min_known_acceptance=.80) -> dict:
    """Choose the least restrictive threshold satisfying the impostor constraint.

    An impostor's statistic is its maximum candidate score, just as at use time.
    Complete score ties are handled together; an exact tie between top candidate
    authors always abstains. Calibration rates weight authors and works equally.
    A threshold is usable only if it also retains enough known calibration text.
    Reported rates describe the candidate threshold even when it is unusable.
    """
    authors = _candidates(authors)
    for name, value in (("max_false_acceptance", max_false_acceptance),
                        ("min_known_acceptance", min_known_acceptance)):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be between zero and one")
    known_rows, impostor_rows = list(known_rows), list(impostor_rows)
    known_info, impostor_info = map(_row_metadata, (known_rows, impostor_rows))
    known_scores = _scores(known_scores, len(authors), len(known_rows))
    impostor_scores = _scores(impostor_scores, len(authors), len(impostor_rows))
    if any(row["author"] not in authors for row in known_rows):
        raise ValueError("known calibration authors must belong to the candidate set")
    if any(row["author"] in authors for row in impostor_rows):
        raise ValueError("impostor authors must be absent from the candidate set")
    if {r["work_id"] for r in known_rows} & {r["work_id"] for r in impostor_rows}:
        raise ValueError("known and impostor calibration works must be disjoint")
    result = {
        "status": "insufficient_calibration", "feasible": False, "threshold": None,
        "known_acceptance": None, "known_correct_acceptance": None,
        "known_wrong_acceptance": None, "unknown_false_acceptance": None,
        "candidate_authors": authors,
        "calibration_counts": {"known": known_info, "impostor": impostor_info},
        "targets": {"max_false_acceptance": float(max_false_acceptance),
                    "min_known_acceptance": float(min_known_acceptance)},
        "weighting": "equal authors, then equal works per author, then equal samples per work",
        "scope": "development only; empirical calibration rates do not certify unseen-author risk",
        "reason": "Need known calibration works for every candidate and at least one separate impostor author.",
    }
    if not impostor_rows or {r["author"] for r in known_rows} != set(authors):
        return result

    known_weights, impostor_weights = _weights(known_rows), _weights(impostor_rows)
    known_top, impostor_top = known_scores.max(axis=1), impostor_scores.max(axis=1)
    # This first threshold accepts every unambiguous observed row. The remaining
    # thresholds reject complete impostor score groups, never fractions of ties.
    candidates = [float(min(known_top.min(), impostor_top.min()))]
    with np.errstate(over="ignore"):
        candidates.extend(float(np.nextafter(score, np.inf)) for score in np.unique(impostor_top)
                          if np.isfinite(np.nextafter(score, np.inf)))
    for threshold in sorted(set(candidates)):
        impostor_accepted = _decisions(impostor_scores, threshold)
        far = _rate(impostor_weights, impostor_accepted)
        if far <= max_false_acceptance:
            known_accepted = _decisions(known_scores, threshold)
            winners = np.asarray(authors)[known_scores.argmax(axis=1)]
            correct = winners == np.array([r["author"] for r in known_rows])
            retention = _rate(known_weights, known_accepted)
            feasible = retention >= min_known_acceptance
            result.update({
                "status": "development_only" if feasible else "no_feasible_threshold",
                "feasible": bool(feasible), "threshold": threshold,
                "known_acceptance": retention,
                "known_correct_acceptance": _rate(known_weights, known_accepted & correct),
                "known_wrong_acceptance": _rate(known_weights, known_accepted & ~correct),
                "unknown_false_acceptance": far,
                "reason": ("Calibration point targets met; independent validation is still required."
                           if feasible else
                           "No threshold meets both calibration targets; all application decisions abstain."),
            })
            return result
    # Finite scores can reach the largest representable float. A threshold above
    # that value cannot be serialized safely; abstain instead of emitting infinity.
    result.update(status="no_feasible_threshold",
                  reason="No finite threshold meets the impostor constraint; all application decisions abstain.")
    return result


def apply_rejection(scores, calibration) -> np.ndarray:
    """Accept only under a feasible development calibration and a unique winner."""
    authors = _candidates(calibration["candidate_authors"])
    scores = _scores(scores, len(authors))
    if not calibration.get("feasible"):
        return np.zeros(len(scores), dtype=bool)
    threshold = calibration.get("threshold")
    if (calibration.get("status") != "development_only" or threshold is None
            or not np.isfinite(threshold)):
        raise ValueError("feasible calibration must have a finite development threshold")
    return _decisions(scores, threshold)
