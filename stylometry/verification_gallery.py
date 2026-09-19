"""Aggregate cross-work verifier evidence and calibrate a complete author gallery.

Pair scores and gallery scores are raw decision statistics, not probabilities.
More passages from a reference work do not create more independent works. Every
candidate must agree across at least two reference works, and operational use
requires calibration on both known and separate unknown authors.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from fractions import Fraction
from numbers import Real

import numpy as np


def _authors(values):
    authors = list(values)
    if (len(authors) < 2 or any(not isinstance(a, str) or not a.strip() for a in authors)
            or len(set(authors)) != len(authors)):
        raise ValueError("gallery needs at least two distinct nonempty candidate authors")
    return authors


def _metadata(rows, *, text_required=False):
    works, languages, ids = {}, set(), set()
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
                                            for key in ("author", "work_id")):
            raise ValueError("each row needs nonempty author and canonical work_id")
        if row["work_id"] in works and works[row["work_id"]] != row["author"]:
            raise ValueError("one canonical work cannot have multiple author labels")
        works[row["work_id"]] = row["author"]
        if text_required or "language" in row:
            if row.get("language") not in ("grc", "hbo", "arb", "eng"):
                raise ValueError("rows need a supported language")
            languages.add(row["language"])
        if text_required and (not isinstance(row.get("text_bare"), str) or not row["text_bare"].strip()):
            raise ValueError("gallery and query rows need nonempty normalized text_bare")
        if "id" in row:
            if not isinstance(row["id"], str) or not row["id"].strip() or row["id"] in ids:
                raise ValueError("passage ids must be nonempty and unique")
            ids.add(row["id"])
    if len(languages) > 1:
        raise ValueError("gallery comparisons must use one language")
    return {"authors": len(set(works.values())), "works": len(works), "samples": len(rows)}


def _check_text_duplicates(rows):
    seen = {}
    for row in rows:
        if not isinstance(row.get("text_bare"), str):
            continue
        normalized = " ".join(row["text_bare"].split())
        if not normalized:
            continue
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if digest in seen and seen[digest] != row["work_id"]:
            raise ValueError("identical normalized text appears under different canonical works")
        seen[digest] = row["work_id"]


def _scores(values, count, rows=None):
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != count:
        raise ValueError("score columns must match the calibrated candidate count")
    if rows is not None and len(scores) != rows:
        raise ValueError("score rows must match calibration rows")
    if not np.isfinite(scores).all():
        raise ValueError("gallery scores must be finite")
    return scores


def build_gallery_scores(verifier, gallery_rows, query_rows) -> dict:
    """Score each query against every passage of each candidate reference work.

    Per-work means receive equal standing: the candidate's statistic is their
    minimum, so one strong work cannot compensate for a weaker reference work.
    Query author labels are used only to reject invalid experimental overlap;
    they never choose candidates, comparisons, weights, or returned scores.
    """
    gallery, queries = list(gallery_rows), list(query_rows)
    _metadata(gallery + queries, text_required=True)
    _check_text_duplicates(gallery + queries)
    authors = _authors(sorted({row["author"] for row in gallery}))
    gallery_works = {row["work_id"] for row in gallery}
    query_works = {row["work_id"] for row in queries}
    if gallery_works & query_works:
        raise ValueError("query and gallery canonical works must be disjoint; no self-comparisons")
    try:
        training_authors = set(verifier.training_authors_)
        training_works = set(verifier.training_work_ids_)
    except (AttributeError, TypeError) as exc:
        raise ValueError("fitted verifier training-author and training-work provenance is required") from exc
    if not training_authors or not training_works:
        raise ValueError("fitted verifier training-author and training-work provenance is required")
    if training_authors & {row["author"] for row in gallery + queries}:
        raise ValueError("gallery and query authors must be absent from verifier training")
    if training_works & (gallery_works | query_works):
        raise ValueError("gallery and query works must be absent from verifier training")
    language = gallery[0]["language"]
    if getattr(verifier, "language_", language) != language:
        raise ValueError("gallery language differs from the fitted verifier")
    grouped = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(gallery):
        grouped[row["author"]][row["work_id"]].append(index)
    if any(len(works) < 2 for works in grouped.values()):
        raise ValueError("each candidate needs at least two canonical reference works")

    scores = np.empty((len(queries), len(authors)), dtype=float)
    evidence = []
    for index, query in enumerate(queries):
        # Score a complete query/gallery batch once; do not repeatedly rebuild
        # the verifier's passage features separately for each reference work.
        all_scores = np.asarray(verifier.score_pairs([query] * len(gallery), gallery), dtype=float)
        if all_scores.shape != (len(gallery),) or not np.isfinite(all_scores).all():
            raise ValueError("verifier must return one finite raw score per comparison")
        candidate_evidence = []
        for column, author in enumerate(authors):
            work_evidence = []
            for work_id, indices in sorted(grouped[author].items()):
                # All complete reference passages are included; upstream corpus
                # sampling determines their cap without inspecting this query.
                pair_scores = all_scores[indices]
                # Divide before summing to avoid overflow for large finite scores.
                work_score = float(np.sum(pair_scores / len(pair_scores)))
                if not np.isfinite(work_score):
                    raise ValueError("reference-work score must be finite")
                work_evidence.append({"work_id": work_id, "score": work_score, "comparisons": len(indices)})
            value = min(work["score"] for work in work_evidence)
            scores[index, column] = value
            candidate_evidence.append({"author": author, "score": value, "reference_works": work_evidence})
        evidence.append({"query_id": query.get("id", f"query:{index}"), "query_work_id": query["work_id"],
                         "candidates": candidate_evidence})
    return {
        "authors": authors, "scores": scores, "evidence": evidence,
        "aggregation": "minimum across canonical reference works of each work's mean passage-comparison score",
        "scope": "raw evidence statistic, not author probability; passage comparisons are not independent works",
    }


def _weights(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        grouped[row["author"]][row["work_id"]].append(index)
    return [(np.asarray(indices), Fraction(1, len(grouped) * len(works)))
            for works in grouped.values() for indices in works.values()]


def _decisions(scores, threshold):
    maxima = scores.max(axis=1)
    unique = (scores == maxima[:, None]).sum(axis=1) == 1
    return unique & (maxima >= threshold)


def _rate(weights, selected):
    # Compute the rational work rates before converting once to float. Repeating
    # a work's identical passages must not change an exact 80% into 79.9999999%
    # and incorrectly turn a feasible threshold into an abstain-all calibration.
    return float(sum((mass * Fraction(int(np.count_nonzero(selected[indices])), len(indices))
                      for indices, mass in weights), Fraction()))


def calibrate_gallery(known_rows, known_scores, unknown_rows, unknown_scores, authors, *,
                      max_false_acceptance=.05, min_correct_acceptance=.80,
                      max_wrong_acceptance=.05) -> dict:
    """Calibrate the gallery's top-score decision, including wrong known matches.

    All rates are conditional empirical calibration rates. They are not a bound
    on unseen-author population risk. A diagnostic threshold can be returned for
    an infeasible calibration; ``apply_gallery_rejection`` then abstains on all
    queries regardless of that threshold's diagnostic rates.
    """
    authors = _authors(authors)
    targets = {"max_false_acceptance": max_false_acceptance, "min_correct_acceptance": min_correct_acceptance,
               "max_wrong_acceptance": max_wrong_acceptance}
    for name, value in targets.items():
        if (isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
                or not 0 <= value <= 1 or (name == "min_correct_acceptance" and value == 0)):
            raise ValueError("error targets must be between zero and one; minimum correct acceptance must be positive")
    known, unknown = list(known_rows), list(unknown_rows)
    known_info, unknown_info = _metadata(known), _metadata(unknown)
    _metadata(known + unknown)
    _check_text_duplicates(known + unknown)
    known_scores = _scores(known_scores, len(authors), len(known))
    unknown_scores = _scores(unknown_scores, len(authors), len(unknown))
    if any(row["author"] not in authors for row in known):
        raise ValueError("known calibration authors must belong to the candidate set")
    if any(row["author"] in authors for row in unknown):
        raise ValueError("unknown calibration authors must be absent from the candidate set")
    if {row["work_id"] for row in known} & {row["work_id"] for row in unknown}:
        raise ValueError("known and unknown calibration works must be disjoint")
    result = {
        "status": "insufficient_calibration", "feasible": False, "threshold": None,
        "candidate_count": len(authors), "candidate_authors": authors,
        "unknown_false_acceptance": None, "known_correct_acceptance": None,
        "known_wrong_acceptance": None, "known_acceptance": None,
        "calibration_counts": {"known": known_info, "unknown": unknown_info},
        "targets": {name: float(value) for name, value in targets.items()},
        "weighting": "equal authors, then equal canonical works per author, then equal passages per work",
        "decision": "unique maximum candidate score at or above threshold; all decisions abstain when infeasible",
        "scope": "development calibration only; no certified bound on future author error",
        "reason": "Need known calibration works for every candidate and at least two separate unknown authors.",
    }
    if {row["author"] for row in known} != set(authors) or unknown_info["authors"] < 2:
        return result

    known_weights, unknown_weights = _weights(known), _weights(unknown)
    known_top, unknown_top = known_scores.max(axis=1), unknown_scores.max(axis=1)
    observed = np.unique(np.concatenate((known_top, unknown_top)))
    thresholds = [float(observed[0])]
    with np.errstate(over="ignore"):
        thresholds.extend(float(np.nextafter(value, np.inf)) for value in observed
                          if np.isfinite(np.nextafter(value, np.inf)))
    correct = np.array(authors)[known_scores.argmax(axis=1)] == np.array([row["author"] for row in known])
    admissible = []
    for threshold in thresholds:
        known_accept = _decisions(known_scores, threshold)
        rates = {
            "threshold": threshold,
            "unknown_false_acceptance": _rate(unknown_weights, _decisions(unknown_scores, threshold)),
            "known_correct_acceptance": _rate(known_weights, known_accept & correct),
            "known_wrong_acceptance": _rate(known_weights, known_accept & ~correct),
            "known_acceptance": _rate(known_weights, known_accept),
        }
        if (rates["unknown_false_acceptance"] <= max_false_acceptance
                and rates["known_wrong_acceptance"] <= max_wrong_acceptance):
            admissible.append(rates)
    if not admissible:
        result.update(status="no_feasible_threshold",
                      reason="No finite threshold meets both error limits; all application decisions abstain.")
        return result
    chosen = max(admissible, key=lambda rates: (rates["known_correct_acceptance"], -rates["threshold"]))
    feasible = chosen["known_correct_acceptance"] >= min_correct_acceptance
    result.update(chosen)
    result.update(
        status="development_only" if feasible else "no_feasible_threshold", feasible=bool(feasible),
        reason=("Calibration point targets met; independent author validation is still required." if feasible else
                "No threshold meets both error limits and useful correct acceptance; returned threshold is diagnostic and all application decisions abstain."),
    )
    return result


def apply_gallery_rejection(scores, calibration) -> np.ndarray:
    """Apply a feasible threshold in the recorded candidate order, or abstain."""
    authors = _authors(calibration["candidate_authors"])
    count = calibration.get("candidate_count")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(authors):
        raise ValueError("calibration candidate count does not match its candidate authors")
    scores = _scores(scores, count)
    if not calibration.get("feasible"):
        return np.zeros(len(scores), dtype=bool)
    threshold = calibration.get("threshold")
    if (calibration.get("status") != "development_only" or isinstance(threshold, bool)
            or not isinstance(threshold, Real) or not np.isfinite(threshold)):
        raise ValueError("feasible calibration must have a finite development threshold")
    return _decisions(scores, threshold)
