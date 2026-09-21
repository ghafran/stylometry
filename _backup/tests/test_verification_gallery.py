"""Gallery-level controls for cross-work evidence and useful, correct rejection."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from stylometry.verification_gallery import apply_gallery_rejection, build_gallery_scores, calibrate_gallery


def passage(author, work, ident, value=0, **changes):
    return {"author": author, "work_id": work, "id": ident, "language": "grc",
            "text_bare": f"και λογοσ {ident}", "value": value, **changes}


class SymmetricVerifier:
    training_authors_ = ["training-a", "training-b"]
    training_work_ids_ = ["train-a1", "train-b1"]
    language_ = "grc"

    def __init__(self):
        self.calls = []

    def score_pairs(self, left, right):
        self.calls.extend(zip(left, right))
        return np.array([a["value"] + b["value"] for a, b in zip(left, right)])


def gallery():
    return [passage("A", "a1", "a1.1", 8), passage("A", "a1", "a1.2", 10),
            passage("A", "a2", "a2.1", 1), passage("B", "b1", "b1.1", 2),
            passage("B", "b2", "b2.1", 3)]


def calibration():
    known = [{"author": "A", "work_id": "ka"}, {"author": "B", "work_id": "kb"}]
    unknown = [{"author": "U", "work_id": "uu"}, {"author": "V", "work_id": "uv"}]
    return known, np.array([[.95, .1], [.1, .95]]), unknown, np.array([[.2, .1], [.1, .3]]), ["A", "B"]


def test_gallery_uses_work_means_then_weakest_work_without_self_comparisons():
    references = gallery()
    queries = [passage("A", "a3", "a3.1"), passage("U", "u1", "u1.1", 1)]
    original = copy.deepcopy(references + queries)
    verifier = SymmetricVerifier()
    result = build_gallery_scores(verifier, references, queries)
    assert result["authors"] == ["A", "B"]
    np.testing.assert_array_equal(result["scores"], [[1, 2], [2, 3]])
    first = result["evidence"][0]["candidates"][0]
    assert first == {"author": "A", "score": 1, "reference_works": [
        {"work_id": "a1", "score": 9, "comparisons": 2}, {"work_id": "a2", "score": 1, "comparisons": 1}]}
    assert len(verifier.calls) == len(references) * len(queries)
    assert all(a["work_id"] != b["work_id"] for a, b in verifier.calls)
    assert references + queries == original
    json.dumps(result["evidence"], allow_nan=False)


def test_one_long_strong_work_cannot_override_a_weaker_reference_work():
    references = gallery() + [passage("A", "a1", f"a1.extra{i}", 1000) for i in range(30)]
    result = build_gallery_scores(SymmetricVerifier(), references, [passage("U", "u1", "query")])
    assert result["scores"][0, 0] == 1
    assert len(result["evidence"][0]["candidates"][0]["reference_works"]) == 2


def test_query_author_label_never_changes_candidate_scores_or_evidence():
    query = passage("A", "query-work", "query")
    known = build_gallery_scores(SymmetricVerifier(), gallery(), [query])
    unknown = build_gallery_scores(SymmetricVerifier(), gallery(), [{**query, "author": "unfamiliar"}])
    np.testing.assert_array_equal(known["scores"], unknown["scores"])
    assert known["evidence"] == unknown["evidence"]


@pytest.mark.parametrize("change", [
    {"work_id": "a1", "author": "A"}, {"author": "training-a"},
    {"work_id": "train-a1"}, {"language": "hbo"},
    {"id": "a1.1"}, {"text_bare": "  και   λογοσ a1.1  "},
])
def test_overlap_language_and_duplicate_text_are_rejected_before_scoring(change):
    verifier = SymmetricVerifier()
    with pytest.raises(ValueError):
        build_gallery_scores(verifier, gallery(), [{**passage("U", "u1", "query"), **change}])
    assert not verifier.calls


def test_candidate_requires_two_canonical_works_not_two_passages():
    references = [row for row in gallery() if row["work_id"] != "a2"]
    with pytest.raises(ValueError, match="two canonical reference works"):
        build_gallery_scores(SymmetricVerifier(), references, [passage("U", "u1", "query")])


def test_canonical_work_cannot_have_different_author_labels():
    references = gallery() + [passage("B", "a1", "mislabelled")]
    with pytest.raises(ValueError, match="canonical work"):
        build_gallery_scores(SymmetricVerifier(), references, [passage("U", "u1", "query")])


@pytest.mark.parametrize("bad_scores", [np.array([np.nan]), np.array([np.inf]), np.ones((2, 1)), np.array(1)])
def test_pair_verifier_must_return_finite_one_dimensional_comparison_scores(bad_scores):
    verifier = SymmetricVerifier()
    verifier.score_pairs = lambda *args: bad_scores
    with pytest.raises(ValueError, match="one finite raw score"):
        build_gallery_scores(verifier, gallery(), [passage("U", "u1", "query")])


def test_fitted_verifier_provenance_is_required():
    with pytest.raises(ValueError, match="provenance"):
        build_gallery_scores(object(), gallery(), [passage("U", "u1", "query")])


def test_empty_query_batch_preserves_candidate_shape_without_scoring():
    verifier = SymmetricVerifier()
    result = build_gallery_scores(verifier, gallery(), [])
    assert result["scores"].shape == (0, 2)
    assert result["evidence"] == [] and verifier.calls == []


def test_feasible_threshold_calibrates_unique_maximum_candidate_score():
    result = calibrate_gallery(*calibration())
    assert result["feasible"] and result["status"] == "development_only"
    assert result["threshold"] == np.nextafter(.3, np.inf)
    assert result["candidate_count"] == 2
    assert result["known_correct_acceptance"] == result["known_acceptance"] == 1
    assert result["unknown_false_acceptance"] == result["known_wrong_acceptance"] == 0
    np.testing.assert_array_equal(apply_gallery_rejection([[.9, .1], [.3, .1], [.99, .99]], result), [True, False, False])
    json.dumps(result, allow_nan=False)


def test_small_per_pair_error_does_not_masquerade_as_small_gallery_error():
    authors = [f"a{i}" for i in range(20)]
    known = [{"author": author, "work_id": f"k-{author}"} for author in authors]
    known_scores = np.eye(20) * .7 + .1
    unknown = [{"author": "U", "work_id": "u1"}, {"author": "V", "work_id": "v1"}]
    unknown_scores = np.full((2, 20), .1)
    unknown_scores[0, 0] = unknown_scores[1, 1] = .9
    assert np.mean(unknown_scores >= .8) == .05
    result = calibrate_gallery(known, known_scores, unknown, unknown_scores, authors)
    assert not result["feasible"]
    assert result["known_correct_acceptance"] == 0
    assert result["unknown_false_acceptance"] == 0
    assert not apply_gallery_rejection(known_scores, result).any()


def imperfect_known():
    known = [{"author": author, "work_id": f"{author}{i}"} for author in ("A", "B") for i in range(5)]
    scores = np.array([[.9, .1]] * 4 + [[.1, .4]] + [[.1, .9]] * 4 + [[.4, .1]])
    unknown = [{"author": "U", "work_id": "u1"}, {"author": "V", "work_id": "v1"}]
    return known, scores, unknown, [[.1, .05], [.05, .1]], ["A", "B"]


def test_known_score_breakpoints_can_reject_wrong_known_matches():
    result = calibrate_gallery(*imperfect_known())
    assert result["feasible"]
    assert result["threshold"] == np.nextafter(.4, np.inf)
    assert result["known_correct_acceptance"] == pytest.approx(.8)
    assert result["known_acceptance"] == pytest.approx(.8)
    assert result["known_wrong_acceptance"] == result["unknown_false_acceptance"] == 0


def test_infeasible_calibration_reports_best_safe_coverage_but_abstains():
    result = calibrate_gallery(*imperfect_known(), min_correct_acceptance=.9)
    assert not result["feasible"]
    assert result["threshold"] == np.nextafter(.4, np.inf)
    assert result["known_correct_acceptance"] == pytest.approx(.8)
    assert "diagnostic" in result["reason"]
    assert not apply_gallery_rejection([[1, .1], [.1, 1]], result).any()


def test_accepting_incorrect_known_authors_cannot_count_as_useful_coverage():
    known, scores, unknown, unknown_scores, authors = calibration()
    result = calibrate_gallery(known, scores[:, ::-1], unknown, unknown_scores, authors)
    assert not result["feasible"] and result["known_correct_acceptance"] == 0
    assert result["known_acceptance"] == 0


def test_calibration_ties_abstain_for_known_and_unknown_authors():
    known, scores, unknown, unknown_scores, authors = calibration()
    tied_known = calibrate_gallery(known, np.full_like(scores, .95), unknown, unknown_scores, authors)
    assert not tied_known["feasible"] and tied_known["known_acceptance"] == 0
    tied_unknown = calibrate_gallery(known, scores, unknown, np.full_like(unknown_scores, 10), authors)
    assert tied_unknown["feasible"] and tied_unknown["unknown_false_acceptance"] == 0


def test_calibration_rates_balance_authors_and_works_despite_unequal_passage_counts():
    known = ([{"author": "A", "work_id": "agood"}] * 20
             + [{"author": "A", "work_id": "abad"}, {"author": "B", "work_id": "bgood"}])
    scores = [[.9, .1]] * 20 + [[.1, .4], [.1, .9]]
    unknown = ([{"author": "U", "work_id": "uhigh"}] * 20
               + [{"author": "U", "work_id": "ulow"}, {"author": "V", "work_id": "vlow"}])
    unknown_scores = [[.8, .1]] * 20 + [[.1, .05], [.05, .1]]
    result = calibrate_gallery(known, scores, unknown, unknown_scores, ["A", "B"],
                               max_false_acceptance=.3, max_wrong_acceptance=.3, min_correct_acceptance=.7)
    assert result["feasible"]
    assert result["known_correct_acceptance"] == pytest.approx(.75)
    assert result["known_wrong_acceptance"] == pytest.approx(.25)
    assert result["unknown_false_acceptance"] == pytest.approx(.25)
    assert result["known_acceptance"] == 1


def test_repeating_passages_cannot_change_feasibility_at_the_exact_coverage_boundary():
    known, scores, unknown, unknown_scores, authors = imperfect_known()
    repeats = np.random.default_rng(0).integers(1, 40, len(known))
    duplicated = [row for row, count in zip(known, repeats) for _ in range(count)]
    duplicated_scores = [values for values, count in zip(scores, repeats) for _ in range(count)]
    result = calibrate_gallery(duplicated, duplicated_scores, unknown, unknown_scores, authors)
    assert result["feasible"] and result["known_correct_acceptance"] == .8
    assert result["threshold"] == np.nextafter(.4, np.inf)


@pytest.mark.parametrize("missing", ["known", "unknown"])
def test_incomplete_known_or_unknown_author_panels_disable_calibration(missing):
    known, scores, unknown, unknown_scores, authors = calibration()
    if missing == "known":
        known, scores = known[:1], scores[:1]
    else:
        unknown[1]["author"] = "U"
    result = calibrate_gallery(known, scores, unknown, unknown_scores, authors)
    assert result["status"] == "insufficient_calibration" and not result["feasible"]
    assert result["threshold"] is None
    assert not apply_gallery_rejection([[1, 0]], result).any()


@pytest.mark.parametrize("change", [{"author": "A"}, {"work_id": "ka"}, {"language": "hbo"}])
def test_calibration_rejects_overlapping_roles_and_languages(change):
    known, scores, unknown, unknown_scores, authors = calibration()
    known = [{**row, "language": "grc"} for row in known]
    unknown = [{**row, "language": "grc"} for row in unknown]
    unknown[0].update(change)
    with pytest.raises(ValueError):
        calibrate_gallery(known, scores, unknown, unknown_scores, authors)


@pytest.mark.parametrize("bad_scores", [np.array([[np.nan, 0]]), np.array([[np.inf, 0]]), np.ones((1, 3)), np.ones(2)])
def test_application_rejects_nonfinite_or_wrong_candidate_count_even_when_infeasible(bad_scores):
    result = calibrate_gallery(*calibration())
    result["feasible"] = False
    with pytest.raises(ValueError):
        apply_gallery_rejection(bad_scores, result)


def test_application_rejects_inconsistent_calibration_candidate_metadata():
    result = calibrate_gallery(*calibration())
    result["candidate_count"] = 3
    with pytest.raises(ValueError, match="candidate count"):
        apply_gallery_rejection([[.9, .1]], result)


def test_calibrated_threshold_transfers_to_new_names_only_with_same_gallery_size():
    result = calibrate_gallery(*calibration())
    transferred = {**result, "candidate_authors": ["new-author-C", "new-author-D"]}
    scores = [[.9, .1], [.1, .9], [.2, .1], [.9, .9]]
    np.testing.assert_array_equal(apply_gallery_rejection(scores, transferred),
                                  apply_gallery_rejection(scores, result))
    assert transferred["threshold"] == result["threshold"]
    larger = {**result, "candidate_authors": ["new-C", "new-D", "new-E"]}
    with pytest.raises(ValueError, match="candidate count"):
        apply_gallery_rejection([[.9, .1, .2]], larger)


@pytest.mark.parametrize("options", [
    {"min_correct_acceptance": 0}, {"min_correct_acceptance": float("nan")},
    {"max_wrong_acceptance": -1}, {"max_false_acceptance": 2},
])
def test_invalid_targets_do_not_allow_all_rejection_to_be_success(options):
    with pytest.raises(ValueError, match="targets"):
        calibrate_gallery(*calibration(), **options)


def test_negative_raw_scores_and_maximum_float_do_not_require_probabilities_or_infinity():
    known, scores, unknown, unknown_scores, authors = calibration()
    result = calibrate_gallery(known, scores - 2, unknown, unknown_scores - 2, authors)
    assert result["feasible"] and result["threshold"] < 0
    maximum = np.finfo(float).max
    impossible = calibrate_gallery(known, scores, unknown, [[maximum, 0], [0, maximum]], authors)
    assert not impossible["feasible"] and impossible["threshold"] is None
    json.dumps(impossible, allow_nan=False)
