import json

import numpy as np
import pytest

from stylometry.rejection import apply_rejection, calibrate_rejection


def row(author, work):
    return {"author": author, "work_id": work}


def calibrate(known_scores=None, impostor_scores=None, **kwargs):
    return calibrate_rejection(
        [row("a", "a1"), row("b", "b1")],
        [[.9, .1], [.1, .9]] if known_scores is None else known_scores,
        [row("c", "c1"), row("c", "c2")],
        [[.6, .2], [.2, .7]] if impostor_scores is None else impostor_scores,
        ["a", "b"], **kwargs)


def test_impostor_maximum_calibration_rejects_unknowns_and_retains_knowns():
    result = calibrate()
    assert result["status"] == "development_only"
    assert result["threshold"] == np.nextafter(.7, np.inf)
    assert result["unknown_false_acceptance"] == 0
    assert result["known_acceptance"] == result["known_correct_acceptance"] == 1
    assert result["known_wrong_acceptance"] == 0
    assert apply_rejection([[.69, .2], [.81, .2]], result).tolist() == [False, True]
    json.dumps(result, allow_nan=False)


def test_complete_impostor_score_ties_are_not_fractionally_accepted():
    result = calibrate(impostor_scores=[[.7, .2], [.2, .7]], max_false_acceptance=.5)
    assert result["threshold"] > .7
    assert result["unknown_false_acceptance"] == 0


def test_exact_top_candidate_ties_always_abstain():
    result = calibrate(impostor_scores=[[.95, .95], [.95, .95]])
    assert result["feasible"]
    assert result["unknown_false_acceptance"] == 0
    assert apply_rejection([[1., 1.], [.95, .94]], result).tolist() == [False, True]
    known_tied = calibrate(known_scores=[[.9, .9], [.1, .9]])
    assert known_tied["known_acceptance"] == .5
    assert not known_tied["feasible"]


def test_calibration_balances_authors_works_and_sample_counts():
    known = [row("a", "a1"), row("b", "b1")]
    # The many low-scoring chunks in c1 cannot dilute the high-scoring c2 work,
    # and c's two works cannot dilute the second impostor author's single work.
    impostors = [row("c", "c1")] * 100 + [row("c", "c2"), row("d", "d1")]
    scores = [[.2, .1]] * 100 + [[.8, .1], [.3, .1]]
    result = calibrate_rejection(known, [[.9, .1], [.1, .9]], impostors, scores,
                                ["a", "b"], max_false_acceptance=.25)
    assert result["threshold"] == np.nextafter(.3, np.inf)
    assert result["unknown_false_acceptance"] == .25
    assert result["calibration_counts"]["impostor"] == {"authors": 2, "works": 3, "samples": 102}


def test_known_acceptance_is_weighted_by_author_and_work():
    known = [row("a", "a1")] * 20 + [row("a", "a2"), row("b", "b1")]
    scores = [[.9, .1]] * 20 + [[.1, .9], [.1, .6]]
    result = calibrate_rejection(known, scores, [row("c", "c1")], [[.7, .1]],
                                ["a", "b"], min_known_acceptance=.5)
    assert result["known_acceptance"] == .5
    assert result["known_correct_acceptance"] == pytest.approx(.25)
    assert result["known_wrong_acceptance"] == pytest.approx(.25)


def test_no_feasible_threshold_causes_all_application_decisions_to_abstain():
    result = calibrate(known_scores=[[.6, .1], [.1, .6]])
    assert result["status"] == "no_feasible_threshold"
    assert result["known_acceptance"] == 0
    assert result["unknown_false_acceptance"] == 0
    assert not apply_rejection([[.99, .1]], result).any()


def test_missing_impostors_and_candidate_coverage_are_insufficient():
    result = calibrate_rejection([row("a", "a1"), row("b", "b1")],
                                [[.9, .1], [.1, .9]], [], np.empty((0, 2)), ["a", "b"])
    assert result["status"] == "insufficient_calibration"
    assert result["threshold"] is None
    assert not apply_rejection([[.99, .1]], result).any()
    result = calibrate_rejection([row("a", "a1")], [[.9, .1]],
                                [row("c", "c1")], [[.3, .1]], ["a", "b"])
    assert result["status"] == "insufficient_calibration"


@pytest.mark.parametrize("known,impostors,authors,match", [
    ([row("a", "same")], [row("c", "same")], ["a", "b"], "disjoint"),
    ([row("c", "c1")], [row("d", "d1")], ["a", "b"], "known calibration"),
    ([row("a", "a1")], [row("a", "a2")], ["a", "b"], "impostor authors"),
    ([{"author": "a"}], [row("c", "c1")], ["a", "b"], "work_id"),
    ([row("a", "a1")], [row("c", "c1")], ["a", "a"], "distinct"),
])
def test_invalid_role_or_identity_metadata_is_rejected(known, impostors, authors, match):
    with pytest.raises(ValueError, match=match):
        calibrate_rejection(known, [[.9, .1]], impostors, [[.3, .1]], authors)


@pytest.mark.parametrize("scores", [[[np.nan, .1], [.1, .9]], [[np.inf, .1], [.1, .9]],
                                    [[.9], [.9]], [[.9, .1]]])
def test_invalid_score_matrices_are_rejected(scores):
    with pytest.raises(ValueError):
        calibrate(known_scores=scores)


def test_application_supports_empty_batches_and_rejects_invalid_scores():
    result = calibrate()
    assert apply_rejection(np.empty((0, 2)), result).shape == (0,)
    with pytest.raises(ValueError, match="finite"):
        apply_rejection([[np.nan, .1]], result)


def test_input_order_and_duplicate_chunk_counts_do_not_change_threshold():
    known, impostors = [row("a", "a1"), row("b", "b1")], [row("c", "c1"), row("c", "c2")]
    first = calibrate_rejection(known, [[.9, .1], [.1, .9]], impostors, [[.6, .1], [.7, .1]], ["a", "b"])
    second = calibrate_rejection(known[::-1], [[.1, .9], [.9, .1]],
                                 [impostors[1]] * 3 + [impostors[0]],
                                 [[.7, .1]] * 3 + [[.6, .1]], ["a", "b"])
    assert first["threshold"] == second["threshold"]
    assert first["known_acceptance"] == second["known_acceptance"]


def test_nonserializable_threshold_is_never_emitted():
    result = calibrate(impostor_scores=[[np.finfo(float).max, .1]] * 2)
    assert result["status"] == "no_feasible_threshold"
    assert result["threshold"] is None
    json.dumps(result, allow_nan=False)
