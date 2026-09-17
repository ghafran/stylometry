"""The accuracy gate must be impossible to satisfy by accident.

A gate is only worth having if it fails when it should.  These tests fix the three ways this one could
quietly stop protecting anything: passing on a point estimate whose interval has not cleared the target,
skipping a metric that was never measured, and letting accuracy alone stand in for a method that also
has to say "none of these authors".
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stylometry.accuracy_gate import (
    DIRECTION,
    evaluate,
    evaluate_metric,
    load_targets,
    render,
)

TARGETS = {
    "goal": "95% author accuracy",
    "select": {"language": "grc", "sample_length": 1000},
    "metrics": {"balanced_accuracy": 0.95, "unknown_false_acceptance": 0.05, "known_acceptance": 0.95},
}


def _estimate(value: float, half_width: float = 0.01) -> dict:
    return {"value": value, "ci95": [value - half_width, value + half_width], "independent_works": 12}


def _report(**metrics) -> dict:
    return {"runs": [{"language": "grc", "sample_length": 1000, "ablation": "full_lexical",
                      "authors": 6, "works": 18, "metrics": metrics}]}


def _passing_metrics() -> dict:
    return {"balanced_accuracy": _estimate(0.97), "unknown_false_acceptance": _estimate(0.02),
            "known_acceptance": _estimate(0.97)}


# --- the decision for one metric ----------------------------------------------------------------

def test_a_minimum_passes_only_when_the_whole_interval_clears_the_target() -> None:
    assert evaluate_metric("balanced_accuracy", 0.95, _estimate(0.97)).status == "passed"
    # point estimate above the target but the interval reaches below it
    assert evaluate_metric("balanced_accuracy", 0.95, _estimate(0.96, 0.03)).status == "inconclusive"
    assert evaluate_metric("balanced_accuracy", 0.95, _estimate(0.85)).status == "failed"


def test_a_maximum_passes_only_when_the_whole_interval_is_under_the_target() -> None:
    assert evaluate_metric("unknown_false_acceptance", 0.05, _estimate(0.02)).status == "passed"
    assert evaluate_metric("unknown_false_acceptance", 0.05, _estimate(0.04, 0.03)).status == "inconclusive"
    assert evaluate_metric("unknown_false_acceptance", 0.05, _estimate(0.90)).status == "failed"


def test_an_unmeasured_metric_fails_rather_than_being_skipped() -> None:
    """Silence must not read as success."""
    assert evaluate_metric("balanced_accuracy", 0.95, None).status == "failed"
    assert evaluate_metric("balanced_accuracy", 0.95, {"value": None, "ci95": None}).status == "failed"


def test_a_measurement_without_an_interval_is_inconclusive_not_passed() -> None:
    got = evaluate_metric("balanced_accuracy", 0.95, {"value": 0.99, "ci95": None})
    assert got.status == "inconclusive" and got.note == "no interval"


def test_gap_is_negative_when_short_in_either_direction() -> None:
    low = evaluate_metric("balanced_accuracy", 0.95, _estimate(0.85))
    loose = evaluate_metric("unknown_false_acceptance", 0.05, _estimate(0.90))
    assert low.gap == pytest.approx(-0.10)
    assert loose.gap == pytest.approx(-0.85), "exceeding a maximum must also count as falling short"


# --- the whole gate ------------------------------------------------------------------------------

def test_the_gate_passes_only_when_every_metric_passes() -> None:
    assert evaluate(_report(**_passing_metrics()), TARGETS).passed


def test_high_accuracy_cannot_buy_a_pass_while_unknown_authors_are_accepted() -> None:
    """The failure the benchmark actually shows: accepting nearly every unseen author."""
    metrics = _passing_metrics()
    metrics["balanced_accuracy"] = _estimate(0.99)
    metrics["unknown_false_acceptance"] = _estimate(0.95)
    outcome = evaluate(_report(**metrics), TARGETS)
    assert not outcome.passed
    assert [o.metric for o in outcome.worst()] == ["unknown_false_acceptance"]


def test_a_missing_metric_fails_the_gate_and_is_named() -> None:
    metrics = _passing_metrics()
    del metrics["unknown_false_acceptance"]
    outcome = evaluate(_report(**metrics), TARGETS)
    assert not outcome.passed and outcome.missing == ["unknown_false_acceptance"]


def test_an_empty_benchmark_report_fails_instead_of_vacuously_passing() -> None:
    outcome = evaluate({"runs": []}, TARGETS)
    assert not outcome.passed and "no benchmark run matched" in outcome.scope


def test_the_gate_reads_the_run_the_targets_select() -> None:
    """Selecting a different language must not silently score the wrong pipeline."""
    report = {"runs": [
        {"language": "grc", "sample_length": 1000, "ablation": "full_lexical", "authors": 6, "works": 18,
         "metrics": {"balanced_accuracy": _estimate(0.80), "unknown_false_acceptance": _estimate(0.02),
                     "known_acceptance": _estimate(0.97)}},
        {"language": "hbo", "sample_length": 1000, "ablation": "full_lexical", "authors": 4, "works": 16,
         "metrics": _passing_metrics()},
    ]}
    assert not evaluate(report, TARGETS).passed, "the Greek run was selected and it falls short"
    hebrew = {**TARGETS, "select": {"language": "hbo", "sample_length": 1000}}
    assert evaluate(report, hebrew).passed


def test_selection_prefers_the_strongest_accuracy_among_matching_runs() -> None:
    report = {"runs": [
        {"language": "grc", "sample_length": 1000, "ablation": "closed_class", "authors": 6, "works": 18,
         "metrics": {"balanced_accuracy": _estimate(0.90), "unknown_false_acceptance": _estimate(0.02),
                     "known_acceptance": _estimate(0.97)}},
        {"language": "grc", "sample_length": 1000, "ablation": "full_lexical", "authors": 6, "works": 18,
         "metrics": _passing_metrics()},
    ]}
    outcome = evaluate(report, TARGETS)
    assert outcome.passed and "full_lexical" in outcome.scope


# --- the shipped targets ---------------------------------------------------------------------------

def test_the_declared_targets_ask_for_95_percent_and_gate_rejection_too() -> None:
    targets = load_targets()
    assert targets["metrics"]["balanced_accuracy"] == 0.95, "the stated goal is 95% author accuracy"
    assert "unknown_false_acceptance" in targets["metrics"], \
        "accuracy without a rejection gate would pass a system that accepts everything"
    assert set(targets["metrics"]) <= set(DIRECTION)


def test_the_declared_targets_are_not_weakened_to_the_pilot_thresholds() -> None:
    """The benchmark's own gates are looser; the declared goal must not silently drift down to them."""
    from stylometry.benchmark import PROTOCOL

    targets = load_targets()
    assert targets["metrics"]["balanced_accuracy"] > PROTOCOL["gates"]["balanced_accuracy"]["minimum"]


def test_report_names_the_measurement_and_the_shortfall() -> None:
    metrics = _passing_metrics()
    metrics["balanced_accuracy"] = _estimate(0.85)
    text = render(evaluate(_report(**metrics), TARGETS))
    assert "FAILED" in text and "grc / 1000-token passages" in text
    assert "balanced_accuracy" in text and "10.0% short of 95%" in text


def test_gate_result_is_written_where_the_next_run_can_read_it(tmp_path: Path) -> None:
    from stylometry.accuracy_gate import as_dict

    payload = as_dict(evaluate(_report(**_passing_metrics()), TARGETS))
    path = tmp_path / "accuracy_gate.json"
    path.write_text(json.dumps(payload))
    back = json.loads(path.read_text())
    assert back["passed"] is True
    assert {m["metric"] for m in back["metrics"]} == set(TARGETS["metrics"])
    assert all(m["ci95"] is not None for m in back["metrics"])


def test_an_exploratory_run_cannot_pass_however_good_the_numbers() -> None:
    """Otherwise the target could always be met by changing the protocol until it was."""
    report = _report(**_passing_metrics())
    report["exploratory_override"] = True
    outcome = evaluate(report, TARGETS)
    assert not outcome.passed and outcome.exploratory
    assert all(o.status == "passed" for o in outcome.outcomes), "the metrics themselves still stand"
    assert "exploratory" in render(outcome)


def test_a_preregistered_run_is_not_marked_exploratory() -> None:
    report = _report(**_passing_metrics())
    report["exploratory_override"] = False
    outcome = evaluate(report, TARGETS)
    assert outcome.passed and not outcome.exploratory
