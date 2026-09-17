"""Safeguards for the real-reference benchmark (fixtures do not prove authorship)."""
import json

import numpy as np
import pytest

from stylometry import benchmark as bm


def reference_works(authors=3, works=3, tokens=180):
    styles = ["και τε και λογος", "δε γαρ δε σοφια", "ουν αλλα ουν πολις", "το του της ουτος"]
    result = []
    for author in range(authors):
        for work in range(works):
            text = (styles[author % len(styles)].split() * tokens)[:tokens - 1] + [f"ιδιον{author}{work}"]
            result.append({"author": f"author{author}", "work": f"work{work}", "language": "grc",
                           "genre": "prose", "topic": "ethics", "source_id": f"source{author}{work}",
                           "text_bare": " ".join(text), "n_tokens": len(text)})
    return result


def test_protocol_gates_are_fixed_and_conservative():
    assert bm.PROTOCOL["gates"]["verification_fpr"] == {"maximum": .05}
    assert bm.PROTOCOL["gates"]["balanced_accuracy"] == {"minimum": .80}
    assert bm.PROTOCOL["sample_lengths"] == [100, 500, 1000]
    assert bm.PROTOCOL["minimum_authors_for_pass"] >= 8


def test_sampling_is_nonoverlapping_uniform_and_exact_length():
    work = reference_works(1, 1, 1000)
    rows = bm.sample_works(work, 100, 4)
    assert [r["token_start"] for r in rows] == [0, 300, 600, 900]
    assert all(len(r["text_bare"].split()) == 100 for r in rows)
    assert all(a["token_end"] <= b["token_start"] for a, b in zip(rows, rows[1:]))
    assert all("author" not in r["id"] and "work" not in r["id"] for r in rows)


@pytest.mark.parametrize("length, maximum", [(0, 4), (-1, 4), (100, 0)])
def test_invalid_sampling_settings_rejected(length, maximum):
    with pytest.raises(ValueError):
        bm.sample_works(reference_works(), length, maximum)


def test_duplicate_corpus_items_cannot_create_fake_independent_works():
    works = reference_works()
    with pytest.raises(ValueError, match="duplicate work identity"):
        bm.sample_works(works + [works[0]], 100)
    alias = {**works[0], "author": "other", "work": "alias"}
    with pytest.raises(ValueError, match="duplicate reference texts"):
        bm.sample_works(works + [alias], 100)


def test_each_work_tested_once_never_crosses_roles():
    samples = bm.sample_works(reference_works(3, 4), 50, 3)
    eligible, partitions, excluded = bm.work_partitions(samples)
    assert not excluded
    observed = []
    for fold in range(3):
        roles = [{s["work_id"] for s in eligible if partitions[s["work_id"]] == (fold + role) % 3}
                 for role in range(3)]
        assert all(not roles[a] & roles[b] for a in range(3) for b in range(a))
        assert all({s["author"] for s in eligible if s["work_id"] in role} == {"author0", "author1", "author2"} for role in roles)
        observed.extend(roles[2])
    assert len(observed) == len(set(observed)) == 12


def test_two_works_cannot_fake_three_independent_partitions():
    samples = bm.sample_works(reference_works(2, 2), 50)
    eligible, assignment, excluded = bm.work_partitions(samples)
    assert eligible == [] and assignment == {}
    assert len(excluded) == 2


def test_calibration_thresholds_handle_ties_without_false_matches():
    rows = [{"author": "a"}] * 10 + [{"author": "b"}] * 10
    scores = np.array([[.8, .2]] * 10 + [[.2, .7]] * 10)
    verification, rejection = bm.calibrate_thresholds(rows, scores, ["a", "b"])
    assert verification > .2
    assert rejection == .7
    assert not np.any(np.array([.2] * 20) >= verification)


def test_entire_work_counts_control_uncertainty_not_chunk_count():
    sparse = [{"work_id": f"w{i}", "author": "a", "error": 0.0} for i in range(4)]
    dense = [row.copy() for row in sparse for _ in range(500)]
    a, b = bm.rate_estimate(sparse, "error"), bm.rate_estimate(dense, "error")
    assert a == b
    assert a["independent_works"] == 4
    assert a["ci95"][1] > .05
    assert bm.gate_result(a, {"maximum": .05}) == "inconclusive"


def test_gate_requires_interval_to_clear_target():
    assert bm.gate_result({"ci95": [.81, .98]}, {"minimum": .8}) == "passed"
    assert bm.gate_result({"ci95": [.61, .79]}, {"minimum": .8}) == "failed"
    assert bm.gate_result({"ci95": [.79, .98]}, {"minimum": .8}) == "inconclusive"
    assert bm.gate_result({"ci95": None}, {"minimum": .8}) == "inconclusive"


def test_work_balanced_scores_do_not_reward_replicating_easy_chunks():
    rows = [{"work_id": "easy", "author": "a", "correct": 1.0},
            {"work_id": "hard", "author": "a", "correct": 0.0}]
    repeated = [rows[0]] * 1000 + [rows[1]]
    assert bm.rate_estimate(rows, "correct")["value"] == .5
    assert bm.rate_estimate(repeated, "correct")["value"] == .5


def test_unknown_author_never_enters_fit_or_calibration(monkeypatch, tmp_path):
    calls = []
    original = bm._fit_scores

    def spy(train, cal, test, ablation, seed):
        assert not {r["work_id"] for r in train} & {r["work_id"] for r in cal}
        assert not {r["work_id"] for r in test} & {r["work_id"] for r in train + cal}
        calls.append(({r["author"] for r in train}, {r["author"] for r in cal}, {r["author"] for r in test}))
        return original(train, cal, test, ablation, seed)

    monkeypatch.setattr(bm, "_fit_scores", spy)
    progress = []
    works = [{**work, "work_id": f"canonical-{i}"} for i, work in enumerate(reference_works())]
    report = bm.run_benchmark(works, tmp_path, sample_lengths=(50,), max_samples_per_work=2, bootstrap_replicates=10, progress=progress.append)
    assert report["exclusions"] == []
    assert len(progress) == 4
    assert progress[0].startswith("Evaluating grc, 50 tokens")
    assert progress[-1].startswith("Completed grc, 50 tokens")
    assert report["status"] != "passed"
    assert report["exploratory_override"]
    assert all(train == cal for train, cal, test in calls)
    open_calls = [(train, test) for train, cal, test in calls if len(train) < len(test)]
    assert len(open_calls) == 18
    assert all(len(test - train) == 1 for train, test in open_calls)
    assert json.loads((tmp_path / "benchmark.json").read_text())["status"] == report["status"]
    assert "does not establish scripture authorship" in (tmp_path / "report.md").read_text()
    for run in report["runs"]:
        assert run["works"] == 9
        assert run["metrics"]["verification_auroc"]["ci95"] is None
        assert not run["adequate_coverage_for_pass"]
        assert run["metrics"]["balanced_accuracy"]["value"] >= .9
        unknown = [r for r in run["open_set_predictions"] if not r["known"]]
        assert len({r["work_id"] for r in unknown}) == 9
        assert all(r["heldout_author"] == r["author"] for r in unknown)
        assert run["subsets"]["shared_genre"][0]["value"] == "prose"
        assert run["subsets"]["same_author_cross_genre"]["status"] == "unsupported"


def test_insufficient_data_report_explains_missing_evidence(tmp_path):
    report = bm.run_benchmark(reference_works(1, 2, 80), tmp_path, bootstrap_replicates=10)
    assert report["status"] == "inconclusive"
    assert all(run["status"] == "inconclusive" for run in report["runs"])
    assert any(e["reason"] == "fewer tokens than one complete sample" for e in report["exclusions"])


@pytest.mark.parametrize("changed", [{"work": "translated title"}, {"author": "alternate attribution"}, {"work": "other edition", "author": "other author"}])
def test_canonical_work_cannot_be_repeated_under_new_title_or_author(changed):
    work = {**reference_works(1, 1)[0], "work_id": "canonical-001"}
    alternate = {**work, **changed, "text_bare": work["text_bare"] + " προσθηκη"}
    with pytest.raises(ValueError, match="duplicate work identity: grc:canonical-001"):
        bm.sample_works([work, alternate], 100)


def test_same_display_title_can_represent_distinct_canonical_works():
    works = reference_works(1, 2)
    for i, work in enumerate(works):
        work.update({"work": "On the same subject", "work_id": f"canonical-{i}"})
    samples = bm.sample_works(works, 100)
    assert {row["work_id"] for row in samples} == {"grc:canonical-0", "grc:canonical-1"}
    assert {row["work"] for row in samples} == {"On the same subject"}


def test_canonical_partitions_ignore_display_title_changes():
    works = reference_works(2, 3)
    for i, work in enumerate(works):
        work["work_id"] = f"canonical-{i}"
    first = bm.work_partitions(bm.sample_works(works, 100))[1]
    renamed = [{**work, "work": "Retitled " + work["work"]} for work in works]
    second = bm.work_partitions(bm.sample_works(renamed, 100))[1]
    assert first == second
    assert len(first) == 6


@pytest.mark.parametrize("invalid", [None, "", " ", 123])
def test_invalid_supplied_canonical_ids_do_not_silently_fall_back(invalid):
    work = {**reference_works(1, 1)[0], "work_id": invalid}
    with pytest.raises(ValueError, match="nonempty canonical identifier"):
        bm.sample_works([work], 100)


def test_canonical_identity_is_namespaced_by_language():
    works = reference_works(1, 2)
    works[0].update({"work_id": "canonical-1", "language": "grc"})
    works[1].update({"work_id": "canonical-1", "language": "hbo"})
    samples = bm.sample_works(works, 100)
    assert {row["work_id"] for row in samples} == {"grc:canonical-1", "hbo:canonical-1"}
