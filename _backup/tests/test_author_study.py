"""Study isolation and lock tests; fixture performance is not author evidence."""
import copy
import json

import numpy as np
import pytest

from stylometry import author_study as study
from stylometry import benchmark_data
from stylometry.author_models import AuthorModel
from stylometry.features import LexicalFeatureExtractor


def corpus(author_count=5, works_per_author=3, prefix="dev", role=None):
    works = []
    for author in range(author_count):
        for work in range(works_per_author):
            item = {
                "author": f"{prefix}-author-{author}", "work": f"Title {work}",
                "work_id": f"{prefix}-{author}-{work}", "language": "grc",
                "genre": "oratory" if author < 2 else "dialogue", "topic": "ethics",
                "text_bare": f"και δε λογος και {prefix}{author}{work} λογος",
                "source_id": f"{prefix}-source-{author}-{work}",
            }
            if role:
                item["evaluation_role"] = role
            works.append(item)
    return works


def fresh_corpus(reference_works=3):
    return (corpus(2, reference_works, "reference", "reference")
            + corpus(2, 3, "impostor", "calibration_unknown")
            + corpus(2, 3, "unknown", "test_unknown"))


class SpyModel:
    instances = []

    def __init__(self, kind, seed=42):
        self.kind, self.seed = kind, seed
        self.score_calls = []
        self.instances.append(self)

    def fit(self, rows):
        self.train = copy.deepcopy(rows)
        self.authors_ = sorted({r["author"] for r in rows})
        self.metadata_ = {"kind": self.kind}
        return self

    def score(self, rows):
        self.score_calls.append(copy.deepcopy(rows))
        scores = np.full((len(rows), len(self.authors_)), .05)
        for index, row in enumerate(rows):
            if row["author"] in self.authors_:
                scores[index, self.authors_.index(row["author"])] = .9
            else:
                scores[index, 0] = .2
        return scores


@pytest.fixture
def light_study(monkeypatch):
    protocol = copy.deepcopy(study.PROTOCOL)
    protocol.update(models=["function_centroid"], lengths=[2], max_samples_per_work=2,
                    final_refit_replicates=3)
    monkeypatch.setattr(study, "PROTOCOL", protocol)
    SpyModel.instances = []
    monkeypatch.setattr(study, "AuthorModel", SpyModel)


@pytest.fixture
def locked_panel(tmp_path, monkeypatch, light_study):
    # The lock uses the actual implementation hashes saved at fixture creation.
    # Parsing/loading is isolated here: source-parser integrity has separate tests.
    works = fresh_corpus()
    manifest = tmp_path / "reserved.json"
    manifest.write_text(json.dumps({"works": works}))
    monkeypatch.setattr(benchmark_data, "read_manifest", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(benchmark_data, "load_benchmark", lambda path, *a, **kw: copy.deepcopy(json.loads(path.read_text())["works"]))
    frozen = {
        "protocol": copy.deepcopy(study.PROTOCOL),
        "selected": {"grc": {"model": "function_centroid", "length": 2}},
        "development_corpus": study._corpus_identity(corpus()),
        "implementation_sha256": study.implementation_hashes(),
        "runtime": {p: study.importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn")},
        "authorship_validated": False,
    }
    model_lock = tmp_path / "model_lock.json"
    model_lock.write_text(json.dumps(frozen))
    study.seal(model_lock, [manifest], tmp_path)
    return {"directory": tmp_path, "manifest": manifest, "model_lock": model_lock,
            "evaluation_lock": tmp_path / "evaluation_lock.json", "works": works}


def test_development_same_genre_models_fit_only_the_matching_corpus(light_study, tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_data, "load_benchmark", lambda *a, **kw: pytest.fail("development must not load reserved texts"))
    report = study.develop(corpus(), tmp_path)
    control = next(c for c in report["runs"][0]["controls"] if c["label"] == "genre" and c["value"] == "oratory")
    assert control["available"] and control["authors"] == 2
    restricted = [m for m in SpyModel.instances if {r["author"] for r in m.train} == {"dev-author-0", "dev-author-1"}]
    assert restricted
    assert all({r["genre"] for r in m.train} == {"oratory"} for m in restricted)
    assert (tmp_path / "model_lock.json").exists()


def test_development_lock_rejects_changed_corpus_in_same_study_directory(light_study, tmp_path):
    works = corpus()
    study.develop(works, tmp_path)
    before = (tmp_path / "model_lock.json").read_bytes()
    changed = copy.deepcopy(works)
    changed[0]["text_bare"] += " επιπλεον"
    with pytest.raises(ValueError, match="existing model lock differs"):
        study.develop(changed, tmp_path)
    assert (tmp_path / "model_lock.json").read_bytes() == before


def test_model_selection_does_not_treat_rejecting_everything_as_feasible():
    def candidate(kind, accuracy, acceptance):
        return {"model": kind, "length": 1000, "closed": {"available": True, "accuracy": {"value": accuracy}},
                "controls": [], "open": {"available": True, "metrics": {
                    "unknown_false_acceptance": {"value": 0}, "known_acceptance": {"value": acceptance},
                    "known_correct_acceptance": {"value": acceptance}, "known_wrong_acceptance": {"value": 0}}}}
    zero_coverage = candidate("function_centroid", 1.0, 0)
    useful = candidate("morph_centroid", .85, .9)
    selected = study.select_candidate([zero_coverage, useful])
    assert selected["model"] == "morph_centroid"
    assert selected["development_operating_targets_met"]
    assert not study.select_candidate([zero_coverage])["development_operating_targets_met"]


def test_open_development_reserves_two_impostors_and_a_separate_test_author(light_study):
    result = study._open_development(study.sample_works(corpus(), 2, 2), "function_centroid", 42)
    assert result["available"]
    assert len(result["scenarios"]) == len(SpyModel.instances) == 15
    for scenario, model in zip(result["scenarios"], SpyModel.instances):
        unknown = scenario["unknown_author"]
        impostors = set(scenario["impostor_authors"])
        assert len(impostors) == 2
        assert unknown not in impostors | set(model.authors_)
        assert not impostors & set(model.authors_)
        known_cal, impostor_cal, test = model.score_calls
        assert unknown not in {r["author"] for r in model.train + known_cal + impostor_cal}
        assert {r["author"] for r in impostor_cal} == impostors
        assert not impostors & {r["author"] for r in test}
        work_roles = [{r["work_id"] for r in rows} for rows in (model.train, known_cal, impostor_cal, test)]
        assert all(not first & second for index, first in enumerate(work_roles) for second in work_roles[index + 1:])


def test_open_development_cannot_use_only_one_calibration_impostor(light_study):
    result = study._open_development(study.sample_works(corpus(4), 2), "function_centroid", 42)
    assert not result["available"]
    assert not SpyModel.instances


def test_final_partitions_preserve_canonical_work_and_author_roles(light_study):
    parts, excluded = study._final_partitions(fresh_corpus(), 2)
    assert not excluded
    role_works = [{r["work_id"] for r in rows} for rows in parts.values()]
    assert all(not first & second for index, first in enumerate(role_works) for second in role_works[index + 1:])
    references = {r["author"] for r in parts["train"]}
    assert references == {r["author"] for r in parts["known_calibration"]} == {r["author"] for r in parts["known_test"]}
    impostors, unknown = ({r["author"] for r in parts[key]} for key in ("impostors", "unknown_test"))
    assert len(impostors) == len(unknown) == 2
    assert not references & (impostors | unknown) and not impostors & unknown
    study._fit_final(parts, "function_centroid", 42)
    model = SpyModel.instances[-1]
    assert not unknown & {r["author"] for r in model.train + model.score_calls[0] + model.score_calls[1]}


@pytest.mark.parametrize("mutation,match", [
    ("model", "model or implementation changed"),
    ("implementation", "model or implementation changed"),
    ("manifest", "manifest changed"),
])
def test_evaluation_rejects_changes_after_sealing_before_loading_texts(locked_panel, monkeypatch, mutation, match):
    panel = locked_panel
    if mutation == "model":
        lock = json.loads(panel["evaluation_lock"].read_text())
        lock["model_lock"]["selected"]["grc"]["length"] = 1
        panel["evaluation_lock"].write_text(json.dumps(lock))
    elif mutation == "implementation":
        changed = study.implementation_hashes()
        changed["author_models.py"] = "0" * 64
        monkeypatch.setattr(study, "implementation_hashes", lambda: changed)
    else:
        panel["manifest"].write_text(panel["manifest"].read_text() + "\n")
    monkeypatch.setattr(benchmark_data, "load_benchmark", lambda *a, **kw: pytest.fail("lock failure must precede text loading"))
    with pytest.raises(ValueError, match=match):
        study.evaluate(panel["evaluation_lock"], panel["directory"] / "cache", panel["directory"] / "result")


@pytest.mark.parametrize("mutation,match", [
    ("author_overlap", "overlaps development"), ("work_overlap", "overlaps development"),
    ("cross_role", "cannot cross"), ("canonical_duplicate", "duplicate canonical"),
    ("one_impostor", "at least two authors"),
])
def test_seal_rejects_invalid_fresh_roles_and_development_overlap(locked_panel, mutation, match):
    panel = locked_panel
    works = copy.deepcopy(panel["works"])
    if mutation == "author_overlap":
        works[0]["author"] = "dev-author-0"
    elif mutation == "work_overlap":
        works[0]["work_id"] = "dev-0-0"
    elif mutation == "cross_role":
        works[1]["evaluation_role"] = "test_unknown"
    elif mutation == "canonical_duplicate":
        works.append({**works[0], "work": "Alternate edition title"})
    else:
        works = [w for w in works if w["author"] != "impostor-author-1"]
    panel["manifest"].write_text(json.dumps({"works": works}))
    with pytest.raises(ValueError, match=match):
        study.seal(panel["model_lock"], [panel["manifest"]], panel["directory"] / "second")


def test_fresh_duplicate_development_text_cannot_enter_evaluation(locked_panel):
    panel = locked_panel
    works = copy.deepcopy(panel["works"])
    works[0]["text_bare"] = corpus()[0]["text_bare"]
    panel["manifest"].write_text(json.dumps({"works": works}))
    second = panel["directory"] / "second"
    study.seal(panel["model_lock"], [panel["manifest"]], second)
    with pytest.raises(ValueError, match="duplicate development"):
        study.evaluate(second / "evaluation_lock.json", panel["directory"] / "cache", second)


def test_whole_work_uncertainty_refits_feature_extraction_and_calibration(light_study, monkeypatch):
    parts, _ = study._final_partitions(fresh_corpus(reference_works=6), 2)
    monkeypatch.setattr(study, "AuthorModel", AuthorModel)
    rows, _, _ = study._fit_final(parts, "function_centroid", 42)
    fits, calibrations = [], []
    original_fit, original_calibration = LexicalFeatureExtractor.fit, study.calibrate_rejection

    def record_fit(self, train):
        fits.append(copy.deepcopy(train))
        return original_fit(self, train)

    def record_calibration(*args, **kwargs):
        calibrations.append(copy.deepcopy(args[0]))
        return original_calibration(*args, **kwargs)

    monkeypatch.setattr(LexicalFeatureExtractor, "fit", record_fit)
    monkeypatch.setattr(study, "calibrate_rejection", record_calibration)
    metrics = study._refit_intervals(parts, "function_centroid", study._metrics(rows), repeats=3, seed=42)
    assert len(fits) == len(calibrations) == 3
    allowed_train_text = {r["text_bare"] for r in parts["train"]}
    allowed_cal_text = {r["text_bare"] for r in parts["known_calibration"]}
    assert all({r["text_bare"] for r in fit} <= allowed_train_text for fit in fits)
    assert all({r["text_bare"] for r in cal} <= allowed_cal_text for cal in calibrations)
    assert all(r["work_id"].startswith("bootstrap:") for fit in fits for r in fit)
    assert all(metric["refit_replicates"] == 3 for metric in metrics.values())


def test_rejecting_everything_keeps_selective_accuracy_unavailable(light_study):
    samples = study.sample_works(corpus(2), 2, 1)
    model = SpyModel("function_centroid").fit(samples)
    rows = study._rows(samples, model.score(samples), model.authors_, np.zeros(len(samples), dtype=bool))
    metrics = study._metrics(rows)
    assert metrics["known_acceptance"]["value"] == 0
    assert metrics["known_correct_acceptance"]["value"] == 0
    assert metrics["accuracy_among_accepted"]["value"] is None


def test_successful_fixture_evaluation_remains_small_pilot_not_broad_pass(locked_panel):
    panel = locked_panel
    report = study.evaluate(panel["evaluation_lock"], panel["directory"] / "cache", panel["directory"] / "result")
    assert report["status"] != "passed"
    assert not report["scripture_attribution_validated"]
    assert not report["runs"][0]["adequate_coverage"]
    assert report["runs"][0]["metrics"]["unknown_false_acceptance"]["value"] == 0
    assert report["runs"][0]["metrics"]["known_acceptance"]["value"] == 1


def test_extraction_implementation_and_control_labels_are_part_of_the_lock():
    assert {"benchmark_data.py", "benchmark_additional.py"} <= set(study.SOURCE_FILES)
    works = corpus()
    before = study._corpus_identity(works)
    changed = copy.deepcopy(works)
    changed[0]["genre"] = "another genre"
    changed[0]["topic"] = "another topic"
    assert study._corpus_identity(changed) != before


def test_empty_reserved_panel_is_rejected(locked_panel):
    panel = locked_panel
    panel["manifest"].write_text(json.dumps({"works": []}))
    with pytest.raises(ValueError, match="no attribution works"):
        study.seal(panel["model_lock"], [panel["manifest"]], panel["directory"] / "empty")


def test_runtime_change_after_sealing_is_rejected_before_text_loading(locked_panel, monkeypatch):
    panel = locked_panel
    original = study.importlib.metadata.version
    monkeypatch.setattr(study.importlib.metadata, "version", lambda package: "changed" if package == "numpy" else original(package))
    monkeypatch.setattr(benchmark_data, "load_benchmark", lambda *a, **kw: pytest.fail("runtime guard must precede loading"))
    with pytest.raises(ValueError, match="runtime changed"):
        study.evaluate(panel["evaluation_lock"], panel["directory"] / "cache", panel["directory"] / "result")


def test_singleton_role_strata_cannot_produce_pass_supporting_intervals(light_study):
    parts, _ = study._final_partitions(fresh_corpus(), 2)
    rows, _, _ = study._fit_final(parts, "function_centroid", 42)
    metrics = study._refit_intervals(parts, "function_centroid", study._metrics(rows), repeats=3, seed=42)
    assert all(metric["ci95"] is None for metric in metrics.values())
    assert all(metric["singleton_strata"] for metric in metrics.values())
    assert metrics["accuracy"]["resampling_range95"] is not None


def test_unfamiliar_author_resampling_keeps_each_authors_works_together(light_study):
    rows = study.sample_works(corpus(2, 3), 2, 1)
    sampled = study._resample_works(rows, np.random.default_rng(42), resample_authors=True)
    assert len({r["author"] for r in sampled}) == 2
    assert all(r["author"].startswith("bootstrap-author:") for r in sampled)
    for author in {r["author"] for r in sampled}:
        assert len({r["work_id"] for r in sampled if r["author"] == author}) == 3


def test_final_same_genre_diagnostics_refit_restricted_reference_models(locked_panel):
    panel = locked_panel
    works = (corpus(4, 3, "reference", "reference")
             + [w for w in panel["works"] if w["evaluation_role"] != "reference"])
    panel["manifest"].write_text(json.dumps({"works": works}))
    second = panel["directory"] / "four-reference-authors"
    study.seal(panel["model_lock"], [panel["manifest"]], second)
    study.evaluate(second / "evaluation_lock.json", panel["directory"] / "cache", second)
    # Main fit + two disjoint genre controls + topic control all refit the model.
    non_bootstrap_models = [m for m in SpyModel.instances if not m.train[0]["work_id"].startswith("bootstrap:")]
    assert len(non_bootstrap_models) >= 4
    oratory = [m for m in non_bootstrap_models if set(m.authors_) == {"reference-author-0", "reference-author-1"}]
    dialogue = [m for m in non_bootstrap_models if set(m.authors_) == {"reference-author-2", "reference-author-3"}]
    assert oratory and dialogue
    assert all({r["genre"] for r in m.train} == {"oratory"} for m in oratory)
    assert all({r["genre"] for r in m.train} == {"dialogue"} for m in dialogue)


def test_many_works_from_two_unknown_authors_cannot_certify_low_population_risk(light_study):
    works = (corpus(2, 6, "reference", "reference")
             + corpus(2, 3, "impostor", "calibration_unknown")
             + corpus(2, 40, "unknown", "test_unknown"))
    parts, _ = study._final_partitions(works, 2)
    rows, _, _ = study._fit_final(parts, "function_centroid", 42)
    point = study._metrics(rows)
    assert point["unknown_false_acceptance"]["value"] == 0
    assert point["unknown_false_acceptance"]["independent_authors"] == 2
    assert point["unknown_false_acceptance"]["independent_works"] == 80
    metrics = study._refit_intervals(parts, "function_centroid", point, repeats=3, seed=42)
    uncertainty = metrics["unknown_false_acceptance"]
    assert not uncertainty["singleton_strata"]
    assert uncertainty["ci95"] is not None
    assert uncertainty["ci95"][1] > .8
