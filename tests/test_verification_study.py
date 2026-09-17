"""Independent checks of source boundaries and development-study orchestration.

The synthetic scores below test experimental plumbing, not empirical authorship
accuracy. Real model behavior is exercised separately by verifier/gallery tests.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import itertools
import json
import math
import random

import numpy as np
import pytest

from stylometry import verification_study as study


@pytest.fixture
def small_protocol(monkeypatch):
    protocol = deepcopy(study.PROTOCOL)
    protocol.update(passage_tokens=2, max_samples_per_work=2)
    monkeypatch.setattr(study, "PROTOCOL", protocol)
    return protocol


def corpus(counts=None, *, split_genres=False):
    counts = counts or [4] * 4 + [6] * 4 + [2] * 4
    works = []
    for author_index, count in enumerate(counts):
        for work_index in range(count):
            number = 1000 * (author_index + 1) + work_index
            works.append({
                "author": f"Author {author_index:02d}", "language": "hbo",
                "work": f"Narrative {number}", "work_id": f"benyehuda.m{number}",
                "source_id": f"benyehuda.m{number}", "role": "attribution",
                "source_path": f"html/p{author_index + 100}/m{number}.html",
                "author_authority": f"https://wikidata.org/wiki/Q{author_index + 10000}",
                "genre": "Literary prose" if not split_genres or author_index < 6 else "Other prose",
                "topic": "Social fiction",
                "text_bare": " ".join(f"word{number}part{part} token" for part in range(4)),
                "provenance": {"source_sha256": hashlib.sha256(str(number).encode()).hexdigest()},
            })
    return works


def authors(rows):
    return {row["author"] for row in rows}


def work_ids(rows):
    return {row["work_id"] for row in rows}


def passage_ids(rows):
    return {row["id"] for row in rows}


def panel_signature(panels):
    return [{"fold": p["fold"], "genre": p["genre"], "excluded": p["excluded"],
             **{role: sorted(passage_ids(p[role])) for role in study.ROLE_KEYS}}
            for p in panels]


def test_preparation_normalizes_labels_caps_passages_and_records_exclusions(small_protocol):
    works = corpus([3, 3])
    works += [{**deepcopy(works[0]), "role": "witness_control"}]
    works += [{**deepcopy(works[1]), "work_id": "benyehuda.m999999", "text_bare": "short"}]
    samples, exclusions = study.prepare_samples(works)
    assert len(samples) == 12
    assert {r["genre"] for r in samples} == {"literary_prose"}
    assert {r["topic"] for r in samples} == {"social_fiction"}
    assert {r["n_tokens"] for r in samples} == {2}
    for work in work_ids(samples):
        ranges = sorted((r["token_start"], r["token_end"]) for r in samples if r["work_id"] == work)
        assert len(ranges) == 2
        assert ranges[0][1] <= ranges[1][0]
    assert len(exclusions) == 2
    assert any("non-attribution" in row["reason"] for row in exclusions)
    assert any("no complete passage" in row["reason"] for row in exclusions)


@pytest.mark.parametrize("identity", ["wikidata", "publisher", "cts"])
def test_preparation_rejects_author_aliases_before_splitting(small_protocol, identity):
    works = corpus([1, 1])
    for work in works:
        work.pop("author_authority")
        work.pop("source_path")
    if identity == "wikidata":
        for work in works:
            work["author_authority"] = "https://wikidata.org/wiki/Q12345"
    elif identity == "publisher":
        for index, work in enumerate(works):
            work["source_path"] = f"html/p123/m{index}.html"
    else:
        for index, work in enumerate(works):
            work["language"] = "grc"
            work["work_id"] = f"tlg1234.tlg00{index + 1}"
    with pytest.raises(ValueError, match="identity alias"):
        study.prepare_samples(works)


def test_preparation_rejects_conflicting_author_authorities(small_protocol):
    works = corpus([2])
    works[1]["author_authority"] = "https://wikidata.org/wiki/Q99999"
    with pytest.raises(ValueError, match="conflicting canonical identities"):
        study.prepare_samples(works)


@pytest.mark.parametrize("collision", ["canonical_work", "full_text", "passage"])
def test_preparation_rejects_duplicate_evidence(small_protocol, collision):
    works = corpus([2])
    if collision == "canonical_work":
        works[1]["work_id"] = works[0]["work_id"]
        works[1]["work"] = "Alternate edition with a different display title"
    elif collision == "full_text":
        works[1]["text_bare"] = " \n ".join(works[0]["text_bare"].split())
    else:
        works[0]["text_bare"] = "shared passage distinct ending"
        works[1]["text_bare"] = "shared passage another ending"
    with pytest.raises(ValueError, match="duplicate|identical normalized passages"):
        study.prepare_samples(works)


@pytest.mark.parametrize("counts,expected_folds", [
    ([4] * 4 + [6] * 4 + [2] * 4, 4),
    ([3] * 9 + [6] * 2 + [2, 2, 1], 6),
])
def test_panels_have_disjoint_authors_works_and_complete_role_coverage(small_protocol, counts, expected_folds):
    samples, _ = study.prepare_samples(corpus(counts))
    panels = study.build_panels(samples)
    assert len(panels) == expected_folds == math.ceil(sum(n >= 3 for n in counts) / 2)
    tested = set()
    weak = {f"Author {index:02d}" for index, count in enumerate(counts) if count < 3}
    for panel in panels:
        domains = [authors(panel["train"]),
                   authors(panel["calibration_gallery"] + panel["calibration_known"] + panel["calibration_unknown"]),
                   authors(panel["test_gallery"] + panel["test_known"] + panel["test_unknown"])]
        assert all(not (a & b) for a, b in itertools.combinations(domains, 2))
        assert len(domains[0]) >= 2
        assert all(len(work_ids([r for r in panel["train"] if r["author"] == author])) >= 2 for author in domains[0])
        assert all(not (work_ids(panel[a]) & work_ids(panel[b]))
                   for a, b in itertools.combinations(study.ROLE_KEYS, 2))
        for stage in ("calibration", "test"):
            gallery, known, unknown = [panel[f"{stage}_{role}"] for role in ("gallery", "known", "unknown")]
            assert len(authors(gallery)) == 2
            assert authors(known) == authors(gallery)
            assert len(authors(unknown)) == 2
            assert not authors(unknown) & authors(gallery)
            assert all(len(work_ids([r for r in gallery if r["author"] == author])) == 2 for author in authors(gallery))
        assert weak <= authors(panel["calibration_unknown"] + panel["test_unknown"])
        assert not panel["excluded"]
        assigned = [r for role in study.ROLE_KEYS for r in panel[role]]
        assert len(assigned) == len(samples)
        assert passage_ids(assigned) == passage_ids(samples)
        tested.update(authors(panel["test_gallery"]))
    assert tested == {f"Author {index:02d}" for index, count in enumerate(counts) if count >= 3}


def test_panel_assignments_are_input_order_independent(small_protocol):
    samples, _ = study.prepare_samples(corpus())
    first = panel_signature(study.build_panels(samples, seed=17))
    random.Random(982).shuffle(samples)
    assert panel_signature(study.build_panels(samples, seed=17)) == first


@pytest.mark.parametrize("invalid", ["passage_id", "work_author", "language"])
def test_invalid_sample_identity_is_rejected(small_protocol, invalid):
    samples, _ = study.prepare_samples(corpus())
    if invalid == "passage_id":
        samples.append(deepcopy(samples[0]))
    elif invalid == "work_author":
        samples[1]["author"] = "Conflicting author"
    else:
        samples[0]["language"] = "grc"
    with pytest.raises(ValueError):
        study.build_panels(samples)


def test_genre_control_requires_all_disjoint_roles_not_only_four_candidates(small_protocol):
    samples, _ = study.prepare_samples(corpus(split_genres=True))
    assert len(authors([r for r in samples if r["genre"] == "literary_prose"])) == 6
    with pytest.raises(ValueError, match="insufficient author/work coverage"):
        study.build_panels(samples, genre="literary prose")


def test_rates_balance_authors_works_and_repeated_folds_not_passage_counts():
    def row(author, work, fold, correct, *, known=True):
        return {"author": author, "work_id": work, "fold": fold, "known": known,
                "correct": correct, "accepted": correct, "correct_accepted": correct if known else False,
                "wrong_accepted": False}
    rows = [row("A", "a1", 0, True)] * 100
    rows += [row("A", "a1", 1, False), row("A", "a2", 0, False), row("B", "b1", 0, True)]
    rows += [row("C", "c1", 0, True, known=False)] * 80
    rows += [row("D", "d1", 0, False, known=False), row("D", "d2", 0, False, known=False)]
    values = study.metrics(rows)
    assert values["accuracy"]["value"] == pytest.approx(.625)
    assert values["unknown_false_acceptance"]["value"] == pytest.approx(.5)
    assert values["accuracy"]["authors"] == 2
    assert values["accuracy"]["works"] == 3
    assert all(metric["ci95"] is None for metric in values.values())
    assert study.metrics(rows * 3) == values


class FitReached(RuntimeError):
    pass


@pytest.mark.parametrize('artifact', ['model_lock.json', 'evaluation_lock.json', 'suite.json', 'development.json', 'development.md'])
def test_cannot_overwrite_another_studys_artifacts(small_protocol, monkeypatch, tmp_path, artifact):
    path = tmp_path / artifact
    path.write_text('existing frozen evidence')
    install_first_fit_probe(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match='separate directory'):
        study.run_study(corpus(), tmp_path)
    assert path.read_text() == 'existing frozen evidence'
    assert not (tmp_path / 'development_lock.json').exists()


def install_first_fit_probe(monkeypatch, output, callback=None):
    class Probe:
        def __init__(self, **kwargs):
            pass

        def fit(self, rows):
            lock_path = output / "development_lock.json"
            assert lock_path.exists(), "protocol and source hashes must precede any model fit"
            if callback:
                callback(json.loads(lock_path.read_text()), rows)
            raise FitReached
    monkeypatch.setattr(study, "PairVerifier", Probe)


def test_protocol_source_hashes_and_partitions_are_locked_before_first_fit(small_protocol, monkeypatch, tmp_path):
    output = tmp_path / "run"
    works = corpus()
    def inspect(lock, training):
        assert lock["protocol"] == study.PROTOCOL
        assert lock["independent_validation"] is False
        assert lock["scripture_attribution_validated"] is False
        assert set(lock["implementation_sha256"]) == set(study.SOURCE_FILES)
        for name, digest in lock["implementation_sha256"].items():
            assert digest == hashlib.sha256((study.ROOT / "stylometry" / name).read_bytes()).hexdigest()
        assert {item["provenance"]["source_sha256"] for item in lock["corpus"]} == {
            work["provenance"]["source_sha256"] for work in works}
        assert lock["panels"][0]["folds"][0]["roles"]["train"] == sorted(work_ids(training))
    install_first_fit_probe(monkeypatch, output, inspect)
    with pytest.raises(FitReached):
        study.run_study(works, output)
    assert not (output / "development.json").exists()


@pytest.mark.parametrize("change", ["text", "source_hash", "protocol", "implementation"])
def test_existing_lock_cannot_be_overwritten_for_changed_inputs(small_protocol, monkeypatch, tmp_path, change):
    output = tmp_path / "run"
    works = corpus()
    fake_root = tmp_path / "implementation"
    (fake_root / "stylometry").mkdir(parents=True)
    code = fake_root / "stylometry" / "fixed.py"
    code.write_text("fixed implementation version one")
    monkeypatch.setattr(study, "ROOT", fake_root)
    monkeypatch.setattr(study, "SOURCE_FILES", ("fixed.py",))
    install_first_fit_probe(monkeypatch, output)
    with pytest.raises(FitReached):
        study.run_study(works, output)
    original = (output / "development_lock.json").read_bytes()
    if change == "text":
        works[0]["text_bare"] += " changed tail"
    elif change == "source_hash":
        works[0]["provenance"]["source_sha256"] = "0" * 64
    elif change == "protocol":
        study.PROTOCOL["seed"] += 1
    else:
        code.write_text("changed implementation version two")
    with pytest.raises(ValueError, match="existing development lock differs"):
        study.run_study(works, output)
    assert (output / "development_lock.json").read_bytes() == original
    assert not (output / "development.json").exists()


@pytest.mark.parametrize("split_genres", [False, True])
def test_run_fits_only_allowed_sources_and_calibrates_before_test_scores(small_protocol, monkeypatch, tmp_path, split_genres):
    works = corpus(split_genres=split_genres)
    samples, _ = study.prepare_samples(works)
    panels = study.build_panels(samples)
    state = {"fold": -1, "pending": None, "calibrated": set()}
    events = []
    original_calibrate = study.calibrate_gallery

    def scores(rows, candidates):
        return np.array([[2.0 if row["author"] == author else -2.0 - index
                          for index, author in enumerate(candidates)] for row in rows])

    class Verifier:
        def __init__(self, **kwargs):
            pass

        def fit(self, rows):
            assert (tmp_path / "development_lock.json").exists()
            state["fold"] += 1
            state["calibrated"] = set()
            panel = panels[state["fold"]]
            assert passage_ids(rows) == passage_ids(panel["train"])
            self.metadata_ = {"kind": "synthetic orchestration probe"}
            self.gallery_calls = 0
            events.append((state["fold"], "verifier_fit"))
            return self

    def gallery(verifier, references, queries):
        stage = "calibration" if verifier.gallery_calls == 0 else "test"
        panel = panels[state["fold"]]
        assert passage_ids(references) == passage_ids(panel[f"{stage}_gallery"])
        assert passage_ids(queries) == passage_ids(panel[f"{stage}_known"] + panel[f"{stage}_unknown"])
        assert not authors(panel["train"]) & authors(references + queries)
        assert not work_ids(references) & work_ids(queries)
        if stage == "test":
            assert "pair" in state["calibrated"]
        else:
            state["pending"] = "pair"
        verifier.gallery_calls += 1
        events.append((state["fold"], f"pair_{stage}_score"))
        candidates = sorted(authors(references))
        return {"authors": candidates, "scores": scores(queries, candidates),
                "evidence": [{} for _ in queries]}

    class Baseline:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, rows):
            panel = panels[state["fold"]]
            self.stage = "calibration" if passage_ids(rows) == passage_ids(panel["calibration_gallery"]) else "test"
            assert passage_ids(rows) == passage_ids(panel[f"{self.stage}_gallery"])
            if self.stage == "test":
                assert "baseline" in state["calibrated"]
            self.authors_ = sorted(authors(rows))
            self.metadata_ = {"kind": "synthetic baseline orchestration probe"}
            return self

        def score(self, rows):
            panel = panels[state["fold"]]
            assert passage_ids(rows) == passage_ids(panel[f"{self.stage}_known"] + panel[f"{self.stage}_unknown"])
            if self.stage == "calibration":
                state["pending"] = "baseline"
            events.append((state["fold"], f"baseline_{self.stage}_score"))
            return scores(rows, self.authors_)

    def calibrate(known, known_scores, unknown, unknown_scores, candidates, **kwargs):
        panel = panels[state["fold"]]
        assert passage_ids(known) == passage_ids(panel["calibration_known"])
        assert passage_ids(unknown) == passage_ids(panel["calibration_unknown"])
        assert set(candidates) == authors(panel["calibration_gallery"])
        assert not authors(known + unknown) & authors(panel["test_gallery"] + panel["test_unknown"])
        np.testing.assert_array_equal(known_scores, scores(known, candidates))
        np.testing.assert_array_equal(unknown_scores, scores(unknown, candidates))
        state["calibrated"].add(state["pending"])
        events.append((state["fold"], f"{state['pending']}_calibrated"))
        state["pending"] = None
        return original_calibrate(known, known_scores, unknown, unknown_scores, candidates, **kwargs)

    monkeypatch.setattr(study, "PairVerifier", Verifier)
    monkeypatch.setattr(study, "AuthorModel", Baseline)
    monkeypatch.setattr(study, "build_gallery_scores", gallery)
    monkeypatch.setattr(study, "calibrate_gallery", calibrate)
    report = study.run_study(works, tmp_path)
    assert report["status"] == "development_only"
    assert not report["independent_validation"]
    assert not report["scripture_attribution_validated"]
    assert len(report["runs"]) == 1
    assert report["runs"][0]["candidate_count"] == 2
    assert state["fold"] + 1 == len(panels)
    for fold in range(len(panels)):
        sequence = [event for index, event in events if index == fold]
        assert sequence.index("pair_calibrated") < sequence.index("pair_test_score")
        assert sequence.index("baseline_calibrated") < sequence.index("baseline_test_score")
    controls = report["unavailable_controls"]
    if split_genres:
        assert {row["genre"] for row in controls} == {"literary_prose", "other_prose"}
        assert all("insufficient author/work coverage" in row["reason"] for row in controls)
    else:
        assert len(controls) == 1
        assert "identical to main panel" in controls[0]["reason"]
    assert (tmp_path / "development.json").exists()
    assert "Development only" in (tmp_path / "development.md").read_text()
