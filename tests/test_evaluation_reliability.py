"""Controls for passage-level evaluation, including an intentionally misleading batch signal."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stylometry.ai.profile import CATEGORICAL_DIMS, NUMERIC_DIMS, PROMPT_VERSION, SCHEMA_VERSION, validate_profiles_for_corpus
from stylometry.ai.compare import (
    ModelSet,
    _fit_tag_vectorizer,
    _grouped_folds,
    _profile_features,
    build,
    discrimination,
    ensemble,
    evaluation_groups,
)


def sample(*, signal=False, n_groups=6, n_verses=12):
    profiles, work_of, chapter_of, verses = {}, {}, {}, []
    rng = np.random.default_rng(42)
    for work_idx, work in enumerate(("A", "B")):
        for chapter in range(n_groups):
            request = f"request-{work}-{chapter}"
            for number in range(n_verses):
                ident = f"grc:{work}.{chapter}.{number}"
                value = float(0.15 + work_idx * 0.7 + rng.uniform(-0.05, 0.05)) if signal else 0.5
                profiles[ident] = {
                    "id": ident,
                    **{d: value for d in NUMERIC_DIMS},
                    **{d: values[0] for d, values in CATEGORICAL_DIMS.items()},
                    "style_tags": [request.lower().replace("-", "_")],
                    "distinctive_phrases": [], "signature": "A synthetic test profile.",
                    "provenance": {
                        "request_group": hashlib.sha256(request.encode()).hexdigest(),
                        "text_sha256": hashlib.sha256(ident.encode()).hexdigest(),
                        "prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION, "generation_settings": {},
                    },
                }
                work_of[ident] = work
                chapter_of[ident] = ("grc", work, str(chapter))
                verses.append({"id": ident, "language": "grc", "work": work, "chapter": str(chapter), "text": ident})
    return profiles, work_of, chapter_of, verses


def test_batch_artifact_looks_predictive_only_with_leaky_verse_splits():
    profiles, works, chapters, _ = sample()
    ids = list(profiles)
    # Demonstrate why the former verse-level CV is misleading: the same arbitrary
    # request tag appears in both training and testing, with no general style signal.
    vocabulary = _fit_tag_vectorizer(profiles, ids)
    X = _profile_features(profiles, ids, vocabulary)
    leaky_scores = cross_val_score(
        make_pipeline(StandardScaler(), LogisticRegression(C=0.5)), X, [works[i] for i in ids],
        cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="accuracy",
    )
    assert leaky_scores.mean() > 0.95
    result = discrimination(profiles, works, ids, chapter_of=chapters)
    assert result["work_cv_available"]
    assert result["work_cv_acc"] == result["chance"] == 0.5
    assert result["work_cv_n_splits"] == 5


def test_real_signal_generalizes_to_unseen_requests_and_chapters():
    profiles, works, chapters, _ = sample(signal=True)
    result = discrimination(profiles, works, list(profiles), chapter_of=chapters)
    assert result["work_cv_acc"] > 0.95
    assert result["work_cv_acc"] > result["chance"] + 0.4
    assert all(fold["accuracy"] > 0.95 for fold in result["work_cv_folds"])


def test_training_vocabulary_ignores_heldout_tags_and_document_counts():
    profiles, _, _, _ = sample(n_groups=1, n_verses=3)
    ids = list(profiles)
    train, test = ids[:3], ids[3:]
    for k, ident in enumerate(train):
        profiles[ident]["style_tags"] = ["common"] + (["rare"] if k < 2 else [])
    for ident in test:
        profiles[ident]["style_tags"] = ["heldout_only", "rare"]
    vocabulary = _fit_tag_vectorizer(profiles, train)
    assert set(vocabulary.vocabulary_) == {"common"}
    train_X, test_X = _profile_features(profiles, train, vocabulary), _profile_features(profiles, test, vocabulary)
    assert train_X.shape[1] == test_X.shape[1]
    assert np.all(train_X[:, -1] == 1)
    assert np.all(test_X[:, -1] == 0)


def test_no_qualifying_tags_still_allows_numeric_evaluation():
    profiles, works, chapters, _ = sample(signal=True)
    for profile in profiles.values():
        profile["style_tags"] = []
    assert _fit_tag_vectorizer(profiles, list(profiles)) is None
    assert discrimination(profiles, works, list(profiles), chapter_of=chapters)["work_cv_acc"] > 0.95


def test_chapters_and_requests_are_never_split_between_train_and_test():
    profiles, works, chapters, _ = sample(signal=True)
    # Requests 0 and 1 span consecutive chapters; chapter 2 contains two requests.
    for ident, profile in profiles.items():
        work, chapter = works[ident], chapters[ident][2]
        profile["provenance"]["request_group"] = f"{work}-{int(chapter) // 2}"
    groups, reason = evaluation_groups([profiles], list(profiles), chapters)
    assert reason is None
    ids = list(profiles)
    folds, reason = _grouped_folds([works[i] for i in ids], [groups[i] for i in ids], seed=0)
    assert reason is None and len(folds) == 3
    for train, test in folds:
        for mapping in (chapters, {i: profiles[i]["provenance"]["request_group"] for i in ids}, groups):
            assert {mapping[ids[i]] for i in train}.isdisjoint(mapping[ids[i]] for i in test)
        assert {works[ids[i]] for i in train} == {works[ids[i]] for i in test} == {"A", "B"}


def test_group_union_tracks_bridges_outside_shared_cohort_and_across_models():
    p = {i: {"provenance": {"request_group": g, "prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION}}
         for i, g in (("a", "x"), ("bridge", "x"), ("b", "y"))}
    q = {i: {"provenance": {"request_group": g, "prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION}}
         for i, g in (("a", "u"), ("bridge", "v"), ("b", "v"))}
    groups, reason = evaluation_groups([p, q], ["a", "b"], {"a": 1, "bridge": 2, "b": 3})
    assert reason is None
    assert groups["a"] == groups["b"]


def test_independent_groups_required_even_with_many_verses():
    profiles, works, chapters, _ = sample(n_groups=1, n_verses=40)
    result = discrimination(profiles, works, list(profiles), chapter_of=chapters)
    assert result["work_cv_available"] is False
    assert "two independent" in result["work_cv_reason"]
    assert "work_cv_acc" not in result
    assert "eta2_mean" in result


def test_legacy_profiles_and_missing_chapters_are_explicitly_unavailable():
    profiles, works, chapters, _ = sample()
    ids = list(profiles)
    missing_chapters = discrimination(profiles, works, ids)
    assert "Chapter metadata" in missing_chapters["work_cv_reason"]
    for profile in profiles.values():
        profile.pop("provenance")
    legacy = discrimination(profiles, works, ids, chapter_of=chapters)
    assert "Request provenance" in legacy["work_cv_reason"]
    assert "work_cv_acc" not in legacy


def test_ensemble_retains_every_member_request_membership():
    profiles, _, _, verses = sample()
    second = {i: {**p, "provenance": {**p["provenance"], "request_group": hashlib.sha256(i.encode()).hexdigest()}}
              for i, p in profiles.items()}
    result = ensemble("a×2", [ModelSet("a", Path("a"), profiles), ModelSet("b", Path("b"), second)])
    validate_profiles_for_corpus(verses, result.profiles)
    ident = next(iter(profiles))
    assert set(result.profiles[ident]["provenance"]["request_groups"]) == {
        profiles[ident]["provenance"]["request_group"], second[ident]["provenance"]["request_group"],
    }


def test_ensemble_tag_votes_remain_valid_when_runs_disagree():
    profiles, _, _, verses = sample()
    second = {i: {**p, "style_tags": [f"other_{j}" for j in range(8)]} for i, p in profiles.items()}
    for profile in profiles.values():
        profile["style_tags"] = [f"first_{j}" for j in range(8)]
    result = ensemble("a×2", [ModelSet("a", Path("a"), profiles), ModelSet("b", Path("b"), second)])
    validate_profiles_for_corpus(verses, result.profiles)
    assert all(profile["style_tags"] == [f"first_{j}" for j in range(8)] for profile in result.profiles.values())


def test_unblinded_profile_version_cannot_claim_work_recovery():
    profiles, works, chapters, _ = sample(signal=True)
    for profile in profiles.values():
        profile["provenance"]["prompt_version"] = "unblinded-v1"
    result = discrimination(profiles, works, list(profiles), chapter_of=chapters)
    assert "current blinded prompt" in result["work_cv_reason"]
    assert "work_cv_acc" not in result


def test_comparison_rejects_profiles_for_changed_corpus_text(tmp_path):
    profiles, _, _, verses = sample(signal=True)
    verses[0]["text"] = "changed after profiling"
    with pytest.raises(ValueError, match="profile text changed"):
        build({"a": ModelSet("a", Path("a"), profiles)}, verses, tmp_path, do_cluster=False)


def test_model_comparison_uses_same_verses_and_folds(tmp_path):
    profiles, _, _, verses = sample(signal=True)
    subset = {i: p for j, (i, p) in enumerate(profiles.items()) if j % 2 == 0}
    result = build({"a": ModelSet("a", Path("a"), profiles), "b": ModelSet("b", Path("b"), subset)},
                   verses, tmp_path, do_cluster=False)
    a, b = (result["models"][name]["discrimination"] for name in ("a", "b"))
    assert a["n"] == b["n"] == len(subset)
    assert a["work_cv_folds"] == b["work_cv_folds"]
    assert result["pilot_ids"] == len(subset)
    report = (tmp_path / "model_comparison.md").read_text()
    assert "accuracy is approximated" not in report
    assert "Most accurate" not in report
    assert "not correctness or authorship accuracy" in report


def test_model_comparison_without_shared_cohort_does_not_rank_recovery(tmp_path):
    profiles, _, _, verses = sample(signal=True)
    first = {i: p for i, p in profiles.items() if ":A." in i}
    second = {i: p for i, p in profiles.items() if ":B." in i}
    result = build({"a": ModelSet("a", Path("a"), first), "b": ModelSet("b", Path("b"), second)},
                   verses, tmp_path, do_cluster=False)
    for model in result["models"].values():
        assert "No shared evaluation cohort" in model["discrimination"]["work_cv_reason"]
        assert "work_cv_acc" not in model["discrimination"]


def test_mixed_languages_are_not_treated_as_work_recovery(tmp_path):
    profiles, works, _, verses = sample(signal=True)
    for verse in verses:
        if works[verse["id"]] == "B":
            verse["language"] = "hbo"
    result = build({"a": ModelSet("a", Path("a"), profiles)}, verses, tmp_path, do_cluster=False)
    assert "mixed languages" in result["models"]["a"]["discrimination"]["work_cv_reason"]
    with pytest.raises(ValueError, match="use --language or --no-cluster"):
        build({"a": ModelSet("a", Path("a"), profiles)}, verses, tmp_path, do_cluster=True)


def test_undefined_numeric_agreement_is_unrankable_and_valid_json(tmp_path):
    profiles, _, _, verses = sample()
    sets = {name: ModelSet(name, Path(name), profiles, cost_per_verse=0.001) for name in ("a", "b", "c")}
    result = build(sets, verses, tmp_path, do_cluster=False)
    assert result["recommendation"] == {}
    assert all(model["score"] is None and not model["pareto"] for model in result["models"].values())
    def reject_nonfinite(value):
        pytest.fail(f"Nonstandard JSON numeric constant: {value}")
    stored = json.loads((tmp_path / "comparison.json").read_text(), parse_constant=reject_nonfinite)
    assert stored["models"]["a"]["consensus"]["r_mean"] is None
    assert stored["models"]["a"]["discrimination"]["eta2_mean"] is None


def test_comparison_of_identical_works_keeps_null_and_marks_fixed_partition_unavailable(tmp_path):
    profiles, _, _, verses = sample(n_groups=2, n_verses=10)
    text = "και ο θεος ειπεν"
    for j, verse in enumerate(verses):
        verse.update({"text": text, "text_bare": text, "n_tokens": 4, "witness": "T", "order": j % 20 + 1,
                      "verse": str(j % 10 + 1), "work_title": verse["work"], "ref": verse["id"],
                      "copyist": None, "group": verse["work"], "has_gap": False, "supplied_frac": 0.0})
        profiles[verse["id"]]["style_tags"] = ["shared_tag"]
        profiles[verse["id"]]["provenance"]["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    sets = {name: ModelSet(name, Path(name), profiles) for name in ("a", "a_retest")}
    result = build(sets, verses, tmp_path)
    assert set(result["clusters"]) == {"lexical-only", "a", "a×2"}
    for cluster in result["clusters"].values():
        assert cluster["k_free"] == 1
        assert cluster["fixed_available"] is False
        assert cluster["ari_fixed"] is None and cluster["purity_fixed"] is None
        assert "Fewer distinct" in cluster["fixed_reason"]
    assert "fixed-k partition is unavailable" in (tmp_path / "model_comparison.md").read_text()
