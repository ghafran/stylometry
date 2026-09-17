"""Engineering safeguards for pair verification; fixtures do not prove authorship."""
import copy
import json
from collections import defaultdict

import numpy as np
import pytest

from stylometry.pair_verifier import PairVerifier


def rows_for(authors=3, works=3, passages=3):
    styles = ("και δε λογος και τε", "μεν ουν πολις γαρ ουν", "οτι δε ανθρωπος οτι γαρ")
    return [{"id": f"a{a}-w{w}-p{p}", "work_id": f"a{a}-w{w}", "author": f"author{a}",
             "language": "grc", "genre": "oratory" if a < 2 else "dialogue", "topic": "ethics",
             "text_bare": f"{styles[a % 3]} ιδιον{a}{w}{p} {styles[a % 3]}"}
            for a in range(authors) for w in range(works) for p in range(passages)]


@pytest.fixture
def trained():
    return PairVerifier().fit(rows_for())


def fresh_row(identifier, text, author="new", work=None):
    return {"id": identifier, "work_id": work or identifier, "author": author, "language": "grc",
            "genre": "unknown", "topic": "unknown", "text_bare": text}


def test_pairs_cross_canonical_works_and_cover_all_positive_and_negative_work_pairs(trained):
    metadata = trained.metadata_
    assert trained.training_authors_ == ["author0", "author1", "author2"]
    assert len(trained.training_work_ids_) == 9
    assert len(metadata["training_work_pairs"]) == 36
    assert sum(p["same_author"] for p in metadata["training_work_pairs"]) == 9
    assert all(p["left_work_id"] != p["right_work_id"] for p in metadata["training_work_pairs"])
    assert all(p["selected_passage_pairs"] == 8 for p in metadata["training_work_pairs"])
    assert metadata["training_pair_counts"] == {"positive": 72, "negative": 216}
    assert metadata["negative_matching_counts"]["same_genre_work_pairs"] == 9
    assert metadata["negative_matching_counts"]["same_topic_work_pairs"] == 27
    assert metadata["feature_count"] == len(trained.feature_names_)
    assert not any("log_tokens" in name for name in trained.feature_names_)
    assert trained.classifier_.C == 1 and trained.classifier_.max_iter == 2000
    assert trained.vectorizer_.ngram_range == (3, 5) and trained.vectorizer_.max_features == 10000
    json.dumps(metadata, allow_nan=False)


def test_pair_scores_are_symmetric_finite_and_not_probabilities(trained):
    rows = rows_for()
    left, right = [rows[0], rows[4], rows[15]], [rows[4], rows[12], rows[21]]
    forward, backward = trained.score_pairs(left, right), trained.score_pairs(right, left)
    np.testing.assert_allclose(forward, backward, atol=1e-12)
    assert forward.shape == (3,) and np.isfinite(forward).all()
    assert "not an author probability" in trained.metadata_["score_type"]
    assert trained.score_pairs([], []).shape == (0,)


def test_scoring_heldout_text_never_adapts_feature_or_classifier_state(trained):
    vocabulary = trained.vectorizer_.vocabulary_.copy()
    idf = trained.vectorizer_.idf_.copy()
    static_mean, static_scale = trained.static_scaler_.mean_.copy(), trained.static_scaler_.scale_.copy()
    pair_mean, pair_scale = trained.pair_scaler_.mean_.copy(), trained.pair_scaler_.scale_.copy()
    coefficients = trained.classifier_.coef_.copy()
    metadata = copy.deepcopy(trained.metadata_)
    scores = trained.score_pairs([fresh_row("left", "陌生陌生 και δε")],
                                  [fresh_row("right", "חדשκκκκ μεν ουν")])
    assert np.isfinite(scores).all()
    assert trained.vectorizer_.vocabulary_ == vocabulary
    assert trained.metadata_ == metadata
    for before, after in ((idf, trained.vectorizer_.idf_), (static_mean, trained.static_scaler_.mean_),
                          (static_scale, trained.static_scaler_.scale_), (pair_mean, trained.pair_scaler_.mean_),
                          (pair_scale, trained.pair_scaler_.scale_), (coefficients, trained.classifier_.coef_)):
        np.testing.assert_array_equal(before, after)


def test_metadata_and_author_labels_are_not_predictive_features(trained):
    left, right = fresh_row("left", "και δε νεος λογος"), fresh_row("right", "μεν ουν ετερος λογος")
    before = trained.score_pairs([left], [right])
    left.update(author="different", genre="forensic_oratory", topic="different_topic")
    right.update(author="same-label", genre="dialogue", topic="another_topic")
    np.testing.assert_array_equal(before, trained.score_pairs([left], [right]))


def test_sampling_is_capped_and_independent_of_input_row_order(trained):
    reversed_model = PairVerifier().fit(rows_for()[::-1])
    first = [(p["left_id"], p["right_id"]) for p in trained.training_pairs_]
    second = [(p["left_id"], p["right_id"]) for p in reversed_model.training_pairs_]
    assert first == second
    assert len(first) == len(set(first))
    np.testing.assert_allclose(trained.classifier_.coef_, reversed_model.classifier_.coef_)
    assert trained.metadata_["training_work_pairs"] == reversed_model.metadata_["training_work_pairs"]


def test_weights_balance_classes_authors_author_pairs_and_canonical_work_pairs():
    rows = rows_for(3, 3, 3)
    rows = [r for r in rows if not (r["author"] == "author0" and r["work_id"] == "a0-w2")]
    rows = [r for r in rows if r["work_id"] != "a1-w0" or r["id"].endswith("p0")]
    model = PairVerifier().fit(rows)
    grouped = defaultdict(float)
    for pair in model.metadata_["training_work_pairs"]:
        group = ("positive", pair["left_author"]) if pair["same_author"] else ("negative", tuple(sorted((pair["left_author"], pair["right_author"]))))
        grouped[group] += pair["weight_sum"]
    assert all(mass == pytest.approx(8 / 2 / 3) for key, mass in grouped.items() if key[0] == "positive")
    assert grouped[("negative", ("author0", "author1"))] == pytest.approx(2)
    assert grouped[("negative", ("author0", "author2"))] == pytest.approx(1)
    assert grouped[("negative", ("author1", "author2"))] == pytest.approx(1)
    sums = model.metadata_["training_pair_weight_sums"]
    assert sums["positive"] == pytest.approx(4)
    assert sums["negative"] == pytest.approx(4)
    assert sums["total"] == pytest.approx(8)
    assert "canonical training works" in model.metadata_["mass_basis"]
    assert model.metadata_["negative_strata"]["same_genre"]["weight_sum"] == pytest.approx(2)
    assert model.metadata_["negative_strata"]["other"]["weight_sum"] == pytest.approx(2)
    assert model.metadata_["missing_negative_strata"] == []
    weights_by_work_pair = defaultdict(float)
    for pair, weight in zip(model.training_pairs_, model.pair_weights_):
        weights_by_work_pair[pair["work_pair_index"]] += weight
    assert all(weights_by_work_pair[i] == pytest.approx(pair["weight_sum"])
               for i, pair in enumerate(model.metadata_["training_work_pairs"]))


def test_idf_documents_and_static_weights_are_canonical_work_based(trained):
    assert trained.training_sample_weights_.sum() == pytest.approx(9)
    assert trained.training_sample_weights_[:3].sum() == pytest.approx(1)
    # This author's distinctive content word occurs in exactly three works.
    index = trained.vectorizer_.vocabulary_["πολις"]
    assert trained.vectorizer_.idf_[index] == pytest.approx(np.log(10 / 4) + 1)


@pytest.mark.parametrize("mutation,match", [
    ("missing_id", "id"), ("same_id", "duplicate"), ("conflicting_id", "conflicting"),
    ("conflicting_work_author", "conflicting authors"), ("duplicate_text", "duplicate normalized"),
    ("language", "fitted language"), ("empty", "at least two"),
])
def test_invalid_training_rows_are_rejected(mutation, match):
    rows = rows_for()
    if mutation == "missing_id":
        del rows[0]["id"]
    elif mutation == "same_id":
        rows.append(rows[0].copy())
    elif mutation == "conflicting_id":
        rows[1]["id"] = rows[0]["id"]
    elif mutation == "conflicting_work_author":
        rows[1]["author"] = "other"
    elif mutation == "duplicate_text":
        rows[3]["text_bare"] = " " + rows[0]["text_bare"] + "  "
    elif mutation == "language":
        rows[0]["language"] = "hbo"
    else:
        rows = []
    with pytest.raises(ValueError, match=match):
        PairVerifier().fit(rows)


def test_same_work_only_training_cannot_create_positive_pairs():
    with pytest.raises(ValueError, match="different canonical works"):
        PairVerifier().fit(rows_for(authors=3, works=1))


def test_scoring_rejects_self_work_identical_text_conflicting_ids_and_languages(trained):
    rows = rows_for()
    with pytest.raises(ValueError, match="different canonical works"):
        trained.score_pairs([rows[0]], [rows[1]])
    duplicate = fresh_row("duplicate", rows[0]["text_bare"])
    with pytest.raises(ValueError, match="identical passage"):
        trained.score_pairs([rows[0]], [duplicate])
    conflicting = {**rows[0], "text_bare": "και δε αλλαγη"}
    with pytest.raises(ValueError, match="training identity"):
        trained.score_pairs([conflicting], [rows[4]])
    foreign = {**fresh_row("foreign", "כי אשר"), "language": "hbo"}
    with pytest.raises(ValueError, match="fitted language"):
        trained.score_pairs([rows[0]], [foreign])
    with pytest.raises(ValueError, match="equal lengths"):
        trained.score_pairs([rows[0]], [])


def test_repeated_query_in_pair_batch_is_supported(trained):
    rows = rows_for()
    scores = trained.score_pairs([rows[0], rows[0]], [rows[4], rows[12]])
    assert scores.shape == (2,) and np.isfinite(scores).all()


@pytest.mark.parametrize("only_stratum", ["same_genre", "other"])
def test_missing_negative_strata_receive_no_manufactured_mass(only_stratum):
    rows = rows_for()
    for row in rows:
        row["genre"] = "one_genre" if only_stratum == "same_genre" else row["author"]
    model = PairVerifier().fit(rows)
    missing = "other" if only_stratum == "same_genre" else "same_genre"
    assert model.metadata_["missing_negative_strata"] == [missing]
    assert model.metadata_["negative_strata"][only_stratum]["weight_sum"] == pytest.approx(4.5)
    assert model.metadata_["negative_strata"][missing] == {
        "available": False, "work_pairs": 0, "passage_pairs": 0, "weight_sum": 0.0,
    }


def test_unique_signature_transform_matches_direct_features_and_keeps_symmetry(trained, monkeypatch):
    rows = rows_for()
    left, right = [rows[0], rows[0], rows[4]], [rows[4], rows[12], rows[12]]
    combined = left + right
    static = trained.static_scaler_.transform(trained.extractor_.transform(combined)[:, trained.static_mask_])
    chars = trained.vectorizer_.transform([r["text_bare"] for r in combined])
    expected_features = trained._pair_features(static[:3], static[3:], chars[:3], chars[3:])
    expected = trained.classifier_.decision_function(trained.pair_scaler_.transform(expected_features))
    calls = []
    original = trained.extractor_.transform

    def record_transform(passages):
        calls.append([r["id"] for r in passages])
        return original(passages)

    monkeypatch.setattr(trained.extractor_, "transform", record_transform)
    forward = trained.score_pairs(left, right)
    backward = trained.score_pairs(right, left)
    np.testing.assert_allclose(forward, expected, atol=1e-12)
    np.testing.assert_allclose(forward, backward, atol=1e-12)
    assert all(len(ids) == len(set(ids)) == 3 for ids in calls)


def test_failed_refit_locks_out_previous_model(trained):
    with pytest.raises(ValueError):
        trained.fit([])
    with pytest.raises(ValueError, match="fit"):
        trained.score_pairs([rows_for()[0]], [rows_for()[4]])


def test_unfitted_model_cannot_score():
    with pytest.raises(ValueError, match="fit"):
        PairVerifier().score_pairs([], [])
