import copy
import json

import numpy as np
import pytest

from stylometry.author_models import AuthorModel, MODEL_KINDS


def row(author, work, text, language="grc"):
    return {"author": author, "work_id": work, "text_bare": text, "language": language}


@pytest.fixture
def train():
    return [
        row("alpha", "a1", "και δε λογος και δε λογος και δε"),
        row("alpha", "a2", "και γαρ λογος και δε λογος"),
        row("beta", "b1", "μεν ουν ανθρωπος μεν ουν ανθρωπος"),
        row("beta", "b2", "μεν τε ανθρωπος μεν ουν ανθρωπος"),
    ]


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_model_produces_candidate_aligned_finite_scores_and_serializable_metadata(kind, train):
    model = AuthorModel(kind).fit(train)
    scores = model.score(train)
    assert model.authors_ == ["alpha", "beta"]
    assert scores.shape == (4, 2)
    assert np.isfinite(scores).all()
    assert scores.argmax(axis=1).tolist() == [0, 0, 1, 1]
    assert model.score([]).shape == (0, 2)
    assert model.metadata_["feature_count"] == len(model.feature_names_)
    json.dumps(model.metadata_, allow_nan=False)


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_scoring_heldout_content_never_changes_training_state(kind, train):
    model = AuthorModel(kind).fit(train)
    baseline = model.score(train)
    metadata = copy.deepcopy(model.metadata_)
    names = model.feature_names_[:]
    if kind == "char_linear":
        vocabulary = model.vectorizer_.vocabulary_.copy()
        idf = model.vectorizer_.idf_.copy()
        coefficients = model.classifier_.coef_.copy()
    else:
        mean, scale = model.scaler_.mean_.copy(), model.scaler_.scale_.copy()
        bigrams = model.bigram_vocabulary_.copy()
    scores = model.score([row("unknown", "heldout", "陌生 κκκκκ και μεν")])
    assert np.isfinite(scores).all()
    assert model.metadata_ == metadata
    assert model.feature_names_ == names
    np.testing.assert_allclose(model.score(train), baseline)
    if kind == "char_linear":
        assert model.vectorizer_.vocabulary_ == vocabulary
        np.testing.assert_array_equal(model.vectorizer_.idf_, idf)
        np.testing.assert_array_equal(model.classifier_.coef_, coefficients)
    else:
        np.testing.assert_array_equal(model.scaler_.mean_, mean)
        np.testing.assert_array_equal(model.scaler_.scale_, scale)
        assert model.bigram_vocabulary_ == bigrams


def test_character_model_uses_fixed_recipe_and_work_level_idf(train):
    model = AuthorModel("char_linear").fit(train)
    vectorizer, classifier = model.vectorizer_, model.classifier_
    assert vectorizer.analyzer == "char_wb"
    assert vectorizer.ngram_range == (3, 5)
    assert vectorizer.max_features == 30000
    assert classifier.C == 1 and classifier.max_iter == 2000 and classifier.solver == "lbfgs"
    gram = vectorizer.vocabulary_["και"]
    assert vectorizer.idf_[gram] == pytest.approx(np.log(5 / 3) + 1)
    scores = model.score(train)
    np.testing.assert_allclose(scores[:, 0], -scores[:, 1])
    assert "not probability" in model.metadata_["score_type"]


def test_character_multiclass_scores_follow_author_order(train):
    train += [row("gamma", "g1", "ητοι αρα παις ητοι αρα παις")]
    model = AuthorModel("char_linear").fit(train)
    assert model.authors_ == ["alpha", "beta", "gamma"]
    assert model.score(train).shape == (5, 3)
    assert np.isfinite(model.score([row("unknown", "u1", "陌生陌生")])).all()


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_duplicate_chunk_counts_do_not_change_author_or_work_weight(kind, train):
    # With no feature pruning, repeated identical chunks also leave work IDF,
    # standardization, centroid fitting, and effective classifier C unchanged.
    ordinary = AuthorModel(kind).fit(train)
    repeated = AuthorModel(kind).fit([train[0]] * 10 + train[1:])
    assert repeated.sample_weight_.sum() == pytest.approx(4)
    assert repeated.sample_weight_[:10].sum() == pytest.approx(1)
    assert repeated.sample_weight_[10:].tolist() == [1, 1, 1]
    np.testing.assert_allclose(ordinary.score(train), repeated.score(train), atol=1e-7)


def test_scaler_weights_authors_then_works(train):
    rows = [train[0]] * 10 + [train[1], train[2]]
    model = AuthorModel("function_centroid").fit(rows)
    assert model.sample_weight_[:10].sum() == pytest.approx(.75)
    assert model.sample_weight_[10:].tolist() == [.75, 1.5]
    features = model._centroid_features(rows)
    expected = np.average(features, axis=0, weights=model.sample_weight_)
    np.testing.assert_allclose(model.scaler_.mean_, expected)


def test_morph_bigrams_capture_adjacent_order_without_heldout_vocabulary_growth(train):
    model = AuthorModel("morph_centroid").fit(train)
    assert ("και", "δε") in model.bigram_vocabulary_
    assert ("και", "μεν") not in model.bigram_vocabulary_
    assert "misc:log_tokens" not in model.feature_names_
    same_bag = [row("unknown", "u1", "και δε και δε"),
                row("unknown", "u2", "δε και δε και")]
    features = model._centroid_features(same_bag)
    column = model.feature_names_.index("fwbigram:και|δε")
    assert features[0, column] == pytest.approx(200 / 3)
    assert features[1, column] == pytest.approx(100 / 3)
    model.score([row("unknown", "u3", "και μεν και μεν")])
    assert ("και", "μεν") not in model.bigram_vocabulary_
    assert "not validated morphology" in model.metadata_["feature_note"]


def test_function_model_uses_only_predefined_words(train):
    model = AuthorModel("function_centroid").fit(train)
    assert all(name.startswith("fw:") for name in model.feature_names_)
    assert not model.bigram_vocabulary_


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_invalid_training_or_scoring_rows_are_rejected(kind, train):
    with pytest.raises(ValueError, match="fit"):
        AuthorModel(kind).score(train)
    with pytest.raises(ValueError, match="two authors"):
        AuthorModel(kind).fit(train[:2])
    with pytest.raises(ValueError, match="same language"):
        AuthorModel(kind).fit(train + [row("third", "c1", "כי אשר", "hbo")])
    with pytest.raises(ValueError, match="multiple author"):
        AuthorModel(kind).fit(train + [row("third", "a1", "και δε")])
    model = AuthorModel(kind).fit(train)
    with pytest.raises(ValueError, match="same language"):
        model.score([row("unknown", "h1", "כי אשר", "hbo")])
    with pytest.raises(ValueError, match="nonempty"):
        model.score([row("unknown", "u1", " ")])
    bad = train[0].copy()
    del bad["work_id"]
    with pytest.raises(ValueError, match="work_id"):
        model.score([bad])


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="kind"):
        AuthorModel("select_the_best")


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_failed_refit_cannot_expose_previous_model(kind, train):
    model = AuthorModel(kind).fit(train)
    with pytest.raises(ValueError):
        model.fit([])
    with pytest.raises(ValueError, match="fit"):
        model.score(train)
