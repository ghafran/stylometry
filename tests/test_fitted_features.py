import numpy as np
import pytest
from stylometry.features import LexicalFeatureExtractor, lexical_features


def samples(texts, language='grc'):
    return [{'language': language, 'text_bare': text} for text in texts]


def test_test_text_cannot_change_vocabulary_idf_or_projection():
    train = samples(['και ο λογοσ', 'δε ο λογοσ', 'και το εργον', 'γαρ το εργον', 'ο λογοσ δε'])
    fitted = LexicalFeatureExtractor().fit(train)
    vocab = dict(fitted.vectorizer_.vocabulary_)
    idf = fitted.vectorizer_.idf_.copy()
    axes = fitted.svd_.components_.copy()
    reference = fitted.transform(samples(['και ο λογοσ']))
    other = fitted.transform(samples(['και ο λογοσ', 'ψψψ ωωω χχχ']))
    np.testing.assert_allclose(reference, other[:1])
    assert fitted.vectorizer_.vocabulary_ == vocab
    np.testing.assert_array_equal(fitted.vectorizer_.idf_, idf)
    np.testing.assert_array_equal(fitted.svd_.components_, axes)
    assert not any('ψ' in item for item in vocab)


def test_heldout_transform_preserves_schema_and_is_not_refit():
    train = samples(['ο και λογοσ', 'δε λογοσ', 'εν ο και', 'και δε εν'])
    heldout = samples(['ψψψψψ', 'χχχχχ', 'ωωωωω'])
    fitted = LexicalFeatureExtractor(svd_dims=2).fit(train)
    result = fitted.transform(heldout)
    assert result.shape == (3, len(fitted.feature_names_))
    assert np.isfinite(result).all()
    assert np.all(result[:, -2:] == 0)  # every held-out character gram is unseen


def test_transductive_wrapper_uses_identical_feature_definition():
    train = samples(['και ο λογοσ', 'δε το εργον', 'ο λογοσ εν τω'])
    result, names = lexical_features(train, svd_dims=2)
    fitted = LexicalFeatureExtractor(svd_dims=2)
    np.testing.assert_allclose(result, fitted.fit_transform(train))
    assert names == fitted.feature_names_


def test_static_ablation_learns_no_text_vocabulary():
    fitted = LexicalFeatureExtractor(svd_dims=0).fit(samples(['και ο λογοσ']))
    assert fitted.vectorizer_ is None
    assert not any(name.startswith('cng:') for name in fitted.feature_names_)
    assert fitted.transform([]).shape == (0, len(fitted.feature_names_))


def test_transform_requires_fit_and_rejects_language_transfer():
    with pytest.raises(ValueError, match='fit'):
        LexicalFeatureExtractor().transform(samples(['και']))
    fitted = LexicalFeatureExtractor().fit(samples(['και']))
    with pytest.raises(ValueError, match='same language'):
        fitted.transform(samples(['בראשית'], language='hbo'))
