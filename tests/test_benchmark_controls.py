import numpy as np
import pytest
from stylometry.benchmark_controls import _units, _orthographic_variant, _fit_reference, app_cluster_controls, perturbation_controls, edition_controls
from stylometry.lang import tokenize


def works():
    output = []
    for author, vocabulary in [('A', 'και ο λογοσ εγενετο'), ('B', 'δε εν το εργον')]:
        for i in range(3):
            output.append({'author': author, 'work': f'{author}{i}', 'work_id': f'{author}{i}', 'language': 'grc',
                           'genre': 'prose', 'text_bare': ' '.join([vocabulary] * 150), 'source_id': f'{author}{i}'})
    return output


def test_reference_units_do_not_overlap_or_fabricate_tokens():
    source = [{'author': 'A', 'work': 'W', 'language': 'grc', 'text_bare': ' '.join(map(str, range(103)))}]
    units = _units(source, 10, 4)
    assert len(units) == 4
    tokens = [w for v in units for w in v['text_bare'].split()]
    assert len(set(tokens)) == 40
    assert tokens == list(map(str, range(30, 70)))


def test_same_display_titles_do_not_merge_independent_works():
    source = [{**w, 'work': 'Same title'} for w in works()]
    units = _units(source, 100, 2)
    assert len({v['work'] for v in units}) == len(source)
    assert len({v['id'] for v in units}) == len(units)
    assert {v['work_title'] for v in units} == {'Same title'}


def test_robustness_predictor_matches_attribution_recipe_with_unequal_work_sizes():
    from sklearn.preprocessing import normalize
    from stylometry.benchmark import _fit_scores
    source = works()
    source[0]['text_bare'] = source[0]['text_bare'][:300]
    units = _units(source, 25, 4)
    train = [{**v, 'author': v['group'], 'work_id': v['work']} for v in units]
    labels, _, scores, _ = _fit_scores(train, train, train, 'full_lexical', 42)
    extractor, scaler, authors, centers = _fit_reference(units, 42)
    actual = normalize(scaler.transform(extractor.transform(units))) @ centers.T
    assert labels == authors
    np.testing.assert_allclose(actual, scores, atol=1e-6)


@pytest.mark.parametrize('lang,text', [('grc', 'και ο λογοσ'), ('hbo', 'בראשית אלהימ'), ('arb', 'بسم الله الرحمن')])
def test_spelling_control_retains_normalized_identity(lang, text):
    assert ' '.join(tokenize(_orthographic_variant(text, lang), lang)) == text


def test_stress_test_uses_disjoint_works_and_does_not_claim_manuscript_validation():
    result = perturbation_controls(works(), size=100, max_units=3)
    assert set(result['train_works']).isdisjoint(result['test_works'])
    assert result['normalization_identical_fraction'] == 1
    assert next(r for r in result['conditions'] if r['condition'] == 'orthography')['prediction_agreement'] == 1
    assert 'not actual manuscript' in result['interpretation']


def test_actual_app_control_exercises_single_writer_and_mixed_writer_inputs(tmp_path):
    result = app_cluster_controls(works(), tmp_path, sample_lengths=(100,), max_units=6)
    assert result['single_author_control_count'] == 2
    assert any(r['control'] == 'all_authors' for r in result['controls'])
    assert any(r['control'] == 'same_genre' for r in result['controls'])
    assert result['authorship_validated'] is False
    assert (tmp_path / 'controls.json').exists()


def test_edition_pair_must_not_be_in_training():
    source = works()
    controls = [{**source[0], 'source_id': f'edition{i}'} for i in range(2)]
    with pytest.raises(ValueError, match='excluded'):
        edition_controls(source, controls, size=100)


def test_same_work_editions_are_one_control_not_independent_authors():
    source = works()
    controls = [{**source[0], 'work': 'heldout', 'work_id': 'heldout', 'source_id': f'edition{i}'} for i in range(2)]
    result = edition_controls(source, controls, size=100)
    assert result['available']
    assert len(result['pairs']) == 1
    assert result['pairs'][0]['normalized_texts_identical']
    assert result['pairs'][0]['same_prediction_across_editions']
