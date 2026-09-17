"""Positive/negative controls for exploratory structure, never proof of authorship."""
import csv
import json

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from stylometry import cluster as cl
from stylometry.continuity import annotate_continuity
from stylometry.report import render
from stylometry.html import build_site
from test_pipeline import _fake_verses


@pytest.mark.parametrize('criterion', list(cl.CRITERIA))
@pytest.mark.parametrize('seed', [0, 7, 23])
def test_single_correlated_gaussian_does_not_require_multiple_hands(criterion, seed):
    rng = np.random.default_rng(seed)
    # A single elongated, rotated cloud is still one population.
    X = rng.normal(size=(300, 5)) @ rng.normal(size=(5, 5))
    chosen, rows = cl.select_k(X, 2, 4, criterion=criterion, seed=seed)
    assert chosen == 1
    assert rows[0]['k'] == 1
    assert not any(r['split_supported'] for r in rows)


def test_clear_two_population_signal_survives_smoothing_and_resampling():
    rng = np.random.default_rng(81)
    labels = np.repeat([0, 1], 150)
    X = rng.normal(0, .25, size=(300, 4)) + labels[:, None] * 5
    verses = annotate_continuity(_fake_verses(300, 'X', chapter_size=30))
    smoothed = cl.smooth(X, verses)
    chosen, _ = cl.select_k(smoothed, 2, 4, X_eval=X)
    assert chosen == 2
    assigned, _, _ = cl.cluster(smoothed, chosen)
    assert adjusted_rand_score(labels, assigned) > .98
    stability = cl.partition_stability(smoothed, assigned, verses, chosen)
    assert stability['min_ari'] > .98


def test_smoothing_cannot_turn_raw_gaussian_noise_into_supported_groups():
    X = np.random.default_rng(12).normal(size=(300, 6))
    verses = annotate_continuity(_fake_verses(300, 'X', chapter_size=30))
    chosen, _ = cl.select_k(cl.smooth(X, verses), 2, 4, X_eval=X)
    assert chosen == 1


def test_single_group_has_no_fabricated_assignment_confidence():
    X = np.arange(12).reshape(6, 2)
    authors, margin, centers = cl.cluster(X, 1)
    assert authors.tolist() == ['A1'] * 6
    assert not margin.any()
    np.testing.assert_allclose(centers, X.mean(axis=0, keepdims=True))


def test_filtered_passages_never_blend_or_merge():
    full = annotate_continuity(_fake_verses(10, 'X'))
    verses = [full[0], full[9]]
    X = np.array([[0.], [10.]])
    np.testing.assert_equal(cl.smooth(X, verses), X)
    runs, _ = cl.segments(verses, np.array(['A1', 'A1']))
    assert len(runs) == 2
    assert [r['n'] for r in runs] == [1, 1]


def test_constant_corpus_single_group_renders_without_nonfinite_output(tmp_path):
    verses = _fake_verses(30, 'X')
    for v in verses:
        v['text'] = v['text_bare'] = 'και ο λογοσ'
        v['n_tokens'] = 3
    summary = cl.run(verses, None, tmp_path, kmax=4)
    assert summary['k_used'] == 1
    assert summary['selection_status'] == 'no_supported_split'
    json.dumps(summary, allow_nan=False)
    report = render(tmp_path)
    assert 'no supported split' in report.lower()
    site = build_site(tmp_path)
    assert (site / 'index.html').exists()
    assert 'not identified authors' in (site / 'index.html').read_text()


def test_explicit_single_group_and_missing_profiles_are_reported(tmp_path):
    verses = _fake_verses(12, 'X')
    summary = cl.run(verses, None, tmp_path, k=1, kmax=3)
    assert summary['k_forced'] is True
    assert summary['selection_status'] == 'forced'
    assert summary['k_used'] == 1
    rows = list(csv.DictReader((tmp_path / 'verse_assignments.csv').open()))
    assert all(float(r['confidence']) == 0 for r in rows)


@pytest.mark.parametrize('kwargs', [{'alpha': 2}, {'window': -1}, {'k': 0}, {'kmin': 0},
                                    {'weights': {'lex': 0}}, {'weights': {'lex': -1}}])
def test_invalid_analysis_settings_fail_loudly(tmp_path, kwargs):
    with pytest.raises(ValueError):
        cl.run(_fake_verses(10, 'X'), None, tmp_path, kmax=3, **kwargs)


def test_empty_profile_file_cannot_silently_enable_lexical_only(tmp_path):
    with pytest.raises(ValueError, match='no AI profiles'):
        cl.run(_fake_verses(10, 'X'), {}, tmp_path)


def test_block_weights_normalize_actual_variance():
    varying = np.array([[-1.], [1.]])
    padded = np.hstack([varying, np.zeros((2, 99))])
    result = cl.combine([('lex', padded, [])], {'lex': 1})
    assert np.var(result, axis=0).sum() == pytest.approx(1)


def test_legacy_report_does_not_claim_new_validation(tmp_path):
    cl.run(_fake_verses(12, 'X'), None, tmp_path, k=1, kmax=2)
    path = tmp_path / 'summary.json'
    summary = json.loads(path.read_text())
    for field in ('selection_thresholds', 'selection_status', 'stability', 'k_selected', 'k_forced'):
        summary.pop(field)
    path.write_text(json.dumps(summary))
    report = render(tmp_path)
    assert 'checks were not recorded' in report
    assert 'Automatic splits must improve BIC' not in report
