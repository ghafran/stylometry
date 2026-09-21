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


# --- stability chooses among candidates rather than vetoing one ------------------------------------

def test_supported_k_order_lists_every_candidate_best_first():
    """select_k returns only its favourite; the search needs the rest of the ranking."""
    table = [{'k': 1, 'silhouette': None, 'split_supported': False},
             {'k': 2, 'silhouette': 0.30, 'split_supported': True},
             {'k': 3, 'silhouette': 0.50, 'split_supported': True},
             {'k': 4, 'silhouette': 0.90, 'split_supported': False},
             {'k': 5, 'silhouette': 0.40, 'split_supported': True}]
    assert cl.supported_k_order(table, 'silhouette') == [3, 5, 2]
    assert cl.supported_k_order(table, 'davies_bouldin') == []


def _structured_verses(n_per: int = 30) -> list[dict]:
    """Three genuinely distinct habits, so several k are BIC-supported and there is a list to search."""
    habits = ["και ο θεος ειπεν και ο θεος ειπεν",
              "δε αυτου εν τω λογω δε αυτου εν τω",
              "εγενετο μεν ουν γαρ εγενετο μεν ουν γαρ"]
    out, order = [], 0
    for h, text in enumerate(habits, start=1):
        for i in range(n_per):
            order += 1
            out.append({
                "id": f"grc:W{h}.1.{i + 1}", "source": "x", "language": "grc", "witness": "T",
                "work": f"W{h}", "work_title": f"W{h}", "collection": "NT", "canon": "NT",
                "group": f"W{h}", "chapter": "1", "verse": str(i + 1), "ref": f"W{h} 1:{i + 1}",
                "text": text, "text_bare": text, "n_tokens": len(text.split()), "copyist": None,
                "supplied_frac": 0.0, "has_gap": False, "duplicate_of": None, "order": order,
            })
    return out


def test_an_unstable_favourite_falls_back_to_a_stable_k_not_to_one_group(tmp_path, monkeypatch):
    """The defect this covers: the criterion's first choice was tested alone and, failing, collapsed
    the whole run to one group - discarding stable partitions sitting further down the candidate list.
    Codex Sinaiticus reported one style group for 51 works by many authors because of it.
    """
    verses = _structured_verses()
    real = cl.partition_stability

    def only_k2_is_stable(X, labels, vs, k, **kwargs):
        out = dict(real(X, labels, vs, k, **kwargs))
        if out.get('available'):
            out.update(mean_ari=0.95, min_ari=0.9, ari=[0.9] * 5) if k == 2 else \
                out.update(mean_ari=0.5, min_ari=0.4, ari=[0.4] * 5)
        return out

    monkeypatch.setattr(cl, 'partition_stability', only_k2_is_stable)
    summary = cl.run(verses, None, tmp_path, kmin=2, kmax=6)
    assert summary['k_used'] == 2, 'a stable candidate must be preferred over collapsing to one group'
    assert summary['selection_status'] == 'supported_style_partition'
    assert summary['stability']['tested_k'] == 2
    passed_over = {r['k'] for r in summary['stability_rejected_k']}
    assert passed_over and 2 not in passed_over, 'the chosen k is not among those rejected'
    assert all(r['reason'] == 'unstable' for r in summary['stability_rejected_k'])


def test_one_group_is_still_reported_when_no_candidate_is_stable(tmp_path, monkeypatch):
    verses = _structured_verses()
    real = cl.partition_stability

    def nothing_is_stable(X, labels, vs, k, **kwargs):
        out = dict(real(X, labels, vs, k, **kwargs))
        if out.get('available'):
            out.update(mean_ari=0.3, min_ari=0.2, ari=[0.2] * 5)
        return out

    monkeypatch.setattr(cl, 'partition_stability', nothing_is_stable)
    summary = cl.run(verses, None, tmp_path, kmin=2, kmax=5)
    assert summary['k_used'] == 1 and summary['selection_status'] == 'split_not_stable'
    assert len(summary['stability_rejected_k']) >= 1
