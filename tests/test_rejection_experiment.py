import json

import numpy as np
import pytest

from stylometry import rejection_experiment as experiment


def works(authors=4):
    return [{'author': f'a{a}', 'work': f'w{w}', 'work_id': f'{a}-{w}', 'language': 'grc',
             'text_bare': 'και δε ' * 100 + f'μοναδικον{a}{w}', 'n_tokens': 201}
            for a in range(authors) for w in range(3)]


def test_every_author_is_outer_unknown_and_separate_impostor_once():
    pairs = experiment.author_roles(['a', 'b', 'c', 'd', 'e'])
    assert {a for a, _ in pairs} == {v for _, v in pairs} == {'a', 'b', 'c', 'd', 'e'}
    assert all(a != v for a, v in pairs)
    assert pairs == experiment.author_roles(['e', 'c', 'd', 'a', 'b'])
    assert experiment.author_roles(['a', 'b', 'c']) == []


def test_separate_impostor_and_outer_unknown_never_leak_across_roles(monkeypatch, tmp_path):
    calls = []
    def fitted_scores(train, cal, test, ablation, seed):
        train_authors = {r['author'] for r in train}
        cal_authors = {r['author'] for r in cal}
        test_authors = {r['author'] for r in test}
        assert len(train_authors) == 2
        assert len(cal_authors - train_authors) == 1
        assert len(test_authors - train_authors) == 1
        assert (cal_authors - train_authors).isdisjoint(test_authors)
        assert (test_authors - train_authors).isdisjoint(cal_authors)
        assert not {r['work_id'] for r in train} & {r['work_id'] for r in cal + test}
        assert not {r['work_id'] for r in cal} & {r['work_id'] for r in test}
        calls.append((train_authors, cal_authors, test_authors))
        authors = sorted(train_authors)
        def score(rows):
            # Impostors are more similar than genuine knowns: no useful threshold exists.
            return np.array([[.9 if a == r['author'] else .1 for a in authors]
                             if r['author'] in authors else [.95, .2] for r in rows])
        return authors, score(cal), score(test), 2
    monkeypatch.setattr(experiment, '_fit_scores', fitted_scores)
    result = experiment.run_experiment(works(), tmp_path, length=50, max_samples_per_work=2)
    assert len(calls) == 12
    assert result['status'] == 'development_only'
    assert result['authorship_validated'] is False
    run = result['runs'][0]
    assert run['feasible_calibration_scenarios'] == 0
    assert run['old_policy']['unknown_false_acceptance']['value'] == 1
    assert run['revised_policy']['unknown_false_acceptance']['value'] == 0
    assert run['revised_policy']['known_acceptance']['value'] == 0
    assert run['revised_policy']['accuracy_among_accepted']['value'] is None
    assert all(r['revised_decision'] is None for r in run['predictions'])
    assert all(r['correct'] is False for r in run['predictions'] if not r['known'])
    assert json.loads((tmp_path / 'experiment.json').read_text())['status'] == 'development_only'
    report = (tmp_path / 'report.md').read_text()
    assert 'Refusing all texts is not a successful' in report
    assert 'Unavailable' in report


def test_empty_or_edition_data_cannot_enter_experiment():
    with pytest.raises(ValueError, match='reference works'):
        experiment.run_experiment([])
    with pytest.raises(ValueError, match='edition controls'):
        experiment.run_experiment([{**works()[0], 'role': 'witness_control'}])


def test_insufficient_authors_is_unsupported_not_passed(tmp_path):
    report = experiment.run_experiment(works(3), tmp_path, length=50)
    assert report['status'] == 'development_only'
    assert not report['runs'][0]['available']
    assert 'at least four authors' in report['runs'][0]['reason']


def test_short_works_remain_visible_as_exclusions(tmp_path):
    report = experiment.run_experiment(works(), tmp_path, length=1000)
    assert len(report['runs'][0]['exclusions']) == 12
    assert all('fewer tokens' in e['reason'] for e in report['runs'][0]['exclusions'])


def test_accepted_accuracy_counts_false_acceptance_as_wrong():
    records = [
        {'author': 'known', 'work_id': 'a', 'known': True, 'correct': True, 'revised_accepted': True},
        {'author': 'unknown', 'work_id': 'b', 'known': False, 'correct': False, 'revised_accepted': True},
    ]
    assert experiment._metrics(records, 'revised')['accuracy_among_accepted']['value'] == .5
