"""Development-only unknown-author calibration; never replaces the frozen benchmark.

The exposed reference corpus is development data. Each outer unknown author and
the separate calibration impostor are excluded from feature/model fitting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .benchmark import _fit_scores, _work_identity, calibrate_thresholds, rate_estimate, sample_works, work_partitions
from .rejection import apply_rejection, calibrate_rejection


PROTOCOL = {
    'version': 'impostor-rejection-development-v1',
    'scope': 'development on exposed reference texts; not independent validation',
    'sample_length': 1000,
    'max_samples_per_work': 12,
    'seed': 42,
    'features': 'full_lexical',
    'max_calibration_false_acceptance': .05,
    'min_calibration_known_acceptance': .80,
    'author_roles': 'Seeded author hash cycle: each outer unknown author uses its successor as calibration impostor; both are excluded from fitting',
    'work_roles': 'Three canonical whole-work buckets rotate through training, known-author calibration and testing',
    'decision': 'Abstain on exact top-score ties or when calibration has no feasible threshold',
    'comparison': 'Old and revised rejection use identical fitted models, candidate authors and test texts',
}


def author_roles(authors, seed=42):
    """Balance impostor participation without consulting text, scores or errors."""
    ordered = sorted(set(authors), key=lambda a: hashlib.sha256(f'{seed}:impostor-cycle:{a}'.encode()).digest())
    if len(ordered) < 4:
        return []
    return [(author, ordered[(i + 1) % len(ordered)]) for i, author in enumerate(ordered)]


def _metrics(records, policy):
    known = [r for r in records if r['known']]
    unknown = [r for r in records if not r['known']]
    accepted = [r for r in records if r[f'{policy}_accepted']]
    converted = [{**r, 'correct_accepted': r['correct'] and r[f'{policy}_accepted'],
                  'wrong_accepted': not r['correct'] and r[f'{policy}_accepted']} for r in known]
    return {
        'unknown_false_acceptance': rate_estimate(unknown, f'{policy}_accepted'),
        'known_acceptance': rate_estimate(known, f'{policy}_accepted'),
        'known_correct_acceptance': rate_estimate(converted, 'correct_accepted'),
        'known_wrong_acceptance': rate_estimate(converted, 'wrong_accepted'),
        'accuracy_among_accepted': rate_estimate(accepted, 'correct'),
        'accepted_decisions': len(accepted),
        'decisions': len(records),
    }


def run_experiment(works, output_dir=None, *, length=1000, max_samples_per_work=12,
                   seed=42, progress=None):
    if not works:
        raise ValueError('rejection experiment needs reference works')
    if any(w.get('role', 'attribution') != 'attribution' for w in works):
        raise ValueError('edition controls cannot enter reference-author fitting')
    samples = sample_works(works, length, max_samples_per_work)
    runs = []
    for language in sorted({w['language'] for w in works}):
        language_samples = [s for s in samples if s['language'] == language]
        eligible, partitions, excluded = work_partitions(language_samples, seed)
        sampled_works = {s['work_id'] for s in language_samples}
        excluded.extend({'author': w['author'], 'work_id': _work_identity(w), 'reason': 'fewer tokens than one complete sample'}
                        for w in works if w['language'] == language and _work_identity(w) not in sampled_works)
        roles = author_roles([r['author'] for r in eligible], seed)
        if not roles:
            runs.append({'language': language, 'available': False, 'reason': 'need at least four authors with three complete independent works each', 'exclusions': excluded})
            continue
        if progress:
            progress(f'Evaluating rejection for {language}: {len(roles)} authors, {length}-token passages')
        records, scenarios = [], []
        for unknown, impostor in roles:
            for fold in range(3):
                known = [s for s in eligible if s['author'] not in (unknown, impostor)]
                train = [s for s in known if partitions[s['work_id']] == fold]
                known_cal = [s for s in known if partitions[s['work_id']] == (fold + 1) % 3]
                impostor_cal = [s for s in eligible if s['author'] == impostor]
                test = [s for s in eligible if s['author'] != impostor and partitions[s['work_id']] == (fold + 2) % 3]
                calibration_rows = known_cal + impostor_cal
                authors, calibration_scores, test_scores, feature_count = _fit_scores(
                    train, calibration_rows, test, 'full_lexical', seed)
                known_scores = calibration_scores[:len(known_cal)]
                impostor_scores = calibration_scores[len(known_cal):]
                _, old_threshold = calibrate_thresholds(known_cal, known_scores, authors)
                revised = calibrate_rejection(
                    known_cal, known_scores, impostor_cal, impostor_scores, authors,
                    max_false_acceptance=PROTOCOL['max_calibration_false_acceptance'],
                    min_known_acceptance=PROTOCOL['min_calibration_known_acceptance'])
                old_accepted = test_scores.max(axis=1) >= old_threshold
                revised_accepted = apply_rejection(test_scores, revised)
                scenario_id = f'{language}:{unknown}:{fold}'
                scenario = {
                    'id': scenario_id, 'fold': fold, 'unknown_test_author': unknown,
                    'calibration_impostor_author': impostor, 'candidate_authors': authors,
                    'train_works': sorted({s['work_id'] for s in train}),
                    'known_calibration_works': sorted({s['work_id'] for s in known_cal}),
                    'impostor_calibration_works': sorted({s['work_id'] for s in impostor_cal}),
                    'test_works': sorted({s['work_id'] for s in test}),
                    'feature_count': feature_count, 'old_threshold': old_threshold,
                    'revised_calibration': revised,
                }
                scenarios.append(scenario)
                for row, scores, old_accept, new_accept in zip(test, test_scores, old_accepted, revised_accepted):
                    nearest = authors[int(scores.argmax())]
                    records.append({
                        'id': row['id'], 'work_id': row['work_id'], 'author': row['author'],
                        'scenario_id': scenario_id, 'known': row['author'] in authors,
                        'nearest_reference_author': nearest, 'correct': nearest == row['author'],
                        'candidate_scores': dict(zip(authors, map(float, scores))),
                        'old_accepted': bool(old_accept), 'revised_accepted': bool(new_accept),
                        'revised_decision': nearest if new_accept else None,
                    })
            if progress:
                progress(f'Completed {language}: outer unknown {unknown}; separate calibration author {impostor}')
        runs.append({
            'language': language, 'available': True, 'authors': len(roles),
            'works': len({r['work_id'] for r in eligible}), 'exclusions': excluded,
            'source_languages': sorted({w.get('source_language', language) for w in works if w['language'] == language}),
            'old_policy': _metrics(records, 'old'), 'revised_policy': _metrics(records, 'revised'),
            'feasible_calibration_scenarios': sum(s['revised_calibration']['feasible'] for s in scenarios),
            'total_scenarios': len(scenarios), 'scenarios': scenarios, 'predictions': records,
        })
    implementation = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                      for name in ('rejection.py', 'rejection_experiment.py', 'benchmark.py', 'features.py', 'lang.py', 'greek.py')}
    report = {
        'status': 'development_only', 'authorship_validated': False,
        'protocol': PROTOCOL, 'settings': {'length': length, 'max_samples_per_work': max_samples_per_work, 'seed': seed},
        'reference_works': [{k: v for k, v in w.items() if k != 'text_bare'} for w in works],
        'implementation_sha256': implementation, 'runs': runs,
        'limitations': [
            'This corpus was already inspected. The experiment is development evidence, not fresh validation or a replacement for the frozen failed benchmark.',
            'One calibration impostor author per scenario cannot characterize the diversity of unseen writers. The 5% calibration target is not a certified bound on future false acceptance.',
            'Abstaining from every text eliminates false acceptance but fails useful known-author coverage. Both outcomes must be reported.',
            'Intervals resample fixed predictions by work; model/calibration refitting and dependencies from shared authors and training works are not modeled.',
            'Old and revised policies share each fitted model and candidate pool. Their scores are not directly comparable to the larger candidate pools in the original benchmark.',
            'Accuracy among accepted texts depends on the designed mix of known and unknown authors; it is not expected precision for a deployment population.',
            'Modern Hebrew is not Biblical Hebrew; scripture transfer and Arabic remain unvalidated.',
        ],
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'experiment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        (out / 'report.md').write_text(render_report(report))
    return report


def render_report(report):
    def percent(metric):
        return 'Unavailable' if metric['value'] is None else f"{metric['value']:.1%}"
    lines = ['# Unknown-author rejection development', '',
             '**Development only. Author attribution remains unvalidated.**', '',
             'The revised rule calibrates against a separate unfamiliar author, using the maximum candidate score—the quantity used to accept a text. It abstains on ties or when no threshold meets both calibration targets: ≤5% impostor acceptance and ≥80% known-author acceptance. The outer unknown test author enters neither training nor calibration.', '',
             'Old and revised decisions below use the same models and test texts. Refusing all texts is not a successful authorship system.', '',
             'Rates average within work and then author. Accepted accuracy includes mistakes on both known and unknown authors; it is unavailable if no text was accepted.', '',
             f"Actual settings: {report['settings']['length']} tokens per passage, at most {report['settings']['max_samples_per_work']} samples per work, seed {report['settings']['seed']}. Protocol values describe defaults; overrides remain development-only.", '',
             '| Language | Rule | Unknown accepted | Known accepted | Known correctly accepted | Known wrongly accepted | Balanced accuracy among accepted |',
             '|---|---|---:|---:|---:|---:|---:|']
    for run in report['runs']:
        if not run['available']:
            lines += ['', f"{run['language']}: unavailable — {run['reason']}."]
            continue
        for key, label in [('old_policy', 'Known-author calibration only'), ('revised_policy', 'Separate unfamiliar-author calibration')]:
            lines.append(f"| {run['language']} | {label} | " + ' | '.join(percent(run[key][m]) for m in ('unknown_false_acceptance', 'known_acceptance', 'known_correct_acceptance', 'known_wrong_acceptance', 'accuracy_among_accepted')) + ' |')
    for run in report['runs']:
        if run['available']:
            lines += ['', f"{run['language']}: {run['feasible_calibration_scenarios']}/{run['total_scenarios']} scenarios had a feasible calibration threshold. Infeasible scenarios abstain from all test texts. Feasibility is a calibration point estimate, not validation. Actual source languages: {', '.join(run['source_languages'])}."]
    lines += ['', '## Limits', ''] + ['- ' + item for item in report['limitations']]
    lines += ['', 'Full decisions, author/work partitions, thresholds, source provenance and implementation hashes are in `experiment.json`.', '']
    return '\n'.join(lines)
