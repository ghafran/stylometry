"""Develop reference models, lock their design, then evaluate fresh authors.

The old benchmark stays frozen. No fresh evaluation text enters model selection.
Final model fitting uses only reserved reference training works; thresholds use
separate known works and at least two unfamiliar calibration authors.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import beta

from .author_models import AuthorModel
from .benchmark import _work_identity, rate_estimate, sample_works, work_partitions
from .rejection import calibrate_rejection, apply_rejection

PROTOCOL = {
    'version': 'passage-author-study-v1', 'seed': 42,
    'models': ['function_centroid', 'morph_centroid', 'char_linear'],
    'lengths': [500, 1000, 2000], 'max_samples_per_work': 12,
    'unknown_calibration_authors': 2,
    'targets': {'accuracy': .80, 'unknown_false_acceptance': .05, 'known_acceptance': .80,
                'known_correct_acceptance': .80, 'known_wrong_acceptance': .05},
    'selection': 'Prefer development unknown acceptance <=5%, known wrong acceptance <=5% and known correct acceptance >=80%; then maximize mean matched-genre closed-set accuracy (overall accuracy if no genre control); ties use declared model order then shorter passages',
    'final_split': 'Fresh authors only; explicit reference/calibration_unknown/test_unknown author roles; reference works in fixed train/calibration/test buckets',
    'final_refit_replicates': 40,
    'acceptance': 'An observed target miss fails this test panel; passing requires all refit intervals, matched-genre accuracy and minimum coverage. Missing uncertainty is inconclusive, never evidence of reliability.',
    'coverage': 'Broad pass requires at least 8 reference authors, 60 reference works, 8 final unknown authors and 60 final unknown works; smaller panels remain pilots',
}
SOURCE_FILES = ('author_study.py', 'author_models.py', 'rejection.py', 'features.py', 'lang.py', 'greek.py',
                'benchmark.py', 'benchmark_data.py', 'benchmark_additional.py')


def implementation_hashes():
    return {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def _write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _corpus_identity(works):
    return [{'author': w['author'], 'work_id': _work_identity(w), 'language': w['language'],
             'genre': w.get('genre'), 'topic': w.get('topic'),
             'text_sha256': hashlib.sha256(w['text_bare'].encode()).hexdigest(),
             'source_id': w.get('source_id'), 'provenance': w.get('provenance', {})} for w in works]


def _rows(test, scores, authors, accepted=None):
    if accepted is None:
        accepted = np.ones(len(test), dtype=bool)
    result = []
    for row, values, accept in zip(test, scores, accepted):
        winner = authors[int(np.argmax(values))]
        result.append({**{k: row[k] for k in ('id', 'author', 'work_id', 'genre', 'topic')},
                       'known': row['author'] in authors, 'correct': winner == row['author'],
                       'accepted': bool(accept), 'correct_accepted': bool(accept and winner == row['author']),
                       'wrong_accepted': bool(accept and winner != row['author']),
                       'nearest_reference_author': winner, 'decision': winner if accept else None,
                       'candidate_scores': dict(zip(authors, map(float, values)))})
    return result


def _metrics(rows):
    known = [r for r in rows if r['known']]
    unknown = [r for r in rows if not r['known']]
    result = {
        'accuracy': rate_estimate(known, 'correct'),
        'unknown_false_acceptance': rate_estimate(unknown, 'accepted'),
        'known_acceptance': rate_estimate(known, 'accepted'),
        'known_correct_acceptance': rate_estimate(known, 'correct_accepted'),
        'known_wrong_acceptance': rate_estimate(known, 'wrong_accepted'),
        'accuracy_among_accepted': rate_estimate([r for r in rows if r['accepted']], 'correct'),
    }
    for key, metric in result.items():
        subset = unknown if key == 'unknown_false_acceptance' else [r for r in rows if r['accepted']] if key == 'accuracy_among_accepted' else known
        metric['independent_authors'] = len({r['author'] for r in subset})
    return result


def _closed(samples, kind, seed):
    eligible, buckets, excluded = work_partitions(samples, seed)
    if len({r['author'] for r in eligible}) < 2:
        return {'available': False, 'reason': 'need two authors with three works in this control', 'exclusions': excluded}
    rows = []
    for fold in range(3):
        train = [r for r in eligible if buckets[r['work_id']] == fold]
        test = [r for r in eligible if buckets[r['work_id']] == (fold + 2) % 3]
        model = AuthorModel(kind, seed=seed).fit(train)
        rows.extend(_rows(test, model.score(test), model.authors_))
    return {'available': True, 'authors': len({r['author'] for r in eligible}),
            'works': len({r['work_id'] for r in eligible}), 'accuracy': rate_estimate(rows, 'correct'),
            'exclusions': excluded, 'predictions': rows}


def _open_development(samples, kind, seed):
    eligible, buckets, excluded = work_partitions(samples, seed)
    authors = sorted({r['author'] for r in eligible}, key=lambda a: hashlib.sha256(f'{seed}:study:{a}'.encode()).digest())
    if len(authors) < 5:
        return {'available': False, 'reason': 'need two reference authors, two calibration impostors and one outer unknown', 'exclusions': excluded}
    rows, scenarios = [], []
    for i, unknown in enumerate(authors):
        impostors = {authors[(i + 1) % len(authors)], authors[(i + 2) % len(authors)]}
        for fold in range(3):
            known = [r for r in eligible if r['author'] not in impostors | {unknown}]
            train = [r for r in known if buckets[r['work_id']] == fold]
            cal = [r for r in known if buckets[r['work_id']] == (fold + 1) % 3]
            other = [r for r in eligible if r['author'] in impostors]
            test = [r for r in eligible if r['author'] not in impostors and buckets[r['work_id']] == (fold + 2) % 3]
            model = AuthorModel(kind, seed=seed).fit(train)
            calibrated = calibrate_rejection(cal, model.score(cal), other, model.score(other), model.authors_,
                                             max_false_acceptance=PROTOCOL['targets']['unknown_false_acceptance'],
                                             min_known_acceptance=PROTOCOL['targets']['known_acceptance'])
            scores = model.score(test)
            rows.extend(_rows(test, scores, model.authors_, apply_rejection(scores, calibrated)))
            scenarios.append({'unknown_author': unknown, 'impostor_authors': sorted(impostors), 'fold': fold,
                              'candidate_authors': model.authors_, 'calibration': calibrated,
                              'train_works': sorted({r['work_id'] for r in train}),
                              'known_calibration_works': sorted({r['work_id'] for r in cal}),
                              'impostor_works': sorted({r['work_id'] for r in other}),
                              'test_works': sorted({r['work_id'] for r in test})})
    return {'available': True, 'metrics': _metrics(rows), 'scenarios': scenarios, 'predictions': rows,
            'feasible_scenarios': sum(s['calibration']['feasible'] for s in scenarios)}


def select_candidate(runs):
    candidates = []
    for run in runs:
        if not run['closed']['available']:
            continue
        matched = [c['accuracy']['value'] for c in run['controls'] if c['label'] == 'genre' and c['available']]
        signal = float(np.mean(matched)) if matched else run['closed']['accuracy']['value']
        opened = run['open']
        feasible = (opened['available'] and opened['metrics']['unknown_false_acceptance']['value'] <= .05
                    and opened['metrics']['known_correct_acceptance']['value'] >= .80
                    and opened['metrics']['known_wrong_acceptance']['value'] <= .05)
        rank = (int(feasible), signal, -PROTOCOL['models'].index(run['model']), -run['length'])
        candidates.append((rank, run))
    if not candidates:
        raise ValueError('no eligible development candidate')
    rank, chosen = max(candidates, key=lambda item: item[0])
    return {'model': chosen['model'], 'length': chosen['length'], 'development_operating_targets_met': bool(rank[0]),
            'selection_accuracy': rank[1], 'selection_basis': PROTOCOL['selection']}


def develop(works, output_dir, *, progress=None):
    out = Path(output_dir)
    runs = []
    for language in sorted({w['language'] for w in works}):
        for length in PROTOCOL['lengths']:
            samples = sample_works([w for w in works if w['language'] == language], length, PROTOCOL['max_samples_per_work'])
            for kind in PROTOCOL['models']:
                if progress:
                    progress(f'Development: {language}, {length} tokens, {kind}')
                controls = []
                for key in ('genre', 'topic'):
                    for value in sorted({s[key] for s in samples} - {'unknown', ''}):
                        control = _closed([s for s in samples if s[key] == value], kind, PROTOCOL['seed'])
                        controls.append({'label': key, 'value': value, **control})
                runs.append({'language': language, 'length': length, 'model': kind,
                             'closed': _closed(samples, kind, PROTOCOL['seed']), 'controls': controls,
                             'open': _open_development(samples, kind, PROTOCOL['seed'])})
    selected = {language: select_candidate([r for r in runs if r['language'] == language])
                for language in sorted({r['language'] for r in runs})}
    frozen = {'protocol': PROTOCOL, 'selected': selected, 'development_corpus': _corpus_identity(works),
              'implementation_sha256': implementation_hashes(),
              'runtime': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn')},
              'authorship_validated': False}
    lock_path = out / 'model_lock.json'
    if lock_path.exists() and json.loads(lock_path.read_text()) != frozen:
        raise ValueError('existing model lock differs; use a new study directory, never overwrite an evaluated design')
    report = {'status': 'development_only', 'protocol': PROTOCOL, 'selected': selected, 'runs': runs}
    _write(out / 'development.json', report)
    _write(lock_path, frozen)
    (out / 'development.md').write_text(render_development(report))
    return report


def seal(model_lock, manifest_paths, output_dir):
    """Lock the model and manifest bytes before any fresh texts are evaluated."""
    from .benchmark_data import read_manifest
    frozen = json.loads(Path(model_lock).read_text())
    if frozen['implementation_sha256'] != implementation_hashes() or frozen['protocol'] != PROTOCOL:
        raise ValueError('implementation changed since model selection; cannot evaluate the reserved corpus')
    if frozen['runtime'] != {p: importlib.metadata.version(p) for p in frozen['runtime']}:
        raise ValueError('runtime changed since model selection')
    previous_authors = {(w['language'], w['author']) for w in frozen['development_corpus']}
    previous_works = {w['work_id'] for w in frozen['development_corpus']}
    manifests, roles, identities = [], {}, set()
    for path in map(Path, manifest_paths):
        manifest = read_manifest(path)
        for w in manifest['works']:
            if w.get('role', 'attribution') != 'attribution':
                continue
            identity = _work_identity(w)
            author = (w['language'], w['author'])
            role = w.get('evaluation_role')
            if role not in ('reference', 'calibration_unknown', 'test_unknown'):
                raise ValueError('every fresh work needs a declared evaluation_role')
            if identity in previous_works or author in previous_authors:
                raise ValueError('fresh evaluation overlaps development authors or works')
            if identity in identities:
                raise ValueError('duplicate canonical fresh work')
            identities.add(identity)
            if author in roles and roles[author] != role:
                raise ValueError('an author cannot cross reference/calibration/test roles')
            roles[author] = role
        manifests.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if not roles:
        raise ValueError('reserved panel has no attribution works')
    for lang in {a[0] for a in roles}:
        if lang not in frozen['selected']:
            raise ValueError('no frozen development model for fresh language')
        for role in ('reference', 'calibration_unknown', 'test_unknown'):
            if sum(a[0] == lang and r == role for a, r in roles.items()) < 2:
                raise ValueError('fresh evaluation needs at least two authors in each role')
    lock = {'model_lock': frozen, 'model_lock_sha256': _digest(frozen), 'manifests': manifests,
            'sealed_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'fresh reference pilot; scripture transfer remains unvalidated'}
    destination = Path(output_dir) / 'evaluation_lock.json'
    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing['model_lock_sha256'] != lock['model_lock_sha256'] or existing['manifests'] != manifests:
            raise ValueError('cannot overwrite a different evaluation lock')
        return existing
    _write(destination, lock)
    return lock


def _final_partitions(works, length):
    samples = sample_works(works, length, PROTOCOL['max_samples_per_work'])
    role_by_id = {_work_identity(w): w['evaluation_role'] for w in works}
    reference = [s for s in samples if role_by_id[s['work_id']] == 'reference']
    eligible, buckets, excluded = work_partitions(reference, PROTOCOL['seed'])
    if len({s['author'] for s in eligible}) < 2:
        raise ValueError('fresh reference panel lacks two authors with three complete independent works')
    known_train = [s for s in eligible if buckets[s['work_id']] == 0]
    known_cal = [s for s in eligible if buckets[s['work_id']] == 1]
    known_test = [s for s in eligible if buckets[s['work_id']] == 2]
    impostors = [s for s in samples if role_by_id[s['work_id']] == 'calibration_unknown']
    unknown = [s for s in samples if role_by_id[s['work_id']] == 'test_unknown']
    if min(len({s['author'] for s in rows}) for rows in (impostors, unknown)) < 2:
        raise ValueError('selected passage length lacks two complete calibration and test unknown authors')
    return {'train': known_train, 'known_calibration': known_cal, 'impostors': impostors,
            'known_test': known_test, 'unknown_test': unknown}, excluded


def _fit_final(parts, kind, seed):
    model = AuthorModel(kind, seed=seed).fit(parts['train'])
    calibration = calibrate_rejection(parts['known_calibration'], model.score(parts['known_calibration']),
                                      parts['impostors'], model.score(parts['impostors']), model.authors_,
                                      max_false_acceptance=PROTOCOL['targets']['unknown_false_acceptance'],
                                      min_known_acceptance=PROTOCOL['targets']['known_acceptance'])
    test = parts['known_test'] + parts['unknown_test']
    scores = model.score(test)
    return _rows(test, scores, model.authors_, apply_rejection(scores, calibration)), calibration, model.metadata_


def _resample_works(rows, rng, *, resample_authors=False):
    by_author = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_author[row['author']][row['work_id']].append(row)
    sampled = []
    authors = sorted(by_author)
    draws = rng.integers(len(authors), size=len(authors)) if resample_authors else range(len(authors))
    for author_copy, author_index in enumerate(draws):
        author = authors[author_index]
        works = by_author[author]
        keys = sorted(works)
        for copy, index in enumerate(rng.integers(len(keys), size=len(keys))):
            sampled.extend({**r, 'author': f'bootstrap-author:{author_copy}:{author}' if resample_authors else author,
                            'work_id': f"bootstrap:{author_copy}:{copy}:{r['work_id']}"} for r in works[keys[index]])
    return sampled


def _refit_intervals(parts, kind, point, *, repeats, seed, progress=None):
    rng = np.random.default_rng(seed)
    values = defaultdict(list)
    for rep in range(repeats):
        sampled = {key: _resample_works(rows, rng, resample_authors=key in ('impostors', 'unknown_test')) for key, rows in parts.items()}
        rows, _, _ = _fit_final(sampled, kind, seed + rep + 1)
        metrics = _metrics(rows)
        for key, metric in metrics.items():
            if metric['value'] is not None:
                values[key].append(metric['value'])
        if progress and (rep + 1) % 10 == 0:
            progress(f'Completed {rep + 1}/{repeats} whole-work resamples with model and calibration refitted')
    result = {}
    singleton_strata = [{'role': role, 'author': author} for role, rows in parts.items()
                        for author in sorted({r['author'] for r in rows})
                        if len({r['work_id'] for r in rows if r['author'] == author}) < 2]
    for key, metric in point.items():
        estimate = metric['value']
        interval = None
        if estimate is not None and len(values[key]) == repeats:
            lo, hi = np.quantile(values[key], [.025, .975])
            n = metric['independent_authors'] if key == 'unknown_false_acceptance' else metric['independent_works']
            if estimate == 0:
                hi = max(hi, beta.ppf(.975, 1, n))
            if estimate == 1:
                lo = min(lo, beta.ppf(.025, n, 1))
            interval = [float(lo), float(hi)]
        result[key] = {**metric, 'ci95': None if singleton_strata else interval,
                       'resampling_range95': interval,
                       'ci_method': 'Conditional work bootstrap with full refits and unfamiliar-author block resampling; known candidate authors fixed. Unavailable with singleton author/role strata.',
                       'singleton_strata': singleton_strata,
                       'refit_replicates': repeats, 'usable_replicates': len(values[key])}
    return result


def evaluate(evaluation_lock, cache_dir, output_dir, *, download=False, progress=None):
    from .benchmark_data import load_benchmark
    from .benchmark import gate_result
    lock = json.loads(Path(evaluation_lock).read_text())
    frozen = lock['model_lock']
    if (_digest(frozen) != lock['model_lock_sha256'] or frozen['implementation_sha256'] != implementation_hashes()
            or frozen['protocol'] != PROTOCOL):
        raise ValueError('model or implementation changed after locking')
    if frozen['runtime'] != {p: importlib.metadata.version(p) for p in frozen['runtime']}:
        raise ValueError('runtime changed after locking')
    works = []
    for item in lock['manifests']:
        path = Path(item['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('reserved manifest changed after locking')
        works.extend(load_benchmark(path, cache_dir, download=download))
    if {w['text_sha256'] for w in frozen['development_corpus']} & {hashlib.sha256(w['text_bare'].encode()).hexdigest() for w in works}:
        raise ValueError('fresh texts duplicate development material')
    runs = []
    for language in sorted({w['language'] for w in works}):
        chosen = frozen['selected'][language]
        if progress:
            progress(f"Fresh evaluation: {language}, frozen {chosen['model']}, {chosen['length']} tokens")
        subset = [w for w in works if w['language'] == language]
        parts, excluded = _final_partitions(subset, chosen['length'])
        rows, calibration, metadata = _fit_final(parts, chosen['model'], PROTOCOL['seed'])
        metrics = _refit_intervals(parts, chosen['model'], _metrics(rows), repeats=PROTOCOL['final_refit_replicates'], seed=PROTOCOL['seed'], progress=progress)
        maxima = {'unknown_false_acceptance', 'known_wrong_acceptance'}
        gates = {key: gate_result(metrics[key], {'maximum' if key in maxima else 'minimum': value})
                 for key, value in PROTOCOL['targets'].items()}
        point_targets = {key: (metrics[key]['value'] is not None and
                              (metrics[key]['value'] <= value if key in maxima else metrics[key]['value'] >= value))
                         for key, value in PROTOCOL['targets'].items()}
        reference_rows = parts['train'] + parts['known_calibration'] + parts['known_test']
        adequate = (len({r['author'] for r in reference_rows}) >= 8 and len({r['work_id'] for r in reference_rows}) >= 60
                    and len({r['author'] for r in parts['unknown_test']}) >= 8
                    and len({r['work_id'] for r in parts['unknown_test']}) >= 60)
        controls = []
        for label in ('genre', 'topic'):
            for value in sorted({r[label] for r in parts['known_test']} - {'unknown', ''}):
                matching_authors = {a for a in {r['author'] for r in parts['train']}
                                    if {r[label] for r in parts['train'] if r['author'] == a} == {value}}
                matched_parts = {key: ([r for r in rs if r['author'] in matching_authors and r[label] == value]
                                      if key in ('train', 'known_calibration', 'known_test') else rs)
                                 for key, rs in parts.items()}
                if (len(matching_authors) >= 2 and
                        all({r['author'] for r in matched_parts[k]} == matching_authors for k in ('train', 'known_calibration', 'known_test'))):
                    matched_rows, _, _ = _fit_final(matched_parts, chosen['model'], PROTOCOL['seed'])
                    matched_metrics = _refit_intervals(matched_parts, chosen['model'], _metrics(matched_rows),
                                                     repeats=PROTOCOL['final_refit_replicates'], seed=PROTOCOL['seed'])
                    controls.append({'label': label, 'value': value, 'candidate_authors': sorted(matching_authors),
                                     'accuracy': matched_metrics['accuracy'], 'metrics': matched_metrics,
                                     'interpretation': 'Same frozen model design refitted exclusively on genre/topic-matched reference works; independent diagnostic, no reselection'})
        matched = [c for c in controls if c['label'] == 'genre']
        gates['matched_genre_accuracy'] = ('inconclusive' if not matched else
                                         'failed' if any(gate_result(c['accuracy'], {'minimum': .8}) == 'failed' for c in matched) else
                                         'passed' if all(gate_result(c['accuracy'], {'minimum': .8}) == 'passed' for c in matched) else 'inconclusive')
        point_targets['matched_genre_accuracy'] = bool(matched) and all(c['accuracy']['value'] >= .8 for c in matched)
        status = ('failed' if not all(point_targets[k] for k in PROTOCOL['targets']) or 'failed' in gates.values() else
                  'passed' if adequate and set(gates.values()) == {'passed'} else 'inconclusive')
        runs.append({'language': language, 'selected': chosen, 'status': status, 'adequate_coverage': adequate,
                     'metrics': metrics, 'gates': gates, 'observed_targets_met': point_targets,
                     'calibration': calibration, 'model_metadata': metadata,
                     'partitions': {key: sorted({r['work_id'] for r in rs}) for key, rs in parts.items()},
                     'counts': {key: {'authors': len({r['author'] for r in rs}), 'works': len({r['work_id'] for r in rs}), 'passages': len(rs)} for key, rs in parts.items()},
                     'controls': controls, 'exclusions': excluded, 'predictions': rows})
    report = {'status': 'failed' if any(r['status'] == 'failed' for r in runs) else 'passed' if runs and all(r['status'] == 'passed' for r in runs) else 'inconclusive',
              'scripture_attribution_validated': False, 'evaluation_lock_sha256': _digest(lock),
              'evaluated_utc': datetime.now(timezone.utc).isoformat(), 'runs': runs,
              'limitations': ['Fresh evaluation is conditional on conventional catalogue attributions and a small author panel.',
                              'Refit intervals resample unfamiliar authors as blocks and whole works within roles; the known candidate panel is fixed. Forty replicates are a rough pilot, not calibrated population uncertainty.',
                              'Singleton work strata cannot supply between-work uncertainty; small calibration and unknown panels cannot establish a 5% population error rate.',
                              'This dataset becomes exposed after evaluation. New tuning requires another final test set.',
                              'Scripture transfer, Arabic, Biblical Hebrew and AI profiles remain unvalidated.']}
    _write(Path(output_dir) / 'fresh_evaluation.json', report)
    Path(output_dir, 'fresh_evaluation.md').write_text(render_final(report))
    return report


def _pct(value):
    return 'Unavailable' if value is None else f'{value:.1%}'


def render_development(report):
    lines = ['# Passage model development', '', '**Development data only; no authorship validation claim.**', '',
             '| Language | Words | Model | All-author accuracy | Matched-genre accuracy | Unknown accepted | Known accepted |', '|---|---:|---|---:|---:|---:|---:|']
    for run in report['runs']:
        closed = run['closed'].get('accuracy', {}).get('value')
        matched = [c['accuracy']['value'] for c in run['controls'] if c['label'] == 'genre' and c['available']]
        opened = run['open'].get('metrics', {})
        lines.append(f"| {run['language']} | {run['length']} | {run['model']} | {_pct(closed)} | {_pct(float(np.mean(matched)) if matched else None)} | {_pct(opened.get('unknown_false_acceptance', {}).get('value'))} | {_pct(opened.get('known_acceptance', {}).get('value'))} |")
    lines += ['', 'Selected designs (locked before fresh evaluation):', '']
    for language, selected in report['selected'].items():
        lines.append(f"- {language}: {selected['model']}, {selected['length']} tokens; development operating targets met = {selected['development_operating_targets_met']}.")
    lines += ['', 'Selection uses matched-genre accuracy after preferring feasible rejection. Zero false acceptance with zero known acceptance is not success. Function-word order and suffixes are structural proxies, not validated syntactic or morphological annotation. All rates are author/work balanced. Detailed matched-topic controls and independent work partitions are in `development.json`.', '']
    return '\n'.join(lines)


def render_final(report):
    lines = ['# Fresh-author evaluation', '', f"**Empirical result: {report['status'].upper()}. Scripture attribution remains unvalidated.**", '',
             'Model family and passage length were frozen on the old development corpus before this independent author panel was scored. Thresholds use two separate unfamiliar calibration authors; final unknown authors enter neither fitting nor calibration.', '',
             '| Language | Model | Tokens | Author accuracy | Unknown accepted | Known accepted | Result |', '|---|---|---:|---:|---:|---:|---|']
    for run in report['runs']:
        metrics = run['metrics']
        lines.append(f"| {run['language']} | {run['selected']['model']} | {run['selected']['length']} | " + ' | '.join(_pct(metrics[k]['value']) for k in ('accuracy', 'unknown_false_acceptance', 'known_acceptance')) + f" | {run['status']} |")
        lines += ['', f"Calibration feasible = {run['calibration']['feasible']}; sufficient coverage for a broad pass = {run['adequate_coverage']}. Known-author accuracy measures the nearest candidate before rejection; useful correct acceptance is {_pct(metrics['known_correct_acceptance']['value'])}.", '']
    lines += ['## Interpretation', '', 'Unknown acceptance and useful coverage must improve together. Abstaining from all texts does not solve attribution. Resampling repeats feature fitting, model fitting and calibration, resamples unfamiliar authors as blocks, and preserves the known candidate panel. Author/role strata containing only one work make confidence intervals unavailable; diagnostic resampling ranges remain in the JSON. Boundary guards for unfamiliar-author acceptance count authors, not chunks or works.', '']
    lines += ['- ' + item for item in report['limitations']]
    lines += ['', 'Full predictions, thresholds, work partitions, matched-genre/topic diagnostics and refit intervals are in `fresh_evaluation.json`.', '']
    return '\n'.join(lines)
