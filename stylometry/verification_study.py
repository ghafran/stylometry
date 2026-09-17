"""Author-disjoint development of a learned, cross-work authorship verifier.

All current reference sources are exposed development data. This module neither
changes the frozen v1 study nor claims a new independent validation result.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import re
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

from .author_models import AuthorModel
from .benchmark import sample_works
from .pair_verifier import PairVerifier
from .verification_gallery import (
    apply_gallery_rejection, build_gallery_scores, calibrate_gallery,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFESTS = tuple(ROOT / 'benchmarks' / name for name in (
    'greek_manifest.json', 'additional_manifest.json',
    'greek_fresh_manifest.json', 'hebrew_fresh_manifest.json',
))
PROTOCOL = {
    'version': 'cross-work-pair-verification-development-v1',
    'seed': 42, 'passage_tokens': 500, 'max_samples_per_work': 8,
    'candidates_per_gallery': 2, 'reference_works_per_candidate': 2,
    'unfamiliar_authors_per_panel': 2,
    'folds': 'ceil(eligible candidate authors / 2); deterministic rotating test pairs, next pair for calibration',
    'author_roles': 'Verifier training, calibration identities and test identities strictly disjoint within each fold; roles may rotate across development folds',
    'work_roles': 'Two canonical works per candidate are references; all other candidate works are queries. No reference work is a query.',
    'unfamiliar_selection': 'Prefer authors with fewer than three works, then seeded identity hash; rotate four selected unfamiliar authors across calibration and test',
    'pair_model': 'Fixed symmetric cross-work verifier; training-only static/character features; balanced canonical work-pairs; equal same-genre/other negative strata when both exist; total fitting weight equals canonical training work count',
    'gallery_score': 'Minimum of per-reference-work mean pair scores; threshold calibrated on the complete two-candidate search',
    'targets': {'known_correct_acceptance': .80, 'known_wrong_acceptance': .05,
                'unknown_false_acceptance': .05},
    'baseline_models': {'grc': 'function_centroid', 'hbo': 'char_linear', 'arb': 'function_centroid'},
    'baseline_scope': 'Frozen v1 model recipe independently refitted on calibration and test reference galleries; a transferred threshold, not a common fitted feature space',
    'genre_control': 'Repeat with all roles restricted to one normalized genre only when author/work coverage permits; identical panels are not independent replications',
    'uncertainty': 'No population confidence interval: overlapping development folds and small author panels. Author/work-balanced point rates are descriptive.',
    'advance_rule': 'Preliminary signal only if all folds calibrate and all per-fold and pooled targets pass, including available per-fold and pooled matched-genre unfamiliar subsets; independent final validation still required',
    'scope': 'Exposed-corpus development only; two-candidate searches; scripture, named historical authorship and larger galleries remain unvalidated',
}
SOURCE_FILES = (
    'verification_study.py', 'pair_verifier.py', 'verification_gallery.py',
    'author_models.py', 'benchmark.py', 'features.py', 'lang.py', 'greek.py',
    'benchmark_data.py', 'benchmark_additional.py',
)
ROLE_KEYS = ('train', 'calibration_gallery', 'calibration_known', 'calibration_unknown',
             'test_gallery', 'test_known', 'test_unknown')


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _label(value):
    return re.sub(r'\W+', '_', (value or 'unknown').strip().lower()).strip('_') or 'unknown'


def _check_author_identities(works):
    """Fail on observed aliases; missing authority data is explicitly recorded."""
    identity_names, name_identities = {}, defaultdict(lambda: defaultdict(set))
    missing = set()
    for work in works:
        author = (work['language'], work['author'])
        identities = []
        canonical = str(work.get('work_id', ''))
        greek = re.match(r'^(tlg\d+)\.', canonical)
        publisher = re.search(r'(?:^|/)html/(p\d+)/', str(work.get('source_path', '')))
        authority = re.search(r'(Q\d+)(?:/)?$', str(work.get('author_authority', '')))
        if greek:
            identities.append(('cts_textgroup', greek.group(1)))
        if publisher:
            identities.append(('benyehuda_author', publisher.group(1)))
        if authority:
            identities.append(('wikidata', authority.group(1)))
        if not identities:
            missing.add(author)
        for kind, value in identities:
            key = (work['language'], kind, value)
            if key in identity_names and identity_names[key] != author:
                raise ValueError('author identity alias crosses display labels; curate canonical identities before splitting')
            identity_names[key] = author
            name_identities[author][kind].add(value)
            if len(name_identities[author][kind]) > 1:
                raise ValueError('one author label maps to conflicting canonical identities')
    return [{'language': lang, 'author': author, 'reason': 'no machine-readable author authority; display label requires independent identity curation'}
            for lang, author in sorted(missing)]


def prepare_samples(works):
    works = list(works)
    selected, exclusions = [], []
    for work in works:
        if work.get('role', 'attribution') != 'attribution':
            exclusions.append({'work_id': work.get('work_id'), 'reason': 'non-attribution witness/edition control'})
            continue
        if any(not isinstance(work.get(key), str) or not work[key].strip()
               for key in ('author', 'language', 'work', 'work_id', 'text_bare')):
            raise ValueError('reference works need canonical work_id, author, language, title and text')
        if work['language'] not in ('grc', 'hbo', 'arb'):
            raise ValueError('unsupported reference language')
        selected.append(work)
    _check_author_identities(selected)
    # This also rejects duplicate canonical works and duplicate normalized full texts.
    samples = sample_works(selected, PROTOCOL['passage_tokens'], PROTOCOL['max_samples_per_work'])
    present = {row['work_id'] for row in samples}
    for work in selected:
        if f"{work['language']}:{work['work_id']}" not in present:
            exclusions.append({'work_id': work['work_id'], 'author': work['author'],
                               'reason': 'no complete passage at the fixed length'})
    fingerprints = {}
    for row in samples:
        row['genre'], row['topic'] = _label(row['genre']), _label(row['topic'])
        key = (row['language'], _hash(' '.join(row['text_bare'].split())))
        if key in fingerprints and fingerprints[key] != row['work_id']:
            raise ValueError('identical normalized passages occur across canonical works; inspect textual reuse before splitting')
        fingerprints[key] = row['work_id']
    return samples, exclusions


def _validate_samples(samples):
    if not samples or len({s['language'] for s in samples}) != 1:
        raise ValueError('panels need nonempty samples from one language')
    works, ids = {}, set()
    for row in samples:
        if any(not isinstance(row.get(key), str) or not row[key].strip()
               for key in ('id', 'author', 'work_id', 'text_bare', 'language')):
            raise ValueError('invalid passage metadata')
        if row['id'] in ids:
            raise ValueError('duplicate passage ID')
        ids.add(row['id'])
        if row['work_id'] in works and works[row['work_id']] != row['author']:
            raise ValueError('canonical work has conflicting author labels')
        works[row['work_id']] = row['author']


def build_panels(samples, seed=42, genre=None):
    """Split authors before any pair creation or feature fitting."""
    samples = list(samples)
    _validate_samples(samples)
    samples.sort(key=lambda row: (row['author'], row['work_id'], row['id']))
    if genre is not None:
        samples = [s for s in samples if _label(s.get('genre')) == _label(genre)]
    by_author = defaultdict(set)
    for row in samples:
        by_author[row['author']].add(row['work_id'])
    eligible = sorted((a for a, ws in by_author.items() if len(ws) >= 3),
                      key=lambda a: _hash(f'{seed}:candidate:{a}'))
    if len(eligible) < 4 or len(by_author) < 10:
        raise ValueError('insufficient author/work coverage: need four candidate authors with >=3 works, four separate unfamiliar authors and >=2 verifier-training authors')
    panels = []
    for fold in range(math.ceil(len(eligible) / 2)):
        known_test = {eligible[(2 * fold + offset) % len(eligible)] for offset in (0, 1)}
        known_cal = {eligible[(2 * fold + offset) % len(eligible)] for offset in (2, 3)}
        remaining = set(by_author) - known_test - known_cal
        unfamiliar = sorted(remaining, key=lambda a: (len(by_author[a]) >= 3, _hash(f'{seed}:unfamiliar:{a}')))[:4]
        unfamiliar = unfamiliar[fold % 4:] + unfamiliar[:fold % 4]
        unknown_cal, unknown_test = set(unfamiliar[:2]), set(unfamiliar[2:])
        training = remaining - unknown_cal - unknown_test
        excluded = [{'author': a, 'works': sorted(by_author[a]), 'reason': 'fewer than two independent works for verifier training'}
                    for a in sorted(training) if len(by_author[a]) < 2]
        training = {a for a in training if len(by_author[a]) >= 2}
        if len(training) < 2:
            raise ValueError('insufficient remaining authors with two works for verifier training')
        panel = {'fold': fold, 'genre': _label(genre) if genre else None, 'excluded': excluded,
                 'train': [r for r in samples if r['author'] in training],
                 'calibration_unknown': [r for r in samples if r['author'] in unknown_cal],
                 'test_unknown': [r for r in samples if r['author'] in unknown_test]}
        for stage, authors in (('calibration', known_cal), ('test', known_test)):
            gallery_ids = set()
            for author in sorted(authors):
                ids = sorted(by_author[author], key=lambda w: _hash(f'{seed}:reference:{w}'))
                gallery_ids.update(ids[:2])
            panel[f'{stage}_gallery'] = [r for r in samples if r['work_id'] in gallery_ids]
            panel[f'{stage}_known'] = [r for r in samples if r['author'] in authors and r['work_id'] not in gallery_ids]
        _assert_panel(panel)
        panels.append(panel)
    return panels


def _assert_panel(panel):
    training_authors = {r['author'] for r in panel['train']}
    if len(training_authors) < 2 or any(len({r['work_id'] for r in panel['train'] if r['author'] == author}) < 2
                                       for author in training_authors):
        raise ValueError('verifier training needs at least two authors with two independent works each')
    domains = [{r['author'] for key in keys for r in panel[key]} for keys in (
        ('train',), ('calibration_gallery', 'calibration_known', 'calibration_unknown'),
        ('test_gallery', 'test_known', 'test_unknown'))]
    if any(a & b for i, a in enumerate(domains) for b in domains[i + 1:]):
        raise ValueError('author leakage between verifier training, calibration and test domains')
    work_roles = [{r['work_id'] for r in panel[key]} for key in ROLE_KEYS]
    if any(a & b for i, a in enumerate(work_roles) for b in work_roles[i + 1:]):
        raise ValueError('canonical work leakage between panel roles')
    for stage in ('calibration', 'test'):
        gallery = panel[f'{stage}_gallery']
        known = panel[f'{stage}_known']
        unknown = panel[f'{stage}_unknown']
        authors = {r['author'] for r in gallery}
        if len(authors) != 2 or {r['author'] for r in known} != authors:
            raise ValueError('each panel requires two reference candidates and separate known query works')
        if len({r['author'] for r in unknown}) != 2 or authors & {r['author'] for r in unknown}:
            raise ValueError('each panel requires two separate unfamiliar authors')
        if any(len({r['work_id'] for r in gallery if r['author'] == a}) != 2 for a in authors):
            raise ValueError('exactly two canonical reference works required per candidate')


def _count(rows):
    return {'authors': len({r['author'] for r in rows}), 'works': len({r['work_id'] for r in rows}), 'passages': len(rows)}


def _rate(rows, key):
    # Repeated folds are averaged inside work; they are not new independent works.
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        grouped[row['author']][row['work_id']][row.get('fold', 0)].append(Fraction(str(float(row[key]))))
    def mean(values):
        return sum(values, Fraction()) / len(values)
    author_means = [mean([mean([mean(vals) for vals in episodes.values()])
                         for episodes in works.values()]) for works in grouped.values()]
    return {'value': float(mean(author_means)) if author_means else None,
            'authors': len(grouped), 'works': sum(len(ws) for ws in grouped.values()),
            'ci95': None, 'ci_method': PROTOCOL['uncertainty']}


def metrics(rows):
    known, unknown = [r for r in rows if r['known']], [r for r in rows if not r['known']]
    return {'accuracy': _rate(known, 'correct'),
            'known_acceptance': _rate(known, 'accepted'),
            'known_correct_acceptance': _rate(known, 'correct_accepted'),
            'known_wrong_acceptance': _rate(known, 'wrong_accepted'),
            'unknown_false_acceptance': _rate(unknown, 'accepted'),
            'accuracy_among_accepted': _rate([r for r in rows if r['accepted']], 'correct')}


def _targets_met(values):
    return all(values[k]['value'] is not None and
               (values[k]['value'] >= target if k == 'known_correct_acceptance' else values[k]['value'] <= target)
               for k, target in PROTOCOL['targets'].items())


def _records(rows, scores, authors, calibration, gallery, fold, evidence=None):
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(rows), len(authors)) or not np.isfinite(scores).all():
        raise ValueError('one finite score per query and candidate is required')
    if evidence is not None and len(evidence) != len(rows):
        raise ValueError('reference evidence must align with query rows')
    accepted = apply_gallery_rejection(scores, calibration)
    reference_genres = {r['genre'] for r in gallery} - {'unknown'}
    reference_topics = {r['topic'] for r in gallery} - {'unknown'}
    result = []
    for index, (row, values) in enumerate(zip(rows, scores)):
        winner = authors[int(np.argmax(values))]
        known = row['author'] in authors
        ambiguous = bool(np.count_nonzero(values == values.max()) > 1)
        correct = known and winner == row['author'] and not ambiguous
        result.append({**{k: row[k] for k in ('id', 'author', 'work_id', 'genre', 'topic')},
                       'fold': fold, 'known': known, 'correct': correct,
                       'ambiguous_top_score': ambiguous,
                       'accepted': bool(accepted[index]), 'correct_accepted': bool(accepted[index] and correct),
                       'wrong_accepted': bool(accepted[index] and not correct),
                       'nearest_reference_author': winner, 'decision': winner if accepted[index] else None,
                       'candidate_scores': dict(zip(authors, map(float, values))),
                       'shares_reference_genre': row['genre'] in reference_genres,
                       'shares_reference_topic': row['topic'] in reference_topics,
                       **({'reference_evidence': evidence[index]} if evidence is not None else {})})
    return result


def _calibrate(known, known_scores, unknown, unknown_scores, authors):
    return calibrate_gallery(known, known_scores, unknown, unknown_scores, authors,
                             max_false_acceptance=PROTOCOL['targets']['unknown_false_acceptance'],
                             min_correct_acceptance=PROTOCOL['targets']['known_correct_acceptance'],
                             max_wrong_acceptance=PROTOCOL['targets']['known_wrong_acceptance'])


def _run_panel(panel, language):
    _assert_panel(panel)
    verifier = PairVerifier(seed=PROTOCOL['seed']).fit(panel['train'])
    cal_rows = panel['calibration_known'] + panel['calibration_unknown']
    test_rows = panel['test_known'] + panel['test_unknown']
    n_known = len(panel['calibration_known'])
    cal = build_gallery_scores(verifier, panel['calibration_gallery'], cal_rows)
    calibration = _calibrate(panel['calibration_known'], cal['scores'][:n_known],
                             panel['calibration_unknown'], cal['scores'][n_known:], cal['authors'])
    # Frozen verifier and threshold now transfer to entirely different identities.
    test = build_gallery_scores(verifier, panel['test_gallery'], test_rows)
    # The decision rule depends on candidate count, not the names used to calibrate it.
    transferred = {**calibration, 'candidate_authors': test['authors']}
    pair_rows = _records(test_rows, test['scores'], test['authors'], transferred,
                         panel['test_gallery'], panel['fold'], test['evidence'])
    pair_cal_rows = _records(cal_rows, cal['scores'], cal['authors'], calibration,
                             panel['calibration_gallery'], panel['fold'], cal['evidence'])
    kind = PROTOCOL['baseline_models'][language]
    base_cal = AuthorModel(kind, seed=PROTOCOL['seed']).fit(panel['calibration_gallery'])
    base_scores = base_cal.score(cal_rows)
    base_threshold = _calibrate(panel['calibration_known'], base_scores[:n_known],
                                panel['calibration_unknown'], base_scores[n_known:], base_cal.authors_)
    base_test = AuthorModel(kind, seed=PROTOCOL['seed']).fit(panel['test_gallery'])
    base_rows = _records(test_rows, base_test.score(test_rows), base_test.authors_,
                         {**base_threshold, 'candidate_authors': base_test.authors_},
                         panel['test_gallery'], panel['fold'])
    return {'fold': panel['fold'], 'genre': panel['genre'], 'exclusions': panel['excluded'],
            'counts': {key: _count(panel[key]) for key in ROLE_KEYS},
            'partitions': {key: {'authors': sorted({r['author'] for r in panel[key]}),
                                 'works': sorted({r['work_id'] for r in panel[key]})} for key in ROLE_KEYS},
            'pair_verifier': {'model_metadata': verifier.metadata_, 'calibration': calibration,
                              'calibration_predictions': pair_cal_rows,
                              'metrics': metrics(pair_rows), 'predictions': pair_rows},
            'baseline': {'model': kind, 'calibration': base_threshold,
                         'calibration_model_metadata': base_cal.metadata_, 'test_model_metadata': base_test.metadata_,
                         'calibration_predictions': _records(cal_rows, base_scores, base_cal.authors_, base_threshold,
                                                             panel['calibration_gallery'], panel['fold']),
                         'metrics': metrics(base_rows), 'predictions': base_rows}}


def _summarize(folds, method):
    rows = [r for f in folds for r in f[method]['predictions']]
    values = metrics(rows)
    subsets = {label: _rate([r for r in rows if not r['known'] and r[key]], 'accepted')
               for label, key in (('genre_matched_unfamiliar', 'shares_reference_genre'),
                                  ('topic_matched_unfamiliar', 'shares_reference_topic'))}
    calibrations = sum(bool(f[method]['calibration']['feasible']) for f in folds)
    matched = subsets['genre_matched_unfamiliar']
    fold_matched = [_rate([r for r in f[method]['predictions'] if not r['known'] and r['shares_reference_genre']], 'accepted')
                    for f in folds]
    preliminary = (bool(folds) and calibrations == len(folds) and _targets_met(values)
                   and all(_targets_met(f[method]['metrics']) for f in folds)
                   and matched['authors'] >= 2 and matched['value'] is not None
                   and matched['value'] <= PROTOCOL['targets']['unknown_false_acceptance']
                   and all(s['value'] is None or s['value'] <= PROTOCOL['targets']['unknown_false_acceptance'] for s in fold_matched))
    return {'metrics': values, 'matched_unfamiliar_subsets': subsets,
            'calibration_feasible_folds': calibrations, 'folds': len(folds),
            'observed_pooled_targets_met': _targets_met(values),
            'preliminary_development_signal': preliminary,
            'per_fold_genre_matched_unfamiliar': fold_matched,
            'interpretation': 'Subset labels mean the unknown query shares a label with at least one gallery work; they do not make the whole experiment topic-controlled.'}


def _corpus_lock(works):
    return [{'author': w['author'], 'work_id': w['work_id'], 'language': w['language'],
             'genre': w.get('genre'), 'topic': w.get('topic'),
             'author_authority': w.get('author_authority'), 'source_path': w.get('source_path'),
             'text_sha256': _hash(' '.join(w['text_bare'].split())), 'provenance': w.get('provenance', {})}
            for w in sorted(works, key=lambda w: (w['language'], w['author'], w['work_id']))
            if w.get('role', 'attribution') == 'attribution']


def run_study(works, output_dir, *, progress=None):
    works = list(works)
    samples, exclusions = prepare_samples(works)
    plans, unavailable = [], []
    for language in sorted({r['language'] for r in samples}):
        language_rows = [r for r in samples if r['language'] == language]
        for genre in [None] + sorted({r['genre'] for r in language_rows} - {'unknown'}):
            if genre and all(r['genre'] == genre for r in language_rows):
                unavailable.append({'language': language, 'genre': genre, 'reason': 'identical to main panel; no independent replication'})
                continue
            try:
                panels = build_panels(language_rows, PROTOCOL['seed'], genre=genre)
            except ValueError as exc:
                unavailable.append({'language': language, 'genre': genre, 'reason': str(exc)})
                continue
            plans.append({'language': language, 'genre': genre, 'panels': panels})
    if not plans:
        raise ValueError('no language supports disjoint verifier training, calibration and test panels')
    lock = {'protocol': PROTOCOL, 'corpus': _corpus_lock(works),
            'implementation_sha256': {name: hashlib.sha256((ROOT / 'stylometry' / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
            'runtime': {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'lxml')}},
            'panels': [{'language': plan['language'], 'genre': plan['genre'],
                        'folds': [{'fold': p['fold'], 'roles': {key: sorted({r['work_id'] for r in p[key]}) for key in ROLE_KEYS}}
                                  for p in plan['panels']]} for plan in plans],
            'source_exclusions': exclusions, 'unavailable_controls': unavailable,
            'missing_author_authorities': _check_author_identities([w for w in works if w.get('role', 'attribution') == 'attribution']),
            'independent_validation': False, 'scripture_attribution_validated': False}
    out = Path(output_dir)
    lock_path = out / 'development_lock.json'
    if any((out / name).exists() for name in ('model_lock.json', 'evaluation_lock.json', 'suite.json')):
        raise ValueError('output directory belongs to another frozen study; choose a separate directory')
    if not lock_path.exists() and any((out / name).exists() for name in ('development.json', 'development.md')):
        raise ValueError('existing development artifacts lack this study lock; choose a separate directory')
    if lock_path.exists():
        existing = json.loads(lock_path.read_text())
        if {k: v for k, v in existing.items() if k != 'sealed_utc'} != lock:
            raise ValueError('existing development lock differs; use a new output directory')
        lock = existing
    else:
        lock['sealed_utc'] = datetime.now(timezone.utc).isoformat()
        _write(lock_path, lock)  # Written before model fitting or performance evaluation.
    runs = []
    for plan in plans:
        folds = []
        for panel in plan['panels']:
            if progress:
                progress(f"{plan['language']} / {plan['genre'] or 'all genres'}: author-disjoint fold {panel['fold'] + 1}/{len(plan['panels'])}")
            folds.append(_run_panel(panel, plan['language']))
        summary = {method: _summarize(folds, method) for method in ('pair_verifier', 'baseline')}
        pair_rate = summary['pair_verifier']['metrics']['known_correct_acceptance']['value']
        base_rate = summary['baseline']['metrics']['known_correct_acceptance']['value']
        runs.append({'language': plan['language'], 'genre': plan['genre'], 'candidate_count': 2,
                     'coverage': _count([r for r in samples if r['language'] == plan['language'] and (not plan['genre'] or r['genre'] == plan['genre'])]),
                     'summary': summary, 'correct_acceptance_difference_vs_baseline': pair_rate - base_rate,
                     'folds': folds})
    report = {'status': 'development_only', 'independent_validation': False,
              'scripture_attribution_validated': False, 'protocol': PROTOCOL,
              'requested_languages': sorted({w['language'] for w in works if w.get('role', 'attribution') == 'attribution'}),
              'development_lock_sha256': hashlib.sha256(lock_path.read_bytes()).hexdigest(),
              'completed_utc': datetime.now(timezone.utc).isoformat(),
              'unavailable_controls': unavailable, 'source_exclusions': exclusions, 'runs': runs,
              'limitations': [
                  'All source corpora were previously exposed; this is development, not a fresh confirmation.',
                  'Two-candidate searches cannot validate four-candidate or unrestricted attribution.',
                  'Rotating folds reuse authors and works. Rates balance author/work/fold, and no population confidence interval is supplied.',
                  'Only two unfamiliar authors calibrate each fold; candidate thresholds do not certify population error rates.',
                  'Shared genre/topic labels and exact duplicate checks do not remove quotations, editorial effects or all content shortcuts.',
                  'Classical Greek and modern Hebrew do not establish scripture transfer, Biblical Hebrew, Arabic or named historical identities.',
              ]}
    _write(out / 'development.json', report)
    (out / 'development.md').write_text(render_report(report))
    return report


def _pct(value):
    return 'Unavailable' if value is None else f'{100 * value:.1f}%'


def render_report(report):
    lines = ['# Cross-work pair-verification development', '',
             '**Development only. No independent authorship or scripture validation.**', '',
             'Verifier training authors, calibration authors and testing authors are disjoint within each fold. Each candidate has two independent reference works; the whole two-candidate search is calibrated. All previous corpora are now exposed development data.', '',
             '| Language / control | Method | Nearest-candidate accuracy | Correct known acceptance | Wrong known acceptance | Unfamiliar acceptance | Feasible calibration folds |',
             '|---|---|---:|---:|---:|---:|---:|']
    for run in report['runs']:
        for method, summary in run['summary'].items():
            m = summary['metrics']
            lines.append(f"| {run['language']} / {run['genre'] or 'all genres'} | {method} | {_pct(m['accuracy']['value'])} | {_pct(m['known_correct_acceptance']['value'])} | {_pct(m['known_wrong_acceptance']['value'])} | {_pct(m['unknown_false_acceptance']['value'])} | {summary['calibration_feasible_folds']}/{summary['folds']} |")
    lines += ['', 'Zero false acceptance with zero correct acceptance is a failure to provide useful attribution. Candidate calibration rates for infeasible thresholds are saved separately from the active abstention decisions.', '',
              'The baseline repeats the old fixed model recipe on these same galleries and queries. Its features/models are independently fitted to each calibration/test gallery; its threshold transfers between them. The pair verifier instead keeps its feature/model fit fixed while transferring to both galleries. These are deliberately different strategies, not identical training regimes.', '',
              '## Comparable-genre controls', '']
    for item in report['unavailable_controls']:
        lines.append(f"- {item['language']} / {item['genre'] or 'main'}: {item['reason']}.")
    lines += ['', 'Unknown-query subsets sharing a genre/topic label with a gallery reference are in the JSON, with author/work counts. They are narrower diagnostics, not fully controlled replications.', '', '## Limits', '']
    lines += ['- ' + limitation for limitation in report['limitations']]
    lines += ['', 'The pre-fit protocol, source hashes, runtime and partitions are in `development_lock.json`. All raw decision scores, work-level reference evidence, candidate calibration thresholds and final abstentions are in `development.json`. Scores are not historical authorship probabilities.', '']
    return '\n'.join(lines)
