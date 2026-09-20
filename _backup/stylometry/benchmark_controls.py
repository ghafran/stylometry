"""Run the app itself on reference texts and measure explicit perturbations.

A single known writer can have multiple styles. Splitting that writer is reported
as an authorship-interpretation risk, not as proof that style clustering is broken.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, normalize

from .cluster import run as cluster_run
from .features import LexicalFeatureExtractor
from .lang import tokenize

CONTROL_PROTOCOL = {
    "version": "reference-text-controls-v2-work-balanced-centroids",
    "cluster_unit_tokens": [25, 500],
    "max_contiguous_units_per_work": 20,
    "smoothing_alphas": [0.0, 0.7],
    "seed": 42,
    "stress_sample_tokens": 500,
    "stress_deletion_fraction": 0.02,
    "stress_quotation_fraction": 0.20,
    "reference_predictor": "Training-only lexical features and standardization; equal-work author centroids; cosine scoring, as in the attribution benchmark",
    "interpretation": "Descriptive controls; style clusters are not necessarily people. Synthetic damage is not manuscript validation.",
}


def _units(works: list[dict], size: int, maximum: int = 20) -> list[dict]:
    units = []
    for work in sorted(works, key=lambda w: (w['author'], w['work'])):
        work_id = work.get('work_id', work['work'])
        words = work['text_bare'].split()
        available = len(words) // size
        count = min(maximum, available)
        first = (available - count) // 2
        for position in range(first, first + count):
            text = ' '.join(words[position * size:(position + 1) * size])
            ident = f"{work['language']}:{work.get('source_id', work_id)}.{size}.{position}"
            units.append({
                'id': ident, 'work': work_id, 'work_title': work.get('title', work['work']),
                'language': work['language'], 'group': work['author'], 'copyist': None,
                'source': work.get('source_id', work['work']), 'witness': work.get('edition', 'reference'),
                'chapter': str(position // 20 + 1), 'verse': str(position % 20 + 1),
                'order': position + 1, 'ref': f"{work['work']} tokens {position * size + 1}–{(position + 1) * size}",
                'text': text, 'text_bare': text, 'n_tokens': size, 'has_gap': False,
                'supplied_frac': 0.0, 'duplicate_of': None,
            })
    return units


def _safe_name(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def _fit_reference(verses: list[dict], seed: int):
    """Match the benchmark's raw standardized, equal-work centroid recipe."""
    extractor = LexicalFeatureExtractor(seed=seed)
    scaler = StandardScaler()
    X = scaler.fit_transform(extractor.fit_transform(verses))
    authors = sorted({v['group'] for v in verses})
    centers = []
    for author in authors:
        work_ids = sorted({v['work'] for v in verses if v['group'] == author})
        centers.append(np.mean([X[[v['work'] == w for v in verses]].mean(axis=0)
                                for w in work_ids], axis=0))
    return extractor, scaler, authors, normalize(np.asarray(centers))


def app_cluster_controls(works: list[dict], out_dir: str | Path, *, seed: int = 42,
                         sample_lengths=(25, 500), max_units: int = 20) -> dict:
    """Measure the actual production clustering, with labels used only to score it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len({w['language'] for w in works}) != 1:
        raise ValueError('app cluster controls need one language')
    rows = []
    by_author = defaultdict(list)
    by_genre = defaultdict(list)
    for work in works:
        by_author[work['author']].append(work)
        if work.get('genre'):
            by_genre[work['genre']].append(work)
    selections = [('all_authors', 'all', works)]
    selections += [('single_author', a, ws) for a, ws in sorted(by_author.items())]
    selections += [('same_genre', g, ws) for g, ws in sorted(by_genre.items())
                   if len({w['author'] for w in ws}) >= 2]
    for size in sample_lengths:
        for kind, label, selected in selections:
            verses = _units(selected, size, max_units)
            for alpha in ((0.0, 0.7) if kind == 'all_authors' else (0.7,)):
                row = {'control': kind, 'label': label, 'unit_tokens': size, 'alpha': alpha,
                       'units': len(verses), 'works': len(set(v['work'] for v in verses)),
                       'authors': len(set(v['group'] for v in verses))}
                if len(verses) < 3:
                    rows.append({**row, 'available': False, 'reason': 'fewer than three complete units'})
                    continue
                folder = out_dir / f'{size}-{alpha}-{kind}-{_safe_name(label)}'
                summary = cluster_run(verses, None, folder, kmax=min(8, max(3, 2 * row['authors'])), seed=seed, alpha=alpha)
                with (folder / 'verse_assignments.csv').open() as fh:
                    assignments = list(csv.DictReader(fh))
                rows.append({**row, 'available': True, 'k': summary['k_used'],
                             'author_ari': summary['validation']['ari_vs_group'],
                             'work_ari': summary['validation']['ari_vs_work'],
                             'selection_status': summary['selection_status'],
                             'single_author_split': kind == 'single_author' and summary['k_used'] > 1,
                             'stability': summary['stability'],
                             'assignments': {v['id']: v['author'] for v in assignments}})
    sensitivity = []
    for size in sample_lengths:
        comparable = [r for r in rows if r['control'] == 'all_authors' and r['unit_tokens'] == size and r['available']]
        if len(comparable) == 2:
            a, b = comparable
            ids = sorted(set(a['assignments']) & set(b['assignments']))
            sensitivity.append({'unit_tokens': size, 'alpha_pair': [a['alpha'], b['alpha']],
                                'assignment_ari': float(adjusted_rand_score([a['assignments'][i] for i in ids],
                                                                         [b['assignments'][i] for i in ids]))})
    for row in rows:
        row.pop('assignments', None)
    singles = [r for r in rows if r['control'] == 'single_author' and r['available']]
    result = {'protocol': CONTROL_PROTOCOL, 'language': works[0]['language'], 'controls': rows,
              'single_author_split_count': sum(r['single_author_split'] for r in singles),
              'single_author_control_count': len(singles), 'smoothing_sensitivity': sensitivity,
              'authorship_validated': False,
              'interpretation': 'Catalog-author labels are withheld from features. Multiple clusters within a single author demonstrate why style groups cannot be counted as people.'}
    (out_dir / 'controls.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


def _orthographic_variant(text: str, lang: str) -> str:
    if lang == 'grc':
        return text.upper().replace('Σ', 'Ϲ').replace('Α', 'Α\u0301')
    if lang == 'hbo':
        finals = str.maketrans({'כ': 'ך', 'מ': 'ם', 'נ': 'ן', 'פ': 'ף', 'צ': 'ץ'})
        return ' '.join((w[:-1] + w[-1].translate(finals) if w else w) for w in text.split()).replace('ב', 'בְ')
    if lang == 'arb':
        return text.replace('ا', 'أ').replace('ب', 'بَ')
    raise ValueError(f'unsupported stress-test language {lang}')


def perturbation_controls(works: list[dict], *, size: int = 500, max_units: int = 12, seed: int = 42) -> dict:
    """Train on separate works; perturb only untouched test text using fixed rates."""
    if not works or len({w['language'] for w in works}) != 1:
        raise ValueError('perturbation controls need nonempty, single-language works')
    by_author = defaultdict(list)
    for w in sorted(works, key=lambda w: w['work']):
        by_author[w['author']].append(w)
    by_author = {a: ws for a, ws in by_author.items() if len(ws) >= 2}
    if len(by_author) < 2:
        return {'available': False, 'reason': 'need two authors with distinct train/test works'}
    train_works = [w for ws in by_author.values() for w in ws[:-1]]
    test_works = [ws[-1] for ws in by_author.values()]
    train, test = _units(train_works, size, max_units), _units(test_works, size, max_units)
    if not train or not test or set(v['group'] for v in train) != set(v['group'] for v in test):
        return {'available': False, 'reason': 'insufficient complete held-out samples'}
    extractor, scaler, authors, centers = _fit_reference(train, seed)
    def predict(vs):
        return np.array(authors)[(normalize(scaler.transform(extractor.transform(vs))) @ centers.T).argmax(axis=1)]
    clean = predict(test)
    truth = [v['group'] for v in test]
    rng = np.random.default_rng(seed)
    changed = {'orthography': [], 'deletion_2pct': [], 'quotation_20pct': []}
    identities = []
    for verse in test:
        words = verse['text_bare'].split()
        canonical = ' '.join(tokenize(_orthographic_variant(verse['text_bare'], verse['language']), verse['language']))
        identities.append(canonical == verse['text_bare'])
        changed['orthography'].append({**verse, 'text_bare': canonical})
        keep = np.ones(len(words), dtype=bool)
        keep[rng.choice(len(words), max(1, int(.02 * len(words))), replace=False)] = False
        changed['deletion_2pct'].append({**verse, 'text_bare': ' '.join(w for w, retain in zip(words, keep) if retain)})
        donor = next(v for v in test if v['group'] != verse['group'])['text_bare'].split()
        n = max(1, int(.2 * len(words)))
        quoted = words[:len(words) // 2] + donor[:n] + words[len(words) // 2 + n:]
        changed['quotation_20pct'].append({**verse, 'text_bare': ' '.join(quoted)})
    rows = [{'condition': 'clean', 'prediction_agreement': 1.0,
             'balanced_accuracy': float(balanced_accuracy_score(truth, clean))}]
    for condition, units in changed.items():
        predicted = predict(units)
        rows.append({'condition': condition, 'prediction_agreement': float(np.mean(clean == predicted)),
                     'balanced_accuracy': float(balanced_accuracy_score(truth, predicted))})
    return {'available': True, 'unit_tokens': size, 'train_works': [w.get('work_id', w['work']) for w in train_works],
            'test_works': [w.get('work_id', w['work']) for w in test_works], 'n_test_units': len(test),
            'normalization_identical_fraction': float(np.mean(identities)), 'conditions': rows,
            'interpretation': 'Synthetic spelling/damage/quotation stress on real text, not actual manuscript/translation invariance. No thresholds tuned on perturbations.'}


def edition_controls(works: list[dict], controls: list[dict], *, size: int = 500, seed: int = 42) -> dict:
    """Compare held-out editions of the same work without calling them two authors."""
    pairs = defaultdict(list)
    for control in controls:
        pairs[control.get('work_id', control['work'])].append(control)
    pairs = {w: editions for w, editions in pairs.items() if len(editions) >= 2}
    if not pairs:
        return {'available': False, 'reason': 'no independently sourced same-work edition pair'}
    train = _units(works, size, 12)
    if not train:
        return {'available': False, 'reason': 'no full reference training units'}
    extractor, scaler, authors, centers = _fit_reference(train, seed)
    rows = []
    training_work_ids = {w.get('work_id', w['work']) for w in works}
    for work_id, editions in sorted(pairs.items()):
        if work_id in training_work_ids:
            raise ValueError('edition control work must be excluded from reference training works')
        vectors, labels, hashes, sources = [], [], [], []
        for edition in editions:
            units = _units([edition], size, 12)
            if not units:
                continue
            vector = scaler.transform(extractor.transform(units)).mean(axis=0)
            vector = normalize(vector.reshape(1, -1))[0]
            vectors.append(vector)
            labels.append(authors[int(np.argmax(vector @ centers.T))])
            hashes.append(hashlib.sha256(edition['text_bare'].encode()).hexdigest())
            sources.append(edition.get('source_id', edition['work']))
        if len(vectors) >= 2:
            rows.append({'work_id': work_id, 'source_ids': sources, 'catalog_author': editions[0]['author'],
                         'nearest_reference_authors': labels, 'same_prediction_across_editions': len(set(labels)) == 1,
                         'cosine_between_first_two_editions': float(vectors[0] @ vectors[1]),
                         'normalized_texts_identical': len(set(hashes)) == 1,
                         'edition_count': len(vectors)})
    return {'available': bool(rows), 'pairs': rows,
            'interpretation': 'Held-out editions share a canonical work and are never independent author labels. This limited edition check does not establish invariance across manuscript traditions or translations.'}
