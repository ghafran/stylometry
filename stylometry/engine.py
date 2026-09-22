"""Unlabelled, language-scoped style discovery with explicit passage evidence.

The model never consumes catalogue authors or hierarchy labels as features. Its
groups are estimates of shared writing style, not historical identities.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import re
import unicodedata

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class Config:
    passage_tokens: int = 1200
    min_tokens: int = 200
    max_styles: int = 20
    fit_passages: int = 2000
    seed: int = 42

    def __post_init__(self):
        if self.min_tokens < 20 or self.passage_tokens < self.min_tokens:
            raise ValueError("Require passage_tokens >= min_tokens >= 20")
        if not 1 <= self.max_styles <= 100 or self.fit_passages < 20:
            raise ValueError("Require 1 <= max_styles <= 100 and fit_passages >= 20")


def tokens(text: str) -> list[str]:
    plain = ''.join(c for c in unicodedata.normalize('NFKD', text.casefold())
                    if not unicodedata.combining(c))
    return re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?", plain, re.UNICODE)


def corpus_fingerprint(verses: list[dict]) -> str:
    fingerprint = hashlib.sha256()
    for verse in verses:
        fingerprint.update((verse['id'] + '\0' + verse['text'] + '\0').encode())
    return fingerprint.hexdigest()


def make_passages(verses: list[dict], config: Config) -> tuple[list[dict], dict[str, int]]:
    """Non-overlapping context, never crossing books, languages or marked gaps.

    Input order is source order. Complete verse/paragraph units are kept intact.
    A small final block joins its preceding block only in the same uninterrupted
    book segment. Chapters may share context; passage membership is exported.
    """
    books = defaultdict(list)
    counts = {}
    for verse in verses:
        words = tokens(verse['text'])
        counts[verse['id']] = len(words)
        books[(verse['language'], verse['collection'], verse['book'])].append((verse, words))
    passages = []

    def emit_segment(segment):
        blocks, current, size = [], [], 0
        for verse, words in segment:
            current.append((verse, words))
            size += len(words)
            if size >= config.passage_tokens:
                blocks.append(current)
                current, size = [], 0
        if current:
            if blocks and size < config.min_tokens:
                blocks[-1].extend(current)
            else:
                blocks.append(current)
        for block in blocks:
            ids = [v['id'] for v, _ in block]
            n = sum(len(t) for _, t in block)
            first = block[0][0]
            passages.append({
                'id': 'P-' + hashlib.sha256('\0'.join(ids).encode()).hexdigest()[:16],
                'language': first['language'], 'collection': first['collection'],
                'book': first['book'], 'verse_ids': ids, 'tokens': n,
                'text': ' '.join(v['text'] for v, _ in block),
                'eligible': n >= config.min_tokens,
            })

    for book in books.values():
        segment = []
        for verse, words in book:
            if verse.get('has_gap') or not words:
                if segment:
                    emit_segment(segment)
                    segment = []
                # Damaged material retains its own context, never bridges a gap.
                emit_segment([(verse, words)])
            else:
                segment.append((verse, words))
        if segment:
            emit_segment(segment)
    return passages, counts


def _features(texts: list[str], fit_indices: np.ndarray, seed: int) -> np.ndarray:
    """Frequent-word rates + character patterns, reduced to a small style space."""
    normalized = [' '.join(tokens(t)) for t in texts]
    training = [normalized[i] for i in fit_indices]
    words = CountVectorizer(max_features=200, token_pattern=r'(?u)\b\w+\b')
    words.fit(training)
    rates = words.transform(normalized).astype(float)
    lengths = np.array([max(1, len(t.split())) for t in normalized])
    rates = sparse.diags(1 / lengths) @ rates
    word_scale = StandardScaler(with_mean=False).fit(rates[fit_indices])
    rates = word_scale.transform(rates)
    # Equal total feature-family scale keeps thousands of ngrams from swamping rates.
    rates = rates / np.sqrt(max(1, rates.shape[1]))
    chars = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                           max_features=3000, sublinear_tf=False)
    chars.fit(training)
    matrix = sparse.hstack([rates, chars.transform(normalized)], format='csr')
    dimensions = min(12, len(fit_indices) - 1, matrix.shape[1] - 1)
    if dimensions < 1:
        return np.zeros((len(texts), 1))
    projection = TruncatedSVD(n_components=dimensions, random_state=seed)
    projection.fit(matrix[fit_indices])
    reduced = projection.transform(matrix)
    # Discard numerical null-space axes before variance scaling; otherwise tiny
    # floating-point noise is amplified into apparently meaningful style splits.
    spread = reduced[fit_indices].std(axis=0)
    meaningful = ((projection.singular_values_ > max(1e-10, projection.singular_values_[0] * 1e-8)) &
                  (spread > max(1e-10, float(spread.max()) * 1e-6)))
    if not meaningful.any():
        return np.zeros((len(texts), 1))
    reduced = reduced[:, meaningful]
    scale = StandardScaler().fit(reduced[fit_indices])
    return np.clip(scale.transform(reduced), -8, 8)


def _discover(passages: list[dict], config: Config) -> tuple[np.ndarray, np.ndarray, dict]:
    n = len(passages)
    if not n:
        return np.array([], int), np.array([]), {
            'reason': 'No passage has enough text.', 'candidate_scores': [],
            'stable': None, 'fit_passages': 0,
        }
    if n < 6:
        return np.zeros(n, int), np.full(n, np.nan), {
            'reason': 'Too few passages to support a split; one provisional group.',
            'candidate_scores': [], 'stable': None, 'fit_passages': n,
        }
    rng = np.random.default_rng(config.seed)
    fit = np.sort(rng.choice(n, min(config.fit_passages, n), replace=False))
    if len({' '.join(tokens(passages[i]['text'])) for i in fit}) == 1:
        return np.zeros(n, int), np.full(n, np.nan), {
            'reason': 'All sampled passages have identical normalized text.',
            'candidate_scores': [], 'stable': None, 'fit_passages': len(fit),
        }
    with threadpool_limits(limits=1):
        x = _features([p['text'] for p in passages], fit, config.seed)
        train = x[fit]
        if np.max(np.std(train, axis=0)) < 1e-7:
            return np.zeros(n, int), np.full(n, np.nan), {
                'reason': 'All passage feature vectors are identical.',
                'candidate_scores': [], 'stable': None, 'fit_passages': len(fit),
            }
        upper = min(config.max_styles, max(1, len(fit) // 8))
        candidates, models = [], {}
        for k in range(1, upper + 1):
            model = GaussianMixture(n_components=k, covariance_type='diag',
                                    reg_covar=0.05, n_init=2, max_iter=200,
                                    random_state=config.seed).fit(train)
            labels = model.predict(train)
            sizes = np.bincount(labels, minlength=k)
            supported = bool(model.converged_ and np.min(sizes) >= 3)
            score = float(model.bic(train))
            candidates.append({'k': k, 'bic': round(score, 3), 'score': round(score, 3),
                               'supported': supported, 'smallest_group': int(min(sizes)),
                               'converged': bool(model.converged_)})
            if supported or k == 1:
                models[k] = model
        best = min(models, key=lambda k: models[k].bic(train))
        baseline = models[1].bic(train)
        gain = float(baseline - models[best].bic(train))
        if gain < 10:
            best = 1
        model = models[best]
        labels = model.predict(x)
        distance = np.linalg.norm(x[:, None, :] - model.means_[None, :, :], axis=2)
        if best > 1:
            own = distance[np.arange(n), labels]
            other = distance.copy()
            other[np.arange(n), labels] = np.inf
            nearest_other = np.min(other, axis=1)
            margins = (nearest_other - own) / np.maximum(1e-9, np.maximum(own, nearest_other))
            fit_labels = labels[fit]
            silhouette = _sampled_silhouette(train, fit_labels, rng)
            agreement = []
            for repeat in range(3):
                indices = rng.choice(len(train), max(best * 3, int(len(train) * .8)), replace=False)
                trial = GaussianMixture(n_components=best, covariance_type='diag',
                                        reg_covar=.05, n_init=1, max_iter=200,
                                        random_state=config.seed + repeat + 1).fit(train[indices])
                agreement.append(float(adjusted_rand_score(fit_labels, trial.predict(train))))
        else:
            margins = np.full(n, np.nan)
            silhouette, agreement = None, []
    selection = {
        'method': 'Minimum diagonal Gaussian-mixture BIC; one-group baseline included',
        'reason': 'Selected using unlabelled text only; catalogue authors were not supplied.',
        'candidate_scores': candidates, 'fit_passages': len(fit), 'selected_k': best,
        'bic_gain_over_one': round(gain, 3), 'silhouette': silhouette,
        'stability_ari': agreement, 'stable': min(agreement) >= .8 if agreement else None,
        'at_search_limit': best == upper and upper > 1,
        'limitations': 'Style groups can reflect genre, topic, editor or transmitter. '
                      'Stability holds features and group count fixed; it is not accuracy against a known author.',
    }
    return labels, margins, selection


def _sampled_silhouette(x: np.ndarray, labels: np.ndarray, rng) -> float:
    # Guarantee minority-group coverage: an unconstrained random sample can
    # contain just one label and make an otherwise valid analysis crash.
    mandatory = np.array([np.flatnonzero(labels == label)[0] for label in np.unique(labels)])
    remaining = np.setdiff1d(np.arange(len(x)), mandatory)
    size = min(1000, len(x))
    chosen = np.concatenate([mandatory, rng.choice(remaining, size - len(mandatory), replace=False)])
    return float(silhouette_score(x[chosen], labels[chosen]))


def make_rollups(verses: list[dict]) -> list[dict]:
    levels = ['language', 'collection', 'book', 'chapter', 'verse']
    grouped = {}
    for v in verses:
        path = (v['language'], v['collection'], v['book'], v['chapter'], v['id'])
        for depth, level in enumerate(levels, 1):
            key = path[:depth]
            if (level, key) not in grouped:
                grouped[level, key] = {
                    'level': level, 'language': v['language'],
                    'collection': v['collection'] if depth >= 2 else None,
                    'book': v['book'] if depth >= 3 else None,
                    'book_title': v['book_title'] if depth >= 3 else None,
                    'chapter': v['chapter'] if depth >= 4 else None,
                    'verse': v['verse'] if depth == 5 else None,
                    'id': v['id'] if depth == 5 else None,
                    'verse_count': 0, 'assigned_verse_count': 0,
                    'insufficient_verse_count': 0, 'low_evidence_verse_count': 0,
                    'style_counts': Counter(),
                }
            row = grouped[level, key]
            row['verse_count'] += 1
            if v['style_id']:
                row['assigned_verse_count'] += 1
                row['style_counts'][v['style_id']] += 1
            else:
                row['insufficient_verse_count'] += 1
            row['low_evidence_verse_count'] += v['status'] == 'low_evidence'
    for row in grouped.values():
        counts = row['style_counts']
        row['style_ids'] = sorted(counts)
        row['style_count'] = len(counts)
        row['dominant_style'] = counts.most_common(1)[0][0] if counts else None
        row['style_counts'] = dict(sorted(counts.items()))
    return list(grouped.values())


def discovery_validation(verses: list[dict]) -> dict:
    """Evaluate completed blind English clustering; never influences discovery."""
    known = [v for v in verses if v['language'] == 'eng' and v.get('reference_author')]
    eligible = [v for v in known if v.get('style_id')]
    truth = sorted({v['reference_author'] for v in known})
    discovered = {v['style_id'] for v in eligible}
    contingency = defaultdict(Counter)
    for v in eligible:
        contingency[v['style_id']][v['reference_author']] += 1
    return {
        'known_author_count': len(truth), 'discovered_groups_on_labelled_text': len(discovered),
        'count_error': len(discovered) - len(truth), 'evaluated_verses': len(eligible),
        'coverage': len(eligible) / len(known) if known else 0,
        'adjusted_rand_index': float(adjusted_rand_score(
            [v['reference_author'] for v in eligible], [v['style_id'] for v in eligible])) if eligible else None,
        'normalized_mutual_information': float(normalized_mutual_info_score(
            [v['reference_author'] for v in eligible], [v['style_id'] for v in eligible])) if eligible else None,
        'group_reference_counts': {k: dict(v) for k, v in sorted(contingency.items())},
        'interpretation': 'Post-hoc verse-weighted cluster agreement, not held-out accuracy. '
                          'Nearby verses share passage predictions. Matching the count alone is insufficient.',
    }


def analyze(verses: list[dict], config: Config | None = None, progress=None) -> dict:
    config = config or Config()
    tagged = [{**v, 'style_id': None, 'status': 'insufficient_text',
               'evidence_tokens': 0, 'passage_id': None, 'distance_margin': None} for v in verses]
    by_id = {v['id']: v for v in tagged}
    if len(by_id) != len(tagged):
        raise ValueError('Verse IDs must be unique')
    language_results, evidence = [], []
    for language in sorted({v['language'] for v in tagged}):
        subset = [v for v in tagged if v['language'] == language]
        passages, counts = make_passages(subset, config)
        eligible = [p for p in passages if p['eligible']]
        if progress:
            progress(f"{language}: {len(subset):,} verses; comparing {len(eligible):,} passages")
        labels, margins, selection = _discover(eligible, config)
        # Largest token-supported group first; all hierarchy levels share these IDs.
        sizes = Counter()
        for passage, label in zip(eligible, labels):
            sizes[int(label)] += passage['tokens']
        names = {label: f'{language}-S{rank:03}' for rank, (label, _) in enumerate(
            sorted(sizes.items(), key=lambda p: (-p[1], p[0])), 1)}
        for passage, label, margin in zip(eligible, labels, margins):
            style_id = names[int(label)]
            passage['style_id'] = style_id
            for ident in passage['verse_ids']:
                v = by_id[ident]
                v.update(style_id=style_id, passage_id=passage['id'], evidence_tokens=passage['tokens'],
                         distance_margin=float(margin) if np.isfinite(margin) else None,
                         status='low_evidence' if counts[ident] < 40 or v.get('has_gap') or
                         selection.get('stable') is not True or (np.isfinite(margin) and margin < .1)
                         else 'assigned')
        for passage in passages:
            for ident in passage['verse_ids']:
                if by_id[ident]['passage_id'] is None:
                    by_id[ident]['passage_id'] = passage['id']
                    by_id[ident]['evidence_tokens'] = passage['tokens']
                by_id[ident]['token_count'] = counts[ident]
            evidence.append({k: v for k, v in passage.items() if k != 'text'})
        language_results.append({'language': language, 'verse_count': len(subset),
                                 'passage_count': len(passages), 'eligible_passage_count': len(eligible),
                                 'estimated_styles': len(names), 'selection': selection})
    styles = defaultdict(list)
    for v in tagged:
        if v['style_id']:
            styles[v['style_id']].append(v)
    style_rows = [{
        'style_id': style, 'language': units[0]['language'], 'verse_count': len(units),
        'chapter_count': len({(v['collection'], v['book'], v['chapter']) for v in units}),
        'book_count': len({(v['collection'], v['book']) for v in units}),
        'collection_count': len({v['collection'] for v in units}),
        'token_count': sum(v['token_count'] for v in units), 'examples': [v['id'] for v in units[:5]],
    } for style, units in sorted(styles.items())]
    return {'schema_version': 1, 'config': asdict(config), 'corpus_sha256': corpus_fingerprint(verses),
            'languages': language_results, 'verses': tagged, 'styles': style_rows,
            'passages': evidence, 'rollups': make_rollups(tagged), 'benchmark': None,
            'discovery_validation': discovery_validation(tagged)}
