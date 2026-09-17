"""Acquire reference data and run an auditable, non-cherry-picked benchmark suite."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import run_benchmark
from .benchmark_controls import app_cluster_controls, perturbation_controls, edition_controls
from .benchmark_data import load_benchmark, read_manifest

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = [REPO / 'benchmarks/greek_manifest.json', REPO / 'benchmarks/additional_manifest.json']


def run_suite(manifests=None, cache_dir=None, out_dir=None, *, download=False, language=None,
              controls=True) -> dict:
    manifests = [Path(p) for p in (manifests or DEFAULT_MANIFESTS)]
    cache_dir = Path(cache_dir or REPO / 'data/raw/benchmarks')
    out_dir = Path(out_dir or REPO / 'output/benchmark')
    works, edition_pairs, source_manifests = [], [], []
    for path in manifests:
        manifest = read_manifest(path)
        if language and not any(w['language'] == language for w in manifest['works']):
            continue
        records = load_benchmark(path, cache_dir, download=download, include_controls=controls)
        records = [w for w in records if language is None or w['language'] == language]
        works.extend(w for w in records if w.get('role', 'attribution') == 'attribution')
        edition_pairs.extend(w for w in records if w.get('role', 'attribution') != 'attribution')
        source_manifests.append({'path': str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
                                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'corpus_id': manifest.get('corpus_id'),
                                 'selection_protocol': manifest.get('selection_protocol'),
                                 'attribution_caveat': manifest.get('attribution_caveat')})
    if not works:
        raise ValueError(f'no reference corpus for {language or "the requested manifests"}; this language remains unvalidated')
    out_dir.mkdir(parents=True, exist_ok=True)
    by_language = defaultdict(list)
    for w in works:
        by_language[w['language']].append(w)
    coverage = {}
    for lang in ('grc', 'hbo', 'arb'):
        subset = by_language.get(lang, [])
        coverage[lang] = {'available': bool(subset), 'authors': len({w['author'] for w in subset}),
                          'works': len(subset), 'tokens': sum(w['n_tokens'] for w in subset),
                          'source_languages': sorted({w.get('source_language', w['language']) for w in subset}),
                          'scripture_transfer_validated': False}
        if not subset:
            coverage[lang]['reason'] = 'not selected or no acquired reference corpus; not a passing result'
    print('Running work-held-out and unknown-author evaluation...', flush=True)
    attribution = run_benchmark(works, out_dir / 'attribution', progress=lambda message: print(message, flush=True))
    actual_controls, stress, editions = {}, {}, {}
    if controls:
        for lang, subset in sorted(by_language.items()):
            print(f'Running actual clustering and robustness controls for {lang}...', flush=True)
            actual_controls[lang] = app_cluster_controls(subset, out_dir / 'app_controls' / lang)
            stress[lang] = perturbation_controls(subset)
            editions[lang] = edition_controls(subset, [w for w in edition_pairs if w['language'] == lang])
    implementation = {}
    for filename in ('benchmark.py', 'benchmark_data.py', 'benchmark_additional.py', 'benchmark_controls.py',
                     'benchmark_suite.py', 'features.py', 'cluster.py', 'lang.py', 'greek.py', 'continuity.py'):
        path = REPO / 'stylometry' / filename
        if path.exists():
            implementation[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        'status': attribution['status'], 'scope': 'reference-corpus lexical evaluation only',
        'scripture_attribution_validated': False, 'ai_profiles_validated': False,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'attribution_protocol': attribution['protocol'], 'attribution_settings': attribution['settings'],
        'exclusions': attribution['exclusions'],
        'coverage': coverage, 'source_manifests': source_manifests,
        'implementation_sha256': implementation,
        'runtime': {'python': platform.python_version(), **{package: importlib.metadata.version(package)
                    for package in ('numpy', 'scikit-learn', 'scipy', 'lxml')}},
        'controls_enabled': controls, 'attribution_runs': [
            {key: value for key, value in run.items() if key not in ('closed_set_predictions', 'open_set_predictions')}
            for run in attribution['runs']],
        'app_controls': actual_controls, 'perturbations': stress, 'edition_controls': editions,
        'limitations': attribution['limitations'],
    }
    (out_dir / 'suite.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (out_dir / 'report.md').write_text(render_suite(result), encoding='utf-8')
    return result


def _pct(value):
    return 'unavailable' if value is None else f'{value:.1%}'


def render_suite(result: dict) -> str:
    lines = ['# Real-text reliability benchmark', '', f"**Empirical acceptance: {result['status'].upper()}.**", '',
             '**Scripture attribution and AI profiles remain unvalidated.** Passing software tests only means the evaluator and safeguards execute correctly.', '',
             'Sources and thresholds were selected before inspecting benchmark scores. These held-out works are now an exposed evaluation set; future model tuning needs a fresh final test set.', '',
             '## Corpus coverage', '', '| Pipeline | Actual source language | Authors | Independent works | Tokens |', '|---|---|---:|---:|---:|']
    for lang, c in result['coverage'].items():
        lines.append(f"| {lang} | {', '.join(c['source_languages']) or 'No corpus'} | {c['authors']} | {c['works']} | {c['tokens']:,} |")
    lines += ['', 'Catalog author labels are external reference assumptions. The Hebrew-script pipeline is tested on modern Hebrew, not Biblical Hebrew or Aramaic. Arabic without a corpus is a validation gap.', '',
              '## Held-out author results', '',
              'Whole works rotate through separate training, calibration and test partitions. Unknown authors are excluded completely from fitting and calibration. The table gives point estimates; intervals, predeclared gates, per-author errors, topic/genre subsets and all predictions are in [the detailed report](attribution/report.md) and [benchmark.json](attribution/benchmark.json).', '',
              '| Language | Tokens | Features | Author accuracy | Wrong-author match rate | Missed same-author rate | Unknown authors accepted | Result |',
              '|---|---:|---|---:|---:|---:|---:|---|']
    for r in result['attribution_runs']:
        if 'metrics' not in r:
            continue
        metrics = r['metrics']
        feature_label = {'closed_class': 'Function/common words', 'full_lexical': 'Full lexical'}.get(r['ablation'], r['ablation'])
        lines.append(f"| {r['language']} | {r['sample_length']} | {feature_label} | " + ' | '.join(_pct(metrics[k]['value']) for k in ('balanced_accuracy', 'verification_fpr', 'verification_fnr', 'unknown_false_acceptance')) + f" | {r['status']} |")
    lines += ['', 'The function/common-word list includes some content verbs; this ablation is not topic-free. Intervals resample whole works with predictions fixed. They do not refit models or account for dependencies through shared training works and candidate authors, so they are conditional descriptive intervals, not uncertainty bounds for historical authorship.']
    lines += ['', '## Production clustering controls', '',
              'These runs invoke the current app directly, without author labels entering features. Multiple styles within one writer are possible; splits here show why cluster count cannot be interpreted as author count.', '',
              '| Language | Control | Unit tokens | Smoothing | Catalog authors | Groups returned | Author ARI |', '|---|---|---:|---:|---:|---:|---:|']
    for lang, controls in result['app_controls'].items():
        for row in controls['controls']:
            if row['available']:
                lines.append(f"| {lang} | {row['control']}: {row['label']} | {row['unit_tokens']} | {row['alpha']} | {row['authors']} | {row['k']} | {row['author_ari']:.3f} |")
    lines += ['', '## Perturbation and edition checks', '',
              'Spelling, 2% word deletion and replacement of 20% of words with another author’s text are synthetic perturbations of real held-out text. They do not substitute for manuscript or translation studies.', '',
              '| Language | Condition | Agreement with clean prediction | Author accuracy |', '|---|---|---:|---:|']
    for lang, test in result['perturbations'].items():
        if test.get('available'):
            for row in test['conditions']:
                lines.append(f"| {lang} | {row['condition']} | {_pct(row['prediction_agreement'])} | {_pct(row['balanced_accuracy'])} |")
    for lang, test in result['edition_controls'].items():
        if test.get('available'):
            for pair in test['pairs']:
                lines += ['', f"Actual {lang} edition pair `{pair['work_id']}`: same nearest reference author across editions = {pair['same_prediction_across_editions']}; cosine similarity {pair['cosine_between_first_two_editions']:.3f}. This is one limited edition control, not broad transmission validation."]
        else:
            lines += ['', f"{lang}: no acquired same-work edition control."]
    lines += ['', '## Limits and reproduction', ''] + ['- ' + item for item in result['limitations']]
    lines += ['', 'Run `uv run stylometry benchmark --download` to fetch the checksum-pinned sources and recreate this report. Add `--check` to fail an empirical acceptance gate when the result is failed or inconclusive. Downloads are cached; no paid API calls are made.', '',
              '`suite.json` records corpus and implementation checksums, package versions, settings and detailed control results. The source manifests retain source URLs, editions, licences and curation caveats.', '']
    return '\n'.join(lines)
