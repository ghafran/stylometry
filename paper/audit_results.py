"""Reproduce descriptive article statistics from the saved analysis; no model refit."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / 'output/report.json').read_text())
benchmark = json.loads((ROOT / 'output/benchmark.json').read_text())
build = json.loads((ROOT / 'data/processed/build_report.json').read_text())
assert report['corpus_sha256'] == benchmark['corpus_sha256'] == build['corpus_sha256']
verses = report['verses']
assert len({v['id'] for v in verses}) == len(verses) == 154158
assert sum(v['token_count'] for v in verses) == 6348728
assert Counter(v['status'] for v in verses) == {'low_evidence': 148621, 'insufficient_text': 5537}
members = [ident for p in report['passages'] for ident in p['verse_ids']]
assert len(members) == len(set(members)) == len(verses)
assert set(members) == {v['id'] for v in verses}
assert json.loads((ROOT / 'output/passages.json').read_text()) == report['passages']
assert report['benchmark'] == {k: v for k, v in benchmark.items() if k not in ['discovery', 'corpus_sha256']} or report['benchmark'] == benchmark

languages = []
for language in report['languages']:
    rows = [v for v in verses if v['language'] == language['language']]
    scores = sorted((x for x in language['selection']['candidate_scores'] if x['supported']), key=lambda x: x['bic'])
    aris = language['selection']['stability_ari']
    languages.append({
        **{k: v for k, v in language.items() if k != 'selection'},
        'tokens': sum(v['token_count'] for v in rows),
        'assigned': sum(bool(v['style_id']) for v in rows),
        'coverage': sum(bool(v['style_id']) for v in rows) / len(rows),
        'bic_gain': language['selection']['bic_gain_over_one'],
        'silhouette': language['selection']['silhouette'],
        'stability_ari': aris,
        'mean_ari': sum(aris) / len(aris),
        'runner_up_k': scores[1]['k'],
        'runner_up_bic_gap': scores[1]['bic'] - scores[0]['bic'],
    })
collections = defaultdict(lambda: {'units': 0, 'assigned_units': 0, 'tokens': 0, 'assigned_tokens': 0, 'style_tokens': Counter(), 'style_units': Counter()})
for v in verses:
    row = collections[v['language'] + '/' + v['collection']]
    row['units'] += 1
    row['tokens'] += v['token_count']
    if v['style_id']:
        row['assigned_units'] += 1
        row['assigned_tokens'] += v['token_count']
        row['style_tokens'][v['style_id']] += v['token_count']
        row['style_units'][v['style_id']] += 1
assert sum(x['correct'] for x in benchmark['held_out_predictions']) == 199
assert len(benchmark['held_out_predictions']) == 238
assert sum(x['correct'] for x in benchmark['book_predictions']) == 24
assert len(benchmark['book_predictions']) == 25
multi = [p for p in report['passages'] if p.get('book_count', 1) > 1]
assert len(multi) == 11
assert collections['hbo/DSS']['assigned_units'] == 197
assert collections['arb/Quran']['style_units'] == {'arb-S003': 2541, 'arb-S005': 3648}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

files = ['output/report.json', 'output/passages.json', 'output/benchmark.json', 'data/processed/build_report.json', 'stylometry/engine.py', 'stylometry/benchmark.py', 'stylometry/corpus.py', 'uv.lock']
md = (ROOT / 'paper/stylometry_research_article.md').read_text()
plain = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', md)
word_count = lambda text: len(re.findall(r"\b[\w]+(?:[’'–-][\w]+)*\b", text))
audit = {
    'scope': 'Descriptive reanalysis of saved artifacts; no discovery or classifier refit',
    'inspected_revision': '276465f9263a083d8806b97b4595582449f32e1f',
    'corpus_sha256': report['corpus_sha256'],
    'file_sha256': {f: digest(ROOT / f) for f in files},
    'languages': languages,
    'collections': dict(collections),
    'english_discovery': report['discovery_validation'],
    'english_attribution': {k: benchmark[k] for k in ['accuracy', 'balanced_accuracy', 'book_accuracy', 'book_balanced_accuracy', 'per_author', 'per_collection']},
    'multiple_book_passages': multi,
    'word_counts': {
        'entire_document': word_count(plain),
        'abstract_through_conclusion_including_tables_and_headings': word_count(plain.split('## Abstract\n', 1)[1].split('## Data and reproducibility', 1)[0]),
        'introduction_through_conclusion_including_tables_and_headings': word_count(plain.split('## 1 Introduction\n', 1)[1].split('## Data and reproducibility', 1)[0]),
    },
}
(ROOT / 'paper/results_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'checks': 'passed', 'word_counts': audit['word_counts'], 'languages': languages}, indent=2))
