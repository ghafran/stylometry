import json

from stylometry.cli import main


def test_analyze_exports_all_units_and_removes_stale_benchmark(tmp_path):
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps(dict(id='one', language='eng', collection='C',
        book='B', book_title='Book', chapter='1', verse='1', text='Short verse')) + '\n')
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'benchmark.json').write_text('{"accuracy": 1.0}')
    assert main(['analyze', '--corpus', str(source), '--out', str(out), '--skip-benchmark']) == 0
    result = json.loads((out / 'report.json').read_text())
    assert result['verses'][0]['status'] == 'insufficient_text'
    assert not (out / 'benchmark.json').exists()
    assert (out / 'index.html').is_file()
    assert (out / 'passages.json').is_file()


def test_missing_language_fails_without_new_output(tmp_path):
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps(dict(id='one', language='eng', collection='C',
        book='B', book_title='Book', chapter='1', verse='1', text='Short verse')) + '\n')
    out = tmp_path / 'out'
    assert main(['analyze', '--corpus', str(source), '--out', str(out), '--language', 'grc']) == 1
    assert not out.exists()


def test_failed_build_keeps_diagnostic_report(tmp_path):
    raw = tmp_path / 'raw'
    (raw / 'quran').mkdir(parents=True)
    corpus = tmp_path / 'processed' / 'verses.jsonl'
    assert main(['build', '--raw', str(raw), '--corpus', str(corpus)]) == 1
    report = json.loads((corpus.parent / 'build_report.json').read_text())
    assert any(s['source'] == 'quran' and s['status'] == 'failed' for s in report['sources'])
    assert not corpus.exists()
