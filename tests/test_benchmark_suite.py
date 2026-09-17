"""The CLI must expose an unsuccessful empirical outcome, not hide it as success."""
import pytest
from stylometry import cli
from stylometry.benchmark_suite import render_suite


@pytest.mark.parametrize('status', ['failed', 'inconclusive'])
def test_cli_empirical_gate_fails_loudly(monkeypatch, tmp_path, status):
    import stylometry.benchmark_suite as suite
    monkeypatch.setattr(suite, 'run_suite', lambda *a, **kw: {'status': status})
    with pytest.raises(SystemExit) as exc:
        cli.main(['benchmark', '--check', '--out', str(tmp_path)])
    assert exc.value.code == 1


def test_cli_does_not_require_success_to_deliver_diagnostic_report(monkeypatch, tmp_path):
    import stylometry.benchmark_suite as suite
    seen = []
    def run(*a, **kw):
        seen.append(kw)
        return {'status': 'failed'}
    monkeypatch.setattr(suite, 'run_suite', run)
    cli.main(['benchmark', '--download', '--language', 'grc', '--out', str(tmp_path)])
    assert seen[0]['download'] is True
    assert seen[0]['language'] == 'grc'


def test_suite_report_keeps_unsupported_languages_and_scope_visible():
    report = {'status': 'failed', 'coverage': {
        'arb': {'source_languages': [], 'authors': 0, 'works': 0, 'tokens': 0}},
        'attribution_runs': [], 'app_controls': {}, 'perturbations': {}, 'edition_controls': {}, 'limitations': []}
    rendered = render_suite(report)
    assert 'FAILED' in rendered
    assert 'Scripture attribution and AI profiles remain unvalidated' in rendered
    assert 'No corpus' in rendered
    assert 'not a passing' not in rendered or 'validation gap' in rendered


def test_no_controls_does_not_require_witness_downloads(monkeypatch, tmp_path):
    import stylometry.benchmark_suite as suite
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    source = {'author': 'A', 'work': 'W', 'language': 'grc', 'n_tokens': 100}
    monkeypatch.setattr(suite, 'read_manifest', lambda path: {'works': [source]})
    included = []
    def load(*args, include_controls, **kwargs):
        included.append(include_controls)
        return [source]
    monkeypatch.setattr(suite, 'load_benchmark', load)
    monkeypatch.setattr(suite, 'run_benchmark', lambda *args, **kwargs: {
        'status': 'inconclusive', 'protocol': {}, 'settings': {}, 'exclusions': [], 'runs': [], 'limitations': []})
    result = suite.run_suite([manifest], out_dir=tmp_path / 'out', controls=False)
    assert included == [False]
    assert result['controls_enabled'] is False
    assert result['app_controls'] == {}
