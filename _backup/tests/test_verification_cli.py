"""CLI checks must not confuse a successful experiment with validated attribution."""
from pathlib import Path

import pytest

from stylometry import cli, verification_study as study, benchmark_data


def install_stubs(monkeypatch, *, signal=False):
    seen = {'loads': [], 'runs': []}
    monkeypatch.setattr(benchmark_data, 'read_manifest', lambda path: {
        'works': [{'language': 'grc' if 'greek' in str(path) else 'hbo'}]})

    def load(path, cache, download=False):
        seen['loads'].append((Path(path), cache, download))
        return [{'language': 'grc' if 'greek' in str(path) else 'hbo', 'manifest': str(path)}]

    def run(works, out, **kwargs):
        seen['runs'].append((works, out))
        return {'status': 'development_only', 'runs': [
            {'genre': None, 'summary': {'pair_verifier': {'preliminary_development_signal': signal}}}]}

    monkeypatch.setattr(benchmark_data, 'load_benchmark', load)
    monkeypatch.setattr(study, 'run_study', run)
    return seen


def test_default_command_uses_only_four_explicitly_exposed_manifests(monkeypatch, tmp_path, capsys):
    seen = install_stubs(monkeypatch)
    cli.main(['verification-study', '--out', str(tmp_path)])
    assert [x[0] for x in seen['loads']] == list(study.DEFAULT_MANIFESTS)
    assert all(not x[2] for x in seen['loads'])
    assert len(seen['runs'][0][0]) == 4
    assert 'independent authorship and scripture validation remain outstanding' in capsys.readouterr().out


def test_language_filter_never_loads_unselected_source_language(monkeypatch, tmp_path):
    seen = install_stubs(monkeypatch)
    cli.main(['verification-study', '--language', 'grc', '--out', str(tmp_path)])
    assert len(seen['loads']) == 2
    assert all('greek' in str(x[0]) for x in seen['loads'])
    assert {w['language'] for w in seen['runs'][0][0]} == {'grc'}


def test_check_fails_after_infeasible_development_result(monkeypatch, tmp_path):
    seen = install_stubs(monkeypatch, signal=False)
    with pytest.raises(SystemExit) as exc:
        cli.main(['verification-study', '--check', '--out', str(tmp_path)])
    assert exc.value.code == 1
    assert seen['runs']


def test_passing_preliminary_check_still_reports_validation_outstanding(monkeypatch, tmp_path, capsys):
    install_stubs(monkeypatch, signal=True)
    cli.main(['verification-study', '--check', '--out', str(tmp_path)])
    assert 'validation remain outstanding' in capsys.readouterr().out


def test_explicit_manifest_and_download_are_forwarded(monkeypatch, tmp_path):
    seen = install_stubs(monkeypatch)
    manifest = tmp_path / 'greek-custom.json'
    cli.main(['verification-study', '--manifest', str(manifest), '--download', '--out', str(tmp_path)])
    assert seen['loads'] == [(manifest, str(cli.RAW / 'benchmarks'), True)]


def test_invalid_lock_error_is_visible_without_overwriting(monkeypatch, tmp_path):
    install_stubs(monkeypatch)

    def refuse(*args, **kwargs):
        raise ValueError('existing development lock differs; use a new output directory')

    monkeypatch.setattr(study, 'run_study', refuse)
    with pytest.raises(SystemExit, match='existing development lock differs'):
        cli.main(['verification-study', '--out', str(tmp_path)])


@pytest.mark.parametrize('missing_language', [False, True])
def test_check_cannot_hide_failed_control_or_missing_language(monkeypatch, tmp_path, missing_language):
    install_stubs(monkeypatch, signal=True)
    result = {'requested_languages': ['grc'], 'runs': [
        {'language': 'grc', 'genre': None,
         'summary': {'pair_verifier': {'preliminary_development_signal': True}}}]}
    if missing_language:
        result['requested_languages'].append('hbo')
    else:
        result['runs'].append({'language': 'grc', 'genre': 'oratory',
                              'summary': {'pair_verifier': {'preliminary_development_signal': False}}})
    monkeypatch.setattr(study, 'run_study', lambda *args, **kwargs: result)
    with pytest.raises(SystemExit) as exc:
        cli.main(['verification-study', '--check', '--out', str(tmp_path)])
    assert exc.value.code == 1
