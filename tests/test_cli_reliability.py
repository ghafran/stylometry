"""CLI-level controls for explicit modes and accurate request previews."""
import pytest

from stylometry import cli
from test_pipeline import _fake_verses


def test_missing_profiles_require_explicit_no_ai(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, '_load_corpus', lambda: _fake_verses(10, 'X'))
    with pytest.raises(SystemExit, match='explicitly use --no-ai'):
        cli.main(['cluster', '--scope', 'all', '--profiles', str(tmp_path / 'missing.jsonl'),
                  '--out', str(tmp_path / 'output')])
    assert not (tmp_path / 'output' / 'summary.json').exists()


def test_profile_preview_uses_complete_requests(monkeypatch, tmp_path, capsys):
    from stylometry import ai_profile
    monkeypatch.setattr(cli, '_load_corpus', lambda: _fake_verses(20, 'X', chapter_size=10))
    received = []
    monkeypatch.setattr(ai_profile, 'run_profile', lambda *a, **kw: received.append(kw) or {'profiled': 0})
    cli.main(['profile', '--scope', 'all', '--limit', '1', '--chunk-size', '25',
              '--profiles', str(tmp_path / 'profiles.jsonl')])
    output = capsys.readouterr().out
    # The chapter is the request unit; the pilot cannot cut it at a different point.
    assert '10 verses in 1 complete requests' in output
    assert received[0]['limit'] == 1


def test_limit_cannot_bypass_spending_guard(monkeypatch, tmp_path):
    from stylometry import ai_profile
    monkeypatch.setattr(cli, '_load_corpus', lambda: _fake_verses(600, 'X', chapter_size=600))
    def unexpected_call(*a, **kw):
        pytest.fail('No request should be sent before the spending guard')
    monkeypatch.setattr(ai_profile, 'run_profile', unexpected_call)
    with pytest.raises(SystemExit, match='more than 500 verses'):
        cli.main(['profile', '--scope', 'all', '--limit', '1', '--chunk-size', '600',
                  '--profiles', str(tmp_path / 'profiles.jsonl')])


def test_compare_language_filter_applies_before_evaluation(monkeypatch, tmp_path):
    from stylometry import compare
    from pathlib import Path
    corpus = _fake_verses(10, 'X') + _fake_verses(10, 'Y', language='hbo')
    monkeypatch.setattr(cli, '_load_corpus', lambda: corpus)
    monkeypatch.setattr(compare, 'load_set', lambda name, path, cost: compare.ModelSet(name, path, {}))
    selected = []
    def capture(sets, verses, *a, **kw):
        selected.extend(verses)
        return {'recommendation': {}}
    monkeypatch.setattr(compare, 'build', capture)
    cli.main(['compare-models', '--set', 'model=fake.jsonl', '--language', 'hbo', '--out', str(tmp_path)])
    assert len(selected) == 10
    assert {v['language'] for v in selected} == {'hbo'}
