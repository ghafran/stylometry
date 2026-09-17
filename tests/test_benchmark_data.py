"""Source-integrity and TEI extraction checks; no network needed for unit tests.

The final test uses real sources only when explicitly acquired beforehand. Passing
these checks establishes data hygiene, not author-attribution accuracy.
"""
from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import pytest

from stylometry import benchmark_data as data
from stylometry.lang import tokenize


def tei(body: str, *, extra: str = "") -> bytes:
    return (f'<TEI xmlns="http://www.tei-c.org/ns/1.0">'
            '<teiHeader><fileDesc><titleStmt><title>Πλάτων</title></titleStmt></fileDesc></teiHeader>'
            '<text><front>Λυσίας</front><body>'
            f'<div type="edition" xml:lang="grc">{body}</div>{extra}'
            '</body><back>Ἰσοκράτης</back></text></TEI>').encode()


def fixture_manifest(tmp_path: Path, *, cached: bool = True) -> tuple[Path, Path, dict]:
    source = tei('<p>καὶ μὲν δὴ λόγος.</p>')
    normalized = ' '.join(tokenize(data.extract_tei_greek(source), 'grc'))
    commit = 'a' * 40
    work = {
        'source_id': 'example.grc1', 'work_id': 'example', 'role': 'attribution',
        'author': 'Example Author', 'work': 'Example Work', 'language': 'grc',
        'genre': 'prose', 'topic': 'ethics', 'url': f'https://example.test/{commit}/source.xml',
        'sha256': hashlib.sha256(source).hexdigest(), 'parser': 'tei-greek-v1',
        'license': 'CC-BY-SA-4.0', 'license_url': 'https://creativecommons.org/licenses/by-sa/4.0/',
        'source_repository': 'https://example.test/repository', 'source_commit': commit,
        'source_path': 'source.xml', 'n_tokens': 4,
        'text_sha256': hashlib.sha256(normalized.encode()).hexdigest(),
    }
    manifest = {'schema_version': 1, 'corpus_id': 'fixture', 'works': [work]}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    cache = tmp_path / 'cache'
    if cached:
        cache.mkdir()
        (cache / 'example.grc1.source').write_bytes(source)
    return path, cache, manifest


def rewrite(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest))


def test_tei_only_extracts_edition_and_excludes_editorial_and_foreign_text():
    source = tei('''<head>Δημοσθένης</head><p><label>ΣΩ.</label>ἀληθὴς
        <persName>Σωκράτης</persName><note>ψευδής <foreign xml:lang="grc">λάθος</foreign></note>
        <foreign xml:lang="eng">English Ἀθήνα</foreign>λόγος.</p>
        <sp><speaker>Πλάτων</speaker><p>καὶ μὲν <bibl>Φαίδων</bibl>δή.</p></sp>''',
        extra='<div type="translation" xml:lang="eng">Ἡρόδοτος English</div>')
    words = tokenize(data.extract_tei_greek(source), 'grc')
    assert words == ['αληθησ', 'σωκρατησ', 'λογοσ', 'και', 'μεν', 'δη']


def test_inline_markup_does_not_split_words_or_merge_paragraphs():
    source = tei('<p>ἀλ<hi>ή</hi>θεια<note>λάθος</note></p><p>καὶ μέν</p>')
    assert tokenize(data.extract_tei_greek(source), 'grc') == ['αληθεια', 'και', 'μεν']


def test_prefers_critical_reading_without_duplicating_alternatives():
    source = tei('<p><choice><sic>λάθος</sic><corr>ἀλήθεια</corr></choice> '
                 '<app><lem>λόγος</lem><rdg>ἄλλος</rdg></app> '
                 '<add>καί</add> <del>ψευδής</del> μέν</p>')
    assert tokenize(data.extract_tei_greek(source), 'grc') == ['αληθεια', 'λογοσ', 'και', 'μεν']


def test_excludes_external_quotations_but_preserves_authored_direct_speech():
    source = tei('<p>καὶ <q>λέγει λόγον</q> '
                 '<quote>Ὅμηρος ἔφη</quote><cit><quote>Ὅμηρος</quote><bibl>Ἰλιάς</bibl></cit> μέν</p>')
    assert tokenize(data.extract_tei_greek(source), 'grc') == ['και', 'λεγει', 'λογον', 'μεν']


def test_deletion_span_can_cross_nested_paragraphs():
    source = tei('<p>ἀλήθεια <delSpan spanTo="#end"/>λάθος</p>'
                 '<p>ψευδής <anchor xml:id="end"/> καὶ μέν</p>')
    assert tokenize(data.extract_tei_greek(source), 'grc') == ['αληθεια', 'και', 'μεν']


@pytest.mark.parametrize('body,extra,error', [
    ('<p>καί <delSpan spanTo="#absent"/></p>', '', 'matching anchor'),
    ('<p><anchor xml:id="old"/>καί <delSpan spanTo="#old"/></p>', '', 'unclosed'),
    ('<p><app><rdg>καί</rdg></app></p>', '', 'preferred reading'),
    ('<p><choice><unknown>καί</unknown></choice></p>', '', 'preferred reading'),
    ('<note>καί</note>', '', 'no Greek prose'),
    ('<p>καί</p>', '<div type="edition" xml:lang="grc"><p>καί</p></div>', 'exactly one'),
])
def test_rejects_ambiguous_or_empty_tei(body, extra, error):
    with pytest.raises(ValueError, match=error):
        data.extract_tei_greek(tei(body, extra=extra))


def test_does_not_expand_external_entities(tmp_path):
    private = tmp_path / 'private.txt'
    private.write_text('ἄγνωστος')
    source = (f'<!DOCTYPE TEI [<!ENTITY hidden SYSTEM "{private.as_uri()}">]>'
              + tei('<p>καὶ &hidden;</p>').decode()).encode()
    with pytest.raises(ValueError, match='entity references'):
        data.extract_tei_greek(source)


def test_load_is_offline_and_records_source_and_parser_provenance(tmp_path, monkeypatch):
    path, cache, manifest = fixture_manifest(tmp_path)
    monkeypatch.setattr(data, 'urlopen', lambda *a, **k: pytest.fail('unexpected network'))
    works = data.load_benchmark(path, cache)
    assert len(works) == 1
    assert works[0]['text_bare'] == 'και μεν δη λογοσ'
    assert works[0]['n_tokens'] == 4
    assert works[0]['provenance']['source_sha256'] == manifest['works'][0]['sha256']
    assert works[0]['provenance']['manifest_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert works[0]['provenance']['parser'] == 'tei-greek-v1'


def test_missing_cache_never_downloads_implicitly(tmp_path, monkeypatch):
    path, cache, _ = fixture_manifest(tmp_path, cached=False)
    monkeypatch.setattr(data, 'urlopen', lambda *a, **k: pytest.fail('unexpected network'))
    with pytest.raises(FileNotFoundError, match='--download'):
        data.load_benchmark(path, cache)


def test_cache_corruption_is_an_error_even_when_download_is_enabled(tmp_path, monkeypatch):
    path, cache, _ = fixture_manifest(tmp_path)
    (cache / 'example.grc1.source').write_bytes(b'altered')
    monkeypatch.setattr(data, 'urlopen', lambda *a, **k: pytest.fail('unexpected network'))
    with pytest.raises(ValueError, match='source checksum mismatch'):
        data.load_benchmark(path, cache, download=True)


def test_download_is_verified_before_atomic_cache_write(tmp_path, monkeypatch):
    path, cache, _ = fixture_manifest(tmp_path, cached=False)
    source = tei('<p>καὶ μὲν δὴ λόγος.</p>')
    monkeypatch.setattr(data, 'urlopen', lambda *a, **k: io.BytesIO(source))
    works = data.load_benchmark(path, cache, download=True)
    assert works[0]['n_tokens'] == 4
    assert (cache / 'example.grc1.source').read_bytes() == source


def test_download_checksum_failure_never_caches_corrupt_bytes(tmp_path, monkeypatch):
    path, cache, _ = fixture_manifest(tmp_path, cached=False)
    monkeypatch.setattr(data, 'urlopen', lambda *a, **k: io.BytesIO(b'wrong download'))
    with pytest.raises(ValueError, match='source checksum mismatch'):
        data.load_benchmark(path, cache, download=True)
    assert not cache.exists()


@pytest.mark.parametrize('field,value,error', [
    ('source_id', '../escape', 'unsafe'),
    ('source_commit', 'master', 'immutable'),
    ('url', 'http://example.test/' + 'a' * 40, 'HTTPS'),
    ('url', 'https://example.test/master/source.xml', 'pinned commit'),
    ('sha256', 'invalid', 'checksum'),
    ('text_sha256', 'invalid', 'normalized text checksum'),
    ('author', '', 'nonempty'),
    ('author', None, 'nonempty'),
    ('n_tokens', 0, 'positive'),
    ('n_tokens', True, 'positive'),
    ('role', 'training_copy', 'role'),
])
def test_invalid_provenance_fails_before_loading(tmp_path, field, value, error):
    path, cache, manifest = fixture_manifest(tmp_path)
    manifest['works'][0][field] = value
    rewrite(path, manifest)
    with pytest.raises(ValueError, match=error):
        data.load_benchmark(path, cache)


@pytest.mark.parametrize('field,value,error', [
    ('n_tokens', 7, 'token count drift'),
    ('text_sha256', 'b' * 64, 'normalized text checksum mismatch'),
    ('parser', 'unversioned', 'unsupported benchmark parser'),
    ('language', 'hbo', "requires language='grc'"),
])
def test_parser_or_normalization_drift_is_not_silently_accepted(tmp_path, field, value, error):
    path, cache, manifest = fixture_manifest(tmp_path)
    manifest['works'][0][field] = value
    rewrite(path, manifest)
    with pytest.raises(ValueError, match=error):
        data.load_benchmark(path, cache)


@pytest.mark.parametrize('changes', [
    {}, {'source_id': 'second', 'sha256': 'b' * 64},
    {'source_id': 'second', 'work_id': 'another'},
])
def test_duplicate_sources_or_editions_cannot_leak_across_work_splits(tmp_path, changes):
    path, cache, manifest = fixture_manifest(tmp_path)
    manifest['works'].append({**manifest['works'][0], **changes})
    rewrite(path, manifest)
    with pytest.raises(ValueError, match='duplicate'):
        data.load_benchmark(path, cache)


def test_witness_controls_are_excluded_from_attribution_by_default(tmp_path):
    path, cache, manifest = fixture_manifest(tmp_path)
    control = {**manifest['works'][0], 'source_id': 'witness', 'role': 'witness_control'}
    manifest['works'].append(control)
    rewrite(path, manifest)
    assert len(data.load_benchmark(path, cache)) == 1
    (cache / 'witness.source').write_bytes((cache / 'example.grc1.source').read_bytes())
    all_works = data.load_benchmark(path, cache, include_controls=True)
    assert [w['role'] for w in all_works] == ['attribution', 'witness_control']
    assert all_works[0]['work_id'] == all_works[1]['work_id']


def test_committed_greek_selection_has_distinct_works_and_shared_genre_challenges():
    manifest = data.read_manifest(data.DEFAULT_MANIFEST)
    works = [w for w in manifest['works'] if w.get('role') == 'attribution']
    assert len(works) == 18
    assert set(Counter(w['author'] for w in works).values()) == {3}
    assert len({w['work_id'] for w in works}) == 18
    assert min(w['n_tokens'] for w in works) >= 2000
    assert len({w['author'] for w in works if w['genre'] == 'forensic_oratory'}) == 3
    assert {w['author'] for w in works if w['topic'] == 'socrates'} == {'Plato', 'Xenophon'}
    assert all(w['license'] == 'CC-BY-SA-4.0' and w['edition']['editors'] for w in works)
    controls = [w for w in manifest['works'] if w.get('role') == 'witness_control']
    assert len(controls) == 2 and len({w['work_id'] for w in controls}) == 1
    assert {w['work_id'] for w in controls}.isdisjoint({w['work_id'] for w in works})


def test_real_cached_greek_sources_match_frozen_parser_output():
    manifest = data.read_manifest(data.DEFAULT_MANIFEST)
    if any(not (data.DEFAULT_CACHE / (w['source_id'] + '.source')).exists() for w in manifest['works']):
        pytest.skip('Run python -m stylometry.benchmark_data --download --include-controls first')
    works = data.load_benchmark(include_controls=True)
    assert len(works) == 20
    assert sum(w['n_tokens'] for w in works if w['role'] == 'attribution') == 199918
    assert all(w['text_bare'] and w['provenance']['source_commit'] for w in works)
