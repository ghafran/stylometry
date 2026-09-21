import json
from pathlib import Path

import pytest

from stylometry.corpus import CorpusBuildError, build_corpus, load_corpus, resolve_witnesses, validate_corpus, write_corpus
from stylometry.importers.english import _federalist, _work, strip_gutenberg
from stylometry.importers.hadith import load as load_hadith


def record(**updates):
    return dict(id='eng:c:b:G:1:1', language='eng', collection='c', book='b', book_title='Book',
                chapter='1', verse='1', text='The complete body text.', source='test', witness='G',
                reference_author=None, has_gap=False, **updates)


def test_jsonl_round_trip_and_duplicates(tmp_path):
    path = tmp_path / 'verses.jsonl'
    rows = [record()]
    write_corpus(rows, path)
    assert load_corpus(path) == rows
    path.write_text(json.dumps(rows[0]) + '\n' + json.dumps(rows[0]))
    with pytest.raises(ValueError, match='Duplicate verse id'):
        load_corpus(path)


def test_schema_rejects_missing_text_and_wrong_language():
    verse = record()
    verse['text'] = None
    with pytest.raises(ValueError, match='text must be'):
        validate_corpus([verse])
    verse = record()
    verse['language'] = 'xx'
    with pytest.raises(ValueError, match='unsupported language'):
        validate_corpus([verse])


def test_primary_witness_preserves_repeat_and_prefers_coverage():
    rows = []
    for witness, text in [('S', 'short fragment'), ('L', 'a b c d e f g h i j'), ('L', 'another repeated reference with ten plain tokens for this fixture')]:
        row = record()
        row.update(witness=witness, text=text, n_tokens=len(text.split()))
        rows.append(row)
    selected, report = resolve_witnesses(rows)
    assert len(selected) == 2
    assert {row['witness'] for row in selected} == {'L'}
    assert selected[1]['id'] == selected[0]['id'] + '#2'
    assert selected[1]['verse'] == selected[0]['verse'] == '1'
    assert report[0]['alternatives'][0]['witness'] == 'S'


def test_federalist_removes_labels_and_ambiguous_truth():
    text = '''Project Gutenberg title metadata
*** START OF THE PROJECT GUTENBERG EBOOK TEST ***
FEDERALIST No. 1
Title and publication date

HAMILTON

To the People of the State of New York:

The first complete paragraph stays here.

PUBLIUS.

FEDERALIST No. 18
Joint paper title

MADISON, with HAMILTON

To the People of the State of New York:

Their jointly authored paragraph stays here.

FEDERALIST No. 49
Disputed paper title

MADISON

A disputed paragraph stays in the corpus.
*** END OF THE PROJECT GUTENBERG EBOOK TEST ***
Licence material
'''
    rows = _federalist(text)
    assert len(rows) == 3
    assert rows[0]['reference_author'] == 'Alexander Hamilton'
    assert rows[1]['reference_author'] is None
    assert rows[2]['reference_author'] is None
    assert all('HAMILTON' not in row['text'] and 'MADISON' not in row['text'] for row in rows)


def test_complete_body_keeps_short_dialogue_and_end_past_old_cap():
    repeated = '\n\n'.join('She said a quiet word. ' * 100 for _ in range(50))
    source = f'''*** START OF THE PROJECT GUTENBERG EBOOK BOOK ***
A publisher preface should not become prose.

CHAPTER I

“Tom!”

No answer.

{repeated}

CHAPTER II

The ending remains present in its entirety.
*** END OF THE PROJECT GUTENBERG EBOOK BOOK ***
Licence text
'''
    rows = _work(source, 74, 'Mark Twain', 'TWAIN-TS', 'Tom Sawyer', 'Cross-genre', 'fiction')
    assert rows[0]['text'] == '“Tom!”'
    assert rows[1]['text'] == 'No answer.'
    assert rows[-1]['text'] == 'The ending remains present in its entirety.'
    assert sum(row['n_tokens'] for row in rows) > 20000
    assert rows[-1]['chapter'] == '2'
    assert not any('publisher' in row['text'] or 'Licence' in row['text'] for row in rows)


def test_transcriber_tail_does_not_become_author_prose():
    source = 'It is a truth universally acknowledged\n\nA final sentence.\n\nTranscriber\'s Note:\n\nCorrections from another hand.'
    rows = _work(source, 1342, 'Jane Austen', 'PP', 'Pride and Prejudice', 'Novels', 'fiction')
    assert rows[-1]['text'] == 'A final sentence.'


def test_present_broken_source_fails_and_retains_diagnostics(tmp_path):
    folder = tmp_path / 'quran'
    folder.mkdir()
    (folder / 'quran-uthmani.txt').write_text('invalid downloaded content')
    with pytest.raises(CorpusBuildError) as caught:
        build_corpus(tmp_path)
    sources = {s['source']: s for s in caught.value.report['sources']}
    assert sources['quran']['status'] == 'failed'
    assert sources['english']['status'] == 'missing'


def test_hadith_preserves_full_text_and_unassigned_reports(tmp_path):
    text = 'حدثنا راو عن راو قال هذا نص كامل'
    (tmp_path / 'ara-bukhari.json').write_text(json.dumps({'hadiths': [
        {'hadithnumber': 1, 'reference': {'book': 0}, 'text': text}]}))
    rows = load_hadith(tmp_path)
    assert rows[0]['work'] == 'BUKH00'
    assert rows[0]['text'] == text


def test_custom_input_defaults_and_empty_verse_are_retained(tmp_path):
    verse = record()
    verse['text'] = ''
    verse.pop('source')
    verse.pop('has_gap')
    path = tmp_path / 'custom.jsonl'
    path.write_text(json.dumps(verse))
    loaded = load_corpus(path)
    assert loaded[0]['text'] == ''
    assert loaded[0]['source'] == 'custom'
    assert loaded[0]['has_gap'] is False


def test_shirley_publisher_advertisements_are_not_authored_body():
    source = 'Of late years an abundant shower of curates\n\nThe real ending.\n\nTHE END.\n\nA publisher advertises another book.'
    rows = _work(source, 30486, 'Charlotte Bronte', 'SH', 'Shirley', 'Novels', 'fiction')
    assert rows[-1]['text'] == 'The real ending.'


def test_oshb_excludes_note_descendants_from_original_reading():
    from lxml import etree
    from stylometry.importers.oshb import _verse_text
    verse = etree.fromstring('''<verse xmlns="http://www.bibletechnologies.net/2003/OSIS/namespace">
      <w>בראשית</w><note><rdg><w>נוסף</w></rdg></note><w>ברא</w></verse>''')
    assert _verse_text(verse) == 'בראשית ברא'
