"""Portable report exports preserve attributions and cannot inject HTML."""
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from stylometry.report import _HTML_END, write_html_report, write_report


def _result():
    return {
        "schema_version": 1,
        "config": {"passage_tokens": 500},
        "languages": [{"language": "english", "verse_count": 2, "passage_count": 1,
                       "estimated_styles": 1, "selection": {"reason": "Fixture"}}],
        "verses": [
            {"id": "english/test/book/1/1", "language": "english", "collection": "test",
             "book": "book", "book_title": "A book", "chapter": 1, "verse": 1,
             "text": 'A sentence, with "quotes".\nAnother line.', "style_id": "english-S01",
             "status": "assigned", "evidence_tokens": 501, "passage_id": "p1",
             "distance_margin": 0.0, "reference_author": "Known writer"},
            {"id": "english/test/book/1/2", "language": "english", "collection": "test",
             "book": "book", "book_title": "A book", "chapter": 1, "verse": 2,
             "text": "…", "style_id": None, "status": "insufficient_text",
             "evidence_tokens": 0, "passage_id": None, "distance_margin": None},
        ],
        "rollups": [{"level": "book", "language": "english", "collection": "test",
                     "book": "book", "chapter": None, "verse": None, "verse_count": 2,
                     "assigned_verse_count": 1, "insufficient_verse_count": 1,
                     "style_count": 1, "style_ids": ["english-S01"],
                     "dominant_style": "english-S01", "style_counts": {"english-S01": 1}}],
        "styles": [{"style_id": "english-S01", "language": "english", "verse_count": 1,
                     "book_count": 1, "collection_count": 1, "token_count": 501,
                     "examples": ["english/test/book/1/1"]}],
        "benchmark": None,
    }


def _embedded_json(html):
    match = re.search(r'<script type="application/json" id="report-data">(.*?)</script>',
                      html, flags=re.DOTALL)
    assert match is not None
    return match.group(1)


def _assert_display_projection(html, source):
    payload = json.loads(_embedded_json(html))
    rows = [dict(zip(payload["verse_fields"], values)) for values in payload["verse_rows"]]
    assert len(rows) == len(source["verses"])
    for original, row in zip(source["verses"], rows):
        for field in payload["verse_fields"]:
            assert row[field] == original.get(field)
    for field in ("languages", "config", "styles", "benchmark", "discovery_validation", "source_coverage"):
        if field in source:
            assert payload[field] == source[field]
    assert payload["rollups"] == [r for r in source["rollups"] if r["level"] != "verse"]
    return payload


def test_portable_report_and_exports_preserve_attributions(tmp_path):
    source = _result()
    path = write_report(source, tmp_path / "nested" / "report")
    assert path.name == "index.html"
    assert path.parent.is_dir()
    assert {p.name for p in path.parent.iterdir()} == {
        "index.html", "analysis.html", "expected.html", "report.json", "verses.csv", "rollups.csv"
    }
    assert json.loads((path.parent / "report.json").read_text()) == source
    html = path.read_text()
    _assert_display_projection(html, source)
    assert 'href="verses.csv" download' in html
    assert 'href="rollups.csv" download' in html
    assert 'href="report.json" download' in html
    assert "fetch(" not in html
    with (path.parent / "verses.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(source["verses"])
    assert rows[0]["text"] == source["verses"][0]["text"]
    assert rows[0]["style_id"] == "english-S01"
    assert rows[0]["distance_margin"] == "0.0"
    assert rows[0]["reference_author"] == "Known writer"
    assert rows[1]["style_id"] == ""
    assert rows[1]["evidence_tokens"] == "0"
    assert rows[1]["distance_margin"] == ""
    assert rows[1]["passage_id"] == ""
    with (path.parent / "rollups.csv").open(encoding="utf-8-sig", newline="") as stream:
        rollup = next(csv.DictReader(stream))
    assert rollup["chapter"] == ""
    assert rollup["verse_count"] == "2"
    assert json.loads(rollup["style_ids"]) == ["english-S01"]
    assert json.loads(rollup["style_counts"]) == {"english-S01": 1}


def test_untrusted_text_cannot_close_the_data_script(tmp_path):
    source = _result()
    attack = '</script><script>alert("x")</script><!-- & <img src=x onerror=alert(1)>\u2028\u2029'
    source["verses"][0]["text"] = attack
    source["verses"][0]["book_title"] = attack
    source["verses"][0]["reference_author"] = attack
    source["config"]["user_label"] = attack
    html = write_report(source, tmp_path).read_text()
    payload = _embedded_json(html)
    _assert_display_projection(html, source)
    assert "<" not in payload
    assert ">" not in payload
    assert "&" not in payload
    assert "\u2028" not in payload
    assert "\u2029" not in payload
    assert html.count("</script>") == 2
    assert "innerHTML" not in html
    analysis = (tmp_path / "analysis.html").read_text()
    assert analysis.count("</script>") == 2
    assert "innerHTML" not in analysis
    for marker in ("<", ">", "&", "\u2028", "\u2029"):
        assert marker not in _embedded_json(analysis)
    assert "node.textContent = str(text)" in html


def test_empty_report_is_complete_and_readable(tmp_path):
    source = {"schema_version": 1, "config": {}, "languages": [], "verses": [],
              "rollups": [], "styles": [], "benchmark": None}
    path = write_report(source, tmp_path)
    html = path.read_text()
    _assert_display_projection(html, source)
    assert "No matching text" in html
    assert "No inferred style groups" in html
    # Validation and method are their own page now, reached from the explorer's top navigation.
    assert 'href="analysis.html#validation"' in html and 'href="analysis.html#method"' in html
    assert "No English validation results" not in html
    analysis = (tmp_path / "analysis.html").read_text()
    assert "No English validation results" in analysis
    assert "verse_rows" not in analysis, "the corpus table never reaches the summary page"
    with (tmp_path / "verses.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        assert "style_id" in reader.fieldnames
        assert list(reader) == []


def test_unicode_text_round_trips(tmp_path):
    source = _result()
    source["verses"][0]["text"] = "בְּרֵאשִׁית ἀρχῇ 中文 العربية"
    path = write_report(source, tmp_path)
    _assert_display_projection(path.read_text(), source)
    with (tmp_path / "verses.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert next(csv.DictReader(stream))["text"] == source["verses"][0]["text"]


def test_html_omits_duplicate_evidence_and_keeps_full_exports(tmp_path):
    source = _result()
    source["passages"] = [{"id": "p1", "verse_ids": [v["id"] for v in source["verses"]]}]
    source["verses"][0]["source_dir"] = "internal/source/location"
    source["verses"][0]["source_reference"] = "Style: book, chapter 1, paragraph 1"
    source["rollups"].append({"level": "verse", "id": source["verses"][0]["id"], "verse_count": 1})
    source["discovery_validation"] = {"known_author_count": 13, "count_error": 6}
    source["source_coverage"] = {"source_policy": "One primary witness per book.", "parsed_units": 3,
                                 "primary_units": 2, "alternate_witness_units": 1}
    original = json.dumps(source)
    path = write_report(source, tmp_path)
    payload = _assert_display_projection(path.read_text(), source)
    assert "passages" not in payload
    assert "verses" not in payload
    assert "source_dir" not in payload["verse_fields"]
    assert "source_reference" in payload["verse_fields"]
    assert json.loads((tmp_path / "report.json").read_text()) == source
    with (tmp_path / "rollups.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert [row["level"] for row in csv.DictReader(stream)] == ["book", "verse"]
    assert json.dumps(source) == original  # Rendering must not alter scientific results.


def test_html_only_refresh_does_not_touch_scientific_exports(tmp_path):
    source = _result()
    write_report(source, tmp_path)
    before = {name: (tmp_path / name).read_bytes() for name in ("report.json", "verses.csv", "rollups.csv")}
    source["source_coverage"] = {"parsed_units": 5, "primary_units": 2, "alternate_witness_units": 3}
    path = write_html_report(source, tmp_path)
    _assert_display_projection(path.read_text(), source)
    assert {name: (tmp_path / name).read_bytes() for name in before} == before


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is optional for browser-code validation")
def test_on_demand_verse_rollups_preserve_counts_and_references():
    source = _result()
    source["verses"].append({**source["verses"][0], "id": "third", "verse": 3, "status": "low_evidence"})
    function = re.search(r"function verseRollup\(row\)\{.*?\}\n", _HTML_END).group(0)
    program = function + "\nprocess.stdout.write(JSON.stringify(" + json.dumps(source["verses"]) + ".map(verseRollup)));"
    actual = json.loads(subprocess.run(["node", "-e", program], check=True, text=True, capture_output=True).stdout)
    for original, row in zip(source["verses"], actual):
        assert row["level"] == "verse"
        assert row["verse_count"] == 1
        for field in ("id", "language", "collection", "book", "book_title", "chapter", "verse"):
            assert row[field] == original[field]
        assert row["assigned_verse_count"] == int(bool(original["style_id"]))
        assert row["insufficient_verse_count"] == int(not original["style_id"])
        assert row["low_evidence_verse_count"] == int(original["status"] == "low_evidence")
        assert row["style_ids"] == ([original["style_id"]] if original["style_id"] else [])
        assert row["style_counts"] == ({original["style_id"]: 1} if original["style_id"] else {})


def _contributions(rows, selection=None, measure='words'):
    if shutil.which('node') is None:
        pytest.skip('Node is optional for browser-code validation')
    function = re.search(r'function contributionSummary\(.*?\n}\n', _HTML_END, re.DOTALL).group(0)
    program = function + '\nprocess.stdout.write(JSON.stringify(contributionSummary(' + ','.join(
        json.dumps(value) for value in (rows, selection or {}, measure)) + ')));'
    return json.loads(subprocess.run(['node', '-e', program], check=True,
                                    text=True, capture_output=True).stdout)


def test_contribution_shares_count_verse_words_once_and_include_unassigned():
    base = dict(language='eng', collection='C', book='B', chapter='1', evidence_tokens=1200)
    rows = [dict(base, style_id='A', token_count=100, status='low_evidence'),
            dict(base, style_id='B', token_count=50, status='assigned'),
            dict(base, style_id=None, token_count=50, status='insufficient_text')]
    # Style/text/evidence selection highlights a contribution; it must not
    # change the full-scope denominator or drop unassigned/uncertain text.
    result = _contributions(rows, {'style': 'A', 'query': 'no match', 'status': 'assigned'})[0]
    assert result['total'] == 200  # Not three copies of the 1200-token context.
    assert {x['style']: x['share'] for x in result['styles']} == {'A': .5, 'B': .25, None: .25}
    assert result['styles'][0]['lowEvidenceUnits'] == 1
    by_verse = _contributions(rows, measure='verses')[0]
    assert by_verse['total'] == 3
    assert all(x['share'] == pytest.approx(1/3) for x in by_verse['styles'])


@pytest.mark.parametrize(('selection', 'expected'), [
    ({'language': 'eng'}, 150),
    ({'language': 'eng', 'collection': 'C1'}, 100),
    ({'language': 'eng', 'collection': 'C1', 'book': 'B1'}, 60),
    ({'language': 'eng', 'collection': 'C1', 'book': 'B1', 'chapter': '1'}, 30),
])
def test_contributions_follow_each_hierarchy_level(selection, expected):
    rows = [dict(language='eng', collection='C1', book='B1', chapter='1', style_id='A', token_count=10),
            dict(language='eng', collection='C1', book='B1', chapter='1', style_id='B', token_count=20),
            dict(language='eng', collection='C1', book='B1', chapter='2', style_id='A', token_count=30),
            dict(language='eng', collection='C1', book='B2', chapter='1', style_id='A', token_count=40),
            dict(language='eng', collection='C2', book='B3', chapter='1', style_id='A', token_count=50),
            dict(language='grc', collection='C1', book='B1', chapter='1', style_id='A', token_count=999)]
    result = _contributions(rows, selection)
    assert len(result) == 1 and result[0]['language'] == 'eng'
    assert result[0]['total'] == expected
    assert sum(x['share'] for x in result[0]['styles']) == pytest.approx(1)
    all_languages = _contributions(rows)
    assert [(x['language'], x['total']) for x in all_languages] == [('eng', 150), ('grc', 999)]


def test_contributions_handle_missing_counts_zero_words_and_empty_scope():
    rows = [dict(language='eng', style_id=None, token_count=0),
            dict(language='eng', style_id='A', evidence_tokens=1200)]
    result = _contributions(rows)[0]
    assert result['total'] == 0
    assert result['missingWords'] == 1
    assert all(x['share'] == 0 for x in result['styles'])
    assert _contributions(rows, {'language': 'grc'}) == []
    assert _contributions(rows, measure='verses')[0]['total'] == 2


def _overview(rows, measure='words', shown=None):
    """Run the browser's aggregate-chart code and return what it would draw.

    ``rows`` is the whole corpus the page loads; ``shown`` is what survives the current
    filters, which is the only thing the chart itself measures.
    """
    if shutil.which('node') is None:
        pytest.skip('Node is optional for browser-code validation')
    summary = re.search(r'function contributionSummary\(.*?\n}\n', _HTML_END, re.DOTALL).group(0)
    block = re.search(r'const stylePalette=.*?(?=function renderOverview\()', _HTML_END, re.DOTALL).group(0)
    program = (
        "const str = value => value == null ? '' : String(value);\n"
        "const count = value => new Intl.NumberFormat('en-US').format(value || 0);\n"
        f"const verses = {json.dumps(rows)};\n"
        f"const shown = {json.dumps(rows if shown is None else shown)};\n" + summary + block +
        f"const measure = {json.dumps(measure)};\n"
        "const groups = contributionSummary(shown, {}, measure);\n"
        "process.stdout.write(JSON.stringify({summary: overviewSummary(groups, measure),"
        " languages: groups.map(group => ({language: group.language, series: overviewSeries(group)}))}));"
    )
    return json.loads(subprocess.run(['node', '-e', program], check=True,
                                     text=True, capture_output=True).stdout)


def _corpus(style_count, words=None):
    rows = []
    for index in range(style_count):
        rows.append(dict(language='eng', collection='C', book='B', chapter='1', status='assigned',
                         style_id=f'eng-S{index + 1:03d}',
                         token_count=(words[index] if words else style_count - index) * 10))
    return rows


def test_aggregate_chart_gives_the_largest_styles_a_fixed_palette_and_folds_the_rest():
    result = _overview(_corpus(9))
    series = result['languages'][0]['series']
    assert [item['label'] for item in series[:6]] == [f'eng-S{i:03d}' for i in range(1, 7)]
    assert len({item['color'] for item in series[:6]}) == 6
    assert series[6]['label'] == '3 further styles'
    assert [entry['style'] for entry in series[6]['members']] == ['eng-S007', 'eng-S008', 'eng-S009']
    assert series[6]['amount'] == sum(entry['amount'] for entry in series[6]['members'])
    assert sum(item['share'] for item in series) == pytest.approx(1)


def test_filtering_to_fewer_styles_never_repaints_the_ones_that_remain():
    rows = _corpus(9)
    full = {item['label']: item['color'] for item in _overview(rows)['languages'][0]['series']}
    # The chart reads an already-filtered corpus, so a filter arrives here as missing rows.
    narrowed = _overview(rows, shown=[row for row in rows if row['style_id'] in {'eng-S002', 'eng-S005'}])
    series = narrowed['languages'][0]['series']
    assert [item['label'] for item in series] == ['eng-S002', 'eng-S005']
    assert [item['color'] for item in series] == [full['eng-S002'], full['eng-S005']]


def test_aggregate_chart_ranks_by_the_whole_corpus_not_by_the_current_filter():
    # eng-S009 is the largest style overall, so it keeps a palette colour when others are filtered out.
    rows = _corpus(9, words=[1, 1, 1, 1, 1, 1, 1, 1, 90])
    series = _overview(rows)['languages'][0]['series']
    assert series[0]['label'] == 'eng-S009'
    assert series[0]['share'] == pytest.approx(900 / 980)
    assert series[-1]['label'] == '3 further styles'


def test_aggregate_summary_reports_scope_concentration_unassigned_and_evidence():
    rows = [dict(language='eng', collection='C', book='B', chapter='1', style_id='eng-S001',
                 token_count=60, status='low_evidence'),
            dict(language='eng', collection='C', book='B', chapter='1', style_id='eng-S002',
                 token_count=20, status='assigned'),
            dict(language='eng', collection='C', book='B', chapter='1', style_id=None,
                 token_count=20, status='insufficient_text'),
            dict(language='grc', collection='C', book='G', chapter='1', style_id='grc-S001',
                 token_count=100, status='assigned')]
    summary = _overview(rows)['summary']
    assert '3 inferred styles across 2 languages' in summary
    assert '4 text units' in summary
    assert '200 words' in summary
    assert 'grc-S001 with 100.0% of Greek' in summary
    assert '1 text units (25.0%) carry no inferred style' in summary
    assert '1 of 3 assigned units are flagged low evidence' in summary
    assert 'No text matches' in _overview([])['summary']


def test_aggregate_summary_flags_a_selection_that_is_entirely_low_evidence():
    rows = [dict(language='eng', collection='C', book='B', chapter='1', style_id='eng-S001',
                 token_count=10, status='low_evidence')]
    assert 'Every assigned unit here is flagged low evidence' in _overview(rows)['summary']


def test_aggregate_summary_switches_units_and_reports_missing_word_counts():
    rows = [dict(language='eng', collection='C', book='B', chapter='1', style_id='eng-S001',
                 status='assigned'),
            dict(language='eng', collection='C', book='B', chapter='1', style_id='eng-S002',
                 token_count=10, status='assigned')]
    assert 'switch the measure' in _overview(rows)['summary']
    by_verse = _overview(rows, measure='verses')
    assert '2 verses / paragraphs' in by_verse['summary']
    assert [item['share'] for item in by_verse['languages'][0]['series']] == [0.5, 0.5]


def test_aggregate_chart_is_visible_above_the_tabs_under_every_filter(tmp_path):
    html = write_report(_result(), tmp_path).read_text()
    overview = html.index('id="overview"')
    assert overview < html.index('role="tablist"'), 'the chart must sit above the tabbed views'
    assert overview > html.index('id="estimate-warnings"')
    assert html.count('id="contribution-measure"') == 1, 'one measure control governs every view'
    assert 'renderMetrics();renderOverview();' in html, 'the chart redraws with every filter change'
    assert '<option value="assigned">Assigned</option>' in html


def _collection_notes(language, counts):
    """Return what the panel would print beneath a language's chart.

    ``counts`` maps a collection to ``{"words": n, "books": {book: [style, ...]}}``,
    optionally with ``bookWords`` and a ``shared`` list of books whose only evidence
    came from a passage shared with a neighbour. This is what the browser accumulates
    while walking the filtered corpus.
    """
    if shutil.which('node') is None:
        pytest.skip('Node is optional for browser-code validation')
    block = re.search(r'const collectionNotes=.*?(?=function appendCollectionNotes\()',
                      _HTML_END, re.DOTALL).group(0)
    program = (
        "const str = value => value == null ? '' : String(value);\n"
        "const count = value => new Intl.NumberFormat('en-US').format(value || 0);\n"
        "const verses = [];  // the control contrast is exercised by its own tests\n"
        "const data = {config: {min_tokens: 200}};\n" + block +
        f"const source = {json.dumps(counts)};\n"
        "const counts = new Map(Object.entries(source).map(([collection, entry]) => [collection, {\n"
        "  units: entry.units || 0, words: entry.words || 0,\n"
        "  styles: new Set(Object.values(entry.books || {}).flat()),\n"
        "  books: new Map(Object.entries(entry.books || {}).map(([book, list]) => [book, new Set(list)])),\n"
        "  titles: new Map(Object.entries(entry.titles || {})),\n"
        "  bookWords: new Map(Object.keys(entry.books || {}).map(book =>\n"
        "    [book, (entry.bookWords || {})[book] || 0])),\n"
        "  alone: new Set(Object.keys(entry.books || {}).filter(book =>\n"
        "    !(entry.shared || []).includes(book))),\n"
        "}]));\n"
        f"const rows = collectionNotesFor({json.dumps(language)}, counts);\n"
        f"process.stdout.write(JSON.stringify({{rows: rows.map(({{styles, books, titles, ...row}}) => row),"
        f" caveats: collectionCaveats({json.dumps(language)}, rows, '')}}));"
    )
    return json.loads(subprocess.run(['node', '-e', program], check=True,
                                     text=True, capture_output=True).stdout)


def test_arabic_collections_say_whose_speech_each_one_reports():
    result = _collection_notes('arb', {
        'Bukhari': {'units': 7589, 'words': 567106,
                    'books': {'b1': ['arb-S001'], 'b2': ['arb-S001', 'arb-S004']},
                    'titles': {'b2': 'Bukhari 65'}},
        'Quran': {'units': 6236, 'words': 77881,
                  'books': {'s1': ['arb-S001'], 's2': ['arb-S001'], 's3': ['arb-S001', 'arb-S002']},
                  'titles': {'s3': 'Sura 28 Al-Qasas'}},
        'Hadith Qudsi': {'units': 40, 'words': 3260, 'books': {'q1': ['arb-S001']}}})
    notes = {row['collection']: row['note'] for row in result['rows']}
    assert [row['collection'] for row in result['rows']] == ['Bukhari', 'Quran', 'Hadith Qudsi']
    assert 'direct speech of God' in notes['Quran']
    assert 'outside the Quran' in notes['Hadith Qudsi'] and 'his own wording' in notes['Hadith Qudsi']
    assert 'own words and actions' in notes['Bukhari'] and 'al-Bukhari' in notes['Bukhari']
    assert all('peace be upon him' in note for note in notes.values() if 'Quran' not in note[:20])
    caveats = ' '.join(result['caveats'])
    assert 'not on its own evidence about a speaker' in caveats
    assert 'chains of transmission' in caveats
    assert 'Hadith Qudsi holds 3,260 words here' in caveats


def test_english_is_labelled_the_control_corpus_not_a_subject():
    result = _collection_notes('eng', {
        'Novels': {'units': 48495, 'words': 2521994, 'books': {'n1': ['eng-S001']}},
        'Cross-genre': {'units': 20508, 'words': 1677606, 'books': {'c1': ['eng-S002']}},
        'Federalist': {'units': 1289, 'words': 189773, 'books': {'f1': ['eng-S003']}}})
    caveats = ' '.join(result['caveats'])
    assert 'control corpus, not part of the scriptural question' in caveats
    assert 'known in advance' in caveats
    # The Federalist is a small share of English but has ample text; it must not be called thin.
    assert 'Federalist holds' not in caveats
    notes = {row['collection']: row['note'] for row in result['rows']}
    assert 'one genre' in notes['Novels'] and 'genre change' in notes['Cross-genre']
    assert 'disputed' in notes['Federalist']


def test_greek_names_its_witnesses_and_warns_that_the_septuagint_is_two_sources():
    result = _collection_notes('grc', {
        'LXX': {'units': 29459, 'words': 592804, 'books': {'g1': ['grc-S001']}},
        'NT': {'units': 7900, 'words': 136295, 'books': {'n1': ['grc-S002']}},
        'noncanonical': {'units': 3767, 'words': 122770, 'books': {'x1': ['grc-S003']}}})
    notes = {row['collection']: row['note'] for row in result['rows']}
    assert 'Codex Sinaiticus' in notes['NT'] and 'earliest surviving complete copy' in notes['NT']
    assert 'Swete' in notes['LXX']
    assert 'never canonised' in notes['noncanonical']
    caveats = ' '.join(result['caveats'])
    assert 'not one source' in caveats, 'a mixed witness can read as style'
    assert 'translated Greek' in caveats


def test_hebrew_notes_separate_the_manuscripts_from_the_authors():
    result = _collection_notes('hbo', {
        'Tanakh': {'units': 23213, 'words': 308575, 'books': {'t1': ['hbo-S001']}},
        'DSS': {'units': 5659, 'words': 150391, 'books': {'d1': ['hbo-S001']}},
        'inscriptions': {'units': 3, 'words': 273, 'books': {'i1': []}}})
    notes = {row['collection']: row['note'] for row in result['rows']}
    assert 'Leningrad Codex' in notes['Tanakh']
    assert 'Dead Sea Scrolls' in notes['DSS'] and 'fragmentary' in notes['DSS']
    assert 'Ketef Hinnom' in notes['inscriptions'] and 'Nash Papyrus' in notes['inscriptions']
    caveats = ' '.join(result['caveats'])
    assert 'difference between manuscripts before' in caveats
    assert 'inscriptions holds 273 words here' in caveats


def test_undescribed_collections_add_nothing_and_a_language_note_can_stand_alone():
    assert _collection_notes('grc', {'Novels': {'units': 9, 'words': 9000}})['rows'] == []
    assert _collection_notes('hbo', {})['caveats'][0].startswith('Hebrew spans')
    thin = _collection_notes('eng', {'Novels': {'units': 1, 'words': 100, 'books': {'b': ['eng-S001']}}})
    assert thin['caveats'][1].startswith('Novels holds 100 words')


def test_each_collection_states_its_expected_authors_beside_the_measured_count():
    result = _collection_notes('arb', {
        'Quran': {'units': 6236, 'words': 77881,
                  'books': {'s1': ['arb-S001'], 's2': ['arb-S001'], 's3': ['arb-S001', 'arb-S002']},
                  'titles': {'s3': 'Sura 28 Al-Qasas'}},
        'Hadith Qudsi': {'units': 40, 'words': 3260, 'books': {'q1': ['arb-S001']}}})
    rows = {row['collection']: row for row in result['rows']}
    assert 'Meccan suras came before the hijra' in rows['Quran']['expected']
    assert 'not of its speaker' in rows['Quran']['expected']
    assert rows['Quran']['measured'] == (
        '2 inferred styles across 3 books, a median of 1 per book'
        ' and Sura 28 Al-Qasas alone carrying 2.')
    assert rows['Hadith Qudsi']['measured'] == '1 inferred style in its single book.'


def test_measured_counts_report_unassigned_books_rather_than_hiding_them():
    result = _collection_notes('hbo', {
        'DSS': {'units': 5659, 'words': 150391,
                'books': {'a': [], 'b': [], 'c': ['hbo-S001'], 'd': ['hbo-S002']}},
        'inscriptions': {'units': 3, 'words': 273, 'books': {'i1': [], 'i2': []}}})
    rows = {row['collection']: row for row in result['rows']}
    assert rows['DSS']['measured'].startswith('2 inferred styles across 4 books')
    # Empty books are usually empty because nothing in them reaches one passage.
    assert '2 of them carrying none at all, every one shorter than the 200-word floor' in rows['DSS']['measured']
    assert rows['inscriptions']['measured'] == (
        'No text here carries an inferred style under the current filters.')


def test_the_control_corpus_reports_how_far_the_method_oversplits_known_authors():
    # Five novelists are known; anything above five is the method splitting one hand.
    result = _collection_notes('eng', {'Novels': {
        'units': 48495, 'words': 2521994,
        'books': {f'b{index}': [f'eng-S{index:03d}', 'eng-S001'] for index in range(1, 16)},
        'titles': {'b7': 'A Tale of Two Cities'}}})
    row = result['rows'][0]
    assert 'Five authors, named on the title pages' in row['expected']
    assert row['measured'].startswith('15 inferred styles across 15 books')
    assert 'a median of 2 per book' in row['measured']


def _control_contrast(rows):
    """Return the sentence the panel derives from the English control corpus."""
    if shutil.which('node') is None:
        pytest.skip('Node is optional for browser-code validation')
    block = re.search(r'const controlContrast=.*?(?=function appendCollectionNotes\()',
                      _HTML_END, re.DOTALL).group(0)
    program = (
        "const str = value => value == null ? '' : String(value);\n"
        "const count = value => new Intl.NumberFormat('en-US').format(value || 0);\n"
        "const languageNotes = {};\n"
        f"const verses = {json.dumps(rows)};\n" + block +
        "process.stdout.write(JSON.stringify({contrast: controlContrast,"
        " arabic: collectionCaveats('arb', [], controlContrast),"
        " english: collectionCaveats('eng', [], controlContrast)}));"
    )
    return json.loads(subprocess.run(['node', '-e', program], check=True,
                                     text=True, capture_output=True).stdout)


def test_other_languages_are_told_how_far_the_method_oversplits_known_authors():
    rows = [dict(language='eng', book=f'b{book}', book_title=f'Book {book}',
                 reference_author=f'Writer {book % 3}', style_id=f'eng-S{index:03d}')
            for book in range(3) for index in range(4)]
    result = _control_contrast(rows)
    assert '3 of them come back as 4 groups' in result['contrast']
    assert 'Book 0 alone is split into 4' in result['contrast']
    # Arabic gets the warning; English does not, since its own cards already show it.
    assert result['arabic'] == [result['contrast']]
    assert result['english'] == []


def test_no_oversplit_warning_when_the_control_recovers_its_authors():
    exact = [dict(language='eng', book=f'b{index}', book_title=f'Book {index}',
                  reference_author=f'Writer {index}', style_id=f'eng-S{index:03d}')
             for index in range(3)]
    assert _control_contrast(exact)['contrast'] == ''
    assert _control_contrast([dict(language='arb', book='b', book_title='B',
                                   reference_author='X', style_id='arb-S001')])['contrast'] == ''


def _hijra(rows):
    """Run the browser's Meccan/Medinan comparison over Quran verses."""
    if shutil.which('node') is None:
        pytest.skip('Node is optional for browser-code validation')
    block = re.search(r'const medinanSuras=.*?(?=function collectionMeasure\()',
                      _HTML_END, re.DOTALL).group(0)
    program = ("const str = value => value == null ? '' : String(value);\n" + block +
               f"process.stdout.write(JSON.stringify(hijraAgreement({json.dumps(rows)})));")
    return json.loads(subprocess.run(['node', '-e', program], check=True,
                                     text=True, capture_output=True).stdout)


def _sura(number, style, words):
    return dict(collection='Quran', book=f'Q{number:03d}', style_id=style, token_count=words)


def test_the_quran_split_is_measured_against_the_meccan_medinan_division():
    # Sura 2 is Medinan, sura 12 Meccan; a perfect split should agree completely.
    perfect = _hijra([_sura(2, 'arb-S003', 100), _sura(12, 'arb-S004', 100)])
    assert perfect['agreement'] == 1 and perfect['baseline'] == 0.5
    # A split that ignores the division scores no better than giving everything one side.
    blind = _hijra([_sura(2, 'arb-S003', 50), _sura(2, 'arb-S004', 50),
                    _sura(12, 'arb-S003', 50), _sura(12, 'arb-S004', 50)])
    assert blind['agreement'] == 0.5 == blind['baseline']
    partial = _hijra([_sura(2, 'arb-S003', 100), _sura(12, 'arb-S004', 60),
                      _sura(12, 'arb-S003', 40)])
    assert partial['agreement'] == pytest.approx(0.8)  # 100 Medinan + 60 of the 100 Meccan
    assert partial['baseline'] == pytest.approx(0.5)


def test_the_hijra_comparison_ignores_everything_that_is_not_a_tagged_sura():
    assert _hijra([_sura(2, 'arb-S003', 10)]) is None, 'one side alone cannot be scored'
    assert _hijra([_sura(2, None, 10), _sura(12, None, 10)]) is None
    assert _hijra([dict(collection='Bukhari', book='BUKH02', style_id='arb-S001', token_count=99),
                   _sura(2, 'arb-S003', 10), _sura(12, 'arb-S004', 10)])['total'] == 20


def test_an_empty_book_that_is_long_enough_is_not_blamed_on_the_passage_floor():
    result = _collection_notes('hbo', {'DSS': {
        'units': 4, 'words': 900,
        'books': {'short': [], 'long': [], 'tagged': ['hbo-S001'], 'other': ['hbo-S002']},
        'bookWords': {'short': 40, 'long': 800, 'tagged': 30, 'other': 30}}})
    measured = result['rows'][0]['measured']
    assert '2 of them carrying none at all, 1 of those shorter than the 200-word floor' in measured


def test_a_book_tagged_only_from_a_shared_passage_is_declared_as_such():
    result = _collection_notes('arb', {'Quran': {
        'units': 6236, 'words': 77881,
        'books': {'Q002': ['arb-S003'], 'Q112': ['arb-S004'], 'Q113': ['arb-S004']},
        'bookWords': {'Q002': 6140, 'Q112': 20, 'Q113': 25},
        'shared': ['Q112', 'Q113']}})
    measured = result['rows'][0]['measured']
    assert '2 tagged only from a passage shared with neighbouring books' in measured
    # A book standing on its own evidence is never counted among them.
    alone = _collection_notes('arb', {'Quran': {
        'units': 10, 'words': 500, 'books': {'Q002': ['arb-S003']}, 'bookWords': {'Q002': 500}}})
    assert 'shared with neighbouring books' not in alone['rows'][0]['measured']
