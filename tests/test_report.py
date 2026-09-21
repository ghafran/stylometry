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
                       "estimated_authors": 1, "selection": {"reason": "Fixture"}}],
        "verses": [
            {"id": "english/test/book/1/1", "language": "english", "collection": "test",
             "book": "book", "book_title": "A book", "chapter": 1, "verse": 1,
             "text": 'A sentence, with "quotes".\nAnother line.', "author_id": "english-A01",
             "status": "assigned", "evidence_tokens": 501, "passage_id": "p1",
             "distance_margin": 0.0, "reference_author": "Known writer"},
            {"id": "english/test/book/1/2", "language": "english", "collection": "test",
             "book": "book", "book_title": "A book", "chapter": 1, "verse": 2,
             "text": "…", "author_id": None, "status": "insufficient_text",
             "evidence_tokens": 0, "passage_id": None, "distance_margin": None},
        ],
        "rollups": [{"level": "book", "language": "english", "collection": "test",
                     "book": "book", "chapter": None, "verse": None, "verse_count": 2,
                     "assigned_verse_count": 1, "insufficient_verse_count": 1,
                     "author_count": 1, "author_ids": ["english-A01"],
                     "dominant_author": "english-A01", "author_counts": {"english-A01": 1}}],
        "authors": [{"author_id": "english-A01", "language": "english", "verse_count": 1,
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
    for field in ("languages", "config", "authors", "benchmark", "discovery_validation", "source_coverage"):
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
        "index.html", "report.json", "verses.csv", "rollups.csv"
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
    assert rows[0]["author_id"] == "english-A01"
    assert rows[0]["distance_margin"] == "0.0"
    assert rows[0]["reference_author"] == "Known writer"
    assert rows[1]["author_id"] == ""
    assert rows[1]["evidence_tokens"] == "0"
    assert rows[1]["distance_margin"] == ""
    assert rows[1]["passage_id"] == ""
    with (path.parent / "rollups.csv").open(encoding="utf-8-sig", newline="") as stream:
        rollup = next(csv.DictReader(stream))
    assert rollup["chapter"] == ""
    assert rollup["verse_count"] == "2"
    assert json.loads(rollup["author_ids"]) == ["english-A01"]
    assert json.loads(rollup["author_counts"]) == {"english-A01": 1}


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
    assert "node.textContent = str(text)" in html


def test_empty_report_is_complete_and_readable(tmp_path):
    source = {"schema_version": 1, "config": {}, "languages": [], "verses": [],
              "rollups": [], "authors": [], "benchmark": None}
    path = write_report(source, tmp_path)
    html = path.read_text()
    _assert_display_projection(html, source)
    assert "No matching text" in html
    assert "No inferred author groups" in html
    assert "No English validation results" in html
    with (tmp_path / "verses.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        assert "author_id" in reader.fieldnames
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
    source["verses"][0]["source_reference"] = "Author: book, chapter 1, paragraph 1"
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
        assert row["assigned_verse_count"] == int(bool(original["author_id"]))
        assert row["insufficient_verse_count"] == int(not original["author_id"])
        assert row["low_evidence_verse_count"] == int(original["status"] == "low_evidence")
        assert row["author_ids"] == ([original["author_id"]] if original["author_id"] else [])
        assert row["author_counts"] == ({original["author_id"]: 1} if original["author_id"] else {})


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
    rows = [dict(base, author_id='A', token_count=100, status='low_evidence'),
            dict(base, author_id='B', token_count=50, status='assigned'),
            dict(base, author_id=None, token_count=50, status='insufficient_text')]
    # Author/text/evidence selection highlights a contribution; it must not
    # change the full-scope denominator or drop unassigned/uncertain text.
    result = _contributions(rows, {'author': 'A', 'query': 'no match', 'status': 'assigned'})[0]
    assert result['total'] == 200  # Not three copies of the 1200-token context.
    assert {x['author']: x['share'] for x in result['authors']} == {'A': .5, 'B': .25, None: .25}
    assert result['authors'][0]['lowEvidenceUnits'] == 1
    by_verse = _contributions(rows, measure='verses')[0]
    assert by_verse['total'] == 3
    assert all(x['share'] == pytest.approx(1/3) for x in by_verse['authors'])


@pytest.mark.parametrize(('selection', 'expected'), [
    ({'language': 'eng'}, 150),
    ({'language': 'eng', 'collection': 'C1'}, 100),
    ({'language': 'eng', 'collection': 'C1', 'book': 'B1'}, 60),
    ({'language': 'eng', 'collection': 'C1', 'book': 'B1', 'chapter': '1'}, 30),
])
def test_contributions_follow_each_hierarchy_level(selection, expected):
    rows = [dict(language='eng', collection='C1', book='B1', chapter='1', author_id='A', token_count=10),
            dict(language='eng', collection='C1', book='B1', chapter='1', author_id='B', token_count=20),
            dict(language='eng', collection='C1', book='B1', chapter='2', author_id='A', token_count=30),
            dict(language='eng', collection='C1', book='B2', chapter='1', author_id='A', token_count=40),
            dict(language='eng', collection='C2', book='B3', chapter='1', author_id='A', token_count=50),
            dict(language='grc', collection='C1', book='B1', chapter='1', author_id='A', token_count=999)]
    result = _contributions(rows, selection)
    assert len(result) == 1 and result[0]['language'] == 'eng'
    assert result[0]['total'] == expected
    assert sum(x['share'] for x in result[0]['authors']) == pytest.approx(1)
    all_languages = _contributions(rows)
    assert [(x['language'], x['total']) for x in all_languages] == [('eng', 150), ('grc', 999)]


def test_contributions_handle_missing_counts_zero_words_and_empty_scope():
    rows = [dict(language='eng', author_id=None, token_count=0),
            dict(language='eng', author_id='A', evidence_tokens=1200)]
    result = _contributions(rows)[0]
    assert result['total'] == 0
    assert result['missingWords'] == 1
    assert all(x['share'] == 0 for x in result['authors'])
    assert _contributions(rows, {'language': 'grc'}) == []
    assert _contributions(rows, measure='verses')[0]['total'] == 2
