"""All explorer views and navigation respect the selected corpus."""
import json
import shutil
import subprocess

import pytest

from stylometry.report import _HTML_END


pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="Node is optional for browser-code validation"
)


def _function(name):
    """Extract the real UI helper, including its nested object literals."""
    start = _HTML_END.index(f"function {name}(")
    opening = _HTML_END.index("{", start)
    depth, quote, escaped = 0, None, False
    for index in range(opening, len(_HTML_END)):
        char = _HTML_END[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "'\"`":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if not depth:
                return _HTML_END[start:index + 1]
    raise AssertionError(f"Unclosed JavaScript function: {name}")


def _run(expression, *helpers):
    program = "const str = value => value == null ? '' : String(value);\n"
    program += "\n".join(_function(name) for name in helpers)
    program += "\nprocess.stdout.write(JSON.stringify(" + expression + "));"
    result = subprocess.run(["node", "-e", program], check=True, text=True,
                            capture_output=True)
    return json.loads(result.stdout)


def _row(ident, **overrides):
    return {"id": ident, "language": "eng", "collection": "C1", "book": "B1",
            "book_title": "First book", "chapter": 1, "verse": 1, "text": "A quiet morning",
            "style_id": "A", "status": "low_evidence", "token_count": 10,
            "evidence_tokens": 1200, "passage_id": "shared-passage", **overrides}


def _rows():
    base = _row("v1")
    return [
        base,
        {**base, "id": "v2", "verse": 2, "style_id": "B", "status": "assigned", "token_count": 20},
        {**base, "id": "v3", "verse": 3, "style_id": None, "status": "insufficient_text", "token_count": 5},
        {**base, "id": "v4", "chapter": 2, "status": "assigned", "token_count": 30},
        {**base, "id": "v5", "book": "B2", "book_title": "Second book", "style_id": "B", "token_count": 40},
        {**base, "id": "v6", "collection": "C2", "token_count": 50},
        {**base, "id": "v7", "language": "grc", "token_count": 90},
    ]


@pytest.mark.parametrize(("selection", "expected"), [
    ({"language": "eng"}, ["v1", "v2", "v3", "v4", "v5", "v6"]),
    ({"language": "eng", "collection": "C1"}, ["v1", "v2", "v3", "v4", "v5"]),
    ({"language": "eng", "collection": "C1", "book": "B1"}, ["v1", "v2", "v3", "v4"]),
    ({"language": "eng", "collection": "C1", "book": "B1", "chapter": "1"}, ["v1", "v2", "v3"]),
    ({"language": "eng", "collection": "C1", "book": "B1", "chapter": "1", "style": "A"}, ["v1"]),
    ({"language": "eng", "status": "assigned"}, ["v2", "v4"]),
    ({"language": "eng", "status": "low_evidence"}, ["v1", "v5", "v6"]),
    ({"language": "eng", "status": "insufficient_text"}, ["v3"]),
    ({"language": "eng", "query": "SECOND BOOK"}, ["v5"]),
    ({"language": "eng", "query": " v2 "}, ["v2"]),
    ({"language": "eng", "focusId": "v4"}, ["v4"]),
    ({"language": "eng", "book": "B2", "focusId": "v4"}, []),
    ({"language": "eng", "status": "assigned", "query": "Second book"}, []),
])
def test_corpus_selection_applies_all_filters_together(selection, expected):
    expression = f"selectCorpus({json.dumps(_rows())}, {json.dumps(selection)}).map(row => row.id)"
    assert _run(expression, "selectCorpus") == expected


@pytest.mark.parametrize("level", ["language", "collection", "book", "chapter"])
def test_parent_rollups_count_only_the_filtered_chapter(level):
    selection = dict(language="eng", collection="C1", book="B1", chapter="1")
    expression = f"rollupSummary(selectCorpus({json.dumps(_rows())}, {json.dumps(selection)}), {json.dumps(level)})"
    rows = _run(expression, "selectCorpus", "verseRollup", "rollupSummary")
    assert len(rows) == 1
    row = rows[0]
    assert row["level"] == level
    assert row["verse_count"] == 3
    assert row["assigned_verse_count"] == 2
    assert row["insufficient_verse_count"] == 1
    assert row["low_evidence_verse_count"] == 1
    assert row["style_count"] == 2
    assert set(row["style_ids"]) == {"A", "B"}
    assert row["style_counts"] == {"A": 1, "B": 1}


def test_rollups_and_style_groups_share_style_and_evidence_filters():
    selection = dict(language="eng", collection="C1", style="A", status="low_evidence")
    expression = "(() => {const rows = selectCorpus(" + json.dumps(_rows()) + "," + json.dumps(selection) + ");return {rollups:rollupSummary(rows,'language'),styles:styleSummary(rows)};})()"
    actual = _run(expression, "selectCorpus", "verseRollup", "rollupSummary", "styleSummary")
    assert actual["rollups"][0]["verse_count"] == 1
    assert actual["rollups"][0]["style_counts"] == {"A": 1}
    assert actual["styles"] == [{"language": "eng", "style_id": "A", "verse_count": 1,
                                  "book_count": 1, "collection_count": 1, "token_count": 10}]


def test_rollup_grouping_preserves_language_and_collection_namespaces():
    rows = _run(f"rollupSummary({json.dumps(_rows())}, 'book')", "verseRollup", "rollupSummary")
    counts = {(row["language"], row["collection"], row["book"]): row["verse_count"] for row in rows}
    assert counts == {("eng", "C1", "B1"): 4, ("eng", "C1", "B2"): 1,
                      ("eng", "C2", "B1"): 1, ("grc", "C1", "B1"): 1}


def test_style_counts_use_selected_words_and_distinct_books():
    actual = _run(f"styleSummary({json.dumps(_rows())})", "styleSummary")
    groups = {(row["language"], row["style_id"]): row for row in actual}
    assert set(groups) == {("eng", "A"), ("eng", "B"), ("grc", "A")}
    assert groups["eng", "A"] == dict(language="eng", style_id="A", verse_count=3,
                                       book_count=2, collection_count=2, token_count=90)
    assert groups["eng", "B"]["token_count"] == 60
    assert groups["eng", "B"]["book_count"] == 2
    assert groups["grc", "A"]["token_count"] == 90


def test_empty_and_unassigned_selections_remain_visible_in_rollups():
    unassigned = [{**_row("empty"), "style_id": None, "status": "insufficient_text"}]
    expression = "({emptyRollups:rollupSummary([], 'book'),emptyAuthors:styleSummary([]),unassignedRollups:rollupSummary(" + json.dumps(unassigned) + ", 'book'),unassignedAuthors:styleSummary(" + json.dumps(unassigned) + ")})"
    actual = _run(expression, "verseRollup", "rollupSummary", "styleSummary")
    assert actual["emptyRollups"] == actual["emptyAuthors"] == actual["unassignedAuthors"] == []
    row = actual["unassignedRollups"][0]
    assert row["verse_count"] == row["insufficient_verse_count"] == 1
    assert row["assigned_verse_count"] == row["style_count"] == 0
    assert row["dominant_style"] is None


def _selection():
    return dict(language="eng", collection="C1", book="B1", chapter="1", style="A",
                status="low_evidence", query="quiet", focusId="v1", level="book", tab="hierarchy")


def test_style_navigation_preserves_the_selected_corpus_and_search():
    selection = _selection()
    expression = "(() => {const original=" + json.dumps(selection) + ";return {next:styleSelection(original,'B','grc'),original};})()"
    actual = _run(expression, "styleSelection")
    assert actual["original"] == selection
    assert actual["next"] == {**selection, "style": "B", "focusId": ""}
    no_language = {**selection, "language": ""}
    actual = _run(f"styleSelection({json.dumps(no_language)},'B','eng')", "styleSelection")
    assert actual == {**no_language, "language": "eng", "style": "B", "focusId": ""}


@pytest.mark.parametrize(("level", "next_level"), [
    ("language", "collection"), ("collection", "book"),
    ("book", "chapter"), ("chapter", "verse"), ("verse", "verse"),
])
def test_drill_preserves_deeper_corpus_filters_style_evidence_and_search(level, next_level):
    selection = _selection()
    row = dict(level=level, language="eng", collection=None, book=None, chapter=None,
               id="v1" if level == "verse" else None)
    expression = "(() => {const original=" + json.dumps(selection) + ";return {next:drillSelection(original," + json.dumps(row) + "),original};})()"
    actual = _run(expression, "drillSelection")
    assert actual["original"] == selection
    assert actual["next"] == {**selection, "level": next_level,
                              "focusId": "v1" if level == "verse" else ""}


def test_drill_only_fills_unselected_hierarchy_fields():
    selection = dict(language="", collection="C1", book="", chapter="1", style="A", status="low_evidence")
    row = dict(level="book", language="eng", collection="C2", book="B1", chapter=2)
    actual = _run(f"drillSelection({json.dumps(selection)}, {json.dumps(row)})", "drillSelection")
    assert actual == {**selection, "language": "eng", "book": "B1", "level": "chapter", "focusId": ""}


@pytest.mark.parametrize("level", ["language", "collection", "book", "chapter", "verse"])
def test_breadcrumbs_only_change_view_level_and_clear_single_verse_focus(level):
    selection = _selection()
    expression = "(() => {const original=" + json.dumps(selection) + ";return {next:hierarchySelection(original," + json.dumps(level) + "),original};})()"
    actual = _run(expression, "hierarchySelection")
    assert actual["original"] == selection
    assert actual["next"] == {**selection, "level": level, "focusId": ""}
