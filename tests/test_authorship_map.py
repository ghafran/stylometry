"""Map graphs retain exact contribution totals within the selected corpus."""
import json
import shutil
import subprocess

import pytest

from stylometry.report import _HTML_END


pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="Node is optional for browser-code validation"
)


def _function(name):
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


def _rows():
    base = {"language": "eng", "collection": "C1", "book": "B1",
            "book_title": "First book", "chapter": 1, "verse": 1,
            "text": "A quiet morning", "author_id": "A", "status": "low_evidence",
            "token_count": 10, "evidence_tokens": 1200, "passage_id": "shared-passage"}
    return [
        {**base, "id": "v1"},
        {**base, "id": "v2", "verse": 2, "author_id": "B", "token_count": 20, "status": "assigned"},
        {**base, "id": "v3", "verse": 3, "author_id": None, "token_count": 5, "status": "insufficient_text"},
        {**base, "id": "v4", "chapter": 2, "token_count": 30},
        {**base, "id": "v5", "book": "B2", "book_title": "Second book", "author_id": "B", "token_count": 40},
        {**base, "id": "v6", "collection": "C2", "token_count": 50},
        {**base, "id": "v7", "language": "grc", "token_count": 90},
    ]


@pytest.mark.parametrize(("level", "group_count"), [
    ("language", 2), ("collection", 3), ("book", 4), ("chapter", 5), ("verse", 7),
])
def test_every_map_level_preserves_all_units_and_their_own_word_counts(level, group_count):
    actual = _run(f"mapBreakdown({json.dumps(_rows())},'{level}')", "mapBreakdown")
    assert len(actual) == group_count
    assert sum(row["verse_count"] for row in actual) == 7
    assert sum(row["token_count"] for row in actual) == 245
    assert sum(row["total"] for row in actual) == 245
    dimensions = ["language", "collection", "book", "chapter", "verse"]
    for row in actual:
        assert row["level"] == level
        assert row["missingWords"] == 0
        assert sum(author["amount"] for author in row["authors"]) == row["total"]
        assert sum(author["units"] for author in row["authors"]) == row["verse_count"]
        assert sum(author["share"] for author in row["authors"]) == pytest.approx(1)
        for dimension in dimensions[dimensions.index(level) + 1:4]:
            assert row[dimension] is None


def test_map_authors_include_low_evidence_and_unassigned_in_word_and_unit_shares():
    expression = "({words:mapBreakdown(" + json.dumps(_rows()) + ",'language'),units:mapBreakdown(" + json.dumps(_rows()) + ",'language','verses')})"
    actual = _run(expression, "mapBreakdown")
    words = next(row for row in actual["words"] if row["language"] == "eng")
    units = next(row for row in actual["units"] if row["language"] == "eng")
    assert (words["verse_count"], words["total"]) == (6, 155)
    assert (units["verse_count"], units["token_count"], units["total"]) == (6, 155, 6)
    assert {entry["author"]: entry["amount"] for entry in words["authors"]} == {"A": 90, "B": 60, None: 5}
    assert {entry["author"]: entry["amount"] for entry in units["authors"]} == {"A": 3, "B": 2, None: 1}
    for group in (words, units):
        for entry in group["authors"]:
            assert entry["share"] == pytest.approx(entry["amount"] / group["total"])


def test_map_namespaces_keep_duplicate_book_chapter_and_verse_ids_separate():
    base = _rows()[0]
    rows = [base, {**base, "language": "grc"},
            {**base, "collection": "C2"}, {**base, "book": "B2"}]
    expected = {("eng", "C1", "B1"), ("grc", "C1", "B1"),
                ("eng", "C2", "B1"), ("eng", "C1", "B2")}
    for level in ("book", "chapter", "verse"):
        actual = _run(f"mapBreakdown({json.dumps(rows)},'{level}')", "mapBreakdown")
        assert {(row["language"], row["collection"], row["book"]) for row in actual} == expected
        assert len(actual) == 4
        assert all(row["total"] == 10 and row["verse_count"] == 1 for row in actual)
        if level == "verse":
            assert all(row["id"] == "v1" and row["verse"] == 1 and
                       row["book_title"] == "First book" for row in actual)


def test_missing_and_invalid_word_counts_never_create_false_graph_shares():
    base = _rows()[0]
    rows = [{**base, "token_count": 0},
            {**base, "id": "missing", "token_count": None, "author_id": None},
            {**base, "id": "negative", "token_count": -5, "author_id": "B"},
            {**base, "id": "string", "token_count": "8", "author_id": "C"}]
    expression = "({words:mapBreakdown(" + json.dumps(rows) + ",'chapter'),units:mapBreakdown(" + json.dumps(rows) + ",'chapter','verses'),empty:mapBreakdown([],'chapter')})"
    actual = _run(expression, "mapBreakdown")
    words, units = actual["words"][0], actual["units"][0]
    assert (words["total"], words["token_count"], words["missingWords"], words["verse_count"]) == (0, 0, 3, 4)
    assert all(entry["share"] == 0 and entry["amount"] == 0 for entry in words["authors"])
    assert units["total"] == 4
    assert all(entry["share"] == 0.25 and entry["amount"] == 1 for entry in units["authors"])
    assert actual["empty"] == []


@pytest.mark.parametrize(("selection", "expected"), [
    ({}, "language"),
    ({"language": "eng"}, "collection"),
    ({"language": "eng", "collection": "C1"}, "book"),
    ({"language": "eng", "collection": "C1", "book": "B1"}, "chapter"),
    ({"language": "eng", "collection": "C1", "book": "B1", "chapter": "1"}, "verse"),
    ({"chapter": "1", "author": "A", "status": "assigned", "query": "quiet"}, "language"),
])
def test_map_starts_at_the_first_unselected_hierarchy_level(selection, expected):
    assert _run(f"mapStartLevel({json.dumps(selection)})", "mapStartLevel") == expected


def test_map_drill_and_back_never_widen_the_shared_corpus_boundary():
    selection = {"language": "eng", "collection": "C1", "chapter": "1"}
    expression = """(() => {
 const original=ROWS, selection=SELECTION;
 const base=selectCorpus(original,selection);
 const book=mapBreakdown(base,'book').find(row=>row.book==='B1');
 const inside=mapScope(base,[book]);
 const verse=mapBreakdown(inside,'verse').find(row=>row.id==='v2');
 const path=[book,verse];
 return {base:base.map(row=>row.id),inside:inside.map(row=>row.id),
  verse:mapScope(base,path).map(row=>row.id),
  back:mapScope(base,path.slice(0,1)).map(row=>row.id),
  root:mapScope(base,[]).map(row=>row.id),pathLength:path.length,selection,original};
})()""".replace("ROWS", json.dumps(_rows())).replace("SELECTION", json.dumps(selection))
    actual = _run(expression, "selectCorpus", "mapBreakdown", "mapScope")
    assert actual["base"] == actual["root"] == ["v1", "v2", "v3", "v5"]
    assert actual["inside"] == actual["back"] == ["v1", "v2", "v3"]
    assert actual["verse"] == ["v2"]
    assert actual["pathLength"] == 2
    assert actual["selection"] == selection
    assert actual["original"] == _rows()


def test_map_path_cannot_reintroduce_author_evidence_or_search_exclusions():
    selection = {"language": "eng", "collection": "C1", "author": "B",
                 "status": "assigned", "query": "quiet", "focusId": "v2"}
    descriptor = {"level": "book", "language": "eng", "collection": "C1", "book": "B1", "chapter": None}
    expression = """(() => {
 const base=selectCorpus(ROWS,SELECTION),path=[DESCRIPTOR];
 return {drill:mapScope(base,path).map(row=>row.id),back:mapScope(base,[]).map(row=>row.id),
  outside:mapScope(base,[{...path[0],language:'grc'}]).map(row=>row.id)};
})()""".replace("ROWS", json.dumps(_rows())).replace("SELECTION", json.dumps(selection)).replace("DESCRIPTOR", json.dumps(descriptor))
    actual = _run(expression, "selectCorpus", "mapScope")
    assert actual == {"drill": ["v2"], "back": ["v2"], "outside": []}


@pytest.mark.parametrize("path", [
    [],
    [{"level": "book", "language": "eng", "collection": "C1", "book": "B1", "chapter": None}],
])
def test_open_matching_text_preserves_an_existing_single_verse_boundary(path):
    selection = {"language": "eng", "collection": "C1", "book": "B1", "chapter": "1",
                 "author": "", "status": "assigned", "query": "quiet", "focusId": "v2"}
    expression = """(() => {
 globalThis.state=SELECTION;globalThis.mapPath=PATH;
 globalThis.mapHighlight={language:'eng',author:'B'};
 for(const name of ['refreshOptions','resetPages','refresh','activate'])globalThis[name]=()=>{};
 openMapText();return state;
})()""".replace("SELECTION", json.dumps(selection)).replace("PATH", json.dumps(path))
    actual = _run(expression, "authorSelection", "drillSelection", "openMapText")
    assert actual["focusId"] == "v2"
    assert actual["author"] == "B"
    for key in ("language", "collection", "book", "chapter", "status", "query"):
        assert actual[key] == selection[key]


@pytest.mark.parametrize(("level", "retained"), [
    ("verse", 3), ("chapter", 3), ("book", 2), ("collection", 1), ("language", 0),
])
def test_changing_map_grouping_retains_ancestors_and_only_trims_coarser_nodes(level, retained):
    path = [{"level": "language", "language": "eng"},
            {"level": "collection", "language": "eng", "collection": "C1"},
            {"level": "book", "language": "eng", "collection": "C1", "book": "B1"}]
    expression = "(() => {const path=" + json.dumps(path) + ";return {next:mapGroupingPath(path," + json.dumps(level) + "),original:path};})()"
    actual = _run(expression, "mapGroupingPath")
    assert actual == {"next": path[:retained], "original": path}
