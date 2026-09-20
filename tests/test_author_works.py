"""Author works expose the selected corpus without losing hierarchy or text."""
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
            "book_title": "First book", "chapter": 2, "verse": 1,
            "text": "A quiet morning", "author_id": "A", "status": "assigned",
            "evidence_tokens": 1200, "passage_id": "shared-passage"}
    return [
        {**base, "id": "late", "chapter": 10, "token_count": 11},
        {**base, "id": "first", "verse": 2, "token_count": 7, "status": "low_evidence"},
        {**base, "id": "next", "token_count": 13},
        {**base, "id": "book", "book": "B2", "book_title": "Second book", "token_count": 17},
        {**base, "id": "collection", "collection": "C2", "token_count": 19},
        {**base, "id": "language", "language": "grc", "token_count": 71},
        {**base, "id": "other", "author_id": "B", "token_count": 29},
        {**base, "id": "unassigned", "author_id": None, "token_count": 101},
    ]


def test_author_works_count_each_unit_once_across_the_complete_hierarchy():
    actual = _run(f"authorWorks({json.dumps(_rows())},'eng','A')", "authorWorks")
    assert {key: actual[key] for key in (
        "language", "author_id", "verse_count", "token_count", "book_count",
        "chapter_count", "collection_count"
    )} == dict(language="eng", author_id="A", verse_count=5, token_count=67,
              book_count=3, chapter_count=4, collection_count=2)
    collections = {row["collection"]: row for row in actual["collections"]}
    assert set(collections) == {"C1", "C2"}
    assert (collections["C1"]["verse_count"], collections["C1"]["token_count"],
            collections["C1"]["book_count"], collections["C1"]["chapter_count"]) == (4, 48, 2, 3)
    assert (collections["C2"]["verse_count"], collections["C2"]["token_count"],
            collections["C2"]["book_count"], collections["C2"]["chapter_count"]) == (1, 19, 1, 1)
    books = {row["book"]: row for row in collections["C1"]["books"]}
    assert (books["B1"]["book_title"], books["B1"]["verse_count"],
            books["B1"]["token_count"], books["B1"]["chapter_count"]) == ("First book", 3, 31, 2)
    chapters = books["B1"]["chapters"]
    assert [str(row["chapter"]) for row in chapters] == ["2", "10"]
    assert [(row["verse_count"], row["token_count"]) for row in chapters] == [(2, 20), (1, 11)]
    assert {row["id"] for row in chapters[0]["verses"]} == {"first", "next"}
    assert any(row["status"] == "low_evidence" for row in chapters[0]["verses"])


def test_author_works_keep_language_namespaces_and_original_complete_verse_rows():
    expression = """(() => {
      const rows = ROWS, before = JSON.stringify(rows);
      const works = authorWorks(rows,'grc','A');
      const verse = works.collections[0].books[0].chapters[0].verses[0];
      return {works, original:verse === rows.find(row => row.id === 'language'),
              unchanged:JSON.stringify(rows) === before};
    })()""".replace("ROWS", json.dumps(_rows()))
    actual = _run(expression, "authorWorks")
    assert actual["original"] and actual["unchanged"]
    works = actual["works"]
    assert (works["verse_count"], works["token_count"], works["book_count"],
            works["chapter_count"], works["collection_count"]) == (1, 71, 1, 1, 1)
    assert works["collections"][0]["books"][0]["chapters"][0]["verses"] == [_rows()[5]]


@pytest.mark.parametrize(("extra_filters", "expected_ids", "expected_words"), [
    ({}, {"first", "next"}, 20),
    ({"status": "low_evidence"}, {"first"}, 7),
    ({"query": " next "}, {"next"}, 13),
    ({"focusId": "first"}, {"first"}, 7),
    ({"author": "B"}, set(), 0),
])
def test_author_works_respect_the_intersection_of_existing_corpus_filters(
    extra_filters, expected_ids, expected_words
):
    selection = dict(language="eng", collection="C1", book="B1", chapter="2", **extra_filters)
    expression = f"authorWorks(selectCorpus({json.dumps(_rows())},{json.dumps(selection)}),'eng','A')"
    actual = _run(expression, "selectCorpus", "authorWorks")
    verses = [verse for collection in actual["collections"] for book in collection["books"]
              for chapter in book["chapters"] for verse in chapter["verses"]]
    assert {row["id"] for row in verses} == expected_ids
    assert actual["verse_count"] == len(expected_ids)
    assert actual["token_count"] == expected_words


@pytest.mark.parametrize(("rows", "author"), [([], "A"), (_rows(), "absent"), (_rows(), None)])
def test_author_works_do_not_invent_works_for_empty_or_unassigned_authors(rows, author):
    actual = _run(f"authorWorks({json.dumps(rows)},'eng',{json.dumps(author)})", "authorWorks")
    assert actual["collections"] == []
    assert all(actual[key] == 0 for key in (
        "verse_count", "token_count", "book_count", "chapter_count", "collection_count"
    ))
