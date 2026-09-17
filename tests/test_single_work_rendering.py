"""Rendering a run that covers a single work, and the divine-name comparison.

With one work the cross-work statistics are degenerate (ARI against work boundaries is 0 by
construction, purity is 1 by construction), so the report and the dashboard switch to a chapter
breakdown instead.  A pooled passage carries only its first source verse's chapter, so that switch is
limited to verse runs.  Nothing exercised any of this, and mutation testing confirmed the whole branch
could be disabled without a single test failing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stylometry.cluster import run
from stylometry.html import build_site
from stylometry.report import render


def _verse(chapter: int, number: int, words: list[str], work: str = "GEN", language: str = "hbo") -> dict:
    text = " ".join(words)
    return {
        "id": f"{language}:{work}.{chapter}.{number}", "language": language, "work": work,
        "work_title": work, "source": "test", "witness": "L", "collection": "Tanakh", "canon": "Tanakh",
        "group": "Torah", "copyist": None, "chapter": str(chapter), "verse": str(number),
        "order": chapter * 1000 + number, "ref": f"{work} {chapter}:{number}",
        "text": text, "text_bare": text, "n_tokens": len(words),
        "has_gap": False, "supplied_frac": 0.0, "duplicate_of": None,
    }


# Two habits that separate cleanly, so the run really does find more than one group.
PLAIN = ["ויאמר", "אל", "הוא", "כי", "בית", "ויאמר", "אל", "הוא"]
LISTY = ["ואלה", "בני", "שם", "ילד", "את", "ואלה", "בני", "שם"]


def _one_work(chapters: int = 6, per_chapter: int = 12) -> list[dict]:
    """One work whose later chapters are written in the other habit."""
    verses = []
    for chapter in range(1, chapters + 1):
        words = LISTY if chapter > chapters // 2 else PLAIN
        for number in range(1, per_chapter + 1):
            verses.append(_verse(chapter, number, words))
    return verses


def _two_works() -> list[dict]:
    a = [_verse(1, i, PLAIN, work="GEN") for i in range(1, 31)]
    b = [_verse(1, i, LISTY, work="EXOD") for i in range(1, 31)]
    return a + b


def _render(verses: list[dict], out: Path, **kwargs) -> tuple[str, str]:
    run(verses, None, out, window=0, alpha=0, kmin=2, kmax=2, k=2, **kwargs)
    report = render(out)
    dashboard = (build_site(out) / "index.html").read_text(encoding="utf-8")
    return report, dashboard


# --- one work: chapters replace works ------------------------------------------------------------

def test_single_work_report_breaks_down_by_chapter(tmp_path: Path) -> None:
    report, _ = _render(_one_work(), tmp_path / "out")
    assert "## Chapters × style groups" in report
    assert "| chapter | n |" in report
    for chapter in range(1, 7):
        assert f"\n| {chapter} |" in report, f"chapter {chapter} missing from the breakdown"


def test_single_work_report_omits_degenerate_cross_work_statistics(tmp_path: Path) -> None:
    report, _ = _render(_one_work(), tmp_path / "out")
    assert "Only one work is in this run" in report
    assert "Adjusted Rand index of style groups vs. work" not in report
    assert "Mean purity across works" not in report


def test_single_work_dashboard_shows_chapters_not_works(tmp_path: Path) -> None:
    _, dashboard = _render(_one_work(), tmp_path / "out")
    assert '<h2 id="works">Chapters</h2>' in dashboard
    assert ">chapters<" in dashboard and ">mixed chapters<" in dashboard
    assert ">largest group<" in dashboard
    assert "ARI vs. traditional groups" not in dashboard, "a degenerate ARI must not be shown as a headline"
    assert "<th>chapter</th>" in dashboard


def test_multi_work_rendering_is_unchanged(tmp_path: Path) -> None:
    report, dashboard = _render(_two_works(), tmp_path / "out")
    assert "## Chapters × style groups" not in report
    assert "## Works × style groups" in report
    assert "Adjusted Rand index of style groups vs. work" in report
    assert '<h2 id="works">Works</h2>' in dashboard
    assert "ARI vs. traditional groups" in dashboard
    assert ">mixed chapters<" not in dashboard


# --- pooled passages keep the works table --------------------------------------------------------

def _to_passage_mode(out: Path) -> None:
    summary = json.loads((out / "summary.json").read_text())
    summary.update(input_unit="pooled_token_passage", passage_tokens=500,
                   source_mapping_file="passages.json")
    (out / "summary.json").write_text(json.dumps(summary))
    (out / "passages.json").write_text(json.dumps({"passages": [], "exclusions": []}))


def test_single_work_passage_run_does_not_claim_a_chapter_breakdown(tmp_path: Path) -> None:
    """A passage spans many chapters but stores only the first, so a chapter table would misreport it."""
    out = tmp_path / "out"
    run(_one_work(), None, out, window=0, alpha=0, kmin=2, kmax=2, k=2)
    _to_passage_mode(out)
    report = render(out)
    dashboard = (build_site(out) / "index.html").read_text(encoding="utf-8")
    assert "## Chapters × style groups" not in report
    assert '<h2 id="works">Works</h2>' in dashboard
    assert ">chapters<" not in dashboard
    # the degenerate statistics are still suppressed, because there is still only one work
    assert "Only one work is in this run" in report
    assert "Compare the chapter table above" not in report, "there is no chapter table to compare with"


# --- divine names --------------------------------------------------------------------------------

def test_divine_name_table_appears_for_hebrew_and_counts_verses(tmp_path: Path) -> None:
    verses = _one_work()
    verses[0]["text"] = verses[0]["text_bare"] = "ויאמר יהוה אל הוא"
    verses[1]["text"] = verses[1]["text_bare"] = "ויאמר אלהים אל הוא"
    report, _ = _render(verses, tmp_path / "out")
    assert "## Divine names by style group" in report
    assert "| style group | verses | YHWH | Elohim | both | neither |" in report
    assert "an independent check" in report and "character n-grams" in report, \
        "the caveat that the names are visible to the features must stay"


def test_divine_name_table_is_absent_for_greek(tmp_path: Path) -> None:
    verses = [_verse(1, i, ["και", "ο", "λογοσ"], work="MARK", language="grc") for i in range(1, 31)]
    verses += [_verse(2, i, ["ινα", "μεν", "ουν"], work="MARK", language="grc") for i in range(1, 31)]
    report, _ = _render(verses, tmp_path / "out")
    assert "Divine names" not in report


def test_divine_name_counts_match_the_text(tmp_path: Path) -> None:
    """The table is read as evidence, so the counts must follow the text rather than the group sizes."""
    from stylometry.cluster import divine_name_crosstab
    import numpy as np

    verses = [
        _verse(1, 1, ["ויאמר", "יהוה", "אל"]),
        _verse(1, 2, ["ויאמר", "אלהים", "אל"]),
        _verse(1, 3, ["וייצר", "יהוה", "אלהים"]),
        _verse(1, 4, ["ויאמר", "אל", "הוא"]),
    ]
    got = divine_name_crosstab(verses, np.array(["A1", "A1", "A1", "A1"]))
    assert got["A1"] == {"YHWH": 1, "Elohim": 1, "both": 1, "neither": 1}
