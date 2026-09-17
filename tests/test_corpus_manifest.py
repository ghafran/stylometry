"""The checked-in manifest must keep describing the corpus that is actually built.

The texts are committed under data/raw and the manifest indexes them. A test that merely read the
manifest would pass forever; these rebuild the coverage table from the corpus on disk and compare, and
skip when the built corpus is absent, since data/processed is not committed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stylometry.manifest import (
    KNOWN_GAPS,
    REQUIRED_WITNESSES,
    SOURCE_LICENCES,
    build_manifest,
    compare,
    coverage,
    witness_table,
)

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "benchmarks" / "corpus_manifest.json"
PROCESSED = REPO / "data" / "processed"


def _load(name: str) -> list[dict]:
    from stylometry.corpus.build import load_corpus

    path = PROCESSED / name
    if not path.exists():
        pytest.skip(f"no {name} under data/processed")
    return load_corpus(path)


# --- the table itself ----------------------------------------------------------------------------

def test_witness_table_counts_units_tokens_and_works() -> None:
    units = [
        {"witness": "S", "work": "MARK", "language": "grc", "n_tokens": 5, "duplicate_of": None},
        {"witness": "S", "work": "JOHN", "language": "grc", "n_tokens": 7, "duplicate_of": None},
        {"witness": "P46", "work": "ROM@P46", "language": "grc", "n_tokens": 3, "duplicate_of": "ROM"},
    ]
    table = witness_table(units)
    assert table["S"] == {"units": 2, "tokens": 12, "works": 2, "languages": ["grc"]}
    assert table["P46"]["works"] == 1, "a secondary witness is counted against the work it witnesses"


def test_coverage_reports_a_missing_manuscript_rather_than_omitting_it() -> None:
    table = {"S": {"units": 10, "tokens": 100, "works": 2, "languages": ["grc"]}}
    got = coverage(table)
    christianity = {e["siglum"]: e for e in got["Christianity"]}
    assert christianity["P46"]["present"] is False and christianity["P46"]["units"] == 0
    assert got["Greek Old Testament"][1]["present"] is True  # Sinaiticus


def test_compare_names_a_manuscript_that_has_disappeared() -> None:
    stored = {"primary_units": 2, "coverage": coverage({"P46": {"units": 5, "tokens": 5, "works": 1,
                                                               "languages": ["grc"]}}), "sources": {}}
    current = {"primary_units": 2, "coverage": coverage({}), "sources": {}}
    problems = compare(current, stored)
    assert any("P46" in p for p in problems), problems


def test_compare_notices_a_changed_source_file() -> None:
    stored = {"primary_units": 1, "coverage": {},
              "sources": {"quran": {"present": True, "sha256": "aaa"}}}
    current = {"primary_units": 1, "coverage": {},
               "sources": {"quran": {"present": True, "sha256": "bbb"}}}
    assert any("checksum differs" in p for p in compare(current, stored))


def test_compare_notices_a_corpus_that_changed_size() -> None:
    stored = {"primary_units": 74130, "coverage": {}, "sources": {}}
    current = {"primary_units": 70000, "coverage": {}, "sources": {}}
    assert any("primary units" in p for p in compare(current, stored))


# --- what is declared ----------------------------------------------------------------------------

def test_every_requested_tradition_is_declared() -> None:
    assert set(REQUIRED_WITNESSES) == {"Judaism", "Greek Old Testament", "Christianity", "Islam"}
    christianity = REQUIRED_WITNESSES["Christianity"]
    for siglum in ("P52", "P104", "P4", "P64", "P66", "P46", "P75", "P45", "P47", "03", "02"):
        assert siglum in christianity, f"{siglum} is in the requested list and must be declared"
    judaism = REQUIRED_WITNESSES["Judaism"]
    for siglum in ("KH", "4Q17", "1Qisaa", "N", "SP", "A", "L"):
        assert siglum in judaism


def test_every_source_carries_a_licence_and_an_attribution() -> None:
    from stylometry.corpus.build import LOADERS

    for subdir, _ in LOADERS:
        assert subdir in SOURCE_LICENCES, f"{subdir} is parsed but its licence is not recorded"
    for subdir, (licence, attribution) in SOURCE_LICENCES.items():
        assert licence.strip() and attribution.strip(), subdir


def test_the_gaps_are_written_down_rather_than_left_to_be_rediscovered() -> None:
    text = json.dumps(KNOWN_GAPS)
    assert "Septuagint" in text and "Quran" in text
    assert "P67" in text and "same codex as P64" in text
    assert all(gap["reason"].strip() for gap in KNOWN_GAPS)


# --- the manifest against the corpus on disk -------------------------------------------------------

def test_manifest_is_checked_in() -> None:
    assert MANIFEST.exists(), "the corpus manifest is the committed evidence of what was analysed"
    stored = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert stored["primary_units"] > 0 and stored["witnesses"] > 0


def test_every_required_manuscript_is_present_in_the_manifest() -> None:
    stored = json.loads(MANIFEST.read_text(encoding="utf-8"))
    missing = [f"{tradition}: {e['name']}" for tradition, entries in stored["coverage"].items()
               for e in entries if not e["present"]]
    assert not missing, "manuscripts the project is organised around are absent: " + "; ".join(missing)


def test_the_built_corpus_still_matches_the_manifest() -> None:
    verses, witness_units = _load("verses.jsonl"), _load("witnesses.jsonl")
    current = build_manifest(verses, witness_units, REPO / "data" / "raw")
    stored = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems = compare(current, stored)
    assert not problems, "corpus no longer matches benchmarks/corpus_manifest.json:\n" + "\n".join(problems)


def test_manifest_unit_counts_agree_with_the_corpus() -> None:
    verses = _load("verses.jsonl")
    stored = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert stored["primary_units"] == len(verses)
    from collections import Counter

    assert stored["primary_units_by_language"] == dict(sorted(Counter(v["language"] for v in verses).items()))
