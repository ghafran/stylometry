"""Load the real files under ``data/processed`` and check they still satisfy the code that reads them.

Every other test builds its own fixtures, so nothing else notices when data written by an earlier
version of the pipeline stops being readable by the current one.  That has already happened once:
profiles written before ``coerce_profile`` required ASCII snake_case style tags became unloadable, and
because the loader raises rather than skipping, a single bad row makes the whole file unusable.

These tests skip when a file is absent, so a fresh clone stays green; they fail only when data that is
actually present no longer matches the code.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from stylometry.ai_profile import CATEGORICAL_DIMS, NUMERIC_DIMS, coerce_profile, load_profiles
from stylometry.corpus.build import load_corpus

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
# work.chapter.verse, where the verse part may itself be dotted (Hermas numbers as 1.1.2).
ID_RE = re.compile(r"^(grc|hbo|arb):[^.]+\.[^.]+\..+$")
REQUIRED_VERSE_KEYS = {
    "id", "language", "witness", "work", "work_title", "chapter", "verse", "ref",
    "text", "text_bare", "n_tokens", "group", "duplicate_of",
}


def _profile_files() -> list[Path]:
    if not PROCESSED.is_dir():
        return []
    return sorted(p for p in PROCESSED.glob("profiles*.jsonl")
                  if not p.stem.endswith(("_runs", "_errors", "_batches", "_rejected")))


def _corpus_files() -> list[Path]:
    if not PROCESSED.is_dir():
        return []
    return sorted(p for p in (PROCESSED / "verses.jsonl", PROCESSED / "witnesses.jsonl") if p.exists())


@pytest.mark.parametrize("path", _profile_files() or [None])
def test_stored_profiles_still_load(path: Path | None) -> None:
    """Every stored profile file loads, so earlier measurements stay usable."""
    if path is None:
        pytest.skip("no profiles under data/processed")
    profiles = load_profiles(path)  # raises with a line number on the first unusable record
    assert profiles, f"{path.name} loaded no profiles"
    for ident, rec in profiles.items():
        assert ID_RE.match(ident), f"{path.name}: malformed profile id {ident!r}"
        for dim in NUMERIC_DIMS:
            assert 0.0 <= rec[dim] <= 1.0
        for dim, allowed in CATEGORICAL_DIMS.items():
            assert rec[dim] in allowed


@pytest.mark.parametrize("path", _profile_files() or [None])
def test_every_stored_profile_row_is_individually_valid(path: Path | None) -> None:
    """Name every unusable row at once instead of stopping at the first, which a loader raise does."""
    if path is None:
        pytest.skip("no profiles under data/processed")
    bad: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        if coerce_profile(rec, allow_legacy="provenance" not in rec) is None:
            bad.append(f"line {line_no} ({rec.get('id', '?')}): tags={rec.get('style_tags')}")
    assert not bad, f"{path.name}: {len(bad)} unusable rows\n" + "\n".join(bad[:10])


@pytest.mark.parametrize("path", _corpus_files() or [None])
def test_stored_corpus_matches_the_verse_schema(path: Path | None) -> None:
    """The corpus on disk still carries every field the feature and report code reads."""
    if path is None:
        pytest.skip("no corpus under data/processed")
    verses = load_corpus(path)
    assert verses, f"{path.name} is empty"
    for v in verses:
        missing = REQUIRED_VERSE_KEYS - v.keys()
        assert not missing, f"{path.name}: {v.get('id', '?')} is missing {sorted(missing)}"
        assert ID_RE.match(v["id"]), f"{path.name}: malformed verse id {v['id']!r}"
        assert isinstance(v["text_bare"], str) and v["text_bare"].strip()
        assert v["n_tokens"] == len(v["text_bare"].split()), f"{path.name}: {v['id']} n_tokens disagrees with text_bare"
        assert v["language"] in ("grc", "hbo", "arb")


def test_primary_corpus_ids_are_unique() -> None:
    """Analysis keys every unit by id, so a collision in verses.jsonl would silently drop a unit.

    witnesses.jsonl is exempt: Sinaiticus really does carry part of 1 Chronicles twice, so the same
    reference legitimately appears at two places in one witness.
    """
    path = PROCESSED / "verses.jsonl"
    if not path.exists():
        pytest.skip("no verses.jsonl under data/processed")
    ids = [v["id"] for v in load_corpus(path)]
    repeated = sorted({i for i in ids if ids.count(i) > 1}) if len(ids) != len(set(ids)) else []
    assert not repeated, f"duplicate verse ids: {repeated[:5]}"


def test_witness_units_are_addressable() -> None:
    """Repeated references inside one witness stay distinguishable by reading order."""
    path = PROCESSED / "witnesses.jsonl"
    if not path.exists():
        pytest.skip("no witnesses.jsonl under data/processed")
    keys = [(v["id"], v["order"]) for v in load_corpus(path)]
    assert len(keys) == len(set(keys)), "two witness units share both id and reading order"


def test_primary_corpus_excludes_secondary_witnesses() -> None:
    """verses.jsonl holds one witness per work; the duplicates live in witnesses.jsonl."""
    primary = PROCESSED / "verses.jsonl"
    if not primary.exists():
        pytest.skip("no verses.jsonl under data/processed")
    verses = load_corpus(primary)
    duplicates = [v["id"] for v in verses if v["duplicate_of"]]
    assert not duplicates, f"{len(duplicates)} secondary-witness units leaked into verses.jsonl, e.g. {duplicates[:3]}"
    per_work = {}
    for v in verses:
        per_work.setdefault((v["language"], v["work"]), set()).add(v["witness"])
    mixed = {k: sorted(w) for k, w in per_work.items() if len(w) > 1}
    assert not mixed, f"works drawn from more than one witness: {list(mixed.items())[:3]}"


def test_profiles_refer_to_units_in_the_corpus() -> None:
    """Profile ids resolve to real verses, so a rebuilt corpus cannot silently orphan measurements."""
    primary = PROCESSED / "verses.jsonl"
    files = _profile_files()
    if not primary.exists() or not files:
        pytest.skip("need verses.jsonl and at least one profiles file")
    known = {v["id"] for v in load_corpus(primary)}
    for path in files:
        ids = set(load_profiles(path))
        orphans = sorted(ids - known)
        assert not orphans, f"{path.name}: {len(orphans)} profiles have no verse, e.g. {orphans[:3]}"
