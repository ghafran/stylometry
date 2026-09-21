"""A checked-in record of which manuscripts the corpus actually contains.

The texts themselves are committed under ``data/raw`` (see ``data/raw/SOURCES.md`` for the licence and
attribution each carries).  This manifest is the index over them: which witnesses were parsed, how much
text each contributed, a checksum per source, and the licence it is used under.  Only files a loader can
read are hashed, so clone metadata or a superseded edition cannot make the corpus look changed when no
parsed text has moved.

``stylometry manifest --write`` regenerates it; ``stylometry manifest --check`` fails when the corpus no
longer matches, which is what the test in ``tests/test_corpus_manifest.py`` runs.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

MANIFEST_PATH = Path("benchmarks") / "corpus_manifest.json"

# Licence and attribution per raw source directory, as recorded in the README's source table.
SOURCE_LICENCES = {
    "codex-sinaiticus": ("CC BY-NC-SA 3.0", "Codex Sinaiticus Project / ITSEE transcription"),
    "cntr": ("CC BY-SA 4.0", "Center for New Testament Restoration, Alan Bunning"),
    "vaticanus": ("CC BY-SA 4.0", "CNTR transcription; Swete's edition of the Greek Old Testament"),
    "first1kgreek": ("CC BY-SA 4.0", "Open Greek and Latin / First1KGreek"),
    "apostolic-fathers/texts": ("CC BY-SA 4.0", "Open Apostolic Fathers, Lake's Greek text"),
    "oshb": ("Public domain", "Open Scriptures Hebrew Bible, Westminster Leningrad Codex"),
    "mam": ("CC BY-SA 4.0", "Miqra according to the Masorah (MAM), Aleppo-based text"),
    "samaritan": ("CC BY-NC 4.0", "Samaritan Pentateuch, DT-UCPH Text-Fabric dataset"),
    "dss": ("CC BY-NC 4.0", "Dead Sea Scrolls, ETCBC Text-Fabric edition of Abegg's transcription"),
    "inscriptions": ("Public domain readings", "Ketef Hinnom and Nash Papyrus, published readings"),
    "quran": ("Tanzil terms: verbatim copy with attribution", "Tanzil.net Uthmani text"),
    "bukhari": ("Unlicense (public domain dedication)", "hadith-api, Fawaz Ahmed; Sahih al-Bukhari, Arabic"),
    "qudsi": ("Unlicense (public domain dedication)", "hadith-api, Fawaz Ahmed; Forty Hadith Qudsi, Arabic"),
    "english": ("Public domain in the United States", "Project Gutenberg texts; PG front and back matter stripped by the loader"),
}

# Manuscripts the project is organised around, as the earliest witness per tradition.
REQUIRED_WITNESSES = {
    "Judaism": {
        "KH": "Ketef Hinnom amulets", "4Q17": "4Q17 / 4QExod-Levᶠ", "1Qisaa": "Great Isaiah Scroll (1QIsaᵃ)",
        "N": "Nash Papyrus", "SP": "Samaritan Pentateuch", "A": "Aleppo Codex", "L": "Leningrad Codex",
    },
    "Greek Old Testament": {"Swete": "Codex Vaticanus (Swete's text)", "S": "Codex Sinaiticus"},
    "Christianity": {
        "P52": "P52 Rylands", "P104": "P104", "P4": "P4", "P64": "P64/P67 (one codex, catalogued as P64)",
        "P66": "P66 Bodmer II", "P46": "P46 Chester Beatty II", "P75": "P75 Bodmer XIV–XV",
        "P45": "P45 Chester Beatty I", "P47": "P47", "03": "Codex Vaticanus (New Testament)",
        "02": "Codex Alexandrinus",
    },
    "Islam": {"T": "Quran, Tanzil Uthmani text", "H": "Sahih al-Bukhari, matn only",
              "Q": "Forty Hadith Qudsi, matn only"},
}

# Requested material with no openly licensed transcription; recorded so it is not rediscovered.
KNOWN_GAPS = [
    {"item": "Early Septuagint fragments (P.Ryl. 458, P.Fouad 266, 4Q119–122, 8HevXIIgr)",
     "reason": "no openly licensed transcription; images are published, the text is not"},
    {"item": "Early Quran manuscripts (Birmingham Mingana 1572a, Ṣanʿāʾ DAM 01-27.1, "
             "Parisino-petropolitanus, Tübingen Ma VI 165, Codex Maʾil, Topkapı, Samarkand)",
     "reason": "no openly licensed transcription; Corpus Coranicum is viewer-only"},
    {"item": "P67", "reason": "not a separate manuscript: the same codex as P64, catalogued under P64"},
]


def _sha256(path: Path, limit_bytes: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        read = 0
        while chunk := fh.read(1 << 20):
            h.update(chunk)
            read += len(chunk)
            if read >= limit_bytes:
                break
    return h.hexdigest()


# Material no loader reads, kept out of both the repository and the checksum so that clone metadata
# or a superseded edition cannot make the corpus look changed when no parsed text has moved.
UNREAD = {
    "codex-sinaiticus": ("archive", "beta-versions_not-for-release"),
    "apostolic-fathers/texts": (),
    "dss": (),
}
UNREAD_SUFFIXES = (".zip",)


def _is_read(relative: Path, subdir: str) -> bool:
    parts = relative.parts
    if any(part == ".git" or part.startswith(".git") for part in parts):
        return False
    if relative.suffix in UNREAD_SUFFIXES:
        return False
    return not any(parts and parts[0] == skip for skip in UNREAD.get(subdir, ()))


def raw_checksums(raw_dir: Path) -> dict:
    """One entry per raw source directory: file count, total bytes and a checksum over the files.

    Only files a loader can read are hashed, so the checksum tracks the text rather than the state of
    whatever working copy the download arrived in.
    """
    out = {}
    for subdir in sorted(SOURCE_LICENCES):
        path = raw_dir / subdir
        if not path.is_dir():
            out[subdir] = {"present": False}
            continue
        files = sorted(p for p in path.rglob("*")
                       if p.is_file() and not p.name.startswith(".")
                       and _is_read(p.relative_to(path), subdir))
        digest = hashlib.sha256()
        total = 0
        for f in files:
            digest.update(f.relative_to(path).as_posix().encode())
            digest.update(_sha256(f).encode())
            total += f.stat().st_size
        licence, attribution = SOURCE_LICENCES[subdir]
        out[subdir] = {"present": True, "n_files": len(files), "bytes": total,
                       "sha256": digest.hexdigest(), "licence": licence, "attribution": attribution}
    return out


def witness_table(witness_units: list[dict]) -> dict:
    """Units and tokens per witness, with the works each one covers."""
    units: Counter = Counter()
    tokens: Counter = Counter()
    works: dict[str, set] = defaultdict(set)
    languages: dict[str, set] = defaultdict(set)
    for v in witness_units:
        w = v["witness"]
        units[w] += 1
        tokens[w] += int(v.get("n_tokens") or 0)
        works[w].add(v.get("duplicate_of") or v["work"])
        languages[w].add(v["language"])
    return {
        w: {"units": units[w], "tokens": tokens[w], "works": len(works[w]),
            "languages": sorted(languages[w])}
        for w in sorted(units, key=lambda x: -units[x])
    }


def coverage(table: dict) -> dict:
    """Whether each manuscript the project is organised around is actually present."""
    out = {}
    for tradition, wanted in REQUIRED_WITNESSES.items():
        entries = []
        for siglum, name in wanted.items():
            got = table.get(siglum)
            entries.append({"siglum": siglum, "name": name, "present": bool(got),
                            "units": (got or {}).get("units", 0), "tokens": (got or {}).get("tokens", 0)})
        out[tradition] = entries
    return out


def build_manifest(verses: list[dict], witness_units: list[dict], raw_dir: Path) -> dict:
    table = witness_table(witness_units)
    by_language = Counter(v["language"] for v in verses)
    dss = [w for w in table if w not in {s for group in REQUIRED_WITNESSES.values() for s in group}
           and any(ch.isdigit() for ch in w) and "Q" in w.upper()]
    return {
        "_comment": "Generated by `stylometry manifest --write`; the sources are under data/raw, see data/raw/SOURCES.md.",
        "primary_units": len(verses),
        "primary_units_by_language": dict(sorted(by_language.items())),
        "witness_units": len(witness_units),
        "witnesses": len(table),
        "dead_sea_scroll_sigla": len(dss),
        "coverage": coverage(table),
        "gaps": KNOWN_GAPS,
        "sources": raw_checksums(raw_dir),
        "witness_units_by_witness": table,
    }


def compare(current: dict, stored: dict) -> list[str]:
    """Differences that matter: a missing manuscript, a lost source, a changed corpus size."""
    problems: list[str] = []
    for tradition, entries in stored.get("coverage", {}).items():
        now = {e["siglum"]: e for e in current.get("coverage", {}).get(tradition, [])}
        for entry in entries:
            got = now.get(entry["siglum"])
            if entry["present"] and not (got and got["present"]):
                problems.append(f"{tradition}: {entry['name']} ({entry['siglum']}) is no longer in the corpus")
    for name, stored_source in stored.get("sources", {}).items():
        got = current.get("sources", {}).get(name, {})
        if stored_source.get("present") and not got.get("present"):
            problems.append(f"source {name} is no longer present")
        elif stored_source.get("present") and got.get("sha256") != stored_source.get("sha256"):
            problems.append(f"source {name} changed on disk (checksum differs from the manifest)")
    if current.get("primary_units") != stored.get("primary_units"):
        problems.append(f"primary units {current.get('primary_units')} != manifest {stored.get('primary_units')}")
    return problems
