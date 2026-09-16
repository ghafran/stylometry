"""Leningrad Codex: Westminster Leningrad Codex via the OSHB OSIS XML (openscriptures/morphhb).

``<verse osisID="Gen.1.1">`` holds ``<w>`` tokens whose text carries ``/`` at morpheme boundaries.
"""
from __future__ import annotations

from pathlib import Path

from lxml import etree

from ..lang import clean_display, tokenize
from .meta import HEBREW_BOOKS, book_code, book_meta


def canonical_files(paths: list[Path]) -> list[Path]:
    """Order per-book files Torah -> Prophets -> Writings rather than by file name."""
    by_code = {book_code(p.stem): p for p in paths}
    ordered = [by_code[c] for c in HEBREW_BOOKS if c in by_code]
    return ordered + [p for p in paths if p not in ordered]

OSIS = "{http://www.bibletechnologies.net/2003/OSIS/namespace}"
SKIP = {f"{OSIS}note"}


def _verse_text(verse: etree._Element) -> str:
    parts: list[str] = []
    join_next = False
    for el in verse.iter():
        if el.tag in SKIP:
            continue
        if el.tag == f"{OSIS}w" and el.text:
            word = el.text.replace("/", "")
            if join_next and parts:
                parts[-1] += word  # word after a maqaf stays attached, as in the manuscript
            else:
                parts.append(word)
            join_next = False
        elif el.tag == f"{OSIS}seg" and el.text:  # punctuation such as maqaf / sof pasuq
            punct = el.text.strip()
            if parts:
                parts[-1] += punct
            else:
                parts.append(punct)
            join_next = "־" in punct
    return " ".join(parts)


def load(dir_path: str | Path, witness: str = "L", source: str = "oshb") -> list[dict]:
    dir_path = Path(dir_path)
    out: list[dict] = []
    order = 0
    for path in canonical_files(sorted(dir_path.glob("*.xml"))):
        tree = etree.parse(str(path))
        for verse in tree.iter(f"{OSIS}verse"):
            osis = verse.get("osisID") or ""
            try:
                book, chapter, vnum = osis.split(".")
            except ValueError:
                continue
            code = book_code(book)
            if not code:
                continue
            text = clean_display(_verse_text(verse), "hbo")
            toks = tokenize(text, "hbo")
            if not toks:
                continue
            title, collection, canon, group = book_meta(code, "hbo")
            order += 1
            out.append(
                {
                    "source": source,
                    "language": "hbo",
                    "witness": witness,
                    "work": code,
                    "work_title": title,
                    "collection": collection,
                    "canon": canon,
                    "group": group,
                    "chapter": chapter,
                    "verse": vnum,
                    "ref": f"{title} {chapter}:{vnum}",
                    "text": text,
                    "text_bare": " ".join(toks),
                    "n_tokens": len(toks),
                    "copyist": None,
                    "supplied_frac": 0.0,
                    "has_gap": False,
                    "duplicate_of": None,
                    "order": order,
                }
            )
    return out
