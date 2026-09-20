"""Aleppo Codex text: "Miqra according to the Masorah" (MAM) single-file OSIS export (bdenckla/MAM-basics).

``<verse osisID="Isa.1.1">`` contains the pointed, cantillated text directly.  MAM follows the Aleppo
Codex where it survives and other Masoretic manuscripts where it does not.
"""
from __future__ import annotations

from pathlib import Path

from lxml import etree

from .unicode import clean_display, tokenize
from .meta import book_code, book_meta


def load(dir_path: str | Path) -> list[dict]:
    dir_path = Path(dir_path)
    files = sorted(dir_path.glob("*.xml"))
    if not files:
        return []
    out: list[dict] = []
    order = 0
    for path in files:
        for _, verse in etree.iterparse(str(path), events=("end",), tag="{*}verse"):
            # the single-file OSIS export is already in canonical order
            osis = verse.get("osisID") or ""
            parts = osis.split(".")
            if len(parts) != 3:
                verse.clear()
                continue
            code = book_code(parts[0])
            if not code:
                verse.clear()
                continue
            text = clean_display(" ".join(t for t in verse.itertext()), "hbo")
            toks = tokenize(text, "hbo")
            verse.clear()
            if not toks:
                continue
            title, collection, canon, group = book_meta(code, "hbo")
            order += 1
            out.append(
                {
                    "source": "mam",
                    "language": "hbo",
                    "witness": "A",
                    "work": code,
                    "work_title": title,
                    "collection": collection,
                    "canon": canon,
                    "group": group,
                    "chapter": parts[1],
                    "verse": parts[2],
                    "ref": f"{title} {parts[1]}:{parts[2]}",
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
