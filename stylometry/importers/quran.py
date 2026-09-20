"""Parse the Tanzil Uthmani Quran text (``sura|aya|text`` lines) plus Tanzil's sura metadata.

Each sura is a work (``Q001`` ... ``Q114``); ``group`` is the traditional Meccan/Medinan classification
from the metadata file, which is the standard external check for any stylistic split of the Quran.
"""
from __future__ import annotations

import re
from pathlib import Path

from lxml import etree

from .unicode import clean_display, tokenize

LINE_RE = re.compile(r"^(\d+)\|(\d+)\|(.*)$")


def load(dir_path: str | Path) -> list[dict]:
    dir_path = Path(dir_path)
    text_files = sorted(dir_path.glob("quran-*.txt"))
    if not text_files:
        return []
    meta: dict[int, dict] = {}
    xml = dir_path / "quran-data.xml"
    if xml.exists():
        for s in etree.parse(str(xml)).iter("sura"):
            meta[int(s.get("index"))] = {
                "name": s.get("tname") or s.get("name"),
                "arabic": s.get("name"),
                "type": s.get("type"),
                "order": int(s.get("order") or 0),
            }
    out: list[dict] = []
    order = 0
    for line in text_files[0].read_text(encoding="utf-8").splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        sura, aya, raw = int(m.group(1)), int(m.group(2)), m.group(3)
        text = clean_display(raw, "arb")
        toks = tokenize(text, "arb")
        if not toks:
            continue
        info = meta.get(sura, {})
        code = f"Q{sura:03d}"
        title = f"Sura {sura} {info.get('name', '')}".strip()
        order += 1
        out.append(
            {
                "source": "tanzil",
                "language": "arb",
                "witness": "T",
                "work": code,
                "work_title": title,
                "collection": "Quran",
                "canon": "Quran",
                "group": info.get("type", "Quran"),
                "chapter": str(sura),
                "verse": str(aya),
                "ref": f"{info.get('name', code)} {sura}:{aya}",
                "text": text,
                "text_bare": " ".join(toks),
                "n_tokens": len(toks),
                "copyist": None,
                "supplied_frac": 0.0,
                "has_gap": False,
                "duplicate_of": None,
                "order": order,
                "chronological_order": info.get("order"),
            }
        )
    return out
