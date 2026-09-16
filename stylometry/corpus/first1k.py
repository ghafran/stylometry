"""Parse First1KGreek EpiDoc/TEI editions of the apocryphal Acts (Bonnet's text, CC BY-SA 4.0).

These editions are divided into chapters/sections of roughly 80-200 words with no verse numbers, so
each chapter is segmented into verse-sized units at sentence boundaries (``.``, ``·``, ``;``).
"""
from __future__ import annotations

import re
from pathlib import Path

from lxml import etree

from ..greek import clean_display, split_sentences, tokenize
from .meta import FIRST1K_WORKS

TEI = "{http://www.tei-c.org/ns/1.0}"
SKIP = {f"{TEI}note", f"{TEI}label", f"{TEI}head", f"{TEI}fw"}
LEADING_NUM_RE = re.compile(r"^\s*\d+\s+")


def _text(el: etree._Element) -> str:
    parts: list[str] = []

    def rec(node: etree._Element) -> None:
        if node.text:
            parts.append(node.text)
        for child in node:
            if child.tag not in SKIP:
                rec(child)
            if child.tail:
                parts.append(child.tail)

    rec(el)
    return " ".join(parts)


def segment(text: str, target: int = 25, min_tokens: int = 10) -> list[str]:
    """Greedy sentence packing into units of about ``target`` tokens."""
    units: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for sent in split_sentences(text):
        n = len(tokenize(sent))
        if cur and cur_n >= target:
            units.append(" ".join(cur))
            cur, cur_n = [], 0
        cur.append(sent)
        cur_n += n
    if cur:
        if units and cur_n < min_tokens:
            units[-1] = units[-1] + " " + " ".join(cur)
        else:
            units.append(" ".join(cur))
    return units


def load(dir_path: str | Path) -> list[dict]:
    dir_path = Path(dir_path)
    out: list[dict] = []
    order = 0
    for tlg, (code, title, group) in FIRST1K_WORKS.items():
        matches = sorted(dir_path.glob(f"{tlg}.*.xml"))
        if not matches:
            continue
        tree = etree.parse(str(matches[0]))
        for div in tree.iter(f"{TEI}div"):
            if div.get("type") != "textpart":
                continue
            chapter = div.get("n") or "?"
            raw = " ".join(_text(p) for p in div.iter(f"{TEI}p"))
            raw = LEADING_NUM_RE.sub("", clean_display(raw))
            if not raw:
                continue
            for i, unit in enumerate(segment(raw), start=1):
                toks = tokenize(unit)
                if not toks:
                    continue
                order += 1
                out.append(
                    {
                        "source": "first1kgreek",
                        "language": "grc",
                        "witness": "Bonnet",
                        "work": code,
                        "work_title": title,
                        "collection": "noncanonical",
                        "canon": "none",
                        "group": group,
                        "chapter": chapter,
                        "verse": str(i),
                        "ref": f"{title} {chapter}.{i}",
                        "text": unit,
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
