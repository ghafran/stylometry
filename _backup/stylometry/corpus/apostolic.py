"""Parse the Open Apostolic Fathers plain-text files (Lake's Greek text, CC BY-SA 4.0).

Each line is ``<ref> <text>`` where ref is ``chapter.verse`` (``1.1``), ``book.chapter.verse`` for the
Shepherd (``1.1.1``) or a label such as ``SB.1`` (subscription) / ``EP.1`` (epilogue).
"""
from __future__ import annotations

import re
from pathlib import Path

from ..greek import clean_display, tokenize
from .meta import APOSTOLIC_WORKS

LINE_RE = re.compile(r"^(\S+)\s+(.*\S)\s*$")


def load(texts_dir: str | Path) -> list[dict]:
    texts_dir = Path(texts_dir)
    out: list[dict] = []
    order = 0
    for stem, (code, title, group, dup) in APOSTOLIC_WORKS.items():
        path = texts_dir / f"{stem}.txt"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            m = LINE_RE.match(line)
            if not m:
                continue
            ref, raw = m.groups()
            if re.fullmatch(r"[A-Z]+", raw.replace(" ", "")):  # Latin editorial headings
                continue
            text = clean_display(raw)
            toks = tokenize(text)
            if not toks:
                continue
            parts = ref.split(".")
            chapter, verse = (parts[0], ".".join(parts[1:])) if len(parts) > 1 else (ref, "1")
            order += 1
            out.append(
                {
                    "source": "apostolic_fathers",
                    "language": "grc",
                    "witness": "Lake",
                    "work": code,
                    "work_title": title,
                    "collection": "noncanonical",
                    "canon": "none",
                    "group": group,
                    "chapter": chapter,
                    "verse": verse,
                    "ref": f"{title} {ref}",
                    "text": text,
                    "text_bare": " ".join(toks),
                    "n_tokens": len(toks),
                    "copyist": None,
                    "supplied_frac": 0.0,
                    "has_gap": False,
                    "duplicate_of": dup,
                    "order": order,
                }
            )
    return out
