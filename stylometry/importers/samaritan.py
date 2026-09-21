"""Samaritan Pentateuch: DT-UCPH/sp Text-Fabric dataset (Schorch's transcription of MS Dublin CBL 751).

Slots are signs; ``word`` nodes are morphemes carrying ``g_cons_utf8`` (consonantal Hebrew square script)
and ``trailer`` (what follows the morpheme: a space at graphic-word boundaries, nothing inside a word).
Licence: CC BY-NC 4.0.
"""
from __future__ import annotations

from pathlib import Path

from .unicode import clean_display, tokenize
from .meta import book_code, book_meta
from .tfutil import load_tf

FEATURES = "otype g_cons_utf8 book chapter verse trailer"


def load(dir_path: str | Path) -> list[dict]:
    tf_dir = Path(dir_path) / "tf"
    if not (tf_dir / "otype.tf").exists():
        return []
    api = load_tf(tf_dir, FEATURES)
    F, L = api.F, api.L
    out: list[dict] = []
    order = 0
    for verse in F.otype.s("verse"):
        book = F.book.v(verse) or (F.book.v(L.u(verse, otype="book")[0]) if L.u(verse, otype="book") else None)
        chapter = F.chapter.v(verse) or (F.chapter.v(L.u(verse, otype="chapter")[0]) if L.u(verse, otype="chapter") else None)
        vnum = F.verse.v(verse)
        code = book_code(str(book)) if book else None
        if not code or chapter is None or vnum is None:
            continue
        parts: list[str] = []
        for w in L.d(verse, otype="word"):
            g = F.g_cons_utf8.v(w)
            if g:
                parts.append(g)
            trailer = F.trailer.v(w)
            if trailer and trailer.strip() == "":
                parts.append(" ")
            elif trailer:
                parts.append(trailer)
        text = clean_display("".join(parts).replace("׃", " "), "hbo")
        toks = tokenize(text, "hbo")
        if not toks:
            continue
        title, collection, canon, group = book_meta(code, "hbo")
        order += 1
        out.append(
            {
                "source": "samaritan_tf",
                "language": "hbo",
                "witness": "SP",
                "work": code,
                "work_title": title,
                "collection": collection,
                "canon": canon,
                "group": group,
                "chapter": str(chapter),
                "verse": str(vnum),
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
