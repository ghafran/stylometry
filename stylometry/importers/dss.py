"""Dead Sea Scrolls: ETCBC/dss Text-Fabric dataset (Abegg's transcriptions, CC BY-NC 4.0).

* Biblical scrolls (feature ``biblical``): words carry Masoretic ``book_etcbc`` / ``chapter`` / ``verse``,
  so they are regrouped into verse units and become *witnesses* of the biblical work (``1Qisaa`` is the
  witness of ISA, the Leningrad Codex the primary).
* Non-biblical scrolls (Community Rule, War Scroll, Hodayot, pesharim, Damascus Document ...) are works
  in their own right, named after the scroll; their lines are packed into ~25-word units.

Abegg's ``word`` nodes are morphemes (prefixed ו/ב/ה/ל/מ/כ are separate nodes); graphic words are
rebuilt with the ``after`` feature.  Reconstructed letters (``rec`` on signs) are kept so that the editors'
readings stay intact, and their share is recorded as ``supplied_frac``; reconstructed letters and tiny fragments are retained and flagged.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .unicode import clean_display, tokenize
from .meta import book_code, book_meta
from .tfutil import load_tf

FEATURES = "otype scroll fragment line book book_etcbc chapter verse glyph type rec after biblical lang"
MIN_SCROLL_TOKENS = 1      # preserve tiny fragments as works of their own
MAX_SUPPLIED = 1.0           # retain reconstructed units; supplied_frac records uncertainty
UNIT_TARGET = 25


def _word(F, L, w) -> tuple[str, int, int]:
    """Consonantal text of a morpheme node plus (reconstructed, total) letter counts."""
    text, rec, tot = [], 0, 0
    for s in L.d(w, otype="sign"):
        if F.type.v(s) != "cons":
            continue
        g = F.glyph.v(s) or ""
        text.append(g)
        tot += 1
        rec += 1 if F.rec.v(s) else 0
    return "".join(text), rec, tot


def _graphic_words(F, L, words) -> tuple[list[str], int, int]:
    out, cur, rec, tot = [], [], 0, 0
    for w in words:
        if F.type.v(w) != "glyph":
            continue
        t, r, n = _word(F, L, w)
        rec += r
        tot += n
        if t:
            cur.append(t)
        if F.after.v(w):
            if cur:
                out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out, rec, tot


def _record(text: str, **kw) -> dict | None:
    text = clean_display(text, "hbo")
    toks = tokenize(text, "hbo")
    if not toks:
        return None
    rec = {
        "source": "etcbc_dss", "language": "hbo", "copyist": None, "has_gap": bool(kw.get("supplied_frac", 0)), "duplicate_of": None,
        "text": text, "text_bare": " ".join(toks), "n_tokens": len(toks),
    }
    rec.update(kw)
    return rec


def load(dir_path: str | Path) -> list[dict]:
    tf_dirs = sorted((Path(dir_path) / "tf").glob("*/otype.tf"))
    if not tf_dirs:
        return []
    api = load_tf(tf_dirs[-1].parent, FEATURES)
    F, L = api.F, api.L
    out: list[dict] = []
    order = 0
    for scroll in F.otype.s("scroll"):
        name = F.scroll.v(scroll)
        words = [w for w in L.d(scroll, otype="word") if F.type.v(w) == "glyph"]
        if not words:
            continue
        biblical = F.biblical.v(scroll)
        if biblical:
            # verse units keyed on the Masoretic reference
            by_verse: dict[tuple[str, int, int], list] = defaultdict(list)
            for w in words:
                b, c, v = F.book_etcbc.v(w), F.chapter.v(w), F.verse.v(w)
                if not (b and c and v):
                    continue
                code = book_code(str(b))
                if not code:
                    continue
                try:
                    by_verse[(code, int(c), int(v))].append(w)
                except ValueError:
                    continue
            for (code, c, v), ws in sorted(by_verse.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
                gw, rec_n, tot = _graphic_words(F, L, ws)
                if not gw or (tot and rec_n / tot > MAX_SUPPLIED):
                    continue
                title, collection, canon, group = book_meta(code, "hbo")
                r = _record(" ".join(gw), witness=name, work=code, work_title=title, collection=collection, canon=canon,
                            group=group, chapter=str(c), verse=str(v), ref=f"{title} {c}:{v} [{name}]",
                            supplied_frac=round(rec_n / tot, 3) if tot else 0.0)
                if r:
                    order += 1
                    r["order"] = order
                    out.append(r)
            if biblical == 1:
                continue  # purely biblical: done
        # non-biblical text of the scroll (all of it, or the non-biblical part of a mixed scroll)
        lines_out: list[tuple[str, str, str, int, int]] = []
        for ln in L.d(scroll, otype="line"):
            ws = [w for w in L.d(ln, otype="word") if F.type.v(w) == "glyph" and not F.book_etcbc.v(w)]
            gw, rec_n, tot = _graphic_words(F, L, ws)
            if gw:
                lines_out.append((" ".join(gw), F.fragment.v(ln) or "", F.line.v(ln) or "", rec_n, tot))
        total_tokens = sum(len(t.split()) for t, *_ in lines_out)
        if total_tokens < MIN_SCROLL_TOKENS:
            continue
        cave = name[:2] if name[:1].isdigit() else name.split("Q")[0]
        unit, unit_n, rec_n, tot, start = [], 0, 0, 0, None
        idx = 0
        for text, frag, line, r, n in lines_out + [("", "", "", 0, 0)]:
            if unit and (not text or unit_n >= UNIT_TARGET):
                if tot == 0 or rec_n / tot <= MAX_SUPPLIED:
                    idx += 1
                    r_ = _record(" ".join(unit), witness="Q", work=name, work_title=f"Dead Sea Scroll {name}", collection="DSS",
                                 canon="none", group=f"DSS {cave}", chapter=start[0], verse=str(idx),
                                 ref=f"{name} {start[0]}:{start[1]}-{line or start[1]}",
                                 supplied_frac=round(rec_n / tot, 3) if tot else 0.0)
                    if r_:
                        order += 1
                        r_["order"] = order
                        out.append(r_)
                unit, unit_n, rec_n, tot, start = [], 0, 0, 0, None
            if not text:
                continue
            if start is None:
                start = (frag or "?", line or "?")
            unit.append(text)
            unit_n += len(text.split())
            rec_n += r
            tot += n
    return out
