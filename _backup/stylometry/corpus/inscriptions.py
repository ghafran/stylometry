"""Tiny pre-Masoretic witnesses: the Nash Papyrus and the two Ketef Hinnom silver amulets.

Readings follow the published editions (Cook 1903 / Albright 1937 for Nash; Barkay et al. 2004 and
Ahituv 2012 for Ketef Hinnom) as reproduced on Wikipedia (CC BY-SA 4.0).  Files hold one physical line
per row: ``NN  text`` with ``[ ]`` around reconstructed letters, ``...`` for lost runs and ``a/b`` for
alternative readings of a damaged letter (the first alternative is kept).  These texts are far too short
to fingerprint an author; they are in the corpus as witnesses, and the ``supplied_frac`` field says how
much of each is reconstruction.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..lang import clean_display, tokenize

LINE_RE = re.compile(r"^(\d\d)\s+(.*?)(?:\s+<-.*)?$")
ALT_RE = re.compile(r"([א-ת])/[א-ת]")


def _clean(raw: str) -> tuple[str, int, int]:
    """Return (text, reconstructed letters, total letters)."""
    raw = ALT_RE.sub(r"\1", raw)
    rec = sum(1 for m in re.finditer(r"\[([^\]]*)\]", raw) for ch in m.group(1) if "א" <= ch <= "ת")
    total = sum(1 for ch in raw if "א" <= ch <= "ת")
    text = re.sub(r"[\[\]]", "", raw).replace("...", " ")
    return text, rec, total


def _parse(path: Path) -> list[tuple[str, list[tuple[str, int, int]]]]:
    """Split a file into (section title, lines) blocks; a file without === headers is one block."""
    blocks: list[tuple[str, list]] = []
    current: list = []
    title = path.stem
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("==="):
            if current:
                blocks.append((title, current))
            title = line.strip("= ").split("(")[0].strip()
            current = []
            continue
        m = LINE_RE.match(line)
        if m:
            current.append(_clean(m.group(2)))
    if current:
        blocks.append((title, current))
    return blocks


WORKS = {
    "nash": ("NASH", "Nash Papyrus", "N"),
    "ketef_hinnom": (None, "Ketef Hinnom", "KH"),
}


def load(dir_path: str | Path) -> list[dict]:
    out: list[dict] = []
    order = 0
    for path in sorted(Path(dir_path).glob("*.txt")):
        meta = WORKS.get(path.stem)
        if not meta:
            continue
        code_default, title_default, witness = meta
        for title, lines in _parse(path):
            code = code_default or title.split()[0].upper()
            work_title = title_default if code_default else f"Ketef Hinnom amulet {code[-1]}"
            text = " ".join(t for t, _, _ in lines)
            rec = sum(r for _, r, _ in lines)
            total = sum(n for _, _, n in lines)
            text = clean_display(text, "hbo")
            toks = tokenize(text, "hbo")
            if not toks:
                continue
            order += 1
            out.append(
                {
                    "source": "inscriptions", "language": "hbo", "witness": witness, "work": code,
                    "work_title": work_title, "collection": "inscriptions", "canon": "none", "group": "Inscriptions",
                    "chapter": "1", "verse": "1", "ref": f"{work_title}", "text": text, "text_bare": " ".join(toks),
                    "n_tokens": len(toks), "copyist": None, "supplied_frac": round(rec / total, 3) if total else 0.0,
                    "has_gap": True, "duplicate_of": None, "order": order,
                }
            )
    return out
