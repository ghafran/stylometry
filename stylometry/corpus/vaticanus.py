"""Codex Vaticanus (GA 03).

New Testament: the Center for New Testament Restoration transcription (CC BY-SA 4.0), one verse per
line ``BBCCCVVV text`` in the MES notation: ``\\NNN`` page, ``|`` column and ``/`` line breaks (also inside
words), ``=`` before a nomen sacrum, ``¯`` for the supralinear nu, ``%`` damaged and ``^`` missing letter,
``~`` supplied, ``+``/``-`` verse present/absent, ``x{...}`` the original hand and ``{...}`` / ``a{...}``
correctors.  We keep the original hand and expand the common nomina sacra so the words match the other
witnesses.

Old Testament: Swete's *The Old Testament in Greek according to the Septuagint* prints the text of
Vaticanus (lacunae supplied from Alexandrinus/Sinaiticus); the nathans/lxx-swete tokenisation
(CC BY-SA 4.0, one ``b.c.v token`` per line) is used.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..greek import clean_display, tokenize
from .meta import book_code, book_meta

CNTR_BOOKS = {
    40: "MATT", 41: "MARK", 42: "LUKE", 43: "JOHN", 44: "ACTS", 45: "ROM", 46: "1COR", 47: "2COR", 48: "GAL",
    49: "EPH", 50: "PHIL", 51: "COL", 52: "1THESS", 53: "2THESS", 54: "1TIM", 55: "2TIM", 56: "TITUS",
    57: "PHLM", 58: "HEB", 59: "JAS", 60: "1PET", 61: "2PET", 62: "1JOHN", 63: "2JOHN", 64: "3JOHN",
    65: "JUDE", 66: "REV",
}

# nomen sacrum (unaccented, plain sigma) -> expansion
NOMINA_SACRA = {
    "ισ": "ιησουσ", "ιυ": "ιησου", "ιν": "ιησουν", "ιη": "ιησου", "ιησ": "ιησουσ",
    "χσ": "χριστοσ", "χυ": "χριστου", "χω": "χριστω", "χν": "χριστον", "χρσ": "χριστοσ",
    "θσ": "θεοσ", "θυ": "θεου", "θω": "θεω", "θν": "θεον", "θε": "θεε",
    "κσ": "κυριοσ", "κυ": "κυριου", "κω": "κυριω", "κν": "κυριον", "κε": "κυριε",
    "πνα": "πνευμα", "πνσ": "πνευματοσ", "πνι": "πνευματι", "πνατα": "πνευματα", "πνατων": "πνευματων", "πνασι": "πνευμασι", "πνασιν": "πνευμασιν",
    "πηρ": "πατηρ", "πρσ": "πατροσ", "πρι": "πατρι", "πρα": "πατερα", "περ": "πατερ", "πρεσ": "πατερεσ", "πρων": "πατερων", "πρασ": "πατερασ", "πρασι": "πατρασι", "πρασιν": "πατρασιν",
    "μηρ": "μητηρ", "μρσ": "μητροσ", "μρι": "μητρι", "μρα": "μητερα",
    "υσ": "υιοσ", "υυ": "υιου", "υω": "υιω", "υν": "υιον", "υε": "υιε", "υοι": "υιοι", "υων": "υιων", "υοισ": "υιοισ", "υουσ": "υιουσ",
    "σηρ": "σωτηρ", "σρσ": "σωτηροσ", "σρι": "σωτηρι", "σρα": "σωτηρα",
    "ανοσ": "ανθρωποσ", "ανου": "ανθρωπου", "ανω": "ανθρωπω", "ανον": "ανθρωπον", "ανε": "ανθρωπε", "ανοι": "ανθρωποι", "ανων": "ανθρωπων", "ανοισ": "ανθρωποισ", "ανουσ": "ανθρωπουσ",
    "ουνοσ": "ουρανοσ", "ουνου": "ουρανου", "ουνω": "ουρανω", "ουνον": "ουρανον", "ουνοι": "ουρανοι", "ουνων": "ουρανων", "ουνοισ": "ουρανοισ", "ουνουσ": "ουρανουσ", "ουνε": "ουρανε",
    "δαδ": "δαυειδ", "ιλημ": "ιερουσαληυμ", "ιηλ": "ισραηλ", "ισλ": "ισραηλ", "ισηλ": "ισραηλ",
    "στροσ": "σταυροσ", "στρου": "σταυρου", "στρω": "σταυρω", "στρον": "σταυρον",
}
NOMINA_SACRA["ιλημ"] = "ιερουσαλημ"
# Three-letter contractions (first two letters plus the inflected ending) used by P46, Bezae, P72 and
# others.  The table is consulted only for tokens the transcription marks with "=", so adding a form
# here cannot touch an ordinary word that happens to look the same (χρω "use" vs Χριστῷ).
NOMINA_SACRA.update({
    "ιηυ": "ιησου", "ιην": "ιησουν", "ιησυ": "ιησου",
    "χρυ": "χριστου", "χρν": "χριστον", "χρω": "χριστω",
    "ιηλμ": "ιερουσαλημ",
})

LINE_RE = re.compile(r"^(\d{2})(\d{3})(\d{3}) (.*)$")
CORR_RE = re.compile(r"(?:x\{[^}]*\}(?: \{[^}]*\})*(?: [a-c]\{[^}]*\})*)|(?:\{[^}]*\}(?: [a-c]\{[^}]*\})*)")
BREAK_RE = re.compile(r"\\\d*|\|\d*|⋄\d+|/")
DROP_RE = re.compile(r"[%^~+\-&*$�]")
LIGATURES = {"ϗ": "και", "": "μου", "⳨": "σταυρ", "¯": "ν"}


def _original_hand(m: re.Match) -> str:
    """Replace a correction group by the original hand's reading."""
    s = m.group(0)
    x = re.match(r"x\{([^}]*)\}", s)
    if x:
        return x.group(1)
    first = re.match(r"\{([^}]*)\}", s)
    return first.group(1) if first else ""


def clean_mes(text: str) -> str:
    text = CORR_RE.sub(_original_hand, text)
    text = BREAK_RE.sub("", text)  # line breaks also fall inside words: "ζα/ρε" -> "ζαρε"
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    words = []
    for tok in text.split():
        tok = DROP_RE.sub("", tok)
        if tok.startswith("="):
            base = tok[1:]
            tok = NOMINA_SACRA.get(base, base)
        if tok:
            words.append(tok)
    return " ".join(words)


def load_nt(path: Path) -> list[dict]:
    out: list[dict] = []
    order = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        book_n, chapter, verse, body = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        code = CNTR_BOOKS.get(book_n)
        if not code:
            continue
        text = clean_display(clean_mes(body))
        toks = tokenize(text)
        if not toks:
            continue
        title, collection, canon, group = book_meta(code, "grc")
        order += 1
        out.append(
            {
                "source": "cntr_vaticanus", "language": "grc", "witness": "B", "work": code, "work_title": title,
                "collection": collection, "canon": canon, "group": group, "chapter": str(chapter), "verse": str(verse),
                "ref": f"{title} {chapter}:{verse}", "text": text, "text_bare": " ".join(toks), "n_tokens": len(toks),
                "copyist": None, "supplied_frac": round(body.count("~") / max(len(toks), 1), 3), "has_gap": "^" in body,
                "duplicate_of": None, "order": order,
            }
        )
    return out


def load_swete(dir_path: Path) -> list[dict]:
    out: list[dict] = []
    order = 0
    for path in sorted(dir_path.glob("*.txt")):
        if not path.name[0].isdigit():
            continue
        name = re.sub(r"^\d+\.", "", path.stem).replace("_", " ")
        code = book_code(name)
        if not code:
            print(f"  swete: no book code for {path.name}; skipped")
            continue
        title, collection, canon, group = book_meta(code, "grc")
        verses: dict[tuple[str, str], list[str]] = {}
        order_keys: list[tuple[str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            ref, tok = parts
            bits = ref.split(".")
            if len(bits) != 3:
                continue
            key = (bits[1], bits[2])
            if key not in verses:
                verses[key] = []
                order_keys.append(key)
            verses[key].append(tok)
        for chapter, verse in order_keys:
            text = clean_display(" ".join(verses[(chapter, verse)]))
            toks = tokenize(text)
            if not toks:
                continue
            order += 1
            out.append(
                {
                    "source": "swete_lxx", "language": "grc", "witness": "Swete", "work": code, "work_title": title,
                    "collection": collection, "canon": canon, "group": group, "chapter": chapter, "verse": verse,
                    "ref": f"{title} {chapter}:{verse}", "text": text, "text_bare": " ".join(toks), "n_tokens": len(toks),
                    "copyist": None, "supplied_frac": 0.0, "has_gap": False, "duplicate_of": None, "order": order,
                }
            )
    return out


def load(dir_path: str | Path) -> list[dict]:
    """Swete's Vaticanus OT; the NT of Vaticanus comes in with the other CNTR witnesses (corpus/cntr.py)."""
    dir_path = Path(dir_path)
    out: list[dict] = []
    nt = dir_path / "cntr_03.txt"
    if nt.exists() and not (dir_path.parent / "cntr" / "class1" / "03.txt").exists():
        out += load_nt(nt)
    swete = dir_path / "swete"
    if swete.exists():
        out += load_swete(swete)
    return out
