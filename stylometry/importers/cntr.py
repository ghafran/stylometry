"""Early New Testament witnesses from the Center for New Testament Restoration transcriptions
(CC BY-SA 4.0, attribution to Alan Bunning / CNTR).

``data/raw/cntr/class1/<GA>.txt`` holds one file per manuscript from the first four centuries: the
papyri (P45, P46, P47, P52, P66, P75 ...) and the great majuscules (02 Alexandrinus, 03 Vaticanus,
04 Ephraemi, 05 Bezae ...).  Each line is ``BBCCCVVV text`` in the MES notation, decoded by
``vaticanus.clean_mes`` (original hand kept, correctors dropped, nomina sacra expanded).

Codex Sinaiticus (01) is skipped here because the ITSEE transcription already provides it.
"""
from __future__ import annotations

from pathlib import Path

from .unicode import clean_display, tokenize
from .meta import book_meta, witness_info
from .vaticanus import CNTR_BOOKS, LINE_RE, clean_mes

SKIP = {"01"}


def load_file(path: Path, witness: str) -> list[dict]:
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
                "source": "cntr", "language": "grc", "witness": witness, "work": code, "work_title": title,
                "collection": collection, "canon": canon, "group": group, "chapter": str(chapter), "verse": str(verse),
                "ref": f"{title} {chapter}:{verse}", "text": text, "text_bare": " ".join(toks), "n_tokens": len(toks),
                "copyist": None, "supplied_frac": round(body.count("~") / max(len(toks), 1), 3), "has_gap": "^" in body,
                "duplicate_of": None, "order": order,
            }
        )
    return out


def load(dir_path: str | Path) -> list[dict]:
    out: list[dict] = []
    for path in sorted((Path(dir_path) / "class1").glob("*.txt")):
        witness = path.stem
        if witness in SKIP:
            continue
        witness_info(witness)  # registers a default description for unknown sigla
        out += load_file(path, witness)
    return out
