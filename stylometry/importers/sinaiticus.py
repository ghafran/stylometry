"""Parse the Codex Sinaiticus Project TEI transcription into verse records.

Encoding notes (see XMLspecs_sinaiticus.pdf in the raw repo):

* ``<div type="book">`` / ``<div type="chapter">`` / ``<ab>`` = book / chapter / verse.
  Verse ids look like ``V-B36K1V1-36-JOHN`` (book 36, chapter 1, verse 1, code JOHN).
* ``<w>`` is a word.  ``@norm`` carries the normalised, accented form (present for 87% of words);
  otherwise we use the manuscript letters, which may be split across lines by ``<lb>`` and
  ``<note type="hyphen"/>``.
* ``<app>`` wraps a correction.  ``<rdg type="orig" hand="firsthand">`` is what the original scribe
  wrote; other ``<rdg>`` are later correctors.  We always take the first hand.
* ``<seg type="margin">`` and ``<note>`` hold Eusebian canon numbers, running titles etc. - skipped.
* ``<pb copyist="A|B|D">`` tells which scribe wrote the page.  Most page breaks fall mid-verse, so the
  copyist is tracked in document order and recorded for the point where each verse begins.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from lxml import etree

from .unicode import bare, clean_display, tokenize
from .meta import SINAITICUS_BOOKS

XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
VERSE_ID_RE = re.compile(r"^V-B(\d+)K(\d+)V(\d+)-(\d+)-([A-Z0-9]+)$")
SKIP_TAGS = {"note", "fw"}


def _word_text(w: etree._Element) -> str:
    norm = w.get("norm")
    if norm:
        return norm
    parts: list[str] = []

    def rec(el: etree._Element) -> None:
        if el.text:
            parts.append(el.text)
        for child in el:
            if child.tag not in SKIP_TAGS:
                rec(child)
            if child.tail:
                parts.append(child.tail)

    rec(w)
    # A word broken across two manuscript lines carries a newline between its halves.
    return re.sub(r"\s+", "", "".join(parts))


def _collect(el: etree._Element, out: list[tuple[str, str, bool]], supplied: bool = False) -> None:
    """Depth-first walk of a verse, emitting (kind, text, supplied) tuples for the first hand only."""
    for child in el:
        tag = child.tag
        if tag in SKIP_TAGS or (tag == "seg" and child.get("type") == "margin"):
            continue
        if tag == "app":
            for rdg in child:
                if rdg.tag == "rdg" and rdg.get("type") == "orig":
                    _collect(rdg, out, supplied)
            continue
        if tag == "w":
            out.append(("w", _word_text(child), supplied))
            continue
        if tag == "pc":
            out.append(("pc", (child.text or "").strip(), supplied))
            continue
        if tag == "gap":
            out.append(("gap", "", supplied))
            continue
        _collect(child, out, supplied or tag == "supplied")


def _render(items: list[tuple[str, str, bool]]) -> tuple[str, int, int, bool]:
    """Join words and punctuation into display text; return (text, n_words, n_supplied, has_gap)."""
    pieces: list[str] = []
    n_words = n_supplied = 0
    has_gap = False
    for kind, text, supplied in items:
        if kind == "w":
            text = text.strip()
            if not text:
                continue
            pieces.append(text)
            n_words += 1
            n_supplied += int(supplied)
        elif kind == "pc" and text:
            if pieces:
                pieces[-1] += text
            else:
                pieces.append(text)
        elif kind == "gap":
            has_gap = True
    return clean_display(" ".join(pieces)), n_words, n_supplied, has_gap


def iter_verses(xml_path: str | Path) -> Iterator[dict]:
    """Stream verse records from the transcription in document order."""
    book_code = book_title = chapter = copyist = None
    order = 0
    context = etree.iterparse(
        str(xml_path), events=("start", "end"), load_dtd=False, resolve_entities=False, huge_tree=True
    )
    verse_copyist = None
    for event, el in context:
        tag = el.tag
        if event == "start":
            if tag == "pb":
                copyist = el.get("copyist") or copyist
            elif tag == "div":
                kind = el.get("type")
                if kind == "book":
                    xid = el.get(XML_ID) or ""
                    book_code = xid.rsplit("-", 1)[-1] if "-" in xid else el.get("n")
                    book_title = el.get("title")
                elif kind == "chapter":
                    chapter = el.get("n")
            elif tag == "ab":
                verse_copyist = copyist
            continue

        if tag == "ab":
            xid = el.get(XML_ID) or ""
            m = VERSE_ID_RE.match(xid)
            if m:
                _, chap, verse, _, code = m.groups()
            else:
                chap, verse, code = chapter or "0", el.get("n") or "0", book_code or "UNK"
            if verse == "0":  # running titles / headings
                el.clear()
                continue
            items: list[tuple[str, str, bool]] = []
            _collect(el, items)
            text, n_words, n_supplied, has_gap = _render(items)
            el.clear()
            if n_words == 0:
                continue
            title, collection, canon, group = SINAITICUS_BOOKS.get(
                code, (book_title or code, "LXX", "OT", f"LXX:{code}")
            )
            text_bare = " ".join(tokenize(text))
            order += 1
            yield {
                "source": "sinaiticus",
                "language": "grc",
                "witness": "S",
                "work": code,
                "work_title": title,
                "collection": collection,
                "canon": canon,
                "group": group,
                "chapter": chap,
                "verse": verse,
                "ref": f"{title} {chap}:{verse}",
                "text": text,
                "text_bare": text_bare,
                "n_tokens": len(text_bare.split()),
                "copyist": verse_copyist,
                "supplied_frac": round(n_supplied / n_words, 3),
                "has_gap": has_gap,
                "duplicate_of": None,
                "order": order,
            }
        elif tag == "div":
            el.clear()


def load(xml_path: str | Path) -> list[dict]:
    return list(iter_verses(xml_path))
