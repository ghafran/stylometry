"""Sahih al-Bukhari: the *matn* of each report, with the isnad and the honorific formulae removed.

Every report in the collection is prefixed by its *isnad*, the chain of transmitters: "A told us, he
said B told us, from C, from D, who said ...". The chain is citation apparatus, not anybody's prose.
It is also enormous - two thirds of the collection by token count - and almost perfectly formulaic, so
leaving it in would separate this collection from the Qur'an on the strength of the footnotes.

What is kept is the *matn*, the body of the report. That is the standard unit in the discipline and it
is what the comparison needs. It is not the same thing as the Prophet's direct speech: the matn also
carries the narrator's framing ("the women said to the Prophet ... so he promised them a day"). The
edition marks direct speech with quotation marks, and those spans are kept separately in
``text_quoted`` so a later test can restrict to them - but they are not used as the boundary, because
measurement showed they do not delimit *his* speech: only 56% of quoted spans in this edition are
preceded by the Prophet named as the speaker. The rest are dialogue - the angel at Hira, Khadija,
Companions - quoted inside a report.

Each *kitab* (book) is a work, ``BUKH01`` ... ``BUKH97``, so the collection has works of comparable
size to the Qur'an's suras rather than one enormous work that whole-work holdout could not use.

311 of the edition's 7,589 records carry ``book: 0``. 307 of them are verbatim duplicates of records
that do carry a book, and 4 are unique. All are dropped: keeping the duplicates would count the same
report twice, and four reports do not make a work. That is 4 reports of about 120 tokens lost out of
339,000, and it is recorded here rather than left to be rediscovered.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..lang import clean_display, tokenize

# The distinctive transmission verbs. `qala` ("he said") is deliberately absent: it is as common in
# the body of a report as in the chain, and including it would let the scan run to the end of the text.
_CHAIN_STEMS = {
    "حدثنا", "حدثني", "حدثتنا", "حدثتني", "حدثه", "حدثهم", "حدثوا", "يحدث", "نحدث",
    "اخبرنا", "اخبرني", "اخبرهم", "اخبره", "انبانا", "انباني", "اخبرتنا", "اخبرتني",
    "ثنا", "سمعت", "سمعه", "سمع", "سمعنا", "عن", "باسناده", "بهذا", "مثله",
}
# Links are joined by waw and fa as often as they stand alone - "wa-akhbarani", "fa-haddathana" -
# and an unprefixed list silently misses them, leaving half a chain at the head of the report.
# `ana` (an abbreviation of "anba'ana") and `kataba` are deliberately absent: they are also the
# ordinary words "I" and "he wrote", so they would extend the chain into the report.
CHAIN_VERBS = _CHAIN_STEMS | {p + v for v in _CHAIN_STEMS for p in ("و", "ف")}
# A real chain has one of those verbs every few words. Once that many words pass without one, the
# chain has ended and the report has begun.
MAX_CHAIN_GAP = 10
# A backstop only, so a record that is nothing but chain cannot be reduced to nothing. It is set high
# on measurement: chains are routinely longer than the report they carry, and at 0.6 the scan stopped
# mid-chain and left transmitters at the head of 11 reports. At 0.9 that falls to 1, and the cost is
# 71 reports of under five words, which are short reports rather than damaged ones.
MAX_CHAIN_SHARE = 0.9
# The last link names its transmitter before the report starts - "from Anas, he said ..." - so the
# boundary has to run past the name. A name ends at the first of these verbs; more than this many
# words without one means the scan has left the chain and the name is abandoned rather than eaten.
HINGE = {"قال", "قالت", "قالوا", "قالا", "يقول", "تقول", "انه", "انها", "انهم", "ان", "فقال", "سمع", "يحدث"}
MAX_NAME_WORDS = 8

# Repeated thousands of times and belonging to the citation apparatus rather than to anyone's style.
HONORIFICS = [
    ("صلي", "الله", "عليه", "وسلم"), ("صلي", "الله", "عليه", "و", "سلم"),
    ("رضي", "الله", "عنهما"), ("رضي", "الله", "عنهم"), ("رضي", "الله", "عنها"),
    ("رضي", "الله", "عنه"), ("رضي", "الله", "عنهن"),
    ("عليه", "السلام"), ("عليها", "السلام"), ("عليهما", "السلام"),
    ("رحمه", "الله"), ("تبارك", "وتعالي"), ("عز", "وجل"), ("سبحانه", "وتعالي"),
]

_BIDI_RE = re.compile(r"[‎‏‪-‮⁦-⁩]")
_QUOTED_RE = re.compile(r'"\s*(.+?)\s*"', re.S)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _BIDI_RE.sub("", text)).strip()


def isnad_end(words: list[str]) -> int:
    """The index in ``words`` (bare forms) where the chain stops and the report begins.

    The chain is dense in transmission verbs and the report is not, so the rule is a density one:
    follow the last link seen, and stop once ``MAX_CHAIN_GAP`` words pass without another.
    """
    limit = max(1, int(len(words) * MAX_CHAIN_SHARE))
    last, gap = None, 0
    for i, word in enumerate(words[:limit]):
        if word in CHAIN_VERBS:
            last, gap = i, 0
        elif last is None:
            if i >= MAX_CHAIN_GAP:
                return 0          # no chain at the head at all; the whole record is the report
        else:
            gap += 1
            if gap > MAX_CHAIN_GAP:
                break
    if last is None:
        return 0
    start = last + 1
    # Run past the last transmitter's name to the verb that introduces the report.
    limit = min(len(words), start + MAX_NAME_WORDS)
    cursor = start
    while cursor < limit and words[cursor] not in HINGE:
        cursor += 1
    if cursor < limit:                 # found the introducing verb; the name lies behind it
        start = cursor
    # "... from Abu Hurayra, he said: the Messenger said" doubles the verb, one belonging to the
    # chain and one to the report. Drop the chain's. A single one is the report's own and is kept,
    # because "the women said to the Prophet ..." loses its subject without it.
    if words[start:start + 2] == ["قال", "قال"]:
        start += 1
    return start


def drop_honorifics(words: list[str], display: list[str]) -> tuple[list[str], list[str]]:
    """Remove the eulogies wherever they fall, keeping the display tokens aligned with the bare ones."""
    out_bare: list[str] = []
    out_display: list[str] = []
    i = 0
    while i < len(words):
        for phrase in HONORIFICS:
            if tuple(words[i:i + len(phrase)]) == phrase:
                i += len(phrase)
                break
        else:
            out_bare.append(words[i])
            out_display.append(display[i])
            i += 1
    return out_bare, out_display


def quoted_speech(text: str) -> str:
    """The spans this edition marks as direct speech, joined. Not used as the isnad boundary."""
    cleaned = _clean(text)
    spans = _QUOTED_RE.findall(cleaned)
    if not spans and cleaned.count('"') == 1:      # a handful of records never close the quote
        spans = [cleaned.split('"', 1)[1]]
    return " ".join(s.strip() for s in spans if s.strip())


def split_matn(text: str) -> tuple[str, str, int]:
    """Return (matn display text, matn bare text, tokens of isnad removed)."""
    cleaned = clean_display(_clean(text), "arb")
    display = cleaned.split()
    # The matching key is letters only: `bare` keeps the commas and quotation marks this edition is
    # full of, and "qala," would then match nothing in the chain-verb set.
    words = ["".join(tokenize(w, "arb")) for w in display]
    keep = [(w, d) for w, d in zip(words, display) if w]
    if not keep:
        return "", "", 0
    words, display = [w for w, _ in keep], [d for _, d in keep]
    # Honorifics come out first, before the boundary is looked for. They pad a transmitter's name -
    # "from Umar b. al-Khattab, may God be pleased with him, on the pulpit, he said" - and a scan
    # that has to step over them runs out of window and leaves the last transmitter in the report.
    words, display = drop_honorifics(words, display)
    start = isnad_end(words)
    return " ".join(display[start:]), " ".join(words[start:]), start


def load(dir_path: str | Path) -> list[dict]:
    dir_path = Path(dir_path)
    edition = dir_path / "ara-bukhari.json"
    if not edition.exists():
        return []
    payload = json.loads(edition.read_text(encoding="utf-8"))
    sections = {int(k): v for k, v in (payload.get("metadata", {}).get("sections") or {}).items()}
    out: list[dict] = []
    order = 0
    for entry in payload.get("hadiths", []):
        book = int((entry.get("reference") or {}).get("book") or 0)
        number = entry.get("hadithnumber")
        if not book or number is None:
            continue
        display, flat, isnad_tokens = split_matn(entry.get("text") or "")
        toks = tokenize(display, "arb")
        if not toks:
            continue
        section = (sections.get(book) or "").strip()
        code = f"BUKH{book:02d}"
        number = f"{number:g}" if isinstance(number, float) else str(number)
        order += 1
        out.append(
            {
                "source": "hadith-api",
                "language": "arb",
                "witness": "H",
                "work": code,
                "work_title": f"Bukhari {book}. {section}".strip().rstrip("."),
                "collection": "Bukhari",
                "canon": "Hadith",
                # The kitab is a topical heading, so it is a genre label, not an authorship one. It is
                # carried as `group` precisely so a style grouping can be tested against it: a split
                # that tracks the kitab is tracking subject matter.
                "group": section or "Bukhari",
                "chapter": str(book),
                "verse": number,
                "ref": f"Bukhari {book}:{number}",
                "text": display,
                "text_bare": " ".join(toks),
                "n_tokens": len(toks),
                "text_quoted": quoted_speech(entry.get("text") or ""),
                "isnad_tokens": isnad_tokens,
                "copyist": None,
                "supplied_frac": 0.0,
                "has_gap": False,
                "duplicate_of": None,
                "order": order,
            }
        )
    return out
