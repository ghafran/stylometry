"""The Forty Hadith Qudsi: reports in which the Prophet relates the speech of God.

A third Arabic category, and the reason it is worth having beside the other two. The Qur'an is, in
the tradition's own account, God's speech verbatim. Sahih al-Bukhari is the Prophet's. A *hadith
qudsi* is God's speech in the Prophet's wording — the same claimed source as the Qur'an, transmitted
the way a hadith is. Whatever separates the Qur'an from Bukhari, this collection sits across it.

It is **small**: 40 reports, about 2,500 tokens of matn, against 78,000 for the Qur'an and 337,000
for Bukhari. That is one work's worth of text, and it is carried as one work for exactly that reason.
Nothing at collection level can be measured from it; a single work cannot be partitioned, and the
explorer will say so. It is here to be read and to be compared against, not to be clustered.

The matn is extracted with the same machinery as Bukhari — the chain is the same apparatus — plus the
closing transmission note this edition appends to 39 of the 40 ("narrated by Muslim, and likewise
al-Bukhari, al-Nasa'i and Ibn Majah"), which is a citation and not part of the report.

Residual: 2 of the 40 keep a citation inside them, because those records carry two variants of the
same report and each is attributed separately, so the note falls in the middle rather than at the end.
That is about 20 words in 2,421. Cutting at the first citation would throw the second variant away,
which is a worse trade than leaving the phrase in.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..lang import clean_display, tokenize
from .bukhari import _clean, marked_spans, split_matn

# "rawahu ..." - who transmitted it, appended after the report. What identifies it is not where it
# falls but what follows: a citation is a short list of names, so anything with a paragraph after it
# is the word used inside the report and is left alone. Two of the forty records carry two variants,
# each attributed separately, and their middle citation is kept for exactly that reason.
SOURCE_NOTE = {"رواه", "اخرجه", "رواة", "رواية"}
NOTE_MAX_NAMES = 12

_TAG_RE = re.compile(r"<[^>]+>")


def strip_source_note(display: list[str], words: list[str]) -> tuple[list[str], list[str]]:
    """Drop the closing "narrated by X" citation, if one is there."""
    for i in range(len(words) - 1, -1, -1):
        if words[i] in SOURCE_NOTE and len(words) - i - 1 <= NOTE_MAX_NAMES:
            return display[:i], words[:i]
    return display, words


def load(dir_path: str | Path) -> list[dict]:
    dir_path = Path(dir_path)
    edition = dir_path / "ara-qudsi.json"
    if not edition.exists():
        return []
    payload = json.loads(edition.read_text(encoding="utf-8"))
    out: list[dict] = []
    order = 0
    for entry in payload.get("hadiths", []):
        number = entry.get("hadithnumber")
        if number is None:
            continue
        raw = _TAG_RE.sub(" ", entry.get("text") or "")      # this edition carries <br> markup
        display_text, _, isnad_tokens = split_matn(raw)
        display = clean_display(display_text, "arb").split()
        words = ["".join(tokenize(w, "arb")) for w in display]
        keep = [(d, w) for d, w in zip(display, words) if w]
        display, words = [d for d, _ in keep], [w for _, w in keep]
        display, words = strip_source_note(display, words)
        if not words:
            continue
        number = f"{number:g}" if isinstance(number, float) else str(number)
        order += 1
        out.append(
            {
                "source": "hadith-api",
                "language": "arb",
                "witness": "Q",
                "work": "QUDSI",
                "work_title": "Forty Hadith Qudsi",
                "collection": "Hadith Qudsi",
                "canon": "Hadith",
                # Every report here is the same kind of thing, which is what makes the collection a
                # category rather than a sample: the Prophet relating the speech of God.
                "group": "Hadith Qudsi",
                "attribution": "divine",
                "chapter": "1",
                "verse": number,
                "ref": f"Hadith Qudsi {number}",
                "text": " ".join(display),
                "text_bare": " ".join(words),
                "n_tokens": len(words),
                "text_quoted": " ".join(s for _, s in marked_spans(_clean(raw))),
                "isnad_tokens": isnad_tokens,
                "copyist": None,
                "supplied_frac": 0.0,
                "has_gap": False,
                "duplicate_of": None,
                "order": order,
            }
        )
    return out
