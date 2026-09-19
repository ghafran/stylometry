"""English: a validation corpus, where the author of every text is known.

The other three languages have no ground truth. Nobody can say who wrote Isaiah, so nothing measured
on it can be scored, and a strategy that looks convincing there might be measuring genre, length or
the editor's punctuation. This corpus exists so a strategy can be shown to work — or shown not to —
before it is pointed at something that matters. Every text here has a settled author, and the three
collections are chosen to test three different things.

**Federalist** (85 papers, 1787-88). The discipline's reference problem, and the one Mosteller and
Wallace built modern stylometry on. Same genre, same purpose, same months, three authors: as close as
history comes to holding everything but authorship constant. 70 papers are undisputed — Hamilton 51,
Madison 14, Jay 5 — and they are the ground truth. The twelve long-disputed papers (49-58, 62, 63)
and the three joint ones (18-20) are labelled as such rather than given an author, so they can be
held back as the classic test instead of quietly scoring as training data.

**Novels** (5 authors, 3 works each). Long single-author texts of one genre, so a strategy can be
tested by holding out a whole book and asking who wrote it — the same whole-work holdout the
scripture runs use, but with a checkable answer.

**Cross-genre** (5 authors, 2 novels and 2 works of non-fiction each). This one exists because of
what this project keeps finding: that genre outweighs authorship. Here the same hand writes fiction
and essays, so the two can be separated. If a strategy groups Twain's travel writing with
Chesterton's essays rather than with Twain's novels, it is measuring genre, and this collection is
where that shows up as a number instead of a suspicion.

Each text contributes ``WORD_BUDGET`` running words, taken from a sixth of the way in. Whole novels
would make English three times the rest of the corpus put together; twenty thousand words is far
above any attribution threshold in the literature and leaves the collections comparable in size to
the scripture ones. Starting past the front matter rather than at the top does two jobs at once: it
steps over the title block, contents, dedication and transcriber's notes without having to recognise
them, and it stops the corpus being a study of how these authors open books.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..lang import clean_display, tokenize

WORD_BUDGET = 20_000
MIN_PARAGRAPH_TOKENS = 5
# Where each work's sample starts, as a share of its paragraphs. Front matter - the title block, the
# contents, a dedication, a note to the third edition, a transcriber's biography of the author - is
# always at the top of these files, so starting a sixth of the way in steps over all of it without
# having to recognise any of it. Recognising it was tried and it is a losing game: every rule that
# caught Chesterton's contents list threw away Jane Austen's first chapter, whose paragraphs are too
# short to look like prose to a length test.
#
# It is also the better sample. Taking each work's opening would have measured openings, which is not
# how these authors write for most of a book.
BODY_SKIP = 0.15

# Printed at the head of all 85 papers. Identical text repeated 85 times is not anybody's style, and
# as its own unit it would put 85 indistinguishable paragraphs into the corpus.
_SALUTATION_RE = re.compile(r"^To the (People|Citizens) of the State of New York", re.I)

# Matter the author did not write, which these files carry both before the work and inside it.
_FRONT_MATTER_RE = re.compile(
    r"project gutenberg|gutenberg-tm|transcriber|electronic text|etext|produced by|proofread|"
    r"scanned|copyright|all rights reserved|\[illustration|this file was|public domain|"
    r"printed in the united states|typographical errors", re.I)
_ROMAN_RE = re.compile(r"\b[IVXLC]{2,}\b")


def is_front_matter(para: str) -> bool:
    """A transcriber's note, or a table of contents read as one paragraph."""
    if _FRONT_MATTER_RE.search(para):
        return True
    if len(_ROMAN_RE.findall(para)) >= 3:
        return True                                   # a contents list, numbered in roman
    digits = sum(c.isdigit() for c in para)
    return len(para) > 40 and digits / len(para) > 0.04

# Project Gutenberg wraps each text in its own front and back matter.
_START_RE = re.compile(r"^\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M)
_END_RE = re.compile(r"^\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M)
_CHAPTER_RE = re.compile(
    r"^\s*(?:"
    r"(?:CHAPTER|Chapter|LETTER|Letter|BOOK|Book|PART|Part|ACT|Act)\s+"
    r"(?:[IVXLCDM]+|\d+|[A-Z][a-z]+)\b.*"
    # A bare roman numeral is as common a section heading as the word "chapter", and missing it sent
    # Chesterton's essays past their first section and into the transcriber's biography of him.
    r"|[IVXLCDM]{1,7}\.?(?:\s*[-—_]*\s*[A-Z].*)?"
    r")$")
_FED_HEAD_RE = re.compile(r"^FEDERALIST\.?\s+No\.\s*(\d+)\s*$")
_FED_AUTHOR_RE = re.compile(r"^(HAMILTON AND MADISON|HAMILTON OR MADISON|HAMILTON|MADISON|JAY)\s*$")

# The settled attribution of the Federalist, which is not what the Gutenberg text prints: that edition
# gives every disputed and joint paper to Madison. Ground truth has to exclude the very papers the
# field is still arguing about, or the benchmark scores itself on its own open question.
FED_DISPUTED = {49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 62, 63}
FED_JOINT = {18, 19, 20}
FED_JAY = {2, 3, 4, 5, 64}
FED_MADISON = {10, 14, *range(37, 49)}

# (gutenberg id, author, short code, title, collection, genre)
CATALOGUE: list[tuple[int, str, str, str, str, str]] = [
    # --- Novels: one genre, several works per author -------------------------------------------------
    (1342, "Jane Austen", "AUSTEN-PP", "Pride and Prejudice", "Novels", "fiction"),
    (161, "Jane Austen", "AUSTEN-SS", "Sense and Sensibility", "Novels", "fiction"),
    (158, "Jane Austen", "AUSTEN-EM", "Emma", "Novels", "fiction"),
    (1400, "Charles Dickens", "DICKENS-GE", "Great Expectations", "Novels", "fiction"),
    (98, "Charles Dickens", "DICKENS-TC", "A Tale of Two Cities", "Novels", "fiction"),
    (730, "Charles Dickens", "DICKENS-OT", "Oliver Twist", "Novels", "fiction"),
    (1260, "Charlotte Bronte", "BRONTE-JE", "Jane Eyre", "Novels", "fiction"),
    (9182, "Charlotte Bronte", "BRONTE-VI", "Villette", "Novels", "fiction"),
    (30486, "Charlotte Bronte", "BRONTE-SH", "Shirley", "Novels", "fiction"),
    (145, "George Eliot", "ELIOT-MM", "Middlemarch", "Novels", "fiction"),
    (550, "George Eliot", "ELIOT-SM", "Silas Marner", "Novels", "fiction"),
    (6688, "George Eliot", "ELIOT-MF", "The Mill on the Floss", "Novels", "fiction"),
    (110, "Thomas Hardy", "HARDY-TD", "Tess of the d'Urbervilles", "Novels", "fiction"),
    (27, "Thomas Hardy", "HARDY-FM", "Far from the Madding Crowd", "Novels", "fiction"),
    (153, "Thomas Hardy", "HARDY-JO", "Jude the Obscure", "Novels", "fiction"),
    # --- Cross-genre: the same hand in fiction and in prose ------------------------------------------
    (74, "Mark Twain", "TWAIN-TS", "The Adventures of Tom Sawyer", "Cross-genre", "fiction"),
    (76, "Mark Twain", "TWAIN-HF", "Adventures of Huckleberry Finn", "Cross-genre", "fiction"),
    (245, "Mark Twain", "TWAIN-LM", "Life on the Mississippi", "Cross-genre", "non-fiction"),
    (3176, "Mark Twain", "TWAIN-IA", "The Innocents Abroad", "Cross-genre", "non-fiction"),
    (1695, "G. K. Chesterton", "CHESTERTON-MT", "The Man Who Was Thursday", "Cross-genre", "fiction"),
    (204, "G. K. Chesterton", "CHESTERTON-FB", "The Innocence of Father Brown", "Cross-genre", "fiction"),
    (470, "G. K. Chesterton", "CHESTERTON-HE", "Heretics", "Cross-genre", "non-fiction"),
    (16769, "G. K. Chesterton", "CHESTERTON-OR", "Orthodoxy", "Cross-genre", "non-fiction"),
    (120, "Robert Louis Stevenson", "STEVENSON-TI", "Treasure Island", "Cross-genre", "fiction"),
    (43, "Robert Louis Stevenson", "STEVENSON-JH", "Dr Jekyll and Mr Hyde", "Cross-genre", "fiction"),
    (535, "Robert Louis Stevenson", "STEVENSON-TD", "Travels with a Donkey", "Cross-genre", "non-fiction"),
    (386, "Robert Louis Stevenson", "STEVENSON-VP", "Virginibus Puerisque", "Cross-genre", "non-fiction"),
    (36, "H. G. Wells", "WELLS-WW", "The War of the Worlds", "Cross-genre", "fiction"),
    (35, "H. G. Wells", "WELLS-TM", "The Time Machine", "Cross-genre", "fiction"),
    (19229, "H. G. Wells", "WELLS-AN", "Anticipations", "Cross-genre", "non-fiction"),
    (7058, "H. G. Wells", "WELLS-MM", "Mankind in the Making", "Cross-genre", "non-fiction"),
    (1661, "Arthur Conan Doyle", "DOYLE-SH", "The Adventures of Sherlock Holmes", "Cross-genre", "fiction"),
    (2852, "Arthur Conan Doyle", "DOYLE-HB", "The Hound of the Baskervilles", "Cross-genre", "fiction"),
    (3069, "Arthur Conan Doyle", "DOYLE-BW", "The Great Boer War", "Cross-genre", "non-fiction"),
    (5317, "Arthur Conan Doyle", "DOYLE-MD", "Through the Magic Door", "Cross-genre", "non-fiction"),
]

FEDERALIST_ID = 1404


def strip_gutenberg(text: str) -> str:
    """Just the work: Project Gutenberg's own front and back matter is not the author's prose."""
    start = _START_RE.search(text)
    if start:
        text = text[start.end():]
    end = _END_RE.search(text)
    if end:
        text = text[:end.start()]
    return text.replace("\r\n", "\n")


def paragraphs(block: str) -> list[str]:
    """Blank-line separated blocks, which is what a paragraph is in these files."""
    out = []
    for raw in re.split(r"\n\s*\n", block):
        para = clean_display(" ".join(raw.split()), "eng")
        if para and not is_front_matter(para) and len(tokenize(para, "eng")) >= MIN_PARAGRAPH_TOKENS:
            out.append(para)
    return out


def is_heading(block: str) -> bool:
    """A standalone chapter or section heading, not a paragraph that happens to start like one."""
    flat = " ".join(block.split())
    return bool(flat) and len(flat) < 90 and bool(_CHAPTER_RE.match(block.strip().split("\n")[0]))


def body_start(blocks: list[str]) -> int:
    """The first paragraph of this work's sample: a sixth of the way in, past the front matter."""
    return int(len(blocks) * BODY_SKIP)


def _record(**kw) -> dict:
    base = {"source": "gutenberg", "language": "eng", "copyist": None, "supplied_frac": 0.0,
            "has_gap": False, "duplicate_of": None}
    return {**base, **kw}


def _federalist(text: str, order_from: int) -> list[dict]:
    lines = strip_gutenberg(text).split("\n")
    heads = [(i, int(m.group(1))) for i, l in enumerate(lines)
             if (m := _FED_HEAD_RE.match(l))]
    out: list[dict] = []
    order = order_from
    for n, (line_no, number) in enumerate(heads):
        stop = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        block = lines[line_no + 1:stop]
        printed = next((m.group(1).title() for l in block if (m := _FED_AUTHOR_RE.match(l))), "")
        body_from = next((i for i, l in enumerate(block) if _FED_AUTHOR_RE.match(l)), -1) + 1
        if number in FED_DISPUTED:
            author, known = "disputed", False
        elif number in FED_JOINT:
            author, known = "joint", False
        elif number in FED_JAY:
            author, known = "John Jay", True
        elif number in FED_MADISON:
            author, known = "James Madison", True
        else:
            author, known = "Alexander Hamilton", True
        kept = [q for q in paragraphs("\n".join(block[body_from:])) if not _SALUTATION_RE.match(q)]
        for i, para in enumerate(kept, start=1):
            toks = tokenize(para, "eng")
            order += 1
            out.append(_record(
                witness="G", work=f"FED{number:02d}", work_title=f"Federalist No. {number}",
                collection="Federalist", canon="Federalist", group=author,
                author=author, known_author=known, genre="essay",
                printed_attribution=printed,
                chapter=str(number), verse=str(i), ref=f"Federalist {number}:{i}",
                text=para, text_bare=" ".join(toks), n_tokens=len(toks), order=order))
    return out


def _work(text: str, author: str, code: str, title: str, collection: str, genre: str,
          order_from: int) -> list[dict]:
    blocks = re.split(r"\n\s*\n", strip_gutenberg(text))
    chapter, index, used, order = 0, 0, 0, order_from
    out: list[dict] = []
    for raw in blocks[body_start(blocks):]:
        flat = " ".join(raw.split())
        if not flat:
            continue
        if is_heading(raw):
            chapter, index = chapter + 1, 0
            continue
        para = clean_display(flat, "eng")
        toks = tokenize(para, "eng")
        if len(toks) < MIN_PARAGRAPH_TOKENS or is_front_matter(para):
            continue
        index += 1
        order += 1
        used += len(toks)
        out.append(_record(
            witness="G", work=code, work_title=f"{author} \u2014 {title}", collection=collection,
            canon=collection, group=author, author=author, known_author=True, genre=genre,
            chapter=str(max(chapter, 1)), verse=str(index),
            ref=f"{title} {max(chapter, 1)}:{index}",
            text=para, text_bare=" ".join(toks), n_tokens=len(toks), order=order))
        if used >= WORD_BUDGET:
            break
    return out


def load(dir_path: str | Path) -> list[dict]:
    dir_path = Path(dir_path)
    out: list[dict] = []
    federalist = dir_path / f"pg{FEDERALIST_ID}.txt"
    if federalist.exists():
        out += _federalist(federalist.read_text(encoding="utf-8", errors="replace"), len(out))
    for gid, author, code, title, collection, genre in CATALOGUE:
        path = dir_path / f"pg{gid}.txt"
        if not path.exists():
            continue
        out += _work(path.read_text(encoding="utf-8", errors="replace"),
                     author, code, title, collection, genre, len(out))
    return out
