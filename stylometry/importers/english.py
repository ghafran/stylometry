"""Complete English body paragraphs and undisputed reference labels.

The pinned Gutenberg editions have explicit opening-text anchors. These remove
publisher/translator prefaces and tables of contents without sampling or dropping
short dialogue. Every body paragraph through the end marker is retained, except
clearly marked editorial illustrations/transcriber notes and standalone headings.
Chapter is a sequential section number; chapter_heading preserves printed labels
(including novels whose chapter numbering restarts in each volume).
"""
from pathlib import Path
import re
from .unicode import clean_display, tokenize
from .english_catalogue import CATALOGUE, FEDERALIST_ID, FED_DISPUTED, FED_JOINT, FED_JAY, FED_MADISON

_START = re.compile(r'^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG (?:EBOOK|ETEXT).*$', re.M | re.I)
_END = re.compile(r'^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG (?:EBOOK|ETEXT).*$', re.M | re.I)
_FED_HEAD = re.compile(r'^\s*FEDERALIST\.?\s+No\.\s*(\d+)\s*$', re.M | re.I)
_FED_AUTHOR = re.compile(r'^\s*(MADISON, with HAMILTON|HAMILTON AND MADISON|HAMILTON OR MADISON|HAMILTON|MADISON|JAY)\s*$', re.M)
_HEADING = re.compile(r'^(?:(?:CHAPTER|BOOK|PART|VOLUME|LETTER|ACT|PHASE)\s+(?:THE\s+)?(?:[IVXLCDM]+|\d+|FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|ONE|TWO|THREE)\b.*|[IVXLCDM]+[.\]]?(?:\s+[^a-z\n]+)?|\d+[.]?|(?:PREFACE|PRELUDE|EPILOGUE|CONCLUSION)[.]?)$', re.I)
# Verified literal opening phrases for the checked-in editions. Matching tolerates
# line wrapping, but fails explicitly when an edition changes rather than silently
# selecting an arbitrary fraction of the book.
OPENINGS = {
 1342: 'It is a truth universally acknowledged', 161: 'The family of Dashwood had long been settled',
 158: 'Emma Woodhouse, handsome, clever, and rich', 1400: 'My father’s family name being Pirrip',
 98: 'It was the best of times, it was the worst of times', 730: 'Among other public buildings in a certain town',
 1260: 'There was no possibility of taking a walk that day', 9182: 'My godmother lived in a handsome house',
 30486: 'Of late years an abundant shower of curates', 145: 'Who that cares much to know the history of man',
 550: 'In the days when the spinning-wheels hummed', 6688: 'A wide plain, where the broadening Floss',
 110: 'On an evening in the latter part of May', 27: 'When Farmer Oak smiled',
 153: 'The schoolmaster was leaving the village', 74: '“Tom!”',
 76: 'You don’t know about me without you have read', 245: 'THE Mississippi is well worth reading about',
 3176: 'For months the great pleasure excursion to Europe', 1695: 'The suburb of Saffron Park',
 204: 'Between the silver ribbon of morning', 470: 'Nothing more strangely indicates an enormous',
 16769: 'The only possible excuse for this book', 120: 'Squire Trelawney, Dr. Livesey, and the rest',
 43: 'Mr. Utterson the lawyer was a man', 535: 'The journey which this little book is to describe',
 386: 'WITH the single exception of Falstaff', 36: 'No one would have believed in the last years',
 35: 'The Time Traveller (for so it will be convenient', 19229: 'It is proposed in this book to present',
 7058: 'Toleration to-day is becoming a different thing', 1661: 'To Sherlock Holmes she is always _the_ woman',
 2852: 'Mr. Sherlock Holmes, who was usually very late', 3069: 'Take a community of Dutchmen of the type',
 5317: 'I care not how humble your bookshelf may be',
}


def strip_gutenberg(text: str) -> str:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    start = _START.search(text)
    if start:
        text = text[start.end():]
    end = _END.search(text)
    if end:
        text = text[:end.start()]
    return text


def _body(text: str, gid: int | None) -> tuple[str, set[str]]:
    text = strip_gutenberg(text)
    # Remove illustration spans before finding paragraphs: captions can include
    # blank lines and therefore cannot safely be filtered paragraph by paragraph.
    text = re.sub(r'\[Illustration\b[^\]]*\]', '', text, flags=re.S | re.I)
    titles = set()
    if gid in OPENINGS:
        pattern = r'\s+'.join(re.escape(p) for p in OPENINGS[gid].split())
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f'pg{gid}: expected opening text absent; review this edition')
        front = text[:match.start()]
        # Contents titles recur as section headings in collections of stories.
        titles = {clean_display(line, 'eng') for line in front.splitlines()
                  if 4 < len(line.strip()) < 100 and not re.search(r'[?!“”"]', line)}
        text = text[match.start():]
        if gid == 30486:
            ending = re.search(r'(?m)^[ \t]*THE END\.[ \t]*$', text)
            if ending:
                text = text[:ending.start()]
        tail = re.search(r'(?mi)^[ \t]*\[?Transcriber[’\x27]s? Note', text)
        if tail:
            text = text[:tail.start()]
    return text, titles


def _is_heading(raw: str, titles: set[str]) -> bool:
    first = raw.strip().splitlines()[0].strip()
    flat = clean_display(raw, 'eng')
    if not flat or len(flat) > 220:
        return False
    if flat in titles:
        return True
    if re.fullmatch(r'[IVXLCDM]+\. [A-Z][^\n]{1,110}', flat):
        return True
    if _HEADING.fullmatch(first):
        # A prose line starting with the pronoun I is not a Roman heading.
        return first == 'I' or not first.startswith('I ') or first[2:].isupper()
    return bool(len(flat) < 100 and len(flat.split()) >= 2 and flat.isupper()
                and not re.search(r'[“”"?!]', flat))


def _record(text: str, code: str, title: str, collection: str, chapter: int,
            verse: int, author: str | None, **extra) -> dict:
    return dict(language='eng', source='gutenberg', witness='G', work=code, work_title=title,
                collection=collection, chapter=str(chapter), verse=str(verse), text=text,
                reference_author=author, has_gap=False, n_tokens=len(tokenize(text, 'eng')),
                ref=f'{title} {chapter}:{verse}', **extra)


def _work(text: str, gid: int, author: str, code: str, title: str, collection: str, genre: str) -> list[dict]:
    body, titles = _body(text, gid)
    chapter, paragraph = 1, 0
    heading = 'Opening section'
    out = []
    for raw in re.split(r'\n\s*\n', body):
        flat = clean_display(raw, 'eng')
        if not flat:
            continue
        if _is_heading(raw, titles):
            if paragraph:
                chapter += 1
                paragraph = 0
            heading = flat
            continue
        if re.match(r'(?i)^(?:\[?transcriber[’\x27]?s? notes?|produced by|end of (?:the )?(?:project gutenberg|transcription))', flat):
            continue
        if not tokenize(flat, 'eng'):
            continue
        # Explicit bylines/signatures and printing imprints are never features.
        if flat.rstrip('.,').casefold() in {author.casefold(), ('by '+author).casefold(), 'the end', 'finis', 'the author'}:
            continue
        paragraph += 1
        out.append(_record(flat, code, title, collection, chapter, paragraph, author,
                           genre=genre, chapter_heading=heading, unit_type='paragraph',
                           source_file=f'english/pg{gid}.txt'))
    return out


def _federalist(text: str) -> list[dict]:
    text = strip_gutenberg(text)
    heads = list(_FED_HEAD.finditer(text))
    if not heads:
        raise ValueError('Federalist text has no numbered papers')
    out = []
    for i, head in enumerate(heads):
        number = int(head.group(1))
        block = text[head.end():heads[i+1].start() if i+1 < len(heads) else len(text)]
        byline = _FED_AUTHOR.search(block)
        if byline is None:
            raise ValueError(f'Federalist {number}: missing printed author boundary')
        # Every paragraph before and including the byline is publication metadata.
        block = block[byline.end():]
        author = None if number in FED_DISPUTED | FED_JOINT else (
            'John Jay' if number in FED_JAY else 'James Madison' if number in FED_MADISON else 'Alexander Hamilton')
        index = 0
        for raw in re.split(r'\n\s*\n', block):
            para = clean_display(raw, 'eng')
            if not para or re.match(r'^To the (?:People|Citizens) of the State of New York', para, re.I):
                continue
            if para.strip(' .') == 'PUBLIUS' or _FED_AUTHOR.fullmatch(para):
                continue
            if not tokenize(para, 'eng'):
                continue
            index += 1
            out.append(_record(para, f'FED{number:02d}', f'Federalist No. {number}', 'Federalist', number, index,
                               author, genre='essay', unit_type='paragraph', source_file='english/pg1404.txt',
                               attribution_status='disputed' if number in FED_DISPUTED else 'joint' if number in FED_JOINT else 'undisputed'))
    return out


def load(path: Path) -> list[dict]:
    out = []
    if (path / f'pg{FEDERALIST_ID}.txt').exists():
        out.extend(_federalist((path / f'pg{FEDERALIST_ID}.txt').read_text(encoding='utf-8')))
    for gid, author, code, title, collection, genre in CATALOGUE:
        file = path / f'pg{gid}.txt'
        if file.exists():
            out.extend(_work(file.read_text(encoding='utf-8'), gid, author, code, title, collection, genre))
    for i, record in enumerate(out, 1):
        record['order'] = i
    return out
