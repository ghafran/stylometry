"""Import complete Arabic reports, including their transmitted framing.

No inferred speaker or isnad boundary is treated as known authorship. The input
text remains available in full; stylistic models can mark short units uncertain.
"""
import json
import re
from pathlib import Path
from .unicode import clean_display, tokenize


def load(path: Path, qudsi: bool = False) -> list[dict]:
    filename = 'ara-qudsi.json' if qudsi else 'ara-bukhari.json'
    payload = json.loads((path / filename).read_text(encoding='utf-8'))
    entries = payload.get('hadiths')
    if not isinstance(entries, list):
        raise ValueError(f'{filename}: missing hadiths list')
    sections = payload.get('metadata', {}).get('sections', {})
    out = []
    for entry in entries:
        number = entry.get('hadithnumber')
        if number is None or not isinstance(entry.get('text'), str):
            raise ValueError(f'{filename}: report missing number or text')
        text = clean_display(re.sub(r'<[^>]+>', ' ', entry['text']), 'arb')
        book = int((entry.get('reference') or {}).get('book') or 0)
        number = f'{number:g}' if isinstance(number, float) else str(number)
        code = 'QUDSI' if qudsi else f'BUKH{book:02d}'
        title = 'Forty Hadith Qudsi' if qudsi else f'Bukhari {book}. {sections.get(str(book), "Unassigned" if book == 0 else "")}'.rstrip('. ')
        out.append(dict(source='hadith-api', language='arb', witness='Q' if qudsi else 'H',
                        work=code, work_title=title, collection='Hadith Qudsi' if qudsi else 'Bukhari',
                        chapter='1' if qudsi else str(book), verse=number, text=text,
                        ref=f'{code} {number}', has_gap=False, n_tokens=len(tokenize(text, 'arb')),
                        text_scope='full transmitted report', order=len(out)+1))
    return out
