"""Build and validate the canonical, verse-level corpus from local raw sources."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Callable
from urllib.parse import quote

from .importers import apostolic, cntr, dss, english, first1k, hadith, inscriptions, mam, oshb, quran, samaritan, sinaiticus, vaticanus
from .importers.unicode import clean_display

LANGUAGES = {'eng', 'grc', 'hbo', 'arb'}
REQUIRED_STRINGS = ('id', 'language', 'collection', 'book', 'book_title', 'chapter', 'verse', 'source')
WITNESS_PRIORITY = ('S', 'B', 'L', 'A', 'SP', 'Swete', 'Lake', 'Bonnet', 'T', 'H', 'Q', 'G')


class CorpusBuildError(ValueError):
    """A present source failed; report remains available for diagnostics."""
    def __init__(self, report: dict):
        self.report = report
        failed = [s for s in report['sources'] if s['status'] == 'failed']
        super().__init__('Corpus build failed: ' + '; '.join(f"{s['source']}: {s['error']}" for s in failed))


def _sinaiticus(path: Path) -> list[dict]:
    files = sorted(path.glob('sinaiticus_full_v*.xml'))
    if not files:
        raise ValueError('missing sinaiticus_full_v*.xml')
    return sinaiticus.load(files[-1])


LOADERS: list[tuple[str, Callable]] = [
    ('codex-sinaiticus', _sinaiticus), ('cntr', cntr.load), ('vaticanus', vaticanus.load),
    ('first1kgreek', first1k.load), ('apostolic-fathers/texts', apostolic.load),
    ('oshb', oshb.load), ('mam', mam.load), ('samaritan', samaritan.load), ('dss', dss.load),
    ('inscriptions', inscriptions.load), ('quran', quran.load), ('bukhari', hadith.load),
    ('qudsi', lambda p: hadith.load(p, qudsi=True)), ('english', english.load),
]


def validate_corpus(verses: list[dict]) -> None:
    if not verses:
        raise ValueError('Corpus contains no verses')
    seen = set()
    for line, verse in enumerate(verses, 1):
        if not isinstance(verse, dict):
            raise ValueError(f'Record {line} must be an object')
        for key in REQUIRED_STRINGS:
            if not isinstance(verse.get(key), str) or not verse[key].strip():
                raise ValueError(f'Record {line}: {key} must be a nonempty string')
        if not isinstance(verse.get('text'), str):
            raise ValueError(f'Record {line}: text must be a string (empty is allowed)')
        if verse['language'] not in LANGUAGES:
            raise ValueError(f"Record {line}: unsupported language {verse['language']!r}")
        if verse['id'] in seen:
            raise ValueError(f"Duplicate verse id: {verse['id']}")
        seen.add(verse['id'])
        author = verse.get('reference_author')
        if author is not None and (not isinstance(author, str) or not author.strip()):
            raise ValueError(f'Record {line}: reference_author must be text or null')
        if author is not None and verse['language'] != 'eng':
            raise ValueError(f'Record {line}: reference authors are supported only for the English benchmark')
        if not isinstance(verse.get('has_gap'), bool):
            raise ValueError(f'Record {line}: has_gap must be boolean')


def _canonical(record: dict, source_dir: str) -> dict:
    verse = dict(record)
    verse['book'] = verse.pop('work')
    verse['book_title'] = verse.pop('work_title')
    verse['language'] = verse.get('language', 'grc')
    verse['text'] = clean_display(verse['text'], verse['language'])
    verse['chapter'], verse['verse'] = str(verse['chapter']), str(verse['verse'])
    verse['source_dir'] = source_dir
    verse['source_reference'] = verse.pop('ref', f"{verse['book']} {verse['chapter']}:{verse['verse']}")
    verse['has_gap'] = bool(verse.get('has_gap', False))
    verse.setdefault('reference_author', None)
    verse.setdefault('witness', '?')
    # The fresh estimator is given the source text; legacy analytical derivatives
    # and conventional authorship groups are not part of the canonical schema.
    for field in ('text_bare', 'duplicate_of', 'group', 'author', 'known_author', 'pos'):
        verse.pop(field, None)
    return verse


def resolve_witnesses(verses: list[dict]) -> tuple[list[dict], list[dict]]:
    """Choose a complete witness per language/book; report every alternative.

    Preference applies only to witnesses with at least 90% of the fullest token
    coverage. Source name breaks ties deterministically when editions share sigla.
    Repeated references within the selected witness remain distinct observations.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for verse in verses:
        grouped[(verse['language'], verse['book'])][(verse['witness'], verse['source'])].append(verse)
    selected = set()
    choices = []
    for (language, book), candidates in sorted(grouped.items()):
        sizes = {key: sum(v.get('n_tokens', len(v['text'].split())) for v in units) for key, units in candidates.items()}
        largest = max(sizes.values())
        eligible = [key for key, count in sizes.items() if count >= .9 * largest]
        def rank(key):
            return (WITNESS_PRIORITY.index(key[0]) if key[0] in WITNESS_PRIORITY else len(WITNESS_PRIORITY), -sizes[key], key)
        best = min(eligible, key=rank)
        selected.add((language, book, *best))
        choices.append(dict(language=language, book=book, witness=best[0], source=best[1],
                            selected_units=len(candidates[best]), alternatives=[
                                dict(witness=k[0], source=k[1], units=len(v), tokens=sizes[k])
                                for k, v in sorted(candidates.items()) if k != best]))
    primary = [v for v in verses if (v['language'], v['book'], v['witness'], v['source']) in selected]
    counts = Counter()
    for verse in primary:
        parts = (verse['language'], verse['collection'], verse['book'], verse['witness'], verse['chapter'], verse['verse'])
        identifier = ':'.join(quote(str(p), safe='-.') for p in parts)
        counts[identifier] += 1
        occurrence = counts[identifier]
        verse['id'] = identifier if occurrence == 1 else f'{identifier}#{occurrence}'
        if occurrence > 1:
            verse['repeated_reference'] = occurrence
    return primary, choices


def build_corpus(raw_dir: Path) -> tuple[list[dict], dict]:
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise ValueError(f'Raw corpus directory does not exist: {raw_dir}')
    verses, sources = [], []
    for subdir, loader in LOADERS:
        path = raw_dir / subdir
        if not path.exists():
            sources.append(dict(source=subdir, status='missing', parsed_units=0, error=None))
            continue
        try:
            loaded = loader(path)
            if not loaded:
                raise ValueError('present source produced no text; check expected raw files')
            canonical = [_canonical(v, subdir) for v in loaded]
            verses.extend(canonical)
            source = dict(source=subdir, status='ok', parsed_units=len(canonical), error=None)
            if subdir == 'english':
                expected = [f'pg{english.FEDERALIST_ID}.txt'] + [f'pg{row[0]}.txt' for row in english.CATALOGUE]
                source['missing_files'] = [name for name in expected if not (path/name).exists()]
                source['files_read'] = [name for name in expected if (path/name).exists()]
                source['reference_authors'] = sorted({v['reference_author'] for v in canonical if v['reference_author']})
                source['policy'] = 'Full body paragraphs, no word cap or percentage skip. Verified opening anchors exclude title blocks, editorial prefaces and contents; explicit Gutenberg/transcriber matter and headings are removed. Short dialogue remains.'
            if subdir in {'bukhari', 'qudsi'}:
                source['policy'] = 'Complete transmitted reports; framing retained. Bukhari book 0 is retained as Unassigned.'
            if subdir == 'dss':
                source['policy'] = 'All nonempty fragments, including reconstructed text; supplied_frac and has_gap flag reconstruction.'
            if subdir == 'cntr':
                source['excluded_files'] = [{'path': 'class1/01.txt', 'reason': 'Sinaiticus available from fuller ITSEE transcription'}] if (path/'class1/01.txt').exists() else []
            sources.append(source)
        except Exception as error:
            sources.append(dict(source=subdir, status='failed', parsed_units=0, error=f'{type(error).__name__}: {error}'))
    covered = {path.split('/')[0] for path, _ in LOADERS}
    for path in sorted(raw_dir.iterdir()):
        if path.is_dir() and path.name not in covered:
            sources.append(dict(source=path.name, status='excluded', parsed_units=0,
                                reason='Auxiliary benchmark downloads require their separate manifests; not a primary text collection' if path.name == 'benchmarks' else 'No supported source format registered', error=None))
    report = dict(schema_version=1, sources=sources, parsed_units=len(verses),
                  witness_policy='Prefer catalogued manuscript only when coverage is at least 90% of the fullest witness; one primary witness per language/book.',
                  warnings=['English paragraphs and prose sections stand in for verses where the edition has no verses.',
                            'Chapter numbers in English are sequential sections; chapter_heading preserves available printed headings.'])
    if any(s['status'] == 'failed' for s in sources):
        raise CorpusBuildError(report)
    primary, choices = resolve_witnesses(verses)
    validate_corpus(primary)
    selected_counts = Counter(v['source_dir'] for v in primary)
    for source in sources:
        source['selected_units'] = selected_counts[source['source']]
        source['alternate_witness_units'] = source['parsed_units'] - source['selected_units']
    report.update(units=len(primary), n_units=len(primary), primary_units=len(primary),
                  alternate_witness_units=len(verses)-len(primary), primary_witnesses=choices,
                  repeated_references=sum('repeated_reference' in v for v in primary),
                  languages=dict(sorted(Counter(v['language'] for v in primary).items())),
                  reference_authors=sorted({v['reference_author'] for v in primary if v['reference_author']}),
                  n_failed=0, n_missing=sum(s['status']=='missing' for s in sources))
    return primary, report


def load_corpus(path: Path) -> list[dict]:
    verses = []
    with Path(path).open(encoding='utf-8') as handle:
        for line, raw in enumerate(handle, 1):
            if raw.strip():
                try:
                    record = json.loads(raw)
                    if isinstance(record, dict):
                        record.setdefault('source', 'custom')
                        record.setdefault('has_gap', False)
                        record.setdefault('reference_author', None)
                    verses.append(record)
                except json.JSONDecodeError as error:
                    raise ValueError(f'{path}:{line}: invalid JSON: {error.msg}') from error
    validate_corpus(verses)
    return verses


def write_corpus(verses: list[dict], path: Path) -> None:
    validate_corpus(verses)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as handle:
            for verse in verses:
                handle.write(json.dumps(verse, ensure_ascii=False) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
