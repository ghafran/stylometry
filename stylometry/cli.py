"""One-command corpus build, style discovery, evaluation and explorer."""
from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys

from . import __version__


def _dump(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def parser():
    root = argparse.ArgumentParser(description='Discover shared writing styles within each language and tag every verse.')
    root.add_argument('--version', action='version', version=__version__)
    commands = root.add_subparsers(dest='command', required=True)
    for name in ['build', 'run']:
        p = commands.add_parser(name, help='Build source corpus' if name == 'build' else 'Build, analyze, benchmark and export')
        p.add_argument('--raw', type=Path, default=Path('data/raw'))
        p.add_argument('--corpus', type=Path, default=Path('data/processed/verses.jsonl'))
    for name in ['analyze', 'benchmark']:
        p = commands.add_parser(name, help='Analyze an existing JSONL corpus' if name == 'analyze' else 'Evaluate held-out English books')
        p.add_argument('--corpus', type=Path, default=Path('data/processed/verses.jsonl'))
    for name in ['run', 'analyze']:
        p = commands.choices[name]
        p.add_argument('--language', action='append', help='Language code; repeat to select multiple languages (default: all)')
        p.add_argument('--out', type=Path, default=Path('output'))
        p.add_argument('--passage-tokens', type=int, default=1200)
        p.add_argument('--min-tokens', type=int, default=200)
        p.add_argument('--max-styles', type=int, default=20, help='Search ceiling, not a supplied style count')
        p.add_argument('--fit-passages', type=int, default=2000)
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--skip-benchmark', action='store_true')
    commands.choices['benchmark'].add_argument('--out', type=Path, default=Path('output/benchmark.json'))
    commands.choices['benchmark'].add_argument('--seed', type=int, default=42)
    p = commands.add_parser('serve', help='Open output through a local web server')
    p.add_argument('--out', type=Path, default=Path('output'))
    p.add_argument('--port', type=int, default=8000)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return _run(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


def _run(args):
    if args.command == 'serve':
        if not (args.out / 'index.html').exists():
            raise ValueError(f'No report at {args.out}; run stylometry run first')
        handler = functools.partial(SimpleHTTPRequestHandler, directory=str(args.out.resolve()))
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
        print(f'Explorer: http://127.0.0.1:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0

    from .corpus import CorpusBuildError, build_corpus, load_corpus, write_corpus
    from .engine import Config, analyze, corpus_fingerprint
    build_report = None
    if args.command in {'build', 'run'}:
        print('Loading the local source texts…', flush=True)
        try:
            verses, build_report = build_corpus(args.raw)
        except CorpusBuildError as error:
            _dump(args.corpus.parent / 'build_report.json', error.report)
            raise
        if not verses:
            raise ValueError(f'No text records found under {args.raw}')
        write_corpus(verses, args.corpus)
        build_report['corpus_sha256'] = corpus_fingerprint(verses)
        _dump(args.corpus.parent / 'build_report.json', build_report)
        print(f'Saved {len(verses):,} text units to {args.corpus}', flush=True)
        if args.command == 'build':
            return 0
    else:
        verses = load_corpus(args.corpus)
        report_path = args.corpus.parent / 'build_report.json'
        if report_path.exists():
            candidate = json.loads(report_path.read_text(encoding='utf-8'))
            if candidate.get('corpus_sha256') == corpus_fingerprint(verses):
                build_report = candidate
    if args.command == 'benchmark':
        from .benchmark import evaluate_english
        score = evaluate_english(verses, seed=args.seed)
        _dump(args.out, score)
        print(f'English benchmark: {args.out}')
        return 0

    from .report import write_report
    if args.language:
        available = {v['language'] for v in verses}
        missing = set(args.language) - available
        if missing:
            raise ValueError(f'Languages absent from this corpus: {", ".join(sorted(missing))}')
        verses = [v for v in verses if v['language'] in args.language]
    if not verses:
        raise ValueError('The input corpus is empty')
    config = Config(passage_tokens=args.passage_tokens, min_tokens=args.min_tokens,
                    max_styles=args.max_styles, fit_passages=args.fit_passages, seed=args.seed)
    result = analyze(verses, config, progress=lambda message: print(message, flush=True))
    if build_report is not None:
        result['source_coverage'] = {key: build_report.get(key) for key in (
            'parsed_units', 'primary_units', 'alternate_witness_units', 'languages', 'corpus_sha256')}
        result['source_coverage']['source_policy'] = build_report['witness_policy']
        result['source_coverage']['analyzed_units'] = len(verses)
    if not args.skip_benchmark and any(v['language'] == 'eng' for v in verses):
        from .benchmark import evaluate_english
        print('Testing English attribution on held-out books…', flush=True)
        result['benchmark'] = evaluate_english(verses, passage_tokens=args.passage_tokens, seed=args.seed)
        result['benchmark']['discovery'] = result['discovery_validation']
        result['benchmark']['corpus_sha256'] = result['corpus_sha256']
    result['runtime'] = {'version': __version__}
    print('Writing verse tags, rollups, evidence and the explorer…', flush=True)
    report = write_report(result, args.out)
    _dump(args.out / 'passages.json', result['passages'])
    if result['benchmark'] is not None:
        _dump(args.out / 'benchmark.json', result['benchmark'])
    else:
        # A reused output directory must not advertise a prior run's accuracy.
        (args.out / 'benchmark.json').unlink(missing_ok=True)
    for row in result['languages']:
        print(f"{row['language']}: {row['estimated_styles']} inferred styles, {row['verse_count']:,} text units")
    print(f'Explorer: {report.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
