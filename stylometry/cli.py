"""Command-line entry point.

    stylometry build-corpus                     parse raw sources -> data/processed/verses.jsonl
    stylometry estimate  [--scope ..] [--model] print the AI-profiling cost estimate
    stylometry profile   [--backend sdk|batch|cli] [--scope ..] [--limit N]   AI style profiles
    stylometry profile-collect                  fetch finished Batch API results
    stylometry cluster   [--k N] [--scope ..] [--no-ai]   discover authors A1..Ak
    stylometry report                           write output/report.md
    stylometry html / witnesses                 HTML site; manuscript-witness comparison
    stylometry compare-models --set NAME=PATH   accuracy, stability and cost of the profiling models
    stylometry accuracy-gate                    measured benchmark accuracy vs the declared targets
    stylometry manifest --write|--check          which manuscripts the corpus contains
    stylometry all                              build-corpus -> profile -> cluster -> report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CORPUS = ROOT / "data" / "processed" / "verses.jsonl"
PROFILES = ROOT / "data" / "processed" / "profiles.jsonl"
OUTPUT = ROOT / "output"


def _load_corpus():
    from .corpus.build import load_corpus

    if not CORPUS.exists():
        sys.exit(f"{CORPUS} not found - run `stylometry build-corpus` first")
    return load_corpus(CORPUS)


LANGS = ["grc", "hbo", "arb"]


def _select(args, verses):
    from .corpus.meta import select_verses

    language = getattr(args, "language", None)
    scope = args.scope or ("christian" if language in (None, "grc") else "all")
    selected = select_verses(verses, scope=scope, works=args.works, language=language)
    chapters = getattr(args, "chapters", None)
    if chapters:
        wanted = {str(c) for c in chapters}
        selected = [v for v in selected if str(v["chapter"]) in wanted]
    return selected


def _lang_out(args) -> Path:
    """Output directory: --out if given, else output/<language>."""
    if getattr(args, "out", None):
        return Path(args.out)
    return OUTPUT / (getattr(args, "language", None) or "grc")


def _profiles_path(args) -> Path:
    return Path(args.profiles) if getattr(args, "profiles", None) else PROFILES


def _add_profiles(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profiles", help=f"profiles file (default {PROFILES}); use one file per model")


def _add_scope(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scope", default=None,
                   choices=["all", "sinaiticus", "lxx", "nt", "christian", "noncanonical", "greek", "hebrew", "arabic", "tanakh", "quran"],
                   help="which part of the corpus to use (default: 'christian' for Greek = everything except the Septuagint; 'all' for other languages)")
    p.add_argument("--language", choices=LANGS, help="grc (Greek), hbo (Hebrew), arb (Arabic); analyses never mix languages")
    p.add_argument("--works", nargs="*", help="restrict to these work codes, e.g. MARK JOHN AJOHN, ISA, Q002")
    p.add_argument("--chapters", nargs="*", help="restrict to these chapters of the selected work(s), e.g. --works GEN --chapters 1")


def cmd_build(args) -> None:
    from .corpus.build import build

    build(args.raw, CORPUS, include_duplicates=args.include_duplicates)


def cmd_estimate(args) -> None:
    from .ai_profile import estimate_cost, load_profiles, chunk_verses

    verses = _select(args, _load_corpus())
    done = load_profiles(_profiles_path(args))
    chunks = [c for c in chunk_verses(verses, args.chunk_size) if any(v["id"] not in done for v in c)]
    todo = [v for c in chunks for v in c]
    if args.model.startswith("deepseek"):
        est = estimate_cost(todo, args.model, args.chunk_size, chunks=chunks)
        print(f"DeepSeek {args.model}: {est['verses']} verses, {est['requests']} requests, "
              f"~{est['est_input_tokens']:,} in / ~{est['est_output_tokens']:,} out tokens, ≈ ${est['est_cost_usd']:.2f} peak "
              f"/ ${est['est_cost_usd'] / 2:.2f} off-peak (--thinking multiplies output tokens by ~6)")
    else:
        for batch in (False, True):
            est = estimate_cost(todo, args.model, args.chunk_size, batch=batch, chunks=chunks)
            print(f"{'batch API' if batch else 'sync API'}: {est['verses']} verses, {est['requests']} requests, "
                  f"~{est['est_input_tokens']:,} in / ~{est['est_output_tokens']:,} out tokens, ≈ ${est['est_cost_usd']:.2f} ({args.model})")
    print("(±50%: thinking effort and Greek tokenisation vary; run a --limit pilot to calibrate)")


def cmd_profile(args) -> None:
    from .ai_profile import (estimate_cost, run_profile, load_profiles, profile_settings,
                             _resume_chunks, OPENAI_PRESETS)

    verses = _select(args, _load_corpus())
    model = args.model
    if args.backend in OPENAI_PRESETS and model.startswith("claude"):
        model = OPENAI_PRESETS[args.backend]["model"] or model
    done = load_profiles(_profiles_path(args))
    settings = profile_settings(args.backend, args.effort, args.thinking, args.base_url)
    chunks = _resume_chunks(verses, done, model, args.backend, settings, args.chunk_size, args.limit)
    todo = [v for chunk in chunks for v in chunk]
    est = estimate_cost(todo, model, args.chunk_size, batch=args.backend == "batch", chunks=chunks)
    print(f"scope={args.scope} works={args.works or 'all'}: {est['verses']} verses in {len(chunks)} complete requests ≈ ${est['est_cost_usd']:.2f} on {model}")
    if not args.yes and est["verses"] > 500:
        sys.exit("more than 500 verses: re-run with --yes to confirm the spend, or use --limit N for a pilot")
    totals = run_profile(
        verses, _profiles_path(args), backend=args.backend, model=model, effort=args.effort,
        chunk_size=args.chunk_size, workers=args.workers, limit=args.limit,
        base_url=args.base_url, api_key_env=args.api_key_env, thinking=args.thinking,
    )
    print(json.dumps(totals, indent=1))


def cmd_collect(args) -> None:
    from .ai_profile import collect_batches

    verses = _load_corpus()
    print(json.dumps(collect_batches(_profiles_path(args), {v["id"]: v for v in verses}, wait=args.wait), indent=1))


def cmd_cluster(args) -> None:
    from .ai_profile import load_profiles
    from .cluster import run

    args.language = args.language or "grc"
    verses = _select(args, _load_corpus())
    if not verses:
        sys.exit(f"no verses for language={args.language} scope={args.scope} works={args.works}")
    profiles = None if args.no_ai else load_profiles(_profiles_path(args))
    if profiles is not None and not profiles:
        sys.exit("no AI profiles found; generate profiles or explicitly use --no-ai")
    if profiles and not any(v["id"] in profiles for v in verses):
        sys.exit("no AI profiles cover this selection; generate profiles or explicitly use --no-ai")
    weights = {"lex": args.w_lex, "ai": args.w_ai, "tags": args.w_tags}
    out = _lang_out(args)
    summary = run(verses, profiles, out, k=args.k, kmin=args.kmin, kmax=args.kmax,
                  window=args.window, alpha=args.alpha, weights=weights, seed=args.seed,
                  criterion=args.k_criterion)
    print(json.dumps({k: v for k, v in summary.items() if k != "validation"}, indent=1))
    print(f"ARI vs work {summary['validation']['ari_vs_work']}, vs group {summary['validation']['ari_vs_group']}, "
          f"mean purity {summary['validation']['mean_purity']}")
    print(f"outputs in {out}")


def cmd_cluster_passages(args) -> None:
    from .continuity import annotate_continuity
    from .passages import build_passages
    from .cluster import run
    from .report import render
    from .corpus.build import load_corpus

    args.language = args.language or 'grc'
    source = load_corpus(args.corpus) if getattr(args, 'corpus', None) else _load_corpus()
    verses = _select(args, annotate_continuity(source))
    result = build_passages(verses, tokens=args.tokens)
    out = Path(args.out) if args.out else OUTPUT / 'passages' / args.language
    out.mkdir(parents=True, exist_ok=True)
    (out / 'passages.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if len(result['passages']) < 3:
        raise SystemExit(f"fewer than three complete passages; exclusions and source mapping saved to {out / 'passages.json'}")
    summary = run(result['passages'], None, out, k=args.k, kmin=1, kmax=args.kmax,
                  window=0, alpha=0, weights={'lex': 1.0}, seed=args.seed,
                  criterion=args.k_criterion)
    summary.update(input_unit='pooled_token_passage', passage_tokens=args.tokens,
                   source_mapping_file='passages.json', input_verses=len(verses))
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    render(out)
    print(f"Analyzed {len(result['passages'])} complete {args.tokens}-token passages. Style groups remain exploratory.")
    print(f"Report: {out / 'report.md'}; source spans and exclusions: {out / 'passages.json'}")


def cmd_report(args) -> None:
    from .report import render

    out = _lang_out(args)
    render(out)
    print(f"wrote {out / 'report.md'}")


def cmd_html(args) -> None:
    from .html import build_root_index, build_site

    out = _lang_out(args)
    site = build_site(out)
    build_root_index(OUTPUT)
    print(f"wrote {site / 'index.html'} (+ authors/ and works/ pages); language index at {OUTPUT / 'index.html'}")


def cmd_witnesses(args) -> None:
    from .corpus.build import load_corpus
    from .witnesses import run

    path = CORPUS.with_name("witnesses.jsonl")
    if not path.exists():
        sys.exit(f"{path} not found - run `stylometry build-corpus` first")
    lang = args.language or "grc"
    out = _lang_out(args)
    summary = run(load_corpus(path), lang, out)
    print(json.dumps(summary, indent=1))
    print(f"outputs: {out / 'witness_summary.csv'}, {out / 'site' / 'witnesses.html'}")


def cmd_manifest(args) -> None:
    """Write or verify the checked-in record of which manuscripts the corpus contains."""
    import json as _json

    from .corpus.build import load_corpus
    from .manifest import MANIFEST_PATH, build_manifest, compare

    witness_path = CORPUS.with_name("witnesses.jsonl")
    if not CORPUS.exists() or not witness_path.exists():
        sys.exit("corpus not built; run `stylometry build-corpus` first")
    current = build_manifest(load_corpus(CORPUS), load_corpus(witness_path), Path(args.raw))
    path = Path(args.path) if args.path else ROOT / MANIFEST_PATH
    if args.write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(current, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {path}")
    missing = [f"{t}: {e['name']}" for t, es in current["coverage"].items() for e in es if not e["present"]]
    print(f"{current['primary_units']:,} primary units, {current['witness_units']:,} witness units, "
          f"{current['witnesses']} witnesses, {current['dead_sea_scroll_sigla']} Dead Sea Scroll sigla")
    for tradition, entries in current["coverage"].items():
        got = sum(1 for e in entries if e["present"])
        print(f"  {tradition}: {got}/{len(entries)} required manuscripts present")
    if missing:
        print("  missing: " + "; ".join(missing))
    if args.check:
        if not path.exists():
            sys.exit(f"{path} not found; run `stylometry manifest --write` first")
        problems = compare(current, _json.loads(path.read_text(encoding="utf-8")))
        if problems:
            print("\ncorpus no longer matches the manifest:")
            for problem in problems:
                print(f"  - {problem}")
            sys.exit(1)
        print("\ncorpus matches the manifest")


def cmd_accuracy_gate(args) -> None:
    """Check measured benchmark accuracy against the declared targets."""
    import json as _json

    from .accuracy_gate import as_dict, evaluate, load_targets, render

    targets_path = Path(args.targets) if args.targets else None
    targets = load_targets(targets_path)
    report_path = Path(args.report)
    if not report_path.exists():
        sys.exit(f"{report_path} not found; run `stylometry benchmark --download` first")
    report = _json.loads(report_path.read_text(encoding="utf-8"))
    outcome = evaluate(report, targets, str(targets_path or ""), str(report_path))
    print(render(outcome))
    out = Path(args.out) if args.out else report_path.with_name("accuracy_gate.json")
    out.write_text(_json.dumps(as_dict(outcome), indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    if not outcome.passed:
        sys.exit(1)


def cmd_compare(args) -> None:
    from .compare import build, load_set, parse_set
    from .corpus.meta import select_verses

    verses = _load_corpus()
    costs = dict(c.split("=", 1) for c in (args.cost or []))
    sets = {}
    for spec in args.set:
        name, path = parse_set(spec)
        sets[name] = load_set(name, path, float(costs[name]) if name in costs else None)
    scopes = {
        "whole corpus": len(verses),
        "Greek christian": len(select_verses(verses, scope="christian", language="grc")),
        "Greek all": len(select_verses(verses, scope="all", language="grc")),
        "Hebrew": len(select_verses(verses, scope="all", language="hbo")),
        "Arabic": len(select_verses(verses, scope="all", language="arb")),
    }
    scopes = {k: v for k, v in scopes.items() if v}
    out = Path(args.out) if args.out else OUTPUT / "models"
    analysis_verses = select_verses(verses, scope="all", language=args.language) if args.language else verses
    res = build(sets, analysis_verses, out, reference=args.reference, scopes=scopes, seed=args.seed, do_cluster=not args.no_cluster,
                ensembles=not args.no_ensemble)
    from .html import build_root_index

    build_root_index(OUTPUT)
    rec = res["recommendation"]
    if rec:
        print(f"highest consensus agreement: {rec['best']}   best value: {rec['value']}   cheapest: {rec['cheapest']}")
    print(f"wrote {out / 'index.html'}, {out / 'model_comparison.md'}, {out / 'models.csv'}")


def cmd_all(args) -> None:
    cmd_build(args)
    from .corpus.build import load_corpus

    present = sorted({v["language"] for v in load_corpus(CORPUS)})
    languages = [args.language] if args.language else [l for l in LANGS if l in present]
    for lang in languages:
        print(f"\n===== {lang} =====")
        args.language = lang
        if not args.no_ai:
            cmd_profile(args)
        cmd_cluster(args)
        cmd_report(args)
        cmd_html(args)
        cmd_witnesses(args)


def cmd_benchmark(args) -> None:
    from .benchmark_suite import run_suite

    try:
        result = run_suite(args.manifest, args.cache, args.out, download=args.download,
                           language=args.language, controls=not args.no_controls)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Empirical acceptance: {result['status']}; scripture attribution remains unvalidated.")
    print(f"Report: {Path(args.out) / 'report.md'}")
    if args.check and result['status'] != 'passed':
        raise SystemExit(1)


def cmd_benchmark_rejection(args) -> None:
    from .benchmark_data import load_benchmark, read_manifest
    from .benchmark_suite import DEFAULT_MANIFESTS
    from .rejection_experiment import run_experiment

    try:
        works = []
        for manifest in args.manifest or DEFAULT_MANIFESTS:
            if args.language and not any(w['language'] == args.language for w in read_manifest(manifest)['works']):
                continue
            works.extend(w for w in load_benchmark(manifest, args.cache, download=args.download)
                         if not args.language or w['language'] == args.language)
        if not works:
            raise ValueError('no selected reference corpus; this language remains unvalidated')
        run_experiment(works, args.out, progress=lambda message: print(message, flush=True))
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    print('Development experiment only; authorship remains unvalidated.')
    print(f"Report: {Path(args.out) / 'report.md'}")


def cmd_author_study(args) -> None:
    from .author_study import develop, seal, evaluate
    from .benchmark_data import load_benchmark, read_manifest
    from .benchmark_suite import DEFAULT_MANIFESTS
    try:
        if args.stage == 'develop':
            works = []
            for manifest in args.manifest or DEFAULT_MANIFESTS:
                if args.language and not any(w['language'] == args.language for w in read_manifest(manifest)['works']):
                    continue
                works.extend(w for w in load_benchmark(manifest, args.cache, download=args.download)
                             if not args.language or w['language'] == args.language)
            if not works:
                raise ValueError('no selected development reference texts')
            develop(works, args.out, progress=lambda s: print(s, flush=True))
            print(f"Development results and model lock saved in {args.out}")
        elif args.stage == 'seal':
            if not args.model_lock or not args.manifest:
                raise ValueError('seal requires --model-lock and one or more fresh --manifest paths')
            seal(args.model_lock, args.manifest, args.out)
            print(f"Locked model and reserved source manifests: {Path(args.out) / 'evaluation_lock.json'}")
        else:
            if not args.evaluation_lock:
                raise ValueError('evaluate requires --evaluation-lock')
            result = evaluate(args.evaluation_lock, args.cache, args.out, download=args.download,
                              progress=lambda s: print(s, flush=True))
            print(f"Fresh evaluation: {result['status']}; scripture attribution remains unvalidated.")
            print(f"Report: {Path(args.out) / 'fresh_evaluation.md'}")
            if args.check and result['status'] != 'passed':
                raise SystemExit(1)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc


def cmd_verification_study(args) -> None:
    from .verification_study import DEFAULT_MANIFESTS, run_study
    from .benchmark_data import load_benchmark, read_manifest

    try:
        works = []
        for manifest in args.manifest or DEFAULT_MANIFESTS:
            if args.language and not any(w['language'] == args.language for w in read_manifest(manifest)['works']):
                continue
            works.extend(w for w in load_benchmark(manifest, args.cache, download=args.download)
                         if not args.language or w['language'] == args.language)
        result = run_study(works, args.out, progress=lambda s: print(s, flush=True))
        print('Pair-verification development completed; independent authorship and scripture validation remain outstanding.')
        print(f"Report: {Path(args.out) / 'development.md'}")
        main_runs = [r for r in result['runs'] if r['genre'] is None]
        missing_languages = (set(result.get('requested_languages', []))
                             - {r.get('language') for r in main_runs})
        if args.check and (not main_runs or missing_languages or
                           not all(r['summary']['pair_verifier']['preliminary_development_signal'] for r in result['runs'])):
            raise SystemExit(1)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, no expansion; existing environment wins."""
    import os

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(prog="stylometry", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-corpus", help="parse raw sources into verses.jsonl")
    b.add_argument("--raw", default=str(RAW))
    b.add_argument("--include-duplicates", action="store_true", help="keep Lake's Barnabas/Hermas alongside the Sinaiticus copies")
    b.set_defaults(func=cmd_build)

    e = sub.add_parser("estimate", help="estimate AI profiling cost")
    _add_scope(e)
    _add_profiles(e)
    e.add_argument("--model", default="claude-opus-5", help="e.g. claude-opus-5, claude-sonnet-5, deepseek-v4-pro, deepseek-flash")
    e.add_argument("--chunk-size", type=int, default=25)
    e.set_defaults(func=cmd_estimate)

    pr = sub.add_parser("profile", help="produce AI style profiles per verse")
    _add_scope(pr)
    _add_profiles(pr)
    pr.add_argument("--backend", default="sdk", choices=["sdk", "batch", "cli", "deepseek", "openai"])
    pr.add_argument("--model", default="claude-opus-5", help="claude-* for sdk/batch/cli; deepseek-v4-pro or deepseek-flash for deepseek")
    pr.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    pr.add_argument("--chunk-size", type=int, default=25, help="verse units per request")
    pr.add_argument("--workers", type=int, default=4, help="parallel requests (sdk/deepseek/openai backends)")
    pr.add_argument("--limit", type=int, help="profile approximately N units, rounded up to preserve complete request context")
    pr.add_argument("--yes", action="store_true", help="confirm spending on a large run")
    pr.add_argument("--base-url", help="openai backend: endpoint base URL")
    pr.add_argument("--api-key-env", help="environment variable holding the API key (default DEEPSEEK_API_KEY / OPENAI_API_KEY)")
    pr.add_argument("--thinking", action="store_true", help="deepseek backend: enable thinking mode (slower, ~2x output tokens)")
    pr.set_defaults(func=cmd_profile)

    pc = sub.add_parser("profile-collect", help="collect finished Batch API results")
    _add_profiles(pc)
    pc.add_argument("--wait", action="store_true", help="poll until every batch has ended")
    pc.set_defaults(func=cmd_collect)

    c = sub.add_parser("cluster", help="explore style groups (one language at a time)")
    _add_scope(c)
    _add_profiles(c)
    c.add_argument("--out", default=None, help="output directory (default output/<language>)")
    c.add_argument("--k", type=int, help="force the number of exploratory style groups (including 1)")
    c.add_argument("--kmin", type=int, default=2)
    c.add_argument("--kmax", type=int, default=20)
    c.add_argument("--window", type=int, default=5, help="neighbour verses on each side for smoothing")
    c.add_argument("--alpha", type=float, default=0.7, help="weight of the neighbourhood mean vs the verse itself")
    c.add_argument("--w-lex", type=float, default=1.0)
    c.add_argument("--w-ai", type=float, default=1.0)
    c.add_argument("--w-tags", type=float, default=0.7)
    c.add_argument("--no-ai", action="store_true", help="ignore AI profiles even if present")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--k-criterion", default="silhouette", choices=["silhouette", "bic", "davies_bouldin", "calinski"],
                   help="rank supported style partitions; single-group and stability checks still apply")
    c.set_defaults(func=cmd_cluster)

    cp = sub.add_parser('cluster-passages', help='pool raw tokens before exploratory lexical style analysis')
    _add_scope(cp)
    cp.add_argument('--tokens', type=int, choices=[500, 1000, 2000], default=1000)
    cp.add_argument('--corpus', help='source verse JSONL (defaults to the built project corpus)')
    cp.add_argument('--out', default=None)
    cp.add_argument('--k', type=int)
    cp.add_argument('--kmax', type=int, default=20)
    cp.add_argument('--seed', type=int, default=42)
    cp.add_argument('--k-criterion', choices=['silhouette', 'bic', 'davies_bouldin', 'calinski'], default='silhouette')
    cp.set_defaults(func=cmd_cluster_passages)

    r = sub.add_parser("report", help="render output/<language>/report.md")
    r.add_argument("--language", choices=LANGS, default="grc")
    r.add_argument("--out", default=None)
    r.set_defaults(func=cmd_report)

    h = sub.add_parser("html", help="render the HTML site (dashboard, per-author and per-work pages)")
    h.add_argument("--language", choices=LANGS, default="grc")
    h.add_argument("--out", default=None)
    h.set_defaults(func=cmd_html)

    wt = sub.add_parser("witnesses", help="compare every manuscript witness against the primary text, verse by verse")
    wt.add_argument("--language", choices=LANGS, default="grc")
    wt.add_argument("--out", default=None)
    wt.set_defaults(func=cmd_witnesses)

    mf = sub.add_parser("manifest", help="record or verify which manuscripts the corpus contains")
    mf.add_argument("--raw", default=str(RAW))
    mf.add_argument("--path", default=None, help="manifest file (default benchmarks/corpus_manifest.json)")
    mf.add_argument("--write", action="store_true", help="regenerate the manifest from the built corpus")
    mf.add_argument("--check", action="store_true", help="exit non-zero if the corpus no longer matches")
    mf.set_defaults(func=cmd_manifest)

    ag = sub.add_parser("accuracy-gate",
                        help="check measured benchmark accuracy against benchmarks/accuracy_targets.json")
    ag.add_argument("--report", default=str(OUTPUT / "benchmark" / "attribution" / "benchmark.json"),
                    help="benchmark.json produced by `stylometry benchmark`")
    ag.add_argument("--targets", default=None, help="targets file (default benchmarks/accuracy_targets.json)")
    ag.add_argument("--out", default=None, help="where to write the gate result (default beside the report)")
    ag.set_defaults(func=cmd_accuracy_gate)

    cm = sub.add_parser("compare-models", help="compare AI profiling models on the verses they have all profiled")
    cm.add_argument("--set", action="append", required=True, metavar="NAME=PATH",
                    help="a profiles file to compare; NAME_retest pairs with NAME as a repeat run")
    cm.add_argument("--reference", help="set to treat as the reference (default: the first --set)")
    cm.add_argument("--language", choices=LANGS, help="restrict comparison to one language (required for multilingual profile sets with clustering)")
    cm.add_argument("--cost", action="append", metavar="NAME=USD_PER_VERSE",
                    help="override the $/verse measured from NAME's *_runs.jsonl sidecar")
    cm.add_argument("--out", default=None, help=f"output directory (default {OUTPUT / 'models'})")
    cm.add_argument("--seed", type=int, default=0)
    cm.add_argument("--no-cluster", action="store_true", help="skip re-running the clustering per model")
    cm.add_argument("--no-ensemble", action="store_true", help="do not add NAME×2 rows (mean of a set and its retest)")
    cm.set_defaults(func=cmd_compare)

    a = sub.add_parser("all", help="run the whole pipeline (every language present unless --language)")
    _add_scope(a)
    _add_profiles(a)
    a.add_argument("--raw", default=str(RAW))
    a.add_argument("--include-duplicates", action="store_true")
    a.add_argument("--backend", default="sdk", choices=["sdk", "batch", "cli", "deepseek", "openai"])
    a.add_argument("--model", default="claude-opus-5")
    a.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    a.add_argument("--chunk-size", type=int, default=25)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--limit", type=int)
    a.add_argument("--yes", action="store_true")
    a.add_argument("--base-url")
    a.add_argument("--api-key-env")
    a.add_argument("--thinking", action="store_true")
    a.add_argument("--out", default=None)
    a.add_argument("--k", type=int)
    a.add_argument("--kmin", type=int, default=2)
    a.add_argument("--kmax", type=int, default=20)
    a.add_argument("--window", type=int, default=5)
    a.add_argument("--alpha", type=float, default=0.7)
    a.add_argument("--w-lex", type=float, default=1.0)
    a.add_argument("--w-ai", type=float, default=1.0)
    a.add_argument("--w-tags", type=float, default=0.7)
    a.add_argument("--no-ai", action="store_true")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--k-criterion", default="silhouette", choices=["silhouette", "bic", "davies_bouldin", "calinski"])
    a.set_defaults(func=cmd_all)

    bm = sub.add_parser("benchmark", help="run real-text author controls with whole-work holdouts")
    bm.add_argument("--manifest", action="append", help="source manifest JSON (repeatable; defaults to bundled Greek and Hebrew manifests)")
    bm.add_argument("--cache", default=str(RAW / "benchmarks"))
    bm.add_argument("--out", default=str(OUTPUT / "benchmark"))
    bm.add_argument("--download", action="store_true", help="download missing checksum-pinned source files")
    bm.add_argument("--language", choices=LANGS, help="evaluate one reference language")
    bm.add_argument("--no-controls", action="store_true", help="run held-out attribution only; skip production clustering and perturbation controls")
    bm.add_argument("--check", action="store_true", help="exit unsuccessfully unless all predeclared empirical gates pass")
    bm.set_defaults(func=cmd_benchmark)

    br = sub.add_parser('benchmark-rejection', help='development-only calibration against separate unfamiliar authors')
    br.add_argument('--manifest', action='append', help='reference manifest JSON (repeatable)')
    br.add_argument('--cache', default=str(RAW / 'benchmarks'))
    br.add_argument('--out', default=str(OUTPUT / 'rejection-development'))
    br.add_argument('--download', action='store_true', help='download missing checksum-pinned reference texts')
    br.add_argument('--language', choices=LANGS)
    br.set_defaults(func=cmd_benchmark_rejection)

    study = sub.add_parser('author-study', help='develop models, seal a design, then evaluate reserved authors')
    study.add_argument('stage', choices=['develop', 'seal', 'evaluate'])
    study.add_argument('--manifest', action='append')
    study.add_argument('--model-lock')
    study.add_argument('--evaluation-lock')
    study.add_argument('--cache', default=str(RAW / 'benchmarks'))
    study.add_argument('--out', default=str(OUTPUT / 'author-study'))
    study.add_argument('--download', action='store_true')
    study.add_argument('--language', choices=LANGS, help='development-language selection')
    study.add_argument('--check', action='store_true', help='fail if fresh evaluation is failed or inconclusive')
    study.set_defaults(func=cmd_author_study)

    verifier = sub.add_parser('verification-study', help='author-disjoint cross-work pair verification on exposed development sources')
    verifier.add_argument('--manifest', action='append', help='exposed development manifest (repeatable; defaults to all four previous reference manifests)')
    verifier.add_argument('--cache', default=str(RAW / 'benchmarks'))
    verifier.add_argument('--out', default=str(OUTPUT / 'verification-development-v1'))
    verifier.add_argument('--download', action='store_true', help='download missing checksum-pinned development texts')
    verifier.add_argument('--language', choices=LANGS)
    verifier.add_argument('--check', action='store_true', help='fail unless all main development panels meet the preliminary signal criteria; never a final validation claim')
    verifier.set_defaults(func=cmd_verification_study)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
