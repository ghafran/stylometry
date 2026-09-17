"""Command-line entry point.

    stylometry build-corpus                     parse raw sources -> data/processed/verses.jsonl
    stylometry estimate  [--scope ..] [--model] print the AI-profiling cost estimate
    stylometry profile   [--backend sdk|batch|cli] [--scope ..] [--limit N]   AI style profiles
    stylometry profile-collect                  fetch finished Batch API results
    stylometry cluster   [--k N] [--scope ..] [--no-ai]   discover authors A1..Ak
    stylometry report                           write output/report.md
    stylometry html / witnesses                 HTML site; manuscript-witness comparison
    stylometry compare-models --set NAME=PATH   accuracy, stability and cost of the profiling models
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
    return select_verses(verses, scope=scope, works=args.works, language=language)


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

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
