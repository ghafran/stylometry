"""Command-line entry point.

    stylometry build-corpus                     parse raw sources -> data/processed/verses.jsonl
    stylometry estimate  [--scope ..] [--model] print the AI-profiling cost estimate
    stylometry profile   [--backend sdk|batch|cli] [--scope ..] [--limit N]   AI style profiles
    stylometry profile-collect                  fetch finished Batch API results
    stylometry cluster   [--k N] [--scope ..] [--with-ai]  exploratory style groups (lexical by default)
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


def _corpus_for(args):
    """The analysed corpus: one primary text per work, or every manuscript that attests it.

    The default corpus holds one witness per work, so an analysis of John is an analysis of whichever
    manuscript was chosen as primary - Sinaiticus - and the other twenty-nine copies of John never
    enter it. ``--witnesses`` reads ``witnesses.jsonl`` instead, where each manuscript's text of a work
    is its own unit, so manuscripts can be compared with each other rather than only against a base.
    """
    from .corpus.build import load_corpus

    if not getattr(args, "witnesses", False):
        return _load_corpus()
    path = CORPUS.with_name("witnesses.jsonl")
    if not path.exists():
        sys.exit(f"{path} not found - run `stylometry build-corpus` first")
    return load_corpus(path)


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
    from .ai.profile import estimate_cost, load_profiles, chunk_verses

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
    from .ai.profile import (estimate_cost, run_profile, load_profiles, profile_settings,
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
    if totals.get("aborted"):
        sys.exit(f"run stopped early: {totals['aborted']}. "
                 f"{totals['profiled']} units were saved; rerun the same command to continue.")


def cmd_collect(args) -> None:
    from .ai.profile import collect_batches

    verses = _load_corpus()
    print(json.dumps(collect_batches(_profiles_path(args), {v["id"]: v for v in verses}, wait=args.wait), indent=1))


def cmd_cluster(args) -> None:
    from .cluster import run

    args.language = args.language or "grc"
    verses = _select(args, _load_corpus())
    if not verses:
        sys.exit(f"no verses for language={args.language} scope={args.scope} works={args.works}")
    # Lexical by default: style is measured from the text alone, so a run needs no key, no network
    # and no spend, and anyone with the corpus can reproduce it exactly. --with-ai opts back into the
    # archived profiling in stylometry/ai/, which is kept but is no longer part of the analysis.
    profiles = None
    if args.with_ai:
        from .ai.profile import load_profiles

        profiles = load_profiles(_profiles_path(args))
        if not profiles:
            sys.exit(f"--with-ai found no profiles in {_profiles_path(args)}; "
                     f"drop the flag to run on lexical features")
        if not any(v["id"] in profiles for v in verses):
            sys.exit(f"--with-ai: no profile in {_profiles_path(args)} covers this selection; "
                     f"drop the flag to run on lexical features")
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
    from .passages import build_passages, pool_profiles
    from .cluster import run
    from .report import render
    from .corpus.build import load_corpus

    # A profiles file alone must not quietly turn the control into an AI run; the opt-in is the
    # only way in, and this is checked before anything is loaded.
    if getattr(args, 'profiles', None) and not args.with_ai:
        raise SystemExit(2)
    args.language = args.language or 'grc'
    source = load_corpus(args.corpus) if getattr(args, 'corpus', None) else _load_corpus()
    bridge = getattr(args, 'bridge_chapters', False)
    verses = _select(args, annotate_continuity(source, bridge_chapters=bridge))
    result = build_passages(verses, tokens=args.tokens, bridge_chapters=bridge)
    out = Path(args.out) if args.out else OUTPUT / 'passages' / args.language
    out.mkdir(parents=True, exist_ok=True)
    (out / 'passages.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if len(result['passages']) < 3:
        raise SystemExit(f"fewer than three complete passages; exclusions and source mapping saved to {out / 'passages.json'}")
    # Passage mode is the AI-free control by default: no model touches it, so its result is
    # independent of any profiling. Pooling existing verse profiles in is available, but only when
    # asked for explicitly, and the summary records which of the two was run.
    pooled = None
    if args.with_ai:
        from .ai.profile import load_profiles

        verse_profiles = load_profiles(_profiles_path(args))
        if not verse_profiles:
            raise SystemExit(f"--with-ai needs verse profiles; none found in {_profiles_path(args)}")
        pooled = pool_profiles(result['passages'], verse_profiles)
        if len(pooled) < 3:
            raise SystemExit(f"only {len(pooled)} of {len(result['passages'])} passages are covered by "
                             f"{_profiles_path(args)}; profile the verses first or drop --with-ai")
    weights = ({'lex': 1.0} if pooled is None
               else {'lex': args.w_lex, 'ai': args.w_ai, 'tags': args.w_tags})
    summary = run(result['passages'], pooled, out, k=args.k, kmin=1, kmax=args.kmax,
                  window=0, alpha=0, weights=weights, seed=args.seed,
                  criterion=args.k_criterion)
    summary.update(input_unit='pooled_token_passage', passage_tokens=args.tokens,
                   source_mapping_file='passages.json', input_verses=len(verses),
                   ai_profiles_pooled=(0 if pooled is None else len(pooled)))
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


def _attributable(labels: list[str], works: list[str]) -> tuple[list[int], list[str]]:
    """Indices that can honestly be scored, and the labels that cannot be.

    Holding out a whole work removes every passage of it. A label carried by only one work therefore
    has nothing left to match against, and every one of its units is wrong by construction. Scoring
    them deflates the result without saying anything, so they are reported and set aside.
    """
    from collections import defaultdict

    works_per_label: dict = defaultdict(set)
    for label, work in zip(labels, works):
        works_per_label[label].add(work)
    unattributable = sorted(l for l, ws in works_per_label.items() if len(ws) < 2)
    keep = [i for i, label in enumerate(labels) if label not in set(unattributable)]
    return keep, unattributable


def _holdout_works(passages: list[dict]) -> list[str]:
    """What must be held out together.

    With ``--witnesses`` the same chapter of John appears once per manuscript, under work codes
    JOHN, JOHN@P66, JOHN@P75 and so on. Holding out by that code would leave P66's John to be judged
    by Sinaiticus's John - nearly the same words - and every method would look superb while having
    learnt nothing. Holding out by the *base* work keeps all copies of a text together, so a
    manuscript can only be recognised from the books it is not being tested on.
    """
    return [str(p.get("duplicate_of") or p["work"]).split("@", 1)[0] for p in passages]


def cmd_delta(args) -> None:
    """Burrows's Delta as a baseline: most-frequent-word rates, z-scored, nearest neighbour.

    Reported against whatever label the corpus already carries, held out by whole work, so a passage
    is never attributed by its own neighbours. No model is involved.
    """
    from collections import Counter, defaultdict

    from .continuity import annotate_continuity
    from .delta import attribute
    from .passages import build_passages

    verses = _select(args, annotate_continuity(_corpus_for(args), bridge_chapters=args.bridge_chapters))
    if not verses:
        sys.exit(f"no verses for language={args.language} scope={args.scope} works={args.works}")
    passages = build_passages(verses, tokens=args.tokens, bridge_chapters=args.bridge_chapters)["passages"]
    if len(passages) < 2:
        sys.exit(f"only {len(passages)} complete {args.tokens}-token passages; "
                 f"try --tokens 500 or --bridge-chapters")
    docs = [p["text_bare"].split() for p in passages]
    labels = [str(p.get(args.label) or "?") for p in passages]
    works = _holdout_works(passages)
    keep, unattributable = _attributable(labels, works)
    if len(keep) < 2:
        sys.exit("every label is carried by a single work, so nothing can be attributed")
    # Set aside before attributing, not after: a label that can never be right must not be offered
    # as a candidate either, or it takes predictions while being unable to earn any.
    dropped = len(docs) - len(keep)
    docs, labels, works = ([x[i] for i in keep] for x in (docs, labels, works))
    predicted = attribute(docs, labels, groups=works, n_words=args.mfw, metric=args.metric)
    keep = list(range(len(docs)))

    scored = [(labels[i], predicted[i]) for i in keep]
    correct = sum(p == t for t, p in scored)
    kept_labels = [labels[i] for i in keep]
    majority = max(Counter(kept_labels).values()) / len(kept_labels)
    print(f"{len(docs)} passages of {args.tokens} tokens, {len(set(labels))} {args.label} values, "
          f"{len(set(works))} works")
    print(f"{args.metric} Delta on {args.mfw} most frequent words, held out by work")
    if unattributable:
        print(f"  {len(unattributable)} {args.label} values come from a single work and cannot be "
              f"attributed once it is held out; {dropped} units set aside")
    print(f"  accuracy {correct / len(scored):.1%} over {len(scored)} units   "
          f"majority baseline {majority:.1%}   chance {1 / len(set(kept_labels)):.1%}")
    per: dict = defaultdict(lambda: [0, 0])
    for truth, pred in scored:
        per[truth][1] += 1
        per[truth][0] += truth == pred
    print(f"\n  {args.label:<26}{'correct':>9}{'n':>6}")
    for value, (hit, total) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        print(f"  {value[:26]:<26}{hit / total:>8.0%}{total:>6}")
    print("\nDelta ranks candidates and never answers 'none of these'; it attributes, it does not verify.")


def cmd_wan(args) -> None:
    """Word adjacency networks: which function word follows which, and how closely."""
    from collections import Counter, defaultdict

    from .continuity import annotate_continuity
    from .wan import attribute_pooled

    verses = _select(args, annotate_continuity(_corpus_for(args), bridge_chapters=args.bridge_chapters))
    if not verses:
        sys.exit(f"no verses for language={args.language} scope={args.scope} works={args.works}")
    language = args.language or "grc"
    if args.whole_works:
        pooled: dict = defaultdict(list)
        for v, base in zip(verses, _holdout_works(verses)):
            pooled[(base, str(v.get(args.label) or "?"))].append(v["text_bare"])
        units = [{"tokens": " ".join(t).split(), "work": w, "label": g} for (w, g), t in pooled.items()]
        units = [u for u in units if len(u["tokens"]) >= 2000]
    else:
        from .passages import build_passages

        built = build_passages(verses, tokens=args.tokens,
                               bridge_chapters=args.bridge_chapters)["passages"]
        units = [{"tokens": p["text_bare"].split(), "work": w,
                  "label": str(p.get(args.label) or "?")}
                 for p, w in zip(built, _holdout_works(built))]
    if len(units) < 2:
        sys.exit(f"only {len(units)} units; try --tokens 500, --bridge-chapters, or --whole-works")

    docs = [u["tokens"] for u in units]
    labels = [u["label"] for u in units]
    works = [u["work"] for u in units]
    keep, unattributable = _attributable(labels, works)
    if len(keep) < 2:
        sys.exit("every label is carried by a single work, so nothing can be attributed")
    dropped = len(docs) - len(keep)
    docs, labels, works = ([x[i] for i in keep] for x in (docs, labels, works))
    predicted = attribute_pooled(docs, labels, language=language, groups=works,
                                 window=args.window, n_markers=args.markers, decay=args.decay)
    scored = [(t, p) for t, p in zip(labels, predicted) if p is not None]
    if not scored:
        sys.exit("every label is carried by a single work, so nothing can be attributed")
    correct = sum(t == p for t, p in scored)
    kept_labels = [t for t, _ in scored]
    print(f"{len(docs)} {'works' if args.whole_works else str(args.tokens) + '-token passages'}, "
          f"{len(set(labels))} {args.label} values, {len(set(works))} works")
    print(f"{args.markers} markers, window {args.window}, {args.decay} decay, held out by work")
    if unattributable:
        print(f"  {len(unattributable)} {args.label} values come from a single work and cannot be "
              f"attributed once it is held out; {dropped} units set aside")
    print(f"  accuracy {correct / len(scored):.1%} over {len(scored)} units   "
          f"majority baseline {max(Counter(kept_labels).values()) / len(kept_labels):.1%}   "
          f"chance {1 / len(set(kept_labels)):.1%}")
    per: dict = defaultdict(lambda: [0, 0])
    for truth, pred in scored:
        per[truth][1] += 1
        per[truth][0] += truth == pred
    print(f"\n  {args.label:<26}{'correct':>9}{'n':>6}")
    for value, (hit, total) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        print(f"  {value[:26]:<26}{hit / total:>8.0%}{total:>6}")


def cmd_analyse(args) -> None:
    """The full battery: every feature family the corpus supports, every method it warrants."""
    from .analysis import analyse, save
    from .continuity import annotate_continuity
    from .passages import build_passages

    verses = _select(args, annotate_continuity(_corpus_for(args), bridge_chapters=args.bridge_chapters))
    if not verses:
        sys.exit(f"no verses for language={args.language} scope={args.scope} works={args.works}")
    language = args.language or "grc"
    passages = build_passages(verses, tokens=args.tokens,
                              bridge_chapters=args.bridge_chapters)["passages"]
    if len(passages) < 4:
        sys.exit(f"only {len(passages)} complete {args.tokens}-token passages; "
                 f"try --tokens 500 or --bridge-chapters")

    # The passage text carries no punctuation, so the original verse text is rejoined for the blocks
    # that need it (sentence length, punctuation, rhythm).
    by_id = {v["id"]: v for v in verses}
    texts = [" ".join(by_id[i]["text"] for i in p["source_verse_ids"] if i in by_id) for p in passages]
    docs = [p["text_bare"].split() for p in passages]
    labels = [str(p.get(args.label) or "?") for p in passages]
    works = _holdout_works(passages)

    keep, unattributable = _attributable(labels, works)
    if len(keep) < 4:
        sys.exit("too few units remain once labels carried by a single work are set aside")
    dropped = len(docs) - len(keep)
    docs, texts, labels, works = ([x[i] for i in keep] for x in (docs, texts, labels, works))
    if unattributable:
        print(f"{len(unattributable)} {args.label} values come from a single work and cannot be "
              f"attributed once it is held out; {dropped} units set aside")

    result = analyse(docs, texts, labels, works, language=language, seed=args.seed,
                     permutations=args.permutations)
    out = Path(args.out) if args.out else OUTPUT / f"{language}-analysis"
    save(result, out)
    print(f"{result['n_units']} units, {result['n_features']} features, {result['n_works']} works")
    print(f"  baselines: majority {result['majority_baseline']:.1%}, chance {result['chance']:.1%}")
    for name, row in result["attribution"].items():
        print(f"  {name:<32}{row['accuracy']:>8.1%}")
    v = result["verification"]
    if v.get("available"):
        print(f"  verification: {v['acceptance_of_genuine']:.0%} of genuine pairings accepted, "
              f"{v['false_acceptance']:.0%} of false ones")
    print(f"wrote {out / 'analysis.md'} and analysis.json")


def cmd_strategies(args) -> None:
    """Run every strategy alone, together, and all-but-one; write the explorer."""
    from .continuity import annotate_continuity
    from .passages import build_passages
    from .strategy_report import save, sweep

    verses = _select(args, annotate_continuity(_corpus_for(args), bridge_chapters=args.bridge_chapters))
    if not verses:
        sys.exit(f"no verses for language={args.language} scope={args.scope} works={args.works}")
    language = args.language or "grc"
    passages = build_passages(verses, tokens=args.tokens,
                              bridge_chapters=args.bridge_chapters)["passages"]
    if len(passages) < 4:
        sys.exit(f"only {len(passages)} complete {args.tokens}-token passages; try --tokens 500")

    by_id = {v["id"]: v for v in verses}
    sources = [[by_id[i] for i in p["source_verse_ids"] if i in by_id] for p in passages]
    texts = [" ".join(v["text"] for v in group) for group in sources]
    # Part-of-speech tags travel with the verse, so a passage's tags are its verses' tags in order.
    pos = [[tag for v in group for tag in (v.get("pos") or [])] for group in sources]
    docs = [p["text_bare"].split() for p in passages]
    labels = [str(p.get(args.label) or "?") for p in passages]
    works = _holdout_works(passages)

    keep, unattributable = _attributable(labels, works)
    if len(keep) < 4:
        sys.exit("too few units remain once labels carried by a single work are set aside")
    if unattributable:
        print(f"{len(unattributable)} {args.label} values come from a single work and cannot be "
              f"attributed once it is held out; {len(docs) - len(keep)} units set aside")
    docs, texts, labels, works, pos = ([x[i] for i in keep] for x in (docs, texts, labels, works, pos))

    result = sweep(docs, texts, labels, works, language=language, pos=pos, seed=args.seed)
    out = Path(args.out) if args.out else OUTPUT / f"{language}-strategies"
    save(result, out, title=args.title or f"Strategies — {language}")
    print(f"{result['n_units']} passages, {result['n_works']} works, "
          f"majority baseline {result['majority_baseline']:.1%}")
    print(f"  all strategies together: {result['combined']['accuracy']:.1%} "
          f"({result['combined']['n_features']} features)")
    ranked = sorted(((v.get("accuracy") or 0, k) for k, v in result["solo"].items()), reverse=True)
    for accuracy, key in ranked[:5]:
        print(f"  {key:<22}{accuracy:>8.1%}")
    if result["unavailable"]:
        print(f"  unavailable for {language}: {', '.join(result['unavailable'])}")
    print(f"wrote {out / 'strategies.html'}")


def cmd_explore(args) -> None:
    """Build the drill-down explorer: language, collections, books, chapters, verses."""
    from .explorer import build
    from .explorer_html import write

    corpus = _corpus_for(args)
    languages = [args.language] if args.language else ["grc", "hbo", "arb"]
    out = Path(args.out) if args.out else OUTPUT / "explorer"
    built = []
    for language in languages:
        verses = _select(argparse.Namespace(scope=args.scope, language=language, works=args.works,
                                            chapters=None), corpus)
        if len(verses) < 10:
            print(f"  {language}: {len(verses)} verses, skipped")
            continue
        print(f"{language}: {len(verses):,} verses")
        data = build(verses, language, seed=args.seed, workers=args.workers,
                     progress=lambda m: print(f'  {m.strip()}', flush=True))
        write(data, out / language)
        built.append((language, data))
        print(f"  wrote {out / language / 'index.html'} "
              f"({len(data['strategies'])} strategies, {sum(len(c['children']) for c in data['tree'])} books)")
    if not built:
        sys.exit("no language had enough text to explore")
    _write_explorer_index(out, built)
    print(f"start at {out / 'index.html'}")


def _write_explorer_index(out: Path, built: list) -> None:
    """The front page: pick a strategy here, pick a language, and carry the choice through."""
    import json as _json

    from .explorer_html import CSS, JS_COMMON, LANGUAGE_NAMES

    # The union, in first-seen order. Hebrew has part-of-speech tags and the others do not, so a
    # strategy offered here is not offered everywhere; the page says which language is missing it
    # rather than quietly dropping the choice.
    strategies: dict = {}
    for _, data in built:
        for s in data["strategies"]:
            strategies.setdefault(s["key"], s)

    def by_strategy(data: dict) -> dict:
        """Keyed by name, not packed by position: this page's key order is the union, not a language's."""
        packed = data["book_groups"]
        return {key: {"k": packed["gk"][i], "reason": packed["gr"][i], "ari": packed["gari"][i],
                      "n": packed["gn"], "sizes": packed["gsz"][i]}
                for i, key in enumerate(data["keys"])}

    payload = {
        "keys": list(strategies),
        "strategies": list(strategies.values()),
        "group_reasons": built[0][1]["group_reasons"],
        "languages": [{"code": lang, "label": LANGUAGE_NAMES.get(lang, lang),
                       "verses": data["n_verses"],
                       "books": sum(len(c["children"]) for c in data["tree"]),
                       "keys": [s["key"] for s in data["strategies"]],
                       "groups": by_strategy(data)}
                      for lang, data in built],
    }
    (out / "index.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Style explorer</title>"
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;600&'
        'family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">'
        f"<style>{CSS}.row{{text-decoration:none;display:grid}}"
        ".row.dim{opacity:.55}.row .w{grid-column:1;font-size:12.5px;color:var(--clay)}"
        ".row .k{grid-row:2;grid-column:2;font-family:'IBM Plex Mono',monospace;font-size:11.5px;"
        "color:var(--muted);white-space:nowrap;align-self:end}"
        ".row .bk{grid-column:1/-1;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--faint)}"
        "</style></head><body><div class=\"wrap\">"
        "<h1>Style explorer</h1>"
        "<p class=\"small\">Pick a strategy, then a language, then drill from collection to book to "
        "chapter to verse. Every level reports how many style groups the units it lists fall into, and "
        "the strategy can be changed on every page — a grouping that survives the change means "
        "something, one that rearranges itself does not.</p>"
        "<div class=\"bar\" id=\"bar\"></div><div id=\"list\"></div>"
        "<p class=\"small\" style=\"margin-top:22px\">A style group is a partition of the units a level "
        "lists, fitted on pooled profiles and kept only if it survives refitting on 80% subsamples at an "
        "adjusted Rand index of 0.8 or better. One group means no split was supported, not that one hand "
        "wrote the text. Positions are the first two principal components of the chosen strategy's "
        "standardised features, comparable within a strategy and never between strategies. No model is "
        "involved at any point.</p>"
        "<script>const DATA = " + _json.dumps(payload, ensure_ascii=False) + ";\n" + JS_COMMON + """
function render() {
  document.getElementById('list').innerHTML = DATA.languages.map(l => {
    const has = l.keys.includes(strategy), g = l.groups[strategy];
    const count = !has ? '' : g.k > 1
      ? `<span class="k">${g.k} style groups</span>`
      : `<span class="k">1 style group</span>`;
    const stack = (!has || g.k < 2) ? '' :
      `<span class="bk">${g.sizes.map((c, i) => `A${i + 1}&nbsp;${c}`).join(' · ')}` +
      `<span class="stack">${g.sizes.map((c, i) =>
        `<span style="flex:${c};background:var(--g${i + 1})" title="A${i + 1}: ${c}"></span>`).join('')}</span></span>`;
    const why = (!has || g.k > 1) ? '' :
      `<span class="w">${(DATA.group_reasons || {})[g.reason] || 'no split is supported'}</span>`;
    return `<a class="row${has ? '' : ' dim'}" href="${l.code}/index.html?s=${encodeURIComponent(strategy)}">
      <span class="t">${l.label}</span>
      <span class="n">${l.verses.toLocaleString()} verses</span>
      <span class="s">${l.books} books · ${l.keys.length} strategies</span>
      ${count}${why}${stack}
      ${has ? '' : `<span class="w">no ${strategy} data here — opens on ${l.keys[0]}</span>`}
    </a>`;
  }).join('');
}
strategyBar(render); render();
</script></div></body></html>""", encoding="utf-8")


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
    from .ai.compare import build, load_set, parse_set
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
    c.add_argument("--with-ai", action="store_true",
                   help="also use the archived AI style profiles (stylometry/ai/). Off by default: "
                        "the analysis is lexical, so it needs no API key, no network and no spend, "
                        "and reproduces exactly from the corpus alone.")
    c.add_argument("--no-ai", action="store_true",
                   help=argparse.SUPPRESS)  # retained so older scripts keep working; lexical is now the default
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--k-criterion", default="silhouette", choices=["silhouette", "bic", "davies_bouldin", "calinski"],
                   help="rank supported style partitions; single-group and stability checks still apply")
    c.set_defaults(func=cmd_cluster)

    cp = sub.add_parser('cluster-passages', help='pool raw tokens before exploratory lexical style analysis')
    _add_scope(cp)
    cp.add_argument('--tokens', type=int, choices=[500, 1000, 2000], default=1000)
    cp.add_argument('--bridge-chapters', action='store_true',
                    help='treat a chapter division as continuous text where the corpus records the two '
                         'verses as adjacent. Chapter and verse numbers are medieval and early-modern '
                         'editorial additions, not manuscript features; breaking on them costs most of '
                         'a complete codex. Leave off for fragmentary sources.')
    cp.add_argument('--corpus', help='source verse JSONL (defaults to the built project corpus)')
    _add_profiles(cp)
    cp.add_argument('--with-ai', action='store_true',
                    help='pool the existing verse AI profiles into each passage and cluster on them '
                         'too. Off by default: passage mode is the AI-free control, and leaving it '
                         'off keeps that result independent of any model.')
    cp.add_argument('--w-lex', type=float, default=1.0)
    cp.add_argument('--w-ai', type=float, default=1.0)
    cp.add_argument('--w-tags', type=float, default=0.7)
    cp.add_argument('--out', default=None)
    cp.add_argument('--k', type=int)
    cp.add_argument('--kmax', type=int, default=20)
    cp.add_argument('--seed', type=int, default=42)
    cp.add_argument('--k-criterion', choices=['silhouette', 'bic', 'davies_bouldin', 'calinski'], default='silhouette')
    cp.set_defaults(func=cmd_cluster_passages)

    dl = sub.add_parser('delta', help="Burrows's Delta baseline: most-frequent-word attribution, no AI")
    _add_scope(dl)
    dl.add_argument('--tokens', type=int, choices=[500, 1000, 2000], default=1000)
    dl.add_argument('--mfw', type=int, default=500, help='how many most frequent words to measure on')
    dl.add_argument('--metric', choices=['cosine', 'classic'], default='cosine',
                    help="cosine (Evert et al. 2017) measures better than Burrows's original on most corpora")
    dl.add_argument('--label', default='group',
                    help='which corpus field to score against, e.g. group, work, collection')
    dl.add_argument('--bridge-chapters', action='store_true',
                    help='treat chapter divisions as continuous text (see cluster-passages)')
    dl.add_argument('--witnesses', action='store_true',
                    help="analyse every manuscript's text rather than one primary witness per work. "
                         "The default corpus holds one witness per work, so John is analysed only as "
                         "Sinaiticus reads it; this reads witnesses.jsonl, where each manuscript is "
                         "its own unit.")
    dl.set_defaults(func=cmd_delta)

    wn = sub.add_parser('wan', help='word adjacency networks: how an author arranges function words')
    _add_scope(wn)
    wn.add_argument('--tokens', type=int, choices=[500, 1000, 2000], default=1000)
    wn.add_argument('--markers', type=int, default=100,
                    help='how many function words form the graph. The matrix is this squared, so more '
                         'markers need far more text; 200 markers on 1,000-token passages is mostly '
                         'smoothing and scores near chance.')
    wn.add_argument('--window', type=int, default=10, help='how far ahead a marker casts weight')
    wn.add_argument('--decay', choices=['inverse', 'uniform'], default='inverse')
    wn.add_argument('--label', default='group', help='which corpus field to score against')
    wn.add_argument('--bridge-chapters', action='store_true')
    wn.add_argument('--witnesses', action='store_true',
                    help="analyse every manuscript's text rather than one primary witness per work. "
                         "The default corpus holds one witness per work, so John is analysed only as "
                         "Sinaiticus reads it; this reads witnesses.jsonl, where each manuscript is "
                         "its own unit.")
    wn.add_argument('--whole-works', action='store_true',
                    help='one document per work rather than fixed-length passages. The method needs '
                         'the text: it measures far better on whole works here.')
    wn.set_defaults(func=cmd_wan)

    an = sub.add_parser('analyse', aliases=['analyze'],
                        help='full battery: every feature family and method the corpus supports')
    _add_scope(an)
    an.add_argument('--tokens', type=int, choices=[500, 1000, 2000], default=1000)
    an.add_argument('--label', default='group', help='which corpus field to score against')
    an.add_argument('--bridge-chapters', action='store_true')
    an.add_argument('--witnesses', action='store_true',
                    help="analyse every manuscript's text rather than one primary witness per work. "
                         "The default corpus holds one witness per work, so John is analysed only as "
                         "Sinaiticus reads it; this reads witnesses.jsonl, where each manuscript is "
                         "its own unit.")
    an.add_argument('--permutations', type=int, default=500,
                    help='permutations for the change-point test')
    an.add_argument('--seed', type=int, default=0)
    an.add_argument('--out', default=None)
    an.set_defaults(func=cmd_analyse)

    st = sub.add_parser('strategies', help='run each strategy alone, together and all-but-one; '
                                           'write an explorer you can steer')
    _add_scope(st)
    st.add_argument('--tokens', type=int, choices=[500, 1000, 2000], default=1000)
    st.add_argument('--label', default='group', help='which corpus field to score against')
    st.add_argument('--bridge-chapters', action='store_true')
    st.add_argument('--witnesses', action='store_true',
                    help="analyse every manuscript's text rather than one primary witness per work")
    st.add_argument('--title', default=None)
    st.add_argument('--seed', type=int, default=0)
    st.add_argument('--out', default=None)
    st.set_defaults(func=cmd_strategies)

    ex = sub.add_parser('explore', help='drill from a language through collections, books, chapters '
                                        'and verses, under any strategy')
    ex.add_argument('--scope', default='all',
                    choices=["all", "sinaiticus", "lxx", "nt", "christian", "noncanonical", "greek",
                             "hebrew", "arabic", "tanakh", "quran"])
    ex.add_argument('--language', choices=LANGS, help='one language; default builds all three')
    ex.add_argument('--works', nargs='*')
    ex.add_argument('--witnesses', action='store_true',
                    help="explore every manuscript's text rather than one witness per work")
    ex.add_argument('--seed', type=int, default=0)
    ex.add_argument('--out', default=None)
    ex.add_argument('--workers', type=int, default=None,
                    help='processes to group strategies across (default: one per strategy, capped at the core count)')
    ex.set_defaults(func=cmd_explore)

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
