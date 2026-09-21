"""Build the unified verse corpus (data/processed/verses.jsonl) from the raw downloads.

Witness resolution
------------------
Several sources can carry the same work in the same language (Isaiah in the Leningrad Codex, the Aleppo
Codex and 1QIsaa; the Gospels in Sinaiticus and Vaticanus).  For authorship analysis one witness per work
is enough, so the fullest witness (most tokens) becomes the *primary* and keeps the plain work code; every
other witness is renamed ``WORK@WITNESS`` and marked ``duplicate_of=WORK``.  Duplicates are dropped unless
``include_duplicates`` is set, in which case they behave as separate works (useful for comparing witnesses).

Verse ids are ``<language>:<work>.<chapter>.<verse>``.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from . import (apostolic, bukhari, cntr, dss, english, first1k, inscriptions, mam, oshb, qudsi,
               quran, samaritan, sinaiticus, vaticanus)

# (subdirectory under data/raw, loader)
LOADERS = [
    ("codex-sinaiticus", lambda p: sinaiticus.load(sorted(p.glob("sinaiticus_full_v*.xml"))[-1]) if list(p.glob("sinaiticus_full_v*.xml")) else []),
    ("cntr", cntr.load),
    ("vaticanus", vaticanus.load),
    ("first1kgreek", first1k.load),
    ("apostolic-fathers/texts", apostolic.load),
    ("oshb", oshb.load),
    ("mam", mam.load),
    ("samaritan", samaritan.load),
    ("dss", dss.load),
    ("inscriptions", inscriptions.load),
    ("quran", quran.load),
    ("bukhari", bukhari.load),
    ("qudsi", qudsi.load),
    ("english", english.load),
]


# Preferred witness order when several carry a work at comparable length: the manuscripts the project
# is about first, then the editions.  A witness only wins on priority if it has at least COVERAGE of
# the fullest witness's tokens, so a fragmentary manuscript never displaces a complete edition.
WITNESS_PRIORITY = ["S", "B", "L", "A", "SP", "Swete", "Lake", "Bonnet", "T", "H", "Q", "G"]
COVERAGE = 0.9


def _rank(witness: str) -> int:
    return WITNESS_PRIORITY.index(witness) if witness in WITNESS_PRIORITY else len(WITNESS_PRIORITY)


def resolve_witnesses(verses: list[dict]) -> list[dict]:
    tokens: dict[tuple[str, str, str], int] = Counter()
    for v in verses:
        tokens[(v["language"], v["work"], v["witness"])] += v["n_tokens"]
    by_work: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for (lang, work, wit), n in tokens.items():
        by_work[(lang, work)].append((wit, n))
    primary: dict[tuple[str, str], str] = {}
    for key, cands in by_work.items():
        fullest = max(n for _, n in cands)
        eligible = [(w, n) for w, n in cands if n >= COVERAGE * fullest]
        primary[key] = min(eligible, key=lambda wn: (_rank(wn[0]), -wn[1]))[0]
    for v in verses:
        v.setdefault("duplicate_of", None)
        if primary[(v["language"], v["work"])] != v["witness"]:
            v["duplicate_of"] = v["work"]
            v["work"] = f"{v['work']}@{v['witness']}"
            v["work_title"] = f"{v['work_title']} [{v['witness']}]"
        v["id"] = f"{v['language']}:{v['work']}.{v['chapter']}.{v['verse']}"
    return verses


def _disambiguate_repeated_references(verses: list[dict]) -> list[dict]:
    """Make ids unique when one manuscript attests the same reference twice.

    This is not a parsing fault. Codex Sinaiticus carries a double text of 1 Chronicles 17-18, where a
    stretch of Chronicles was copied a second time, and Vaticanus gives two readings at Romans 4:4-5;
    the two copies differ in wording (απεναντι against απεναντιον). Both belong in a witness corpus,
    so neither is dropped. Later occurrences take a ``#2`` suffix, which keeps ids unique without
    hiding that the manuscript really does read the verse twice.
    """
    seen: dict[str, int] = {}
    repeated = []
    for verse in verses:
        ident = verse.get("id")
        if ident is None:
            continue
        count = seen.get(ident, 0) + 1
        seen[ident] = count
        if count > 1:
            repeated.append({"id": ident, "witness": verse.get("witness"), "ref": verse.get("ref"),
                             "occurrence": count})
            verse["id"] = f"{ident}#{count}"
            verse["repeated_reference"] = count
    if repeated:
        print(f"  {len(repeated)} references attested more than once in their own manuscript "
              f"(kept, ids suffixed): "
              + ", ".join(sorted({f"{r['witness']} {str(r['ref']).rsplit(':', 1)[0]}" for r in repeated})))
    return verses


def build(raw_dir: str | Path, out_path: str | Path, include_duplicates: bool = False) -> list[dict]:
    raw_dir = Path(raw_dir)
    verses: list[dict] = []
    # One broken source must not kill the whole build, but it must not vanish either: every source's
    # outcome is recorded and written to build_report.json beside the corpus.
    sources: list[dict] = []
    for subdir, loader in LOADERS:
        path = raw_dir / subdir
        if not path.exists():
            print(f"  (no {subdir}; skipped)")
            sources.append({"source": subdir, "n_units": 0, "status": "absent", "error": None})
            continue
        print(f"parsing {subdir} ...", flush=True)
        try:
            got = loader(path)
        except Exception as e:
            print(f"  ERROR in {subdir}: {e}")
            sources.append({"source": subdir, "n_units": 0, "status": "failed", "error": f"{type(e).__name__}: {e}"})
            continue
        print(f"  {len(got)} units")
        sources.append({"source": subdir, "n_units": len(got),
                        "status": "ok" if got else "empty", "error": None})
        verses += got

    for v in verses:
        v.setdefault("language", "grc")
        v.setdefault("witness", "?")
        v.setdefault("duplicate_of", None)
    verses = resolve_witnesses(verses)
    _disambiguate_repeated_references(verses)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failed = [s for s in sources if s["status"] == "failed"]
    empty = [s for s in sources if s["status"] == "empty"]
    report = {"sources": sources, "n_units": len(verses), "n_failed": len(failed), "n_empty": len(empty)}
    (out_path.parent / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    # Every witness, for textual comparison
    with out_path.with_name("witnesses.jsonl").open("w", encoding="utf-8") as fh:
        for v in verses:
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")
    if not include_duplicates:
        verses = [v for v in verses if not v["duplicate_of"]]
    with out_path.open("w", encoding="utf-8") as fh:
        for v in verses:
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")

    by_lang: dict[str, Counter] = defaultdict(Counter)
    tok: dict[str, Counter] = defaultdict(Counter)
    for v in verses:
        by_lang[v["language"]][v["work"]] += 1
        tok[v["language"]][v["work"]] += v["n_tokens"]
    print(f"wrote {len(verses)} verse units to {out_path}")
    for s in failed:
        print(f"  !! {s['source']} FAILED and contributed nothing: {s['error']}")
    for s in empty:
        print(f"  !! {s['source']} parsed without error but produced no units")
    if failed or empty:
        print(f"  {len(failed)} failed, {len(empty)} empty; see {out_path.parent / 'build_report.json'}")
    for lang, counts in by_lang.items():
        print(f"[{lang}] {sum(counts.values())} units, {sum(tok[lang].values())} tokens, {len(counts)} works")
        for work, n in counts.items():
            print(f"  {work:12s} {n:6d} units {tok[lang][work]:8d} tokens")
    return verses


def load_corpus(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
