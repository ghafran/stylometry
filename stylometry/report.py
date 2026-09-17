"""Render output/report.md from the clustering outputs."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .cluster import author_key


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _fmt_marker(name: str, eff: float) -> str:
    return f"`{name}` {eff:+.2f}σ"


def render(out_dir: str | Path) -> str:
    out_dir = Path(out_dir)
    summary = json.loads((out_dir / "summary.json").read_text())
    authors = json.loads((out_dir / "authors.json").read_text())
    ktable = _read_csv(out_dir / "k_selection.csv")
    crosstab = _read_csv(out_dir / "work_by_author.csv")
    segs = _read_csv(out_dir / "segments.csv")
    outliers = _read_csv(out_dir / "outliers.csv")
    author_ids = sorted(authors, key=author_key)
    val = summary["validation"]

    md: list[str] = []
    md.append("# Exploratory verse-level style analysis\n")
    md.append(
        f"{summary['n_verses']} verse units from {summary['n_works']} works were profiled with "
        f"{'lexical statistics plus AI style profiles' if summary['used_ai_profiles'] else 'lexical statistics only (no AI profiles found)'}, "
        f"smoothed over a ±{summary['window']}-verse window (α={summary['alpha']}), reduced to {summary['pca_dims']} principal components "
        f"({summary['pca_explained_variance']:.0%} of variance) and clustered with k-means.\n"
    )
    selected = summary.get("k_selected", summary.get("k_selected_by_silhouette"))
    md.append(f"**Result: {summary['k_used']} exploratory style groups (A1–A{summary['k_used']}).** "
              f"Selection criterion: {summary.get('k_criterion', 'silhouette')}; preferred k={selected}. "
              + ("The number of groups was forced. " if summary.get("k_forced") else "")
              + "These are not identified authors.")
    status = summary.get("selection_status", "legacy_unvalidated")
    md.append(f"Selection status: **{status.replace('_', ' ')}**. One group means no supported split, not one proven author.")
    stability = summary.get("stability", {})
    if stability.get("available"):
        md.append(f"Partition tested at k={stability.get('tested_k', summary['k_used'])}: passage subsampling mean ARI {stability['mean_ari']:.3f}, minimum {stability['min_ari']:.3f} "
                  f"over {stability['repeats']} repeats. The feature map is fixed; this is partition sensitivity, not held-out accuracy.")
    if summary.get("profile_metadata", {}).get("unverified_profiles"):
        md.append("**Legacy/unverified AI profiles:** blinding and input provenance are not verified; regenerate profiles before using these results for validation.")
    if summary.get("n_missing_profiles"):
        md.append(f"{summary['n_missing_profiles']} input verses had no profile and were excluded; missing passages break continuity.")
    md.append("Assignment margins measure distance to cluster centers, not probabilities of authorship; single-group margins are zero.")
    hdb = summary.get("hdbscan", {})
    if hdb.get("available"):
        md.append(f"Density clustering (HDBSCAN, min cluster {hdb['min_cluster_size']}) found {hdb['n_clusters']} dense groups with {hdb['noise_fraction']:.0%} of verses unassigned, as a second opinion.\n")

    md.append("\n## How many style groups? (k selection)\n")
    if "selection_thresholds" in summary:
        md.append("For each k the partition is fitted on smoothed vectors and scored on unsmoothed vectors. "
                  "Higher silhouette / Calinski-Harabasz and lower Davies-Bouldin / BIC are better. "
                  "Automatic splits must improve BIC over a single Gaussian by at least 10 on a fixed sample of unsmoothed vectors, "
                  "have positive raw silhouette, and retain ARI of at least 0.8 in all five passage subsamples. "
                  "These are diagnostic thresholds, not a significance test or proof of authorship.\n")
    else:
        md.append("Legacy output: single-group BIC and passage-resampling checks were not recorded. "
                  "Regenerate this analysis with the current pipeline before interpreting cluster support.\n")
    md.append(_table(["k", "silhouette", "Calinski-Harabasz", "Davies-Bouldin", "GMM BIC"],
                     [[r["k"], r["silhouette"], r["calinski_harabasz"], r["davies_bouldin"], r["gmm_bic"]] for r in ktable]))

    md.append("\n\n## Style groups at a glance\n")
    rows = []
    for a in author_ids:
        e = authors[a]
        top_works = ", ".join(f"{w} ({n})" for w, n in list(e["works"].items())[:5])
        ai = e.get("ai_profile_means")
        ai_s = (f"reg {ai['register']:.2f} · sem {ai['semitic_interference']:.2f} · hyp {ai['hypotaxis']:.2f} · "
                f"lex {ai['lexical_richness']:.2f} · rhet {ai['rhetorical_polish']:.2f}") if ai else "—"
        rows.append([a, e["n_verses"], f"{e['share']:.1%}", f"{e['mean_confidence']:.2f}", top_works, ai_s])
    md.append(_table(["author", "verses", "share", "mean margin", "main works", "AI style means"], rows))

    md.append("\n\n## Style-group markers\n")
    md.append("Markers are the features whose mean inside the cluster differs most from the corpus mean (in standard deviations). "
              "`fw:` closed-class word rate, `sfx:` word-ending rate, `misc:` length/connective habits, `cng:` character n-gram axis, "
              "`ai:` model-rated style scale, `cat:` model-assigned category, `tag:` model-assigned device tag.\n")
    for a in author_ids:
        e = authors[a]
        md.append(f"\n### {a} — {e['n_verses']} verses ({e['share']:.1%})\n")
        md.append("**Works:** " + ", ".join(f"{w} {n}" for w, n in e["works"].items()) + "\n")
        if e.get("copyists"):
            md.append("**Sinaiticus scribes:** " + ", ".join(f"{s} {n}" for s, n in sorted(e["copyists"].items())) + "\n")
        md.append("**Over-represented:** " + ", ".join(_fmt_marker(n, x) for n, x in e["markers_high"][:10]) + "\n")
        md.append("**Under-represented:** " + ", ".join(_fmt_marker(n, x) for n, x in e["markers_low"][:8]) + "\n")
        if e.get("tag_lift"):
            md.append("**Device tags (lift ×, count):** " + ", ".join(f"{t} ×{l} ({c})" for t, l, c in e["tag_lift"][:10]) + "\n")
        if e.get("phrases"):
            md.append("**Diagnostic phrases:** " + ", ".join(f"«{p}» ({c})" for p, c in e["phrases"][:8]) + "\n")
        if e.get("discourse_modes"):
            md.append("**Discourse modes:** " + ", ".join(f"{m} {n}" for m, n in e["discourse_modes"].items()) + "\n")
        md.append("**Closest to centroid:**\n")
        for r in e["representative"][:3]:
            md.append(f"- {r['ref']}: {r['text']}")
        md.append("")

    md.append("\n## Works × style groups\n")
    md.append("Each row is a work; columns count how many of its verses each author received. "
              "`purity` is the share held by the work's majority author.\n")
    headers = ["work", "group", "n"] + author_ids + ["majority", "purity"]
    md.append(_table(headers, [[r["work"], r["group"], r["n"]] + [r[a] for a in author_ids] + [r["majority"], r["purity"]] for r in crosstab]))

    md.append("\n\n## Agreement with traditional attributions\n")
    md.append(f"- Adjusted Rand index of authors vs. work: **{val['ari_vs_work']}**; vs. traditional author group: **{val['ari_vs_group']}** "
              "(1 = identical partition, 0 = chance).")
    md.append(f"- Mean purity across works: **{val['mean_purity']}**.")
    group_ct: dict[str, Counter] = {}
    for r in crosstab:
        c = group_ct.setdefault(r["group"], Counter())
        for a in author_ids:
            c[a] += int(r[a])
    rows = []
    for g, c in sorted(group_ct.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(c.values())
        rows.append([g, tot] + [f"{c[a] / tot:.0%}" if c[a] else "" for a in author_ids])
    md.append("\n" + _table(["traditional group", "n"] + author_ids, rows))
    if val.get("author_by_scribe"):
        md.append("\n\nSinaiticus was copied by three scribes (A, B, D). If the discovered authors tracked the *scribes* rather than the *composers*, "
                  "the table below would be block-diagonal; a mixed table alone cannot rule out scribal or edition effects.\n")
        scribes = sorted({s for c in val["author_by_scribe"].values() for s in c})
        md.append(_table(["author"] + scribes, [[a] + [val["author_by_scribe"].get(a, {}).get(s, 0) for s in scribes] for a in author_ids]))

    md.append("\n\n## Passages that break from their work's main style group\n")
    md.append("Runs of at least three consecutive verses assigned to an author other than the work's majority author. "
              "These are the candidates for interpolation, embedded sources, or a change of style worth reading closely.\n")
    minority = [s for s in segs if s["is_majority"] == "False" and int(s["n"]) >= 3]
    minority.sort(key=lambda s: -int(s["n"]))
    md.append(_table(["work", "from", "to", "verses", "author"], [[s["work"], s["start"], s["end"], s["n"], s["author"]] for s in minority[:60]]))
    if len(minority) > 60:
        md.append(f"\n… {len(minority) - 60} more in `segments.csv`.")

    md.append("\n\n## Individual outlier verses\n")
    md.append("Verses whose own (unsmoothed) style is more than 2.5 SD from their assigned author's centre.\n")
    md.append(_table(["ref", "author", "z", "text"], [[o["ref"], o["author"], o["z"], o["text"][:90]] for o in outliers[:40]]))

    md.append("\n\n## Caveats\n")
    md.append("- Unsupervised clusters are exploratory style groups, not proven persons. Genre (narrative vs. letter vs. vision) is the strongest stylistic signal in any corpus and will shape the top-level split; read the works×authors table with that in mind.")
    md.append("- The number of authors depends on the selection criterion; the k-selection table shows how sharp (or flat) the optimum is.")
    md.append("- Single verses are short; per-verse labels inherit their neighbourhood through smoothing. Treat minority runs as hypotheses for further study; smoothing itself induces runs, and quotations or genre shifts can explain them.")
    lang = summary.get("language", "grc")
    if lang == "grc":
        md.append("- Manuscript texts (Sinaiticus, Vaticanus) are the first hand of one copy; LXX books are translations, so their 'authors' are translators.")
    elif lang == "hbo":
        md.append("- The Masoretic text is the primary witness; Qumran and Samaritan copies are witnesses of the same works unless the corpus was built with duplicates. Non-biblical scrolls are fragmentary and partly reconstructed; units that are mostly reconstruction were dropped.")
    elif lang == "arb":
        md.append("- Each sura is treated as a work; the Meccan/Medinan column is the traditional classification, not a stylistic result.")
    text = "\n".join(md) + "\n"
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    return text
