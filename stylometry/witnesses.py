"""Witness comparison: how each manuscript reads against the primary witness of the same verses.

Works on ``data/processed/witnesses.jsonl`` (every witness, primaries and duplicates).  For every verse
attested in a secondary witness the bare tokens are aligned against the primary with a sequence
matcher; the similarity ratio (1.0 = identical) and a token-level diff are recorded.  Differences are
whatever the bare form keeps: wording, word order, presence/absence and consonantal spelling (so Qumran
plene spellings and itacisms that survive the normalisation show up as differences by design).
"""
from __future__ import annotations

import csv
import difflib
import html as h
import json
from collections import Counter, defaultdict
from pathlib import Path

from .corpus.meta import witness_info


def _base(work: str) -> str:
    return work.split("@", 1)[0]


def _slug(code: str) -> str:
    """File-system-safe form of a witness siglum (scroll names such as 5/6Hev1a contain slashes)."""
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", code)


def compare(verses: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (per-witness summary rows, per-verse rows) for one language."""
    by_key: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for v in verses:
        by_key[(_base(v["work"]), v["chapter"], v["verse"])][v["witness"]] = v
    primary_of: dict[str, str] = {}
    for v in verses:
        if not v.get("duplicate_of"):
            primary_of[v["work"]] = v["witness"]

    verse_rows: list[dict] = []
    stats: dict[str, dict] = defaultdict(lambda: {"n": 0, "works": Counter(), "sim": 0.0, "identical": 0, "tokens": 0})
    for (work, chapter, verse), wits in by_key.items():
        prim_wit = primary_of.get(work)
        prim = wits.get(prim_wit) if prim_wit else None
        for wit, rec in wits.items():
            st = stats[wit]
            st["n"] += 1
            st["works"][work] += 1
            st["tokens"] += rec["n_tokens"]
            if prim is None or wit == prim_wit:
                continue
            a, b = prim["text_bare"].split(), rec["text_bare"].split()
            sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
            ratio = sm.ratio()
            st["sim"] += ratio
            st["identical"] += int(ratio == 1.0)
            diff = []
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op == "equal":
                    diff.append(" ".join(a[i1:i2]))
                else:
                    if i2 > i1:
                        diff.append("[-" + " ".join(a[i1:i2]) + "-]")
                    if j2 > j1:
                        diff.append("{+" + " ".join(b[j1:j2]) + "+}")
            verse_rows.append(
                {
                    "work": work, "chapter": chapter, "verse": verse, "ref": rec["ref"], "witness": wit,
                    "primary_witness": prim_wit, "similarity": round(ratio, 3), "primary_text": prim["text"],
                    "witness_text": rec["text"], "diff": " ".join(diff), "order": rec["order"],
                }
            )
    summary: list[dict] = []
    for wit, st in stats.items():
        name, date, year = witness_info(wit)
        compared = st["n"] - sum(1 for w in primary_of.values() if w == wit) if wit in primary_of.values() else st["n"]
        n_cmp = sum(1 for r in verse_rows if r["witness"] == wit)
        summary.append(
            {
                "witness": wit, "name": name, "date": date, "year": year if year is not None else "",
                "verses": st["n"], "works": len(st["works"]), "tokens": st["tokens"],
                "is_primary_for": ", ".join(sorted(w for w, p in primary_of.items() if p == wit)),
                "compared_verses": n_cmp,
                "mean_similarity": round(st["sim"] / n_cmp, 3) if n_cmp else "",
                "pct_identical": round(100 * st["identical"] / n_cmp, 1) if n_cmp else "",
                "top_works": ", ".join(f"{w} {n}" for w, n in st["works"].most_common(6)),
            }
        )
    summary.sort(key=lambda r: (r["year"] == "", r["year"] if r["year"] != "" else 0, -r["verses"]))
    verse_rows.sort(key=lambda r: (r["witness"], r["order"]))
    return summary, verse_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _diff_html(diff: str) -> str:
    out = h.escape(diff)
    out = out.replace("[-", '<del style="color:#c0392b;background:rgba(192,57,43,.12)">').replace("-]", "</del>")
    out = out.replace("{+", '<ins style="color:#1e8449;background:rgba(30,132,73,.12);text-decoration:none">').replace("+}", "</ins>")
    return out


def render_html(out_dir: Path, summary: list[dict], verse_rows: list[dict], language: str) -> None:
    from .html import page

    site = out_dir / "site"
    (site / "witnesses").mkdir(parents=True, exist_ok=True)
    rtl = language in ("hbo", "arb")
    gk = "gk rtl" if rtl else "gk"
    rows = []
    for s in summary:
        link = f'<a href="witnesses/{_slug(s["witness"])}.html">{h.escape(s["witness"])}</a>' if s["compared_verses"] else h.escape(s["witness"])
        rows.append(
            f"<tr><td>{link}</td><td>{h.escape(s['name'])}</td><td>{h.escape(s['date'])}</td>"
            f'<td class="num">{s["verses"]:,}</td><td class="num">{s["works"]}</td><td>{h.escape(s["is_primary_for"][:60])}</td>'
            f'<td class="num">{s["compared_verses"]:,}</td><td class="num">{s["mean_similarity"]}</td><td class="num">{s["pct_identical"]}</td>'
            f"<td>{h.escape(s['top_works'])}</td></tr>"
        )
    body = (
        '<p class="sub">Every witness in this language, oldest first. Similarity is the token-level alignment ratio between the witness and the '
        "primary witness of the same verse (1 = identical after normalisation); manuscripts that are primary for a work are not compared for it.</p>"
        '<table><thead><tr><th>siglum</th><th>witness</th><th>date</th><th class="num">verses</th><th class="num">works</th><th>primary for</th>'
        '<th class="num">compared</th><th class="num">mean sim.</th><th class="num">% identical</th><th>largest contents</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )
    (site / "witnesses.html").write_text(page("Witnesses", body, "", "Manuscripts and editions of the same verses, side by side"), encoding="utf-8")

    by_wit: dict[str, list[dict]] = defaultdict(list)
    for r in verse_rows:
        by_wit[r["witness"]].append(r)
    for wit, rs in by_wit.items():
        name, date, _ = witness_info(wit)
        n_diff = sum(1 for r in rs if r["similarity"] < 1.0)
        parts = [f'<div class="tiles"><div class="tile"><div class="v">{len(rs):,}</div><div class="l">verses compared</div></div>'
                 f'<div class="tile"><div class="v">{n_diff:,}</div><div class="l">verses with a difference</div></div>'
                 f'<div class="tile"><div class="v">{sum(r["similarity"] for r in rs) / len(rs):.3f}</div><div class="l">mean similarity</div></div></div>',
                 '<p class="sub"><del style="color:#c0392b">red</del> = in the primary witness but not here; <ins style="color:#1e8449;text-decoration:none">green</ins> = here but not in the primary. '
                 'Identical verses are listed without a diff line.</p>']
        for r in rs:
            same = r["similarity"] >= 1.0
            parts.append(
                f'<div class="verse" style="border-left-color:{"var(--border)" if same else "var(--a2)"}"><span class="ref">{h.escape(r["ref"])}<br>'
                f'<span class="conf">{r["primary_witness"]} → {h.escape(wit)} · {r["similarity"]:.2f}</span></span><span></span>'
                f'<span class="{gk}" dir="auto">{h.escape(r["witness_text"])}'
                + ("" if same else f'<br><span class="conf" dir="auto" style="font-size:13px">{_diff_html(r["diff"])}</span>')
                + "</span></div>"
            )
        (site / "witnesses" / f"{_slug(wit)}.html").write_text(page(f"{wit} · {name}", "\n".join(parts), "../", h.escape(date)), encoding="utf-8")


def run(all_verses: list[dict], language: str, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    verses = [v for v in all_verses if v["language"] == language]
    summary, verse_rows = compare(verses)
    _write_csv(out_dir / "witness_summary.csv", summary)
    _write_csv(out_dir / "witness_verses.csv", verse_rows)
    render_html(out_dir, summary, verse_rows, language)
    return {"witnesses": len(summary), "compared_verses": len(verse_rows),
            "with_differences": sum(1 for r in verse_rows if r["similarity"] < 1.0)}
