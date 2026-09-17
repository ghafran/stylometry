"""Static HTML site rendered from the clustering outputs.

    output/site/index.html          dashboard: every style group, corpus share, and work counts
    output/site/authors/A1.html     verses assigned to one style group, in reading order
    output/site/works/MARK.html     one work, verse by verse, with each assigned style group

No external assets: one inline stylesheet, a few lines of JavaScript for tooltips.
"""
from __future__ import annotations

import csv
import html as h
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from .cluster import author_key

# Categorical palette (validated, fixed order).  Authors beyond the eighth get the neutral slot and are
# always identified by their text label as well, so identity never rides on colour alone.
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

CSS = """
:root { color-scheme: light; --surface:#fcfcfb; --surface-2:#f0efec; --text:#0b0b0b; --text-2:#52514e;
  --muted:#8a8985; --border:#e3e2de; --link:#1c5cab; --a-other:#8a8985; %(light)s }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --surface:#1a1a19;
  --surface-2:#262624; --text:#ffffff; --text-2:#c3c2b7; --muted:#8f8e88; --border:#383835; --link:#86b6ef; %(dark)s } }
:root[data-theme="dark"] { color-scheme: dark; --surface:#1a1a19; --surface-2:#262624; --text:#ffffff; --text-2:#c3c2b7;
  --muted:#8f8e88; --border:#383835; --link:#86b6ef; %(dark)s }
* { box-sizing: border-box; }
body { margin:0; background:var(--surface); color:var(--text); font:15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 16px; }
nav { display:flex; gap:16px; padding:12px 16px; border-bottom:1px solid var(--border); font-size:14px; flex-wrap:wrap; }
nav a { color:var(--link); text-decoration:none; }
a { color:var(--link); }
h1 { font-size:24px; margin:12px 0 4px; } h2 { font-size:18px; margin:28px 0 8px; } h3 { font-size:15px; margin:20px 0 6px; }
.sub { color:var(--text-2); margin:0 0 12px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:12px 0 20px; }
.tile { background:var(--surface-2); border-radius:8px; padding:10px 12px; }
.tile .v { font-size:26px; font-weight:600; line-height:1.1; } .tile .l { color:var(--text-2); font-size:13px; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--border); vertical-align:top; }
th { color:var(--text-2); font-weight:600; font-size:13px; }
td.num, th.num { text-align:right; font-variant-numeric: tabular-nums; }
.chip { display:inline-block; padding:0 6px 0 4px; border-radius:4px; background:var(--surface-2); color:var(--text); font-size:12px;
  font-weight:600; line-height:1.7; text-decoration:none; white-space:nowrap; }
.chip i { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
.sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }
.bar { position:relative; height:14px; background:var(--surface-2); border-radius:4px; overflow:hidden; min-width:120px; }
.bar > span { position:absolute; left:0; top:0; bottom:0; border-radius:0 4px 4px 0; }
.stack { display:flex; height:16px; gap:2px; border-radius:4px; overflow:hidden; }
.stack > span { display:block; height:100%; min-width:1px; }
.legend { display:flex; flex-wrap:wrap; gap:6px 16px; font-size:13px; color:var(--text-2); margin:8px 0 12px; }
.verse { display:grid; grid-template-columns:110px 56px 1fr; gap:8px; padding:5px 0 5px 10px; border-left:4px solid transparent; }
.verse .ref { color:var(--text-2); font-size:12px; white-space:nowrap; } .verse .ref a { color:inherit; text-decoration:none; }
.verse.passage .ref { white-space:normal; overflow-wrap:anywhere; }
.verse .gk { font-family:"SBL Greek","Gentium Plus","Palatino Linotype",Palatino,Georgia,serif; font-size:16px; }
.verse .gk.rtl { font-family:"SBL Hebrew","Ezra SIL","Noto Serif Hebrew","Amiri","Scheherazade New","Noto Naskh Arabic",serif; font-size:18px; direction:rtl; text-align:right; }
.verse.flag .gk::after { content:" ⚑"; color:var(--muted); }
.conf { color:var(--muted); font-size:11px; }
.work-head { margin-top:22px; padding-top:8px; border-top:1px solid var(--border); }
.markers { display:flex; flex-wrap:wrap; gap:6px; } .markers code { background:var(--surface-2); padding:1px 6px; border-radius:4px; font-size:12px; }
#tip { position:fixed; pointer-events:none; background:var(--text); color:var(--surface); padding:4px 8px; border-radius:4px;
  font-size:12px; display:none; z-index:9; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }
@media (max-width:640px) { .verse { grid-template-columns:1fr; gap:2px; } .verse .ref { order:0; } }
"""

JS = """
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{
  el.addEventListener('mousemove',e=>{tip.style.display='block';tip.textContent=el.dataset.tip;
    tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';});
  el.addEventListener('mouseleave',()=>{tip.style.display='none';});
});
"""


def _vars(palette: list[str]) -> str:
    return " ".join(f"--a{i + 1}:{c};" for i, c in enumerate(palette))


def color_var(author: str) -> str:
    i = author_key(author)
    return f"var(--a{i})" if i <= len(LIGHT) else "var(--a-other)"


def chip(author: str, rel: str = "") -> str:
    return f'<a class="chip" href="{rel}authors/{author}.html"><i style="background:{color_var(author)}"></i>{author}</a>'


def page(title: str, body: str, rel: str, subtitle: str = "", nav: str | None = None) -> str:
    css = CSS.replace("%(light)s", _vars(LIGHT)).replace("%(dark)s", _vars(DARK))
    if nav is None:
        nav = (f'<a href="{rel}index.html">Dashboard</a><a href="{rel}index.html#authors">Style groups</a><a href="{rel}index.html#works">Works</a>\n'
               f'<a href="{rel}witnesses.html">Witnesses</a><a href="{rel}../report.md">Report (markdown)</a><a href="{rel}../../index.html">All languages</a>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{h.escape(title)}</title><style>{css}</style></head>
<body><nav>{nav}</nav>
<main><h1>{h.escape(title)}</h1>{f'<p class="sub">{subtitle}</p>' if subtitle else ''}
{body}</main><div id="tip"></div><script>{JS}</script></body></html>"""


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def stack_bar(counts: Counter, total: int, authors: list[str], label: str, units: str = "verses") -> str:
    segs = []
    for a in authors:
        n = counts.get(a, 0)
        if n:
            segs.append(
                f'<span style="width:{100 * n / total:.2f}%;background:{color_var(a)}" '
                f'data-tip="{h.escape(label)} · {a}: {n} {units} ({_pct(n / total)})"></span>'
            )
    return f'<div class="stack">{"".join(segs)}</div>'


def legend(authors: list[str]) -> str:
    items = "".join(f'<span><i class="sw" style="background:{color_var(a)}"></i>{a}</span>' for a in authors)
    return f'<div class="legend">{items}</div>'


def build_site(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    site = out_dir / "site"
    (site / "authors").mkdir(parents=True, exist_ok=True)
    (site / "works").mkdir(parents=True, exist_ok=True)

    summary = json.loads((out_dir / "summary.json").read_text())
    authors_meta = json.loads((out_dir / "authors.json").read_text())
    rows = _read_csv(out_dir / "verse_assignments.csv")
    authors = sorted(authors_meta, key=author_key)
    total = len(rows)
    lang = summary.get("language", "grc")
    lang_name = {"grc": "Greek", "hbo": "Hebrew", "arb": "Arabic"}.get(lang, lang)
    gk = "gk rtl" if lang in ("hbo", "arb") else "gk"
    passage_mode = summary.get("input_unit") == "pooled_token_passage"
    unit, units = ("passage", "passages") if passage_mode else ("verse", "verses")
    unit_class = "verse passage" if passage_mode else "verse"

    def mapping_note(rel):
        if not passage_mode:
            return ""
        filename = quote(Path(summary.get('source_mapping_file', 'passages.json')).name)
        return (f'<p class="sub"><a href="{rel}{filename}">Source mappings and exclusions</a>: original verse IDs, '
                'normalized token offsets, and excluded material. Labels describe whole passages, '
                'not the authorship of individual source verses.</p>')

    by_work: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_work[r["work"]].append(r)
    work_order = list(by_work)
    work_counts = {w: Counter(r["author"] for r in rs) for w, rs in by_work.items()}
    work_group = {w: rs[0]["group"] for w, rs in by_work.items()}
    majority = {w: c.most_common(1)[0][0] for w, c in work_counts.items()}

    # ---- dashboard -----------------------------------------------------------------------------
    val = summary["validation"]
    single_work = len(work_order) == 1
    by_chapter: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_chapter[r["chapter"]].append(r)
    chapter_order = sorted(by_chapter, key=lambda c: int(c) if c.isdigit() else 0)
    # A passage carries only its first source verse's chapter, so chapters are reported for verse runs only.
    chapter_breakdown = single_work and not passage_mode
    unit_tile = f"{summary['passage_tokens']}-token passages" if passage_mode else "verse units"
    if single_work:  # one work: ARI and purity against work boundaries are degenerate
        biggest = max((authors_meta[a] for a in authors), key=lambda m: m["n_verses"])
        tiles = [
            (f"{total:,}", unit_tile),
            (str(len(authors)), "style groups"),
            (f"{biggest['share']:.0%}", "largest group"),
        ]
        if chapter_breakdown:
            mixed = sum(1 for c in chapter_order if len({r["author"] for r in by_chapter[c]}) > 1)
            tiles.insert(1, (str(len(chapter_order)), "chapters"))
            tiles.append((str(mixed), "mixed chapters"))
    else:
        tiles = [
            (f"{total:,}", unit_tile),
            (str(len(work_order)), "works"),
            (str(len(authors)), "style groups"),
            (f"{val['ari_vs_group']:.2f}", "ARI vs. traditional groups"),
            (f"{val['mean_purity']:.0%}", "mean purity per work"),
        ]
    tiles_html = "".join(f'<div class="tile"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in tiles)

    author_rows = []
    for a in authors:
        m = authors_meta[a]
        touched = [w for w in work_order if work_counts[w].get(a, 0)]
        substantial = [w for w in touched if work_counts[w][a] / len(by_work[w]) >= 0.10]
        main_hand = [w for w in work_order if majority[w] == a]
        top = ", ".join(f'<a href="works/{w}.html">{w}</a> {n}' for w, n in list(m["works"].items())[:6])
        author_rows.append(
            f"<tr><td>{chip(a)}</td>"
            f'<td class="num">{m["n_verses"]:,}</td>'
            f'<td><div class="bar" data-tip="{a}: {_pct(m["share"])} of all {units}"><span style="width:{100 * m["share"]:.2f}%;background:{color_var(a)}"></span></div></td>'
            f'<td class="num">{_pct(m["share"])}</td>'
            f'<td class="num">{len(main_hand)}</td><td class="num">{len(substantial)}</td><td class="num">{len(touched)}</td>'
            f"<td>{top}</td></tr>"
        )
    authors_table = (
        f'<table><thead><tr><th>style group</th><th class="num">{units}</th><th style="width:22%">share</th><th class="num">%</th>'
        '<th class="num">main group of</th><th class="num">≥10% of</th><th class="num">appears in</th><th>largest contributions</th></tr></thead>'
        f'<tbody>{"".join(author_rows)}</tbody></table>'
        f'<p class="sub">"main group of" counts works where this style group contains the most {units}; "≥10% of" counts works where it contains at least a tenth; "appears in" counts any assigned {unit}.</p>'
    )

    work_rows = []
    for w in work_order:
        n = len(by_work[w])
        purity = work_counts[w][majority[w]] / n
        work_rows.append(
            f'<tr><td><a href="works/{w}.html">{h.escape(w)}</a><br><span class="sub" style="font-size:12px">{h.escape(work_group[w])}</span></td>'
            f'<td class="num">{n:,}</td><td style="width:40%">{stack_bar(work_counts[w], n, authors, w, units)}</td>'
            f'<td>{chip(majority[w])}</td><td class="num">{purity:.0%}</td></tr>'
        )
    works_table = (
        f'<table><thead><tr><th>work</th><th class="num">{units}</th><th>composition</th><th>main group</th><th class="num">purity</th></tr></thead>'
        f'<tbody>{"".join(work_rows)}</tbody></table>'
    )

    if chapter_breakdown:  # the informative breakdown of one work is by chapter
        w0 = work_order[0]
        chapter_rows = []
        for ch in chapter_order:
            rs = by_chapter[ch]
            cnt = Counter(r["author"] for r in rs)
            maj, majn = cnt.most_common(1)[0]
            chapter_rows.append(
                f'<tr><td><a href="works/{w0}.html#{h.escape(rs[0]["id"])}">{h.escape(rs[0]["ref"].rsplit(":", 1)[0])}</a></td>'
                f'<td class="num">{len(rs)}</td><td style="width:40%">{stack_bar(cnt, len(rs), authors, ch, units)}</td>'
                f'<td>{chip(maj)}</td><td class="num">{majn / len(rs):.0%}</td></tr>'
            )
        works_table = (
            f'<table><thead><tr><th>chapter</th><th class="num">{units}</th><th>composition</th><th>main group</th>'
            '<th class="num">purity</th></tr></thead>'
            f'<tbody>{"".join(chapter_rows)}</tbody></table>'
        )

    k_note = (
        f"{summary['k_used']} exploratory style groups were used"
        + (f" (chosen by {summary.get('k_criterion', 'silhouette')})" if not summary.get("k_forced", summary["k_used"] != summary["k_selected_by_silhouette"]) else " (forced with --k)")
        + f"; features: {'lexical statistics + AI style profiles' if summary['used_ai_profiles'] else 'lexical statistics only'}. "
        + "These are not identified authors. "
        + h.escape(summary.get("selection_status", "legacy_unvalidated").replace("_", " "))
        + ". One group means no supported split, not one proven author."
    )
    if summary.get("profile_metadata", {}).get("unverified_profiles"):
        k_note += " Legacy/unverified AI profiles: regenerate for validation."
    if passage_mode:
        k_note += (f" Raw tokens were pooled into non-overlapping {summary['passage_tokens']}-token passages before feature extraction. "
                   + ("No additional smoothing was applied." if not summary['alpha'] or not summary['window'] else
                      f"Additional smoothing used a ±{summary['window']}-passage window (α={summary['alpha']})."))
    body = (
        f'<div class="tiles">{tiles_html}</div><p class="sub">{k_note}</p>'
        f'{mapping_note("../")}'
        f'<h2 id="authors">Style groups</h2>{legend(authors)}{authors_table}'
        f'<h2 id="works">{"Chapters" if chapter_breakdown else "Works"}</h2>{legend(authors)}{works_table}'
    )
    title = f"Passage-level style groups · {lang_name}" if passage_mode else f"Style groups · {lang_name}"
    (site / "index.html").write_text(page(title, body, "", f"Exploratory style analysis in the {lang_name} corpus · <a href=\"../../index.html\">all languages</a>"), encoding="utf-8")

    # ---- per-author pages ------------------------------------------------------------------------
    for a in authors:
        m = authors_meta[a]
        mine = [r for r in rows if r["author"] == a]
        works_here = Counter(r["work"] for r in mine)
        parts = [f'<div class="tiles"><div class="tile"><div class="v">{m["n_verses"]:,}</div><div class="l">{units} ({_pct(m["share"])})</div></div>'
                 f'<div class="tile"><div class="v">{len(works_here)}</div><div class="l">works</div></div>'
                 f'<div class="tile"><div class="v">{m["mean_confidence"]:.2f}</div><div class="l">mean assignment margin</div></div></div>']
        parts.append("<h2>Style markers</h2><p class=\"sub\">Features whose average inside this style group differs most from the corpus average (standard deviations).</p>")
        parts.append('<div class="markers">' + "".join(f"<code>{h.escape(n)} {x:+.2f}σ</code>" for n, x in m["markers_high"][:12]) + "</div>")
        parts.append('<p class="sub">Under-represented:</p><div class="markers">' + "".join(f"<code>{h.escape(n)} {x:+.2f}σ</code>" for n, x in m["markers_low"][:8]) + "</div>")
        if m.get("ai_profile_means"):
            ai = m["ai_profile_means"]
            parts.append("<h3>AI style profile (mean 0–1)</h3><table><tbody>" + "".join(
                f'<tr><td>{h.escape(k.replace("_", " "))}</td><td style="width:50%"><div class="bar"><span style="width:{100 * v:.1f}%;background:{color_var(a)}"></span></div></td><td class="num">{v:.2f}</td></tr>'
                for k, v in ai.items()) + "</tbody></table>")
        if m.get("tag_lift"):
            parts.append("<h3>Characteristic devices</h3><div class=\"markers\">" + "".join(f"<code>{h.escape(t)} ×{l}</code>" for t, l, _ in m["tag_lift"][:12]) + "</div>")
        if m.get("phrases"):
            parts.append("<h3>Diagnostic phrases</h3><div class=\"markers\">" + "".join(f"<code>{h.escape(p)} ({c})</code>" for p, c in m["phrases"][:12]) + "</div>")
        parts.append("<h2>Works</h2>" + legend([a]) + f"<table><thead><tr><th>work</th><th class=\"num\">{units} in this style group</th><th>share of the work</th></tr></thead><tbody>" + "".join(
            f'<tr><td><a href="../works/{w}.html">{h.escape(w)}</a></td><td class="num">{n}</td><td style="width:40%"><div class="bar"><span style="width:{100 * n / len(by_work[w]):.1f}%;background:{color_var(a)}"></span></div> {_pct(n / len(by_work[w]))}</td></tr>'
            for w, n in works_here.most_common()) + "</tbody></table>")
        parts.append('<p class="sub">Exploratory style group, not an identified author.</p>')
        parts.append(mapping_note("../../"))
        parts.append(f"<h2>{units.capitalize()} assigned to this style group</h2><p class=\"sub\">In reading order; the margin is relative centroid separation, not a probability of authorship. Zero denotes a tie or a single-group run.</p>")
        for w in work_order:
            vs = [r for r in mine if r["work"] == w]
            if not vs:
                continue
            work_label = w if passage_mode else vs[0]["ref"].rsplit(" ", 1)[0]
            parts.append(f'<h3 class="work-head"><a href="../works/{w}.html">{h.escape(work_label)}</a> · {len(vs)} of {len(by_work[w])} {units}</h3>')
            for r in vs:
                flag = " flag" if float(r["outlier_z"]) > 2.5 else ""
                parts.append(f'<div class="{unit_class}{flag}" style="border-left-color:{color_var(a)}"><span class="ref"><a href="../works/{w}.html#{h.escape(r["id"])}">{h.escape(r["ref"])}</a></span>'
                             f'<span class="conf">{float(r["confidence"]):.2f}</span><span class="{gk}" dir="auto">{h.escape(r["text"])}</span></div>')
        (site / "authors" / f"{a}.html").write_text(page(f"Style group {a}", "\n".join(parts), "../", f"{m['n_verses']:,} {units} · {_pct(m['share'])} of the corpus"), encoding="utf-8")

    # ---- per-work pages --------------------------------------------------------------------------
    for w in work_order:
        vs = by_work[w]
        n = len(vs)
        title = w if passage_mode else vs[0]["ref"].rsplit(" ", 1)[0]
        counts = work_counts[w]
        parts = [f'<div class="tiles"><div class="tile"><div class="v">{n:,}</div><div class="l">{units}</div></div>'
                 f'<div class="tile"><div class="v">{chip(majority[w], "../")}</div><div class="l">main group ({counts[majority[w]] / n:.0%})</div></div>'
                 f'<div class="tile"><div class="v">{len(counts)}</div><div class="l">style groups present</div></div></div>']
        parts.append('<p class="sub">Exploratory style groups, not identified authors. Genre, topic and witnesses may explain differences.</p>')
        parts.append(mapping_note("../../"))
        parts.append(legend([a for a in authors if counts.get(a)]) + stack_bar(counts, n, authors, w, units))
        parts.append(f"<table><thead><tr><th>group</th><th class=\"num\">{units}</th><th class=\"num\">share</th></tr></thead><tbody>" + "".join(
            f'<tr><td>{chip(a, "../")}</td><td class="num">{c}</td><td class="num">{_pct(c / n)}</td></tr>' for a, c in counts.most_common()) + "</tbody></table>")
        parts.append(f"<h2>{unit.capitalize()} by {unit}</h2><p class=\"sub\">Coloured rule = assigned group. ⚑ marks a {unit} whose own style is far from its group's centre (an exploratory outlier).</p>")
        for r in vs:
            a = r["author"]
            flag = " flag" if float(r["outlier_z"]) > 2.5 else ""
            parts.append(f'<div class="{unit_class}{flag}" id="{h.escape(r["id"])}" style="border-left-color:{color_var(a)}"><span class="ref">{h.escape(r["ref"])}</span>'
                         f'<span>{chip(a, "../")}</span><span class="{gk}" dir="auto">{h.escape(r["text"])}</span></div>')
        (site / "works" / f"{w}.html").write_text(page(title, "\n".join(parts), "../", f"{h.escape(work_group[w])} · main group {majority[w]}"), encoding="utf-8")
    return site


def build_root_index(output_dir: str | Path) -> Path | None:
    """One page linking every language dashboard that has been rendered."""
    output_dir = Path(output_dir)
    names = {"grc": "Greek", "hbo": "Hebrew", "arb": "Arabic"}
    cards = []
    has_passages = False
    for lang_dir in sorted(output_dir.glob("*/")):
        summary_path = lang_dir / "summary.json"
        if not summary_path.exists() or not (lang_dir / "site" / "index.html").exists():
            continue
        s = json.loads(summary_path.read_text())
        lang = s.get("language", lang_dir.name)
        units = "passages" if s.get("input_unit") == "pooled_token_passage" else "verses"
        has_passages = has_passages or units == "passages"
        label = names.get(lang, lang)
        if lang_dir.name != lang:  # e.g. output/hbo-genesis: a run over part of one language
            label += " · " + lang_dir.name.removeprefix(lang).lstrip("-_").replace("-", " ").title()
        cards.append(
            f'<div class="tile"><div class="v"><a href="{lang_dir.name}/site/index.html">{label}</a></div>'
            f'<div class="l">{s["n_verses"]:,} {units} · {s["n_works"]} works · {s["k_used"]} style groups · '
            f'{"AI + lexical" if s.get("used_ai_profiles") else "lexical only"}</div></div>'
        )
    if not cards:
        return None
    body = '<div class="tiles">' + "".join(cards) + "</div><p class=\"sub\">Each language is analysed on its own. These are exploratory style groups, not identified authors.</p>"
    nav = '<a href="index.html">All languages</a>'
    if (output_dir / "models" / "index.html").exists():
        body += '<p><a href="models/index.html">Which model should profile the verses?</a> - agreement, repeatability and cost of every model tried.</p>'
        nav += '<a href="models/index.html">Models</a>'
    path = output_dir / "index.html"
    subtitle = "Exploratory style analysis across languages" if has_passages else "Exploratory style analysis across Greek, Hebrew and Arabic scripture"
    path.write_text(page("Style groups", body, "", subtitle, nav=nav), encoding="utf-8")
    return path
