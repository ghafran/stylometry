"""The explorer as a set of pages: a language, its collections, and a page per book.

Kept as separate pages on purpose. Everything in one file would be twenty megabytes of markup for
Greek, which a browser will parse but not enjoy; split, the overview is small and each book carries
only its own verses.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LANGUAGE_NAMES = {"grc": "Koine Greek", "hbo": "Biblical Hebrew", "arb": "Quranic Arabic"}
RTL = {"hbo", "arb"}

CSS = """
:root{--paper:#FAF9F5;--ink:#141413;--muted:#56554F;--faint:#807E76;--line:#DCD9CE;--clay:#A8482A;
--card:#fff;--panel:#F3F1E9;--dot:#A8482A}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#16161A;--ink:#EDEBE4;
--muted:#B4B1A7;--faint:#93918A;--line:#33322D;--clay:#E0906F;--card:#1F1F23;--panel:#232329;--dot:#E0906F}}
:root[data-theme="dark"]{--paper:#16161A;--ink:#EDEBE4;--muted:#B4B1A7;--faint:#93918A;--line:#33322D;
--clay:#E0906F;--card:#1F1F23;--panel:#232329;--dot:#E0906F}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:'IBM Plex Sans',system-ui,sans-serif;line-height:1.5}
.wrap{max-width:1240px;margin:0 auto;padding:32px 16px 72px}
a{color:var(--clay)}
h1{font-family:'Spectral',Georgia,serif;font-size:clamp(24px,3.6vw,34px);font-weight:600;margin:0 0 4px;letter-spacing:-.01em}
.crumb{font-size:13px;color:var(--faint);margin-bottom:14px}
.crumb a{color:var(--muted);text-decoration:none}.crumb a:hover{color:var(--clay)}
.mono{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:12px 14px;margin-bottom:16px;position:sticky;top:0;z-index:5}
.bar label{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);font-family:'IBM Plex Mono',monospace}
select{font:inherit;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--paper);color:var(--ink);max-width:100%}
.measures{font-size:13px;color:var(--muted);flex-basis:100%;margin-top:2px}
.note{font-size:12.5px;color:var(--faint);flex-basis:100%;border-left:2px solid var(--line);padding-left:9px}
.cols{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,440px);gap:20px}
@media (max-width:900px){.cols{grid-template-columns:1fr}}
.row{display:grid;grid-template-columns:1fr auto;gap:2px 12px;width:100%;text-align:left;background:var(--card);
border:1px solid var(--line);border-radius:9px;padding:10px 13px;margin-bottom:7px;font:inherit;color:inherit;cursor:pointer}
.row:hover{border-color:var(--clay)}
.row .t{font-weight:600;font-size:14.5px}
.row .s{grid-column:1;font-size:12.5px;color:var(--faint)}
.row .n{grid-row:1;grid-column:2;font-family:'IBM Plex Mono',monospace;font-size:12.5px;color:var(--muted);white-space:nowrap}
.card{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:16px 18px;position:sticky;top:78px}
svg{width:100%;height:auto;display:block;background:var(--panel);border-radius:8px}
.mk{display:flex;justify-content:space-between;font-size:12.5px;padding:5px 0;border-top:1px solid var(--line);gap:12px}
.mk span:first-child{font-family:'IBM Plex Mono',monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mk span:last-child{font-family:'IBM Plex Mono',monospace;color:var(--clay)}
.vtext{font-size:15px;line-height:1.75;margin-top:3px}
.vtext.rtl{direction:rtl;text-align:right;font-size:17px}
h2{font-family:'Spectral',Georgia,serif;font-size:19px;margin:0 0 8px;font-weight:600}
.small{font-size:12.5px;color:var(--faint)}
"""

JS_COMMON = """
const qs = new URLSearchParams(location.search);
let strategy = qs.get('s') || localStorage.getItem('styl.strategy') || DATA.keys[0];
if (!DATA.keys.includes(strategy)) strategy = DATA.keys[0];
const idx = () => DATA.keys.indexOf(strategy);
const xyOf = o => { const i = idx() * 2; return [o.xy[i], o.xy[i + 1]]; };
const vxyOf = v => { const i = idx() * 2; return [v[3][i], v[3][i + 1]]; };

function strategyBar(onChange) {
  const meta = () => DATA.strategies.find(s => s.key === strategy) || {};
  const bar = document.getElementById('bar');
  bar.innerHTML = `<label for="sel">Strategy</label>
    <select id="sel">${DATA.strategies.map(s =>
      `<option value="${s.key}"${s.key === strategy ? ' selected' : ''}>${s.name} — ${s.status}</option>`).join('')}</select>
    <div class="measures" id="ms"></div><div class="note" id="nt"></div>`;
  const paint = () => {
    document.getElementById('ms').textContent = meta().measures || '';
    const n = document.getElementById('nt');
    n.textContent = meta().note || '';
    n.style.display = meta().note ? '' : 'none';
  };
  document.getElementById('sel').addEventListener('change', e => {
    strategy = e.target.value;
    localStorage.setItem('styl.strategy', strategy);
    const u = new URL(location); u.searchParams.set('s', strategy); history.replaceState(null, '', u);
    paint(); onChange();
  });
  paint();
}

function plot(points, holder, onPick) {
  const W = 420, H = 300, P = 26;
  if (!points.length) { holder.innerHTML = '<p class="small">Nothing to plot here.</p>'; return; }
  const xs = points.map(p => p.xy[0]), ys = points.map(p => p.xy[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const sx = v => P + (x1 - x0 < 1e-9 ? (W - 2 * P) / 2 : (v - x0) / (x1 - x0) * (W - 2 * P));
  const sy = v => H - P - (y1 - y0 < 1e-9 ? (H - 2 * P) / 2 : (v - y0) / (y1 - y0) * (H - 2 * P));
  const r = points.length > 400 ? 2 : points.length > 90 ? 3 : 5;
  holder.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Style positions under the chosen strategy">
    <line x1="${P}" y1="${H - P}" x2="${W - P}" y2="${H - P}" stroke="var(--line)"/>
    <line x1="${P}" y1="${P}" x2="${P}" y2="${H - P}" stroke="var(--line)"/>
    ${points.map((p, i) => `<circle cx="${sx(p.xy[0]).toFixed(1)}" cy="${sy(p.xy[1]).toFixed(1)}" r="${r}"
      fill="var(--dot)" fill-opacity="${points.length > 400 ? .45 : .8}" data-i="${i}"><title>${p.label}</title></circle>`).join('')}
    <text x="${W - P}" y="${H - 8}" text-anchor="end" font-size="9" fill="var(--faint)" font-family="monospace">component 1</text>
    <text x="8" y="${P}" font-size="9" fill="var(--faint)" font-family="monospace">component 2</text>
  </svg>`;
  if (onPick) holder.querySelectorAll('circle').forEach(c =>
    c.addEventListener('click', () => onPick(points[+c.dataset.i])));
}

function markerList(node, holder) {
  const m = (node.markers || {})[strategy] || [];
  holder.innerHTML = m.length
    ? '<h2>What stands out here</h2>' + m.map(([n, z]) =>
        `<div class="mk"><span>${n}</span><span>${z > 0 ? '+' : ''}${z.toFixed(2)}σ</span></div>`).join('')
      + '<p class="small" style="margin-top:10px">Standard deviations from the corpus mean, under this strategy.</p>'
    : '<h2>What stands out here</h2><p class="small">Nothing in this strategy separates this selection from the corpus.</p>';
}
"""


def _slug(code: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", code) or "work"


def _head(title: str, extra: str = "") -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
            'family=Spectral:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&'
            'family=IBM+Plex+Mono:wght@400;500&display=swap">'
            f"<style>{CSS}{extra}</style></head><body><div class=\"wrap\">")


def write(data: dict, out_dir: str | Path) -> Path:
    """One overview page per language plus one page per book."""
    out = Path(out_dir)
    (out / "works").mkdir(parents=True, exist_ok=True)
    language = data["language"]
    name = LANGUAGE_NAMES.get(language, language)
    rtl = "rtl" if language in RTL else ""

    # --- overview: collections, their works, and the chapters of each --------------------------------
    slim = {
        "keys": data["keys"], "strategies": data["strategies"], "language": language,
        "tree": [{"label": c["label"], "xy": c["xy"], "markers": c["markers"],
                  "n_verses": c["n_verses"], "n_tokens": c["n_tokens"],
                  "children": [{"label": w["label"], "code": w["code"], "xy": w["xy"],
                                "markers": w["markers"], "n_verses": w["n_verses"],
                                "n_tokens": w["n_tokens"], "slug": _slug(w["code"]),
                                "n_chapters": len(w["children"])}
                               for w in c["children"]]}
                 for c in data["tree"]],
    }
    page = _head(f"{name} — style explorer")
    page += f"""<div class="crumb"><a href="../index.html">All languages</a></div>
<h1>{name}</h1>
<p class="small">{data['n_verses']:,} verses · {len(data['strategies'])} strategies available"""
    if data["unavailable"]:
        page += f" · no data for: {', '.join(data['unavailable'])}"
    page += """</p>
<div class="bar" id="bar"></div>
<div class="cols"><div id="list"></div><div><div class="card" id="side"></div></div></div>
<script>const DATA = """ + json.dumps(slim, ensure_ascii=False) + ";\n" + JS_COMMON + """
let collection = null;
function render() {
  const list = document.getElementById('list'), side = document.getElementById('side');
  const nodes = collection === null ? DATA.tree : DATA.tree[collection].children;
  const here = collection === null ? null : DATA.tree[collection];
  list.innerHTML = (collection === null
      ? '<h2>Collections</h2>'
      : `<h2>${DATA.tree[collection].label} — books</h2><button class="row" id="up"><span class="t">← back to collections</span><span class="n"></span></button>`)
    + nodes.map((n, i) => `<button class="row" data-i="${i}">
        <span class="t">${n.label}</span>
        <span class="n">${n.n_verses.toLocaleString()} verses</span>
        <span class="s">${n.code ? n.code + ' · ' + n.n_chapters + ' chapters · ' : ''}${n.n_tokens.toLocaleString()} tokens</span>
      </button>`).join('');
  if (collection !== null) document.getElementById('up').addEventListener('click', () => { collection = null; render(); });
  list.querySelectorAll('.row[data-i]').forEach(b => b.addEventListener('click', () => {
    const n = nodes[+b.dataset.i];
    if (collection === null) { collection = +b.dataset.i; render(); }
    else location.href = 'works/' + n.slug + '.html?s=' + encodeURIComponent(strategy);
  }));
  const holder = document.createElement('div');
  side.innerHTML = '';
  side.appendChild(holder);
  plot(nodes.map(n => ({ label: n.label, xy: xyOf(n) })), holder, null);
  const mk = document.createElement('div');
  side.appendChild(mk);
  markerList(here || { markers: {} }, mk);
  if (!here) mk.insertAdjacentHTML('afterbegin', '<p class="small">Each point is a collection. Open one to see its books.</p>');
}
strategyBar(render); render();
</script></div></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")

    # --- one page per book ---------------------------------------------------------------------------
    for collection in data["tree"]:
        for work in collection["children"]:
            payload = {
                "keys": data["keys"], "strategies": data["strategies"], "language": language,
                "work": {"label": work["label"], "code": work["code"], "xy": work["xy"],
                         "markers": work["markers"], "n_verses": work["n_verses"]},
                "collection": collection["label"],
                "chapters": [{"label": ch["label"], "xy": ch["xy"], "markers": ch["markers"],
                              "n_verses": ch["n_verses"], "v": ch["v"]} for ch in work["children"]],
            }
            body = _head(f"{work['label']} — style explorer")
            body += f"""<div class="crumb"><a href="../../index.html">All languages</a> ·
<a href="../index.html">{name}</a> · {collection['label']}</div>
<h1>{work['label']}</h1>
<p class="small">{work['code']} · {work['n_verses']:,} verses · {len(work['children'])} chapters</p>
<div class="bar" id="bar"></div>
<div class="cols"><div id="list"></div><div><div class="card" id="side"></div></div></div>
<script>const DATA = """ + json.dumps(payload, ensure_ascii=False) + ";\n" + JS_COMMON + f"""
const RTL = {'true' if rtl else 'false'};
let chapter = null, verse = null;
function render() {{
  const list = document.getElementById('list'), side = document.getElementById('side');
  if (chapter === null) {{
    list.innerHTML = '<h2>Chapters</h2>' + DATA.chapters.map((c, i) =>
      `<button class="row" data-i="${{i}}"><span class="t">Chapter ${{c.label}}</span>
       <span class="n">${{c.n_verses}} verses</span></button>`).join('');
    list.querySelectorAll('.row').forEach(b => b.addEventListener('click', () => {{
      chapter = +b.dataset.i; verse = null; render(); }}));
    const h = document.createElement('div'); side.innerHTML = ''; side.appendChild(h);
    plot(DATA.chapters.map(c => ({{ label: 'Chapter ' + c.label, xy: xyOf(c) }})), h, null);
    const mk = document.createElement('div'); side.appendChild(mk); markerList(DATA.work, mk);
    mk.insertAdjacentHTML('afterbegin', '<p class="small">Each point is a chapter of this book.</p>');
    return;
  }}
  const ch = DATA.chapters[chapter];
  list.innerHTML = `<h2>Chapter ${{ch.label}} — verses</h2>
    <button class="row" id="up"><span class="t">← back to chapters</span><span class="n"></span></button>`
    + ch.v.map((v, i) => `<button class="row" data-i="${{i}}">
        <span class="t">${{ch.label}}:${{v[0]}}</span><span class="n">${{v[2]}} tokens</span>
        <span class="s vtext ${{RTL ? 'rtl' : ''}}">${{v[1]}}</span></button>`).join('');
  document.getElementById('up').addEventListener('click', () => {{ chapter = null; verse = null; render(); }});
  list.querySelectorAll('.row[data-i]').forEach(b => b.addEventListener('click', () => {{
    verse = +b.dataset.i; render(); }}));
  const h = document.createElement('div'); side.innerHTML = ''; side.appendChild(h);
  plot(ch.v.map(v => ({{ label: ch.label + ':' + v[0], xy: vxyOf(v) }})), h, p => {{
    verse = ch.v.findIndex(v => ch.label + ':' + v[0] === p.label); render(); }});
  const mk = document.createElement('div'); side.appendChild(mk);
  if (verse !== null) {{
    const v = ch.v[verse], xy = vxyOf(v);
    mk.innerHTML = `<h2>${{ch.label}}:${{v[0]}}</h2>
      <div class="vtext ${{RTL ? 'rtl' : ''}}">${{v[1]}}</div>
      <div class="mk"><span>position under this strategy</span><span>${{xy[0].toFixed(1)}}, ${{xy[1].toFixed(1)}}</span></div>
      <div class="mk"><span>tokens</span><span>${{v[2]}}</span></div>
      <p class="small" style="margin-top:10px">A single verse is far too short to characterise a hand.
      Its position is where this strategy places it, not a claim about who wrote it.</p>`;
  }} else {{
    markerList(ch, mk);
    mk.insertAdjacentHTML('afterbegin', '<p class="small">Each point is a verse. Pick one to read it.</p>');
  }}
}}
strategyBar(render); render();
</script></div></body></html>"""
            (out / "works" / f"{_slug(work['code'])}.html").write_text(body, encoding="utf-8")
    return out
