"""The explorer as a set of pages: a language, its collections, and a page per book.

Kept as separate pages on purpose. Everything in one file would be twenty megabytes of markup for
Greek, which a browser will parse but not enjoy; split, the overview is small and each book carries
only its own verses.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LANGUAGE_NAMES = {"grc": "Koine Greek", "hbo": "Biblical Hebrew", "arb": "Quranic Arabic",
                  "eng": "English (known authors)"}
RTL = {"hbo", "arb"}

CSS = """
:root{--paper:#FAF9F5;--ink:#141413;--muted:#56554F;--faint:#807E76;--line:#DCD9CE;--clay:#A8482A;
--card:#fff;--panel:#F3F1E9;--dot:#A8482A;
--g1:#2a78d6;--g2:#eb6834;--g3:#1baf7a;--g4:#eda100;--g5:#e87ba4;--g6:#008300;--g7:#4a3aa7;--g8:#e34948;--gx:#8a8985}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#16161A;--ink:#EDEBE4;
--muted:#B4B1A7;--faint:#93918A;--line:#33322D;--clay:#E0906F;--card:#1F1F23;--panel:#232329;--dot:#E0906F;
--g1:#3987e5;--g2:#d95926;--g3:#199e70;--g4:#c98500;--g5:#d55181;--g6:#008300;--g7:#9085e9;--g8:#e66767;--gx:#8f8e88}}
:root[data-theme="dark"]{--paper:#16161A;--ink:#EDEBE4;--muted:#B4B1A7;--faint:#93918A;--line:#33322D;
--clay:#E0906F;--card:#1F1F23;--panel:#232329;--dot:#E0906F;
--g1:#3987e5;--g2:#d95926;--g3:#199e70;--g4:#c98500;--g5:#d55181;--g6:#008300;--g7:#9085e9;--g8:#e66767;--gx:#8f8e88}
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
.bar .count{font-size:12.5px;color:var(--faint)}
.bar .miss{flex-basis:100%;font-size:12.5px;color:var(--clay);border-left:2px solid var(--clay);padding-left:9px}
.bar select{font-weight:600;border-color:var(--clay)}
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
.gsum{display:block;font-family:'IBM Plex Sans',system-ui,sans-serif;font-size:13px;font-weight:400;
color:var(--muted);margin:-4px 0 9px}
.gsum b{color:var(--ink);font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:5px 14px;font-size:12px;color:var(--muted);margin:0 0 11px}
.legend i,.gb i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.vrow{display:grid;grid-template-columns:78px 40px minmax(0,1fr);gap:3px 11px;align-items:start;
width:100%;text-align:left;background:var(--card);border:1px solid var(--line);border-left:3px solid transparent;
border-radius:9px;padding:9px 13px;margin-bottom:7px;font:inherit;color:inherit;cursor:pointer}
.vrow:hover{border-color:var(--clay)}
.vrow .vref{font-family:'IBM Plex Mono',monospace;font-size:12.5px;font-weight:600;color:var(--ink);white-space:nowrap}
.vrow .vtok{grid-row:2;grid-column:1;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--faint)}
.vrow .vgrp{font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:600;white-space:nowrap}
.vrow .vgrp i{display:block;width:100%;height:3px;border-radius:2px;margin-top:3px}
.vrow .vbody{grid-row:1/span 2;grid-column:3}
.vhead{display:grid;grid-template-columns:78px 40px minmax(0,1fr);gap:11px;padding:0 13px 5px;
font-family:'IBM Plex Mono',monospace;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
.sp{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.03em;
padding:1px 6px;border-radius:3px;background:var(--panel);color:var(--muted);white-space:nowrap}
.sp.sp-p{background:color-mix(in srgb,var(--g1) 18%,transparent);color:var(--g1)}
.sp.sp-d{background:color-mix(in srgb,var(--g4) 22%,transparent);color:var(--g4)}
.gin{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);margin-left:7px;font-weight:400}
.gin i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.gb.lg{grid-row:1;grid-column:2;align-self:start;color:var(--ink);font-weight:600}
.gb{grid-row:2;grid-column:2;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);
white-space:nowrap;align-self:end}
.headline{background:var(--panel);border-radius:9px;padding:11px 14px;margin:0 0 16px;font-size:13.5px;color:var(--muted)}
.headline b{color:var(--ink)}
.caveat{border-left:2px solid var(--clay);padding:3px 0 3px 10px;margin:0 0 11px;font-size:12.5px;color:var(--muted)}
.caveat b{color:var(--clay)}
.bars{margin:2px 0 4px}
.brow{display:grid;grid-template-columns:30px 1fr auto;align-items:center;gap:9px;margin-bottom:5px}
.blab{font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:600;color:var(--ink)}
.btrack{height:11px;background:var(--panel);border-radius:3px;overflow:hidden}
.bfill{display:block;height:100%;border-radius:0 3px 3px 0;min-width:2px}
.bval{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--muted);font-variant-numeric:tabular-nums}
.brow.mine .blab::after{content:"\\25C0";margin-left:5px;font-size:9px;color:var(--clay);vertical-align:1px}
.brow.mine .bval{color:var(--ink);font-weight:600}
.brow.mine .btrack{outline:1px solid var(--clay);outline-offset:1px}
.stack{display:flex;height:9px;gap:2px;border-radius:3px;overflow:hidden;margin:5px 0 1px;max-width:260px}
.stack>span{display:block;height:100%;min-width:2px}
.rbars{grid-column:1/-1;display:block;margin:7px 0 1px;max-width:340px}
.rcap{display:block;font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.05em;
text-transform:uppercase;color:var(--faint);margin-bottom:4px}
.rrow{display:grid;grid-template-columns:24px 1fr auto;align-items:center;gap:7px;margin-bottom:3px}
.rlab{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:600}
.rtrack{height:8px;background:var(--panel);border-radius:2px;overflow:hidden}
.rfill{display:block;height:100%;border-radius:0 2px 2px 0;min-width:2px}
.rval{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:var(--muted);font-variant-numeric:tabular-nums}
.actl{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:0 0 11px}
.actl label{font-size:13px;color:var(--muted)}.actl label b{color:var(--ink);font-variant-numeric:tabular-nums;font-size:15px}
.actl .st{color:var(--faint);margin-left:4px}
.actl input[type=range]{flex:1;min-width:130px;accent-color:var(--clay)}
.actl .small{flex-basis:100%}
.fit{font-size:12.5px;color:var(--muted);background:var(--panel);border-left:3px solid var(--clay);
border-radius:0 7px 7px 0;padding:8px 11px;margin:0 0 12px}
.fit b{color:var(--ink);font-weight:600}
.kn{grid-column:1/-1;font-size:12px;color:var(--muted);margin-top:2px}
.kn::before{content:'known author: ';color:var(--faint)}
.gb.au i{border-radius:3px}
.row{border-left:3px solid transparent}
.vtext.grouped{padding-left:9px;border-left:3px solid var(--line)}
"""

JS_COMMON = """
const qs = new URLSearchParams(location.search);
let strategy = qs.get('s') || localStorage.getItem('styl.strategy') || DATA.keys[0];
const unavailable = DATA.keys.includes(strategy) ? null : strategy;
if (unavailable) strategy = DATA.keys[0];
const idx = () => DATA.keys.indexOf(strategy);
const xyOf = o => { const i = idx() * 2; return [o.xy[i], o.xy[i + 1]]; };
const vxyOf = v => { const i = idx() * 2; return [v[3][i], v[3][i + 1]]; };

function strategyBar(onChange) {
  const meta = () => DATA.strategies.find(s => s.key === strategy) || {};
  const bar = document.getElementById('bar');
  bar.innerHTML = `<label for="sel">Strategy</label>
    <select id="sel">${DATA.strategies.map(s =>
      `<option value="${s.key}"${s.key === strategy ? ' selected' : ''}>${s.name} — ${s.status}</option>`).join('')}</select>
    <span class="count">${DATA.strategies.length} to choose from — everything below re-plots</span>
    <div class="measures" id="ms"></div><div class="note" id="nt"></div>
    ${unavailable ? `<div class="miss">There is no <b>${unavailable}</b> data for this selection, so it opened on the first strategy instead.</div>` : ''}`;
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

// Authors are inferred from style. `gk` is how many the node's children fall to under each strategy
// and `g` which one a child is in, both flat in `keys` order. The inferred count is the largest that
// survives resampling; where the authors are actually known - English - the known count is used.
const G = i => (i >= 1 && i <= 8) ? `var(--g${i})` : 'var(--gx)';
const A = i => 'Author ' + i;
const AU = A;
const groupsOf = n => ({ k: n.gk ? n.gk[idx()] : 1, reason: n.gr ? n.gr[idx()] : null,
                         ari: n.gari ? n.gari[idx()] : null, n: n.gn || 0, tokens: n.gtok || 0,
                         sizes: (n.gsz ? n.gsz[idx()] : null) || [] });
const groupOf = c => (c.g ? c.g[idx()] : 1);
const why = g => (DATA.group_reasons || {})[g.reason] || 'no division is supported';

// Who is speaking, where the source says. Only the Arabic hadith collections carry this.
const SPEAKER = {
  prophet: ['quote of the Prophet', 'sp-p'],
  other: ['quoted, another speaker', 'sp-o'],
  report: ["narrator's report", 'sp-r'],
  divine: ['speech of God, related by the Prophet', 'sp-d'],
};
const speakerTag = code => {
  const s = SPEAKER[code];
  return s ? `<span class="sp ${s[1]}">${s[0]}</span>` : '';
};

// The inferred count for a level that has no override: chapters of a book, verses of a chapter.
function groupSummary(node, unit) {
  const g = groupsOf(node);
  if (!g.n) return '';
  const head = g.k > 1
    ? `<span class="gsum"><b>${g.k} authors</b> among ${g.n} ${unit}, inferred from style` +
      (g.ari != null ? ` · survives resampling at ARI ${g.ari}` : '') + `</span>`
    : `<span class="gsum"><b>1 author</b> among ${g.n} ${unit} — ${why(g)}</span>`;
  return head + shortUnitWarning(g, unit);
}

// A division of units this short can be perfectly stable and still be arithmetic: a hapax ratio over
// eighteen tokens takes few distinct values, so it clusters cleanly. Say so where it applies.
function shortUnitWarning(g, unit) {
  const floor = DATA.reliable_tokens || 1000;
  if (g.k < 2 || !g.tokens || g.tokens >= floor) return '';
  return `<div class="caveat">These ${unit} run about <b>${g.tokens.toLocaleString()} tokens</b> each.
    This project measured that units below ${floor.toLocaleString()} cannot support attribution, which is
    why the analysis pools text into passages first. Read this division as a property of the measure at
    this length, not as different hands.</div>`;
}

function groupLegend(node) {
  const g = groupsOf(node);
  if (g.k < 2) return '';
  return '<div class="legend">' + Array.from({ length: g.k }, (_, i) =>
    `<span><i style="background:${G(i + 1)}"></i>${A(i + 1)}</span>`).join('') + '</div>';
}

// How the units divide between the authors. Author 1 is always the largest: the clustering ranks
// them by size before naming them, so a long tail of one-member authors is visible at a glance.
function barChart(sizes, unit, heading, note, mark, lab) {
  lab = lab || A;
  if (!sizes.length) return '';
  const max = Math.max(...sizes), total = sizes.reduce((a, b) => a + b, 0);
  if (!total) return '';
  return `<h2>${heading}</h2>` + (note || '') + `<div class="bars">` + sizes.map((c, i) => `
    <div class="brow${mark === i + 1 ? ' mine' : ''}">
      <span class="blab">${lab(i + 1)}</span>
      <span class="btrack"><span class="bfill" style="width:${(c / max * 100).toFixed(1)}%;background:${G(i + 1)}"></span></span>
      <span class="bval">${c.toLocaleString()} · ${(c / total * 100).toFixed(0)}%</span>
    </div>`).join('') + `</div><p class="small">${total.toLocaleString()} ${unit}.</p>`;
}

function groupBars(node, unit, heading) {
  const g = groupsOf(node);
  const note = g.k > 1 ? '' : `<p class="small">${why(g)}.</p>`;
  return barChart(g.sizes, unit, heading || 'Authors', note);
}

// A chart small enough to sit inside a row, so a list can be read for how each entry divides without
// selecting it first. Same bars as the side panel, tighter.
function rowBars(sizes, caption, lab) {
  lab = lab || A;
  const total = sizes.reduce((a, b) => a + b, 0);
  // One occupied author is an answer - "all nineteen of them Author 2" - not an absence.
  if (!total) return '';
  const max = Math.max(...sizes);
  return `<span class="rbars"><span class="rcap">${caption}</span>` + sizes.map((c, i) => c ? `
    <span class="rrow">
      <span class="rlab" style="color:${G(i + 1)}">${lab(i + 1)}</span>
      <span class="rtrack"><span class="rfill" style="width:${(c / max * 100).toFixed(1)}%;background:${G(i + 1)}"></span></span>
      <span class="rval">${c.toLocaleString()}</span>
    </span>` : '').join('') + `</span>`;
}

// How this node's own children divide: a book's chapters, a chapter's verses.
function ownBars(node, unit) {
  const g = groupsOf(node);
  return g.k > 1 ? rowBars(g.sizes, `its ${g.n} ${unit} · ${g.k} authors`) : '';
}

const groupStack = (sizes, lab) => {
  lab = lab || A;
  if (sizes.length < 2) return '';
  return '<span class="stack">' + sizes.map((c, i) =>
    `<span style="flex:${c};background:${G(i + 1)}" title="${lab(i + 1)}: ${c}"></span>`).join('') + '</span>';
};

// The name is on every row beside the swatch: the lighter hues fall under 3:1 against the page, so
// colour is never the only thing carrying which author a unit is.
const groupBadge = (node, child) =>
  groupsOf(node).k > 1 ? `<span class="gb"><i style="background:${G(groupOf(child))}"></i>${A(groupOf(child))}</span>` : '';
const groupEdge = (node, g) => groupsOf(node).k > 1 ? ` style="border-left-color:${G(g)}"` : '';

// ---- books: the count can be set ------------------------------------------------------------------
// `al` holds one partition per author count, from one up to the number of books, because the inferred
// count is not to be trusted: on the English corpus, where the authors are known, inference returned
// 2 whether the truth was 3, 5 or 13. So the page opens at the known count where there is one, at the
// inferred count where there is not, and lets the reader move it.
const assumed = {};

const authorsOf = node => ({
  n: (node && node.an) || 0,
  range: (node && node.arange) || [],
  labels: node && node.al ? node.al[idx()] : null,
  fit: node && node.af ? node.af[idx()] : null,
  known: (node && node.aknown) || [],
});

function authorDefault(node) {
  const a = authorsOf(node);
  return a.fit ? a.fit.true_k : Math.max(1, groupsOf(node).k || 1);
}

function authorCount(node, key) {
  const a = authorsOf(node);
  if (!a.range.length) return groupsOf(node).k || 1;
  const lo = a.range[0], hi = a.range[1];
  const want = assumed[key] != null ? assumed[key] : authorDefault(node);
  return Math.min(Math.max(want, lo), hi);
}

// 'known' - the real count; 'inferred' - what the text supported on its own; 'set' - the reader's.
function authorStatus(node, key) {
  const a = authorsOf(node), n = authorCount(node, key);
  if (a.fit && n === a.fit.true_k) return 'known';
  if (!a.fit && n === authorDefault(node)) return 'inferred';
  return 'set';
}
const statusText = st => st === 'known' ? ' — the known count'
                       : st === 'inferred' ? ', inferred from style' : ', as you have set it';

const authorLabels = (node, key) => {
  const a = authorsOf(node);
  return a.labels ? (a.labels[String(authorCount(node, key))] || null) : null;
};

function authorSizes(node, key) {
  const l = authorLabels(node, key);
  if (!l) return groupsOf(node).sizes;
  const n = authorCount(node, key), out = new Array(n).fill(0);
  l.forEach(g => { if (g >= 1 && g <= n) out[g - 1]++; });
  return out;
}

function authorSummary(node, key, unit) {
  const a = authorsOf(node);
  if (!a.range.length) return groupSummary(node, unit);
  const n = authorCount(node, key), st = authorStatus(node, key), g = groupsOf(node);
  const reason = (n === 1 && st === 'inferred') ? ` — ${why(g)}` : '';
  return `<span class="gsum"><b>${n} author${n === 1 ? '' : 's'}</b> among ${a.n} ${unit}` +
    statusText(st) + reason + `</span>` + shortUnitWarning(g, unit);
}

// The slider. Its range runs to as many hands as there are books, because nothing in the text rules
// out any of them; the only stop is how much fits in the page.
function authorControl(node, key) {
  const a = authorsOf(node);
  if (!a.range.length) return '';
  const n = authorCount(node, key), st = authorStatus(node, key);
  return `<div class="actl">
    <label for="an-${key}"><b id="anv-${key}">${n}</b> author${n === 1 ? '' : 's'}
      <span class="st" id="ast-${key}">${statusText(st).replace(/^[ ,—-]+/, '')}</span></label>
    <input type="range" id="an-${key}" min="${a.range[0]}" max="${a.range[1]}" value="${n}">
    <span class="small">move to assume a different number of hands, ${a.range[0]}–${a.range[1]}</span></div>`;
}

function bindAuthorControl(key, onChange) {
  const el = document.getElementById('an-' + key);
  if (!el) return;
  el.addEventListener('input', e => {
    assumed[key] = +e.target.value;
    const v = document.getElementById('anv-' + key);
    if (v) v.textContent = e.target.value;
    const st = document.getElementById('ast-' + key);
    if (st) st.textContent = 'as you have set it';
  });
  el.addEventListener('change', onChange);
}

// What the inference is worth, measured where it can be. Everywhere else it says so instead.
function authorFitNote(node, key) {
  const f = authorsOf(node).fit;
  if (!f) {
    return `<div class="caveat">No author here is known, so nothing on this page can be checked.
      Where they are known — English — the same procedure, given the right number of hands, put most
      authors in a group that was not mostly theirs (0 of 13 over the whole corpus; 6 of 13 fitting
      each collection on its own). Read these labels as inferred from style, not as attributions.</div>`;
  }
  const n = authorCount(node, key);
  const at = n === f.true_k ? '' :
    `<br><span class="small">You are looking at ${n}; the figures above are for the known ${f.true_k}.</span>`;
  return `<div class="fit"><b>Checked: ${f.n_known} of these works have a known author.</b>
    At the known count of ${f.true_k}, this strategy puts <b>${f.recovered} of ${f.true_k}</b>
    in a group that is both mostly theirs and mostly no one else's — adjusted Rand ${f.ari}.${at}</div>`;
}

// The name, where it is actually known. This is the only thing on any of these pages that is not an
// inference, which is why it is marked differently from everything around it.
const knownAuthor = w => w && w.author
  ? `<span class="kn">${w.author}${w.genre ? ' · ' + w.genre : ''}</span>` : '';

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
      fill="${p.g ? G(p.g) : 'var(--dot)'}" fill-opacity="${points.length > 400 ? .45 : .8}" data-i="${i}"><title>${p.label}${p.g ? ' — group ' + p.g : ''}</title></circle>`).join('')}
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


def _author_card(data: dict, i: int) -> dict | None:
    """The assumed-author view of a whole language at the count the page opens on.

    The same rule the page uses: where the authors are known, open at the truth; where they are not,
    open at however many the conservative test inferred, because that is the only count the text
    itself offers. Nothing here trusts that number - see `explorer.author_partitions`.
    """
    groups = data["author_groups"]
    span = groups.get("arange") or []
    if not span:
        return None
    fit = groups["af"][i]
    k = fit["true_k"] if fit else max(2, data["book_groups"]["gk"][i])
    k = min(max(k, span[0]), span[1])
    labels = groups["al"][i].get(str(k)) or []
    # Two different questions, and they get different answers. Fitting every book of the language at
    # once asks one partition to hold every hand in it; fitting each collection on its own asks a
    # much easier question and does better. Neither is the "real" number, so the page carries both -
    # quoting only the higher one would be the flattering half of a measurement.
    per = [c["af"][i] for c in data["tree"] if c["af"][i]]
    apart = {"recovered": sum(f["recovered"] for f in per),
             "true_k": sum(f["true_k"] for f in per),
             "collections": len(per)} if per else None
    return {"k": k, "range": span, "fit": fit, "apart": apart,
            "sizes": [labels.count(g) for g in range(1, k + 1)] if labels else []}


def write(data: dict, out_dir: str | Path) -> Path:
    """One overview page per language plus one page per book."""
    out = Path(out_dir)
    (out / "works").mkdir(parents=True, exist_ok=True)
    language = data["language"]
    name = LANGUAGE_NAMES.get(language, language)
    rtl = "rtl" if language in RTL else ""
    shared = {"keys": data["keys"], "strategies": data["strategies"], "language": language,
              "group_reasons": data["group_reasons"], "reliable_tokens": data["reliable_tokens"]}

    # --- overview: collections, their works, and the chapters of each --------------------------------
    keep = ("gk", "gr", "gari", "gn", "gtok", "gsz", "g", "gl", "glsz",
            "an", "arange", "al", "af", "aknown", "bi", "author", "genre")
    slim = {
        **shared,
        "book_groups": data["book_groups"], "collection_groups": data["collection_groups"],
        "author_groups": data["author_groups"],
        "tree": [{"label": c["label"], "xy": c["xy"], "markers": c["markers"],
                  "n_verses": c["n_verses"], "n_tokens": c["n_tokens"],
                  **{f: c[f] for f in keep if f in c},
                  "children": [{"label": w["label"], "code": w["code"], "xy": w["xy"],
                                "markers": w["markers"], "n_verses": w["n_verses"],
                                "n_tokens": w["n_tokens"], "slug": _slug(w["code"]),
                                "n_chapters": len(w["children"]),
                                **{f: w[f] for f in keep if f in w}}
                               for w in c["children"]]}
                 for c in data["tree"]],
    }
    page = _head(f"{name} — explorer")
    page += f"""<div class="crumb"><a href="../index.html">All languages</a></div>
<h1>{name}</h1>
<p class="small">{data['n_verses']:,} verses · {len(data['strategies'])} strategies available"""
    if data["unavailable"]:
        page += f" · no data for: {', '.join(data['unavailable'])}"
    page += """</p>
<div class="bar" id="bar"></div>
<div class="headline" id="head"></div>
<div class="cols"><div id="list"></div><div><div class="card" id="side"></div></div></div>
<script>const DATA = """ + json.dumps(slim, ensure_ascii=False) + ";\nconst LANG = " \
        + json.dumps(name, ensure_ascii=False) + ";\n" + JS_COMMON + """
let collection = null;

// How a collection's books fall across the language's authors, for the collections list.
function collectionAcrossAuthors(c) {
  const l = authorLabels(DATA.author_groups, 'lang');
  const n = authorCount(DATA.author_groups, 'lang');
  if (!l || !c.children) return [];
  const out = new Array(n).fill(0);
  c.children.forEach(w => { const g = l[w.bi]; if (g >= 1 && g <= n) out[g - 1]++; });
  return out;
}

function render() {
  const list = document.getElementById('list'), side = document.getElementById('side');
  const top = collection === null;
  const nodes = top ? DATA.tree : DATA.tree[collection].children;
  const here = top ? null : DATA.tree[collection];
  const ag = DATA.author_groups;

  // The headline is the number worth quoting: every book of the language, and how many hands.
  const acount = authorCount(ag, 'lang'), st = authorStatus(ag, 'lang');
  const sizes = authorSizes(ag, 'lang'), inferred = groupsOf(DATA.book_groups);
  document.getElementById('head').innerHTML =
    `The <b>${ag.an} books</b> of this language fall to <b>${acount} author${acount === 1 ? '' : 's'}</b>` +
    statusText(st) +
    (acount > 1 ? `: ${sizes.map((c, i) => `${A(i + 1)} ${c}`).join(' · ')}. ${groupStack(sizes)}`
                : (st === 'inferred' ? ` — ${why(inferred)}.` : '.')) +
    ` Change the strategy and see whether that survives.`;

  const alang = authorLabels(ag, 'lang');
  list.innerHTML = (top
      ? '<h2>Collections</h2>'
      : `<h2>${here.label} — books</h2>` + authorSummary(here, 'c' + collection, 'books') + groupLegend(here))
    + (top ? '' : '<button class="row" id="up"><span class="t">← back to collections</span><span class="n"></span></button>')
    + nodes.map((n, i) => {
        // A book's row carries its author among every book of the language, which is the only label
        // comparable between collections: one fitted inside a collection starts at Author 1 whatever
        // the neighbouring collection's does.
        const mine = (!top && alang && n.bi != null) ? alang[n.bi] : 0;
        const edge = mine ? ` style="border-left-color:${G(mine)}"` : '';
        const badge = mine ? `<span class="gb au"><i style="background:${G(mine)}"></i>${A(mine)} in ${LANG}</span>` : '';
        return `<button class="row" data-i="${i}"${edge}>
        <span class="t">${n.label}</span>
        <span class="n">${top ? n.n_verses.toLocaleString() + ' verses' : badge}</span>
        <span class="s">${n.code ? n.code + ' · ' + n.n_chapters + ' chapters · ' : ''}${n.n_tokens.toLocaleString()} tokens${top ? '' : ' · ' + n.n_verses.toLocaleString() + ' verses'}</span>
        ${knownAuthor(n)}
        ${top ? rowBars(collectionAcrossAuthors(n), `its ${n.children ? n.children.length : 0} books across the language's ${acount} author${acount === 1 ? '' : 's'}`)
              : ownBars(n, 'chapters')}
      </button>`;}).join('');
  if (!top) document.getElementById('up').addEventListener('click', () => { collection = null; render(); });
  list.querySelectorAll('.row[data-i]').forEach(b => b.addEventListener('click', () => {
    const n = nodes[+b.dataset.i];
    if (top) { collection = +b.dataset.i; render(); }
    else location.href = 'works/' + n.slug + '.html?s=' + encodeURIComponent(strategy);
  }));

  const holder = document.createElement('div');
  side.innerHTML = '';
  side.appendChild(holder);
  plot(nodes.map(n => ({ label: n.label, xy: xyOf(n),
                         g: (!top && alang && n.bi != null) ? alang[n.bi] : 0 })), holder, null);
  const bars = document.createElement('div');
  side.appendChild(bars);
  const node = top ? ag : here, key = top ? 'lang' : 'c' + collection;
  bars.innerHTML = authorControl(node, key) + authorFitNote(node, key)
    + barChart(authorSizes(node, key), top ? 'books of this language' : `books of ${here.label}`,
               top ? 'Authors across the language' : `Authors within ${here.label}`)
    + (top ? '' : `<p class="small">Fitted among ${here.label}'s books alone, so these cannot be
        compared with another collection's — both start at Author 1. The label on each row is the
        language-wide one, which can.</p>`);
  bindAuthorControl(key, render);
  const mk = document.createElement('div');
  side.appendChild(mk);
  markerList(here || { markers: {} }, mk);
  if (!here) mk.insertAdjacentHTML('afterbegin', '<p class="small">Each point is a collection. Open one to see its books.</p>');
}
strategyBar(render); render();
</script></div></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")

    # --- one page per book ---------------------------------------------------------------------------
    cards = [_author_card(data, i) for i in range(len(data["keys"]))]
    for collection in data["tree"]:
        for work in collection["children"]:
            # Where this book sits among the language's authors, at the count the language page opens
            # on - the known count where there is one, the inferred count where there is not - so the
            # two pages never disagree about how many hands there are.
            lang_authors = []
            for i, card in enumerate(cards):
                if not card:
                    lang_authors.append(None)
                    continue
                labels = data["author_groups"]["al"][i].get(str(card["k"])) or []
                lang_authors.append({"k": card["k"], "sizes": card["sizes"], "known": bool(card["fit"]),
                                     "mine": labels[work["bi"]] if work["bi"] < len(labels) else 0})
            payload = {
                "lang_authors": lang_authors,
                **shared,
                "work": {"label": work["label"], "code": work["code"], "xy": work["xy"],
                         "markers": work["markers"], "n_verses": work["n_verses"],
                         **{f: work[f] for f in keep if f in work}},
                "collection": collection["label"],
                "language_name": name,
                # Where this book sits: among every book of the language, and among its collection's.
                "book_groups": data["book_groups"],
                "collection_books": {f: collection[f] for f in ("gk", "gr", "gari", "gn", "gsz")
                                     if f in collection},
                "chapters": [{"label": ch["label"], "xy": ch["xy"], "markers": ch["markers"],
                              "n_verses": ch["n_verses"], "v": ch["v"],
                              **{f: ch[f] for f in keep if f in ch}} for ch in work["children"]],
            }
            body = _head(f"{work['label']} — explorer")
            body += f"""<div class="crumb"><a href="../../index.html">All languages</a> ·
<a href="../index.html">{name}</a> · {collection['label']}</div>
<h1>{work['label']}</h1>
<p class="small">{work['code']} · {work['n_verses']:,} verses · {len(work['children'])} chapters</p>
<div class="bar" id="bar"></div>
<div class="headline" id="head"></div>
<div class="cols"><div id="list"></div><div><div class="card" id="side"></div></div></div>
<script>const DATA = """ + json.dumps(payload, ensure_ascii=False) + ";\n" + JS_COMMON + f"""
const RTL = {'true' if rtl else 'false'};
const vGroupOf = v => v[4][idx()];
const langAuthor = () => (DATA.lang_authors || [])[idx()] || null;
const mineAcross = () => {{ const la = langAuthor(); return la ? la.mine : (DATA.work.gl ? DATA.work.gl[idx()] : 0); }};
const mineInCollection = () => (DATA.work.g ? DATA.work.g[idx()] : 0);

// Where this book itself sits. Everything below is about its insides; this is the one line that
// places the book among its peers, and without it the page never says whose hand the book is in.
function placement() {{
  const la = langAuthor(), across = groupsOf(DATA.book_groups), mine = mineAcross();
  const within = groupsOf(DATA.collection_books), inner = mineInCollection();
  const k = la ? la.k : across.k, sizes = la ? la.sizes : across.sizes;
  let out = k > 1 && mine
    ? `Under this strategy this book is <b>${{A(mine)}}</b> of the ${{k}} authors of
       ${{DATA.language_name}}${{la && la.known ? ' — the known count' : ', inferred from style'}} —
       a hand holding ${{sizes[mine - 1]}} of its <b>${{across.n}} books</b>.`
    : `The ${{across.n}} books of ${{DATA.language_name}} fall to one author under this strategy:
       ${{why(across)}}.`;
  out += within.k > 1 && inner
    ? ` Among the ${{within.n}} books of ${{DATA.collection}} alone it is <b>${{A(inner)}}</b>.`
    : ` The ${{within.n}} books of ${{DATA.collection}} do not divide among themselves.`;
  return out;
}}

let chapter = null, verse = null;
function render() {{
  document.getElementById('head').innerHTML = placement();
  const list = document.getElementById('list'), side = document.getElementById('side');
  if (chapter === null) {{
    list.innerHTML = '<h2>Chapters</h2>' + groupSummary(DATA.work, 'chapters') + groupLegend(DATA.work)
      + DATA.chapters.map((c, i) =>
      `<button class="row" data-i="${{i}}"${{groupEdge(DATA.work, groupOf(c))}}>
       <span class="t">Chapter ${{c.label}}</span>
       <span class="n">${{c.n_verses}} verses</span>${{groupBadge(DATA.work, c)}}
       ${{ownBars(c, 'verses')}}</button>`).join('');
    list.querySelectorAll('.row').forEach(b => b.addEventListener('click', () => {{
      chapter = +b.dataset.i; verse = null; render(); }}));
    const h = document.createElement('div'); side.innerHTML = ''; side.appendChild(h);
    const many = groupsOf(DATA.work).k > 1;
    plot(DATA.chapters.map(c => ({{ label: 'Chapter ' + c.label, xy: xyOf(c), g: many ? groupOf(c) : 0 }})), h, null);
    const bars = document.createElement('div'); side.appendChild(bars);
    const la = langAuthor();
    bars.innerHTML = groupBars(DATA.work, 'chapters of this book', 'Authors among the chapters')
      + barChart(la ? la.sizes : groupsOf(DATA.book_groups).sizes, `books of ${{DATA.language_name}}`,
                 'Where this book sits', `<p class="small">Authors over every book of the
                 language. The marked row is this book's.</p>`, mineAcross());
    const mk = document.createElement('div'); side.appendChild(mk); markerList(DATA.work, mk);
    mk.insertAdjacentHTML('afterbegin', '<p class="small">Each point is a chapter of this book.</p>');
    return;
  }}
  const ch = DATA.chapters[chapter];
  const many = groupsOf(ch).k > 1;
  list.innerHTML = `<h2>Chapter ${{ch.label}} — verses</h2>` + groupSummary(ch, 'verses') + groupLegend(ch)
    + `<button class="row" id="up"><span class="t">← back to chapters</span><span class="n"></span></button>`
    + `<div class="vhead"><span>verse</span><span>${{many ? 'author' : ''}}</span><span>text</span></div>`
    + ch.v.map((v, i) => `<button class="vrow" data-i="${{i}}"${{groupEdge(ch, vGroupOf(v))}}>
        <span class="vref">${{ch.label}}:${{v[0]}}</span>
        <span class="vtok">${{v[2]}} tok</span>
        <span class="vgrp" style="color:${{many ? G(vGroupOf(v)) : 'var(--faint)'}}">${{many ? A(vGroupOf(v)) : '—'}}
          ${{many ? `<i style="background:${{G(vGroupOf(v))}}"></i>` : ''}}</span>
        <span class="vbody">${{speakerTag(v[5])}}
          <span class="vtext ${{RTL ? 'rtl' : ''}}">${{v[1]}}</span></span>
      </button>`).join('');
  document.getElementById('up').addEventListener('click', () => {{ chapter = null; verse = null; render(); }});
  list.querySelectorAll('.vrow[data-i]').forEach(b => b.addEventListener('click', () => {{
    verse = +b.dataset.i; render(); }}));
  const h = document.createElement('div'); side.innerHTML = ''; side.appendChild(h);
  plot(ch.v.map(v => ({{ label: ch.label + ':' + v[0], xy: vxyOf(v), g: many ? vGroupOf(v) : 0 }})), h, p => {{
    verse = ch.v.findIndex(v => ch.label + ':' + v[0] === p.label); render(); }});
  const bars = document.createElement('div'); side.appendChild(bars);
  bars.innerHTML = groupBars(ch, `verses of chapter ${{ch.label}}`, 'Authors among the verses');
  const mk = document.createElement('div'); side.appendChild(mk);
  if (verse !== null) {{
    const v = ch.v[verse], xy = vxyOf(v);
    mk.innerHTML = `<h2>${{ch.label}}:${{v[0]}}</h2>
      ${{speakerTag(v[5])}}
      <div class="vtext ${{RTL ? 'rtl' : ''}}">${{v[1]}}</div>
      ${{many ? `<div class="mk"><span>author in this chapter</span>
        <span><i style="display:inline-block;width:9px;height:9px;border-radius:2px;
        margin-right:5px;background:${{G(vGroupOf(v))}}"></i>${{A(vGroupOf(v))}} of ${{groupsOf(ch).k}}</span></div>` : ''}}
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

    # What the front page needs to list this language, written beside the page itself. The front page
    # is shared, and rebuilding one language must not drop the others from it; reading this back is
    # how a language that was not rebuilt keeps its place.
    (out / "card.json").write_text(json.dumps({
        "code": language,
        "label": name,
        "verses": data["n_verses"],
        "books": sum(len(c["children"]) for c in data["tree"]),
        "keys": [s["key"] for s in data["strategies"]],
        "groups": {key: {"k": data["book_groups"]["gk"][i], "reason": data["book_groups"]["gr"][i],
                         "ari": data["book_groups"]["gari"][i], "n": data["book_groups"]["gn"],
                         "sizes": data["book_groups"]["gsz"][i]}
                   for i, key in enumerate(data["keys"])},
        "strategies": data["strategies"],
        "group_reasons": data["group_reasons"],
        # How well the assumed-author view does here, where it can be checked at all. Null for every
        # scripture corpus, which is why the front page can say what the procedure is worth only on
        # the one language that carries known authors.
        "authors": {key: data["author_groups"]["af"][i]
                    for i, key in enumerate(data["keys"]) if data["author_groups"]["af"][i]},
        # The assumed-author view as the front page opens it, so that page can report hands where it
        # reports groups instead of quoting a style-group count under an authorship heading.
        "author_groups": {key: _author_card(data, i) for i, key in enumerate(data["keys"])},
    }, ensure_ascii=False), encoding="utf-8")
    return out
