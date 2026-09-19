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
.modebar{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin:0 0 11px}
.mb{font:inherit;font-size:13px;padding:6px 13px;border:1px solid var(--line);border-radius:999px;
background:var(--card);color:var(--muted);cursor:pointer}
.mb.on{background:var(--clay);border-color:var(--clay);color:var(--paper);font-weight:600}
.modebar .small{flex-basis:100%;margin-top:1px}
.actl{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:0 0 11px}
.actl label{font-size:13px;color:var(--muted)}.actl label b{color:var(--ink);font-variant-numeric:tabular-nums}
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

// A style group is a partition of whatever the level lists. `gk` is how many groups the node's
// children fall into under each strategy, `g` is which group a child is in, both flat in `keys` order.
const G = i => (i >= 1 && i <= 8) ? `var(--g${i})` : 'var(--gx)';
// The same names the rest of this project uses for a style group, largest first: A1, A2, ...
const A = i => 'A' + i;
const groupsOf = n => ({ k: n.gk ? n.gk[idx()] : 1, reason: n.gr ? n.gr[idx()] : null,
                         ari: n.gari ? n.gari[idx()] : null, n: n.gn || 0, tokens: n.gtok || 0,
                         sizes: (n.gsz ? n.gsz[idx()] : null) || [] });
const groupOf = c => (c.g ? c.g[idx()] : 1);

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

function groupSummary(node, unit) {
  const g = groupsOf(node);
  if (!g.n) return '';
  const head = g.k > 1
    ? `<span class="gsum"><b>${g.k} style groups</b> among ${g.n} ${unit}` +
      (g.ari != null ? ` · survives resampling at ARI ${g.ari}` : '') + `</span>`
    : `<span class="gsum"><b>1 style group</b> among ${g.n} ${unit} — ` +
      `${(DATA.group_reasons || {})[g.reason] || 'no split is supported'}</span>`;
  return head + shortUnitWarning(g, unit);
}

// A split of units this short can be perfectly stable and still be arithmetic: a hapax ratio over
// eighteen tokens takes few distinct values, so it clusters cleanly. Say so where it applies.
function shortUnitWarning(g, unit) {
  const floor = DATA.reliable_tokens || 1000;
  if (g.k < 2 || !g.tokens || g.tokens >= floor) return '';
  return `<div class="caveat">These ${unit} run about <b>${g.tokens.toLocaleString()} tokens</b> each.
    This project measured that units below ${floor.toLocaleString()} cannot support attribution, which is
    why the analysis pools text into passages first. Read this split as a property of the measure at
    this length, not as a hand.</div>`;
}

function groupLegend(node) {
  const g = groupsOf(node);
  if (g.k < 2) return '';
  return '<div class="legend">' + Array.from({ length: g.k }, (_, i) =>
    `<span><i style="background:${G(i + 1)}"></i>${A(i + 1)}</span>`).join('') + '</div>';
}

// How the units divide between the groups. A1 is always the largest: the clustering ranks them by
// size before naming them, so a long tail of one-member groups is visible at a glance rather than
// hidden inside a count of "5 style groups".
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
  const why = g.k > 1 ? '' :
    `<p class="small">${(DATA.group_reasons || {})[g.reason] || 'no split is supported'}.</p>`;
  return barChart(g.sizes, unit, heading || 'Style groups', why);
}

// The same chart for a collection's books placed in the language-wide grouping, which is the only
// grouping comparable between collections.
function acrossChart(node, label) {
  const sizes = (node.glsz ? node.glsz[idx()] : null) || [];
  return barChart(sizes, `books of ${label}`, "Where its books sit in the language's groups",
    `<p class="small">Groups fitted over every book of the language, so these are comparable
     with the other collections. The chart above groups ${label}'s books among themselves.</p>`);
}

// A chart small enough to sit inside a row, so a list can be read for how each entry divides without
// selecting it first. Same bars as the side panel, tighter.
function rowBars(sizes, caption, lab) {
  lab = lab || A;
  const total = sizes.reduce((a, b) => a + b, 0);
  // One occupied group is an answer - "all nineteen of them in A2" - not an absence. Requiring two
  // left the whole noncanonical collection with no chart at all under the default strategy.
  if (!total) return '';
  const max = Math.max(...sizes);
  return `<span class="rbars"><span class="rcap">${caption}</span>` + sizes.map((c, i) => c ? `
    <span class="rrow">
      <span class="rlab" style="color:${G(i + 1)}">${lab(i + 1)}</span>
      <span class="rtrack"><span class="rfill" style="width:${(c / max * 100).toFixed(1)}%;background:${G(i + 1)}"></span></span>
      <span class="rval">${c.toLocaleString()}</span>
    </span>` : '').join('') + `</span>`;
}

// A collection's books, placed in the grouping of *every* book of the language. This is the only
// comparison that can show whether a grouping follows the collections or cuts across them; a book's
// group within its own collection cannot, because every collection's labels start at A1.
function acrossLanguage(node) {
  const sizes = (node.glsz ? node.glsz[idx()] : null) || [];
  return rowBars(sizes, `its ${sizes.reduce((a, b) => a + b, 0)} books across the language's groups`);
}

// How this node's own children divide: a collection's books among themselves, a book's chapters,
// a chapter's verses.
function ownBars(node, unit) {
  const g = groupsOf(node);
  return g.k > 1 ? rowBars(g.sizes, `its ${g.n} ${unit} in ${g.k} groups`) : '';
}

const groupStack = node => {
  const g = groupsOf(node);
  if (g.k < 2 || !g.sizes.length) return '';
  const total = g.sizes.reduce((a, b) => a + b, 0);
  return '<span class="stack">' + g.sizes.map((c, i) =>
    `<span style="flex:${c};background:${G(i + 1)}" title="${A(i + 1)}: ${c}"></span>`).join('') + '</span>';
};

// The number is on every row beside the swatch: the lighter hues fall under 3:1 against the page,
// so colour is never the only thing carrying which group a unit is in.
const groupBadge = (node, child) =>
  groupsOf(node).k > 1 ? `<span class="gb"><i style="background:${G(groupOf(child))}"></i>${A(groupOf(child))}</span>` : '';
// A book's group among every book of the language. Always defined, and the one that can be compared
// between collections, so it is on the row even when the collection does not split internally.
const langOf = n => (n.gl ? n.gl[idx()] : 0);
const langBadge = (n, language) => langOf(n)
  ? `<span class="gb lg"><i style="background:${G(langOf(n))}"></i>${A(langOf(n))} in ${language}</span>` : '';

// The row itself is tinted, so a list of text can be read down the left edge for where a group changes.
const groupEdge = (node, g) => groupsOf(node).k > 1 ? ` style="border-left-color:${G(g)}"` : '';

// ---- the author view -----------------------------------------------------------------------------
// The same books under a different question. A style group asks "is a split supported"; an assumed
// author asks "if there are N hands, whose is this". `al` carries one partition per count, because
// the count cannot be read off the text: on the English corpus, where the authors are known,
// selecting it by silhouette returned 2 whether the truth was 3, 5 or 13. So it is the reader's to
// set, and everything below is "under an assumption of N", never "there are N".
let mode = (qs.get('m') === 'authors') ? 'authors' : 'style';
const AU = i => 'Author ' + i;
const assumed = {};

const authorsOf = node => ({
  n: (node && node.an) || 0,
  range: (node && node.arange) || [],
  labels: node && node.al ? node.al[idx()] : null,
  fit: node && node.af ? node.af[idx()] : null,
  known: (node && node.aknown) || [],
});

function authorCount(node, key) {
  const a = authorsOf(node);
  if (!a.range.length) return 0;
  const lo = a.range[0], hi = a.range[1];
  if (assumed[key] != null) return Math.min(Math.max(assumed[key], lo), hi);
  // Where the authors are known, open at the truth, so the page shows what the procedure does when
  // it is given every advantage. Where they are not, open at the number of style groups the
  // conservative test supports - the only count the text itself offers.
  const start = a.fit ? a.fit.true_k : Math.max(2, groupsOf(node).k || 2);
  return Math.min(Math.max(start, lo), hi);
}

const authorLabels = (node, key) => {
  const a = authorsOf(node);
  return a.labels ? (a.labels[String(authorCount(node, key))] || null) : null;
};

function authorSizes(node, key) {
  const l = authorLabels(node, key);
  if (!l) return [];
  const n = authorCount(node, key), out = new Array(n).fill(0);
  l.forEach(g => { if (g >= 1 && g <= n) out[g - 1]++; });
  return out;
}

// The slider. Its range runs to as many hands as there are books, because nothing in the text rules
// out any of them; the only stop is how much fits in the page.
function authorControl(node, key) {
  const a = authorsOf(node);
  if (!a.range.length) return '';
  const n = authorCount(node, key);
  return `<div class="actl">
    <label for="an-${key}">Assume <b id="anv-${key}">${n}</b> authors</label>
    <input type="range" id="an-${key}" min="${a.range[0]}" max="${a.range[1]}" value="${n}">
    <span class="small">${a.range[0]}–${a.range[1]} over ${a.n} books</span></div>`;
}

function bindAuthorControl(key, onChange) {
  const el = document.getElementById('an-' + key);
  if (!el) return;
  el.addEventListener('input', e => {
    assumed[key] = +e.target.value;
    const v = document.getElementById('anv-' + key);
    if (v) v.textContent = e.target.value;
  });
  el.addEventListener('change', onChange);
}

// What this view is worth, measured where it can be. Everywhere else it says so instead.
function authorFitNote(node, key) {
  const f = authorsOf(node).fit;
  if (!f) {
    return `<div class="caveat">No author here is known, so nothing on this page can be checked.
      Where the authors <i>are</i> known - the English corpus - this same procedure, handed the right
      number of hands, put most authors in a group that was not mostly theirs. Read the labels below
      as the shape of an assumption, not as attributions.</div>`;
  }
  const n = authorCount(node, key);
  const at = n === f.true_k ? '' :
    `<br><span class="small">You are looking at ${n}; the figures above are for the true ${f.true_k}.</span>`;
  return `<div class="fit"><b>Checked: ${f.n_known} of these works have a known author.</b>
    Given the true count of ${f.true_k}, this strategy puts <b>${f.recovered} of ${f.true_k}</b>
    in a group that is both mostly theirs and mostly no one else's — adjusted Rand ${f.ari}.${at}</div>`;
}

const authorBadge = (node, key, i) => {
  const l = authorLabels(node, key);
  return l && l[i] ? `<span class="gb au"><i style="background:${G(l[i])}"></i>${AU(l[i])}</span>` : '';
};

// The name, where it is actually known. This is the only thing on any of these pages that is not an
// inference, which is why it is marked differently from everything around it.
const knownAuthor = w => w && w.author
  ? `<span class="kn">${w.author}${w.genre ? ' · ' + w.genre : ''}</span>` : '';

function modeToggle(onChange) {
  const el = document.getElementById('mode');
  if (!el) return;
  el.innerHTML = ['style', 'authors'].map(m =>
    `<button class="mb${m === mode ? ' on' : ''}" data-m="${m}">${
      m === 'style' ? 'Style groups' : 'Assumed authors'}</button>`).join('')
    + `<span class="small">${mode === 'style'
      ? 'Splits kept only where they survive resampling. One group means no split was supported.'
      : 'One style, one hand — assumed, not shown. You set how many hands; the text cannot say.'}</span>`;
  el.querySelectorAll('.mb').forEach(b => b.addEventListener('click', () => {
    mode = b.dataset.m;
    const u = new URL(location); u.searchParams.set('m', mode); history.replaceState(null, '', u);
    modeToggle(onChange); onChange();
  }));
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
    page = _head(f"{name} — style explorer")
    page += f"""<div class="crumb"><a href="../index.html">All languages</a></div>
<h1>{name}</h1>
<p class="small">{data['n_verses']:,} verses · {len(data['strategies'])} strategies available"""
    if data["unavailable"]:
        page += f" · no data for: {', '.join(data['unavailable'])}"
    page += """</p>
<div class="bar" id="bar"></div>
<div class="modebar" id="mode"></div>
<div class="headline" id="head"></div>
<div class="cols"><div id="list"></div><div><div class="card" id="side"></div></div></div>
<script>const DATA = """ + json.dumps(slim, ensure_ascii=False) + ";\nconst LANG = " \
        + json.dumps(name, ensure_ascii=False) + ";\n" + JS_COMMON + """
let collection = null;
function render() {
  const list = document.getElementById('list'), side = document.getElementById('side');
  const top = collection === null;
  const nodes = top ? DATA.tree : DATA.tree[collection].children;
  const here = top ? null : DATA.tree[collection];
  // At the top the page lists collections, so that is the partition it reports; the headline above
  // it is the one worth quoting, every book of the language grouped together.
  const parent = top ? DATA.collection_groups : here;
  const unit = top ? 'collections' : 'books';

  const books = groupsOf(DATA.book_groups);
  const acount = authorCount(DATA.author_groups, 'lang');
  document.getElementById('head').innerHTML = mode === 'authors'
    ? `Assuming one style is one hand, the <b>${DATA.author_groups.an} books</b> of this language
       divide among <b>${acount} assumed authors</b>:
       ${authorSizes(DATA.author_groups, 'lang').map((c, i) => `${AU(i + 1)} ${c}`).join(' · ')}.
       The count is yours to set: nothing in the text supplies it.`
    : books.k > 1
    ? `Under this strategy the <b>${books.n} books</b> of this language fall into
       <b>${books.k} style groups</b>${books.ari != null ? ` (stable to ARI ${books.ari})` : ''}:
       ${books.sizes.map((c, i) => `${A(i + 1)} ${c}`).join(' · ')}. ${groupStack(DATA.book_groups)}
       Change the strategy and see whether that survives.`
    : `Under this strategy the <b>${books.n} books</b> of this language do not split:
       ${(DATA.group_reasons || {})[books.reason] || 'no split is supported'}.`;

  list.innerHTML = (top
      ? '<h2>Collections</h2>'
      : `<h2>${DATA.tree[collection].label} — books</h2>`)
    + groupSummary(parent, unit) + groupLegend(parent)
    + (top ? '' : '<button class="row" id="up"><span class="t">← back to collections</span><span class="n"></span></button>')
    + nodes.map((n, i) => {
        // On the books list the edge follows the language-wide group: it is always defined, and it
        // is the only one that means the same thing in another collection.
        const alang = authorLabels(DATA.author_groups, 'lang');
        const mine = (!top && alang && n.bi != null) ? alang[n.bi] : 0;
        const edge = mode === 'authors'
          ? (mine ? ` style="border-left-color:${G(mine)}"` : '')
          : top ? groupEdge(parent, groupOf(n))
                : (langOf(n) ? ` style="border-left-color:${G(langOf(n))}"` : '');
        // In the author view the row carries the language-wide assumed hand, for the same reason the
        // style view carries the language-wide group: a label fitted inside one collection cannot be
        // compared with another collection's, because both start at 1.
        const badge = mode === 'authors'
          ? (mine ? `<span class="gb au"><i style="background:${G(mine)}"></i>${AU(mine)} in ${LANG}</span>` : '')
          : (top ? groupBadge(parent, n) : langBadge(n, LANG));
        return `<button class="row" data-i="${i}"${edge}>
        <span class="t">${n.label}</span>
        <span class="n">${top ? n.n_verses.toLocaleString() + ' verses' : badge}</span>
        <span class="s">${n.code ? n.code + ' · ' + n.n_chapters + ' chapters · ' : ''}${n.n_tokens.toLocaleString()} tokens${top ? '' : ' · ' + n.n_verses.toLocaleString() + ' verses'}</span>
        ${knownAuthor(n)}
        ${mode === 'authors' ? (top ? rowBars(collectionAcrossAuthors(n), `its ${n.children ? n.children.length : 0} books across the language's assumed authors`, AU) : '')
                             : groupBadge(parent, n) + (top ? acrossLanguage(n) + ownBars(n, 'books') : ownBars(n, 'chapters'))}
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
  const many = groupsOf(parent).k > 1;
  const alang2 = authorLabels(DATA.author_groups, 'lang');
  plot(nodes.map(n => ({
    label: n.label, xy: xyOf(n),
    g: mode === 'authors' ? ((!top && alang2 && n.bi != null) ? alang2[n.bi] : 0)
                          : (many ? groupOf(n) : 0) })), holder, null);
  const bars = document.createElement('div');
  side.appendChild(bars);
  // At the top the list is three collections, which is too few to partition and says so; the chart
  // worth showing there is the one the headline reports, every book of the language.
  if (mode === 'authors') {
    const node = top ? DATA.author_groups : here, key = top ? 'lang' : 'c' + collection;
    bars.innerHTML = authorControl(node, key) + authorFitNote(node, key)
      + barChart(authorSizes(node, key), top ? 'books of this language' : `books of ${here.label}`,
                 top ? 'Assumed authors across the language' : `Assumed authors within ${here.label}`,
                 '', 0, AU)
      + (top ? '' : `<p class="small">Fitted among ${here.label}'s books alone, so these numbers
          cannot be compared with another collection's — both start at Author 1. The label on each
          row is the language-wide one, which can.</p>`);
    bindAuthorControl(key, render);
  } else {
  bars.innerHTML = top
    ? groupBars(DATA.book_groups, 'books of this language', 'Style groups across the language')
    : groupBars(parent, `books of ${here.label}`, `Style groups within ${here.label}`)
      + acrossChart(here, here.label);
  }
  const mk = document.createElement('div');
  side.appendChild(mk);
  markerList(here || { markers: {} }, mk);
  if (!here) mk.insertAdjacentHTML('afterbegin', '<p class="small">Each point is a collection. Open one to see its books.</p>');
}
// How a collection's books fall across the language's assumed authors, for the collections list.
function collectionAcrossAuthors(c) {
  const l = authorLabels(DATA.author_groups, 'lang');
  const n = authorCount(DATA.author_groups, 'lang');
  if (!l || !c.children) return [];
  const out = new Array(n).fill(0);
  c.children.forEach(w => { const g = l[w.bi]; if (g >= 1 && g <= n) out[g - 1]++; });
  return out;
}
strategyBar(render); modeToggle(render); render();
</script></div></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")

    # --- one page per book ---------------------------------------------------------------------------
    for collection in data["tree"]:
        for work in collection["children"]:
            payload = {
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
            body = _head(f"{work['label']} — style explorer")
            body += f"""<div class="crumb"><a href="../../index.html">All languages</a> ·
<a href="../index.html">{name}</a> · {collection['label']}</div>
<h1>{work['label']}</h1>
<p class="small">{work['code']} · {work['n_verses']:,} verses · {len(work['children'])} chapters</p>
<div class="bar" id="bar"></div>
<div class="modebar" id="mode"></div>
<div class="headline" id="head"></div>
<div class="cols"><div id="list"></div><div><div class="card" id="side"></div></div></div>
<script>const DATA = """ + json.dumps(payload, ensure_ascii=False) + ";\n" + JS_COMMON + f"""
const RTL = {'true' if rtl else 'false'};
const vGroupOf = v => v[4][idx()];
const mineAcross = () => (DATA.work.gl ? DATA.work.gl[idx()] : 0);
const mineInCollection = () => (DATA.work.g ? DATA.work.g[idx()] : 0);

// Where this book itself sits. Everything below is about its insides; this is the one line that
// places the book among its peers, and without it the page never says which group the book is in.
function placement() {{
  const across = groupsOf(DATA.book_groups), mine = mineAcross();
  const within = groupsOf(DATA.collection_books), inner = mineInCollection();
  let out = across.k > 1 && mine
    ? `Under this strategy this book is <b>${{A(mine)}}</b> of the ${{across.k}} style groups fitted over
       all <b>${{across.n}} books</b> of ${{DATA.language_name}} — a group holding
       ${{across.sizes[mine - 1]}} of them.`
    : `The ${{across.n}} books of ${{DATA.language_name}} do not divide under this strategy:
       ${{(DATA.group_reasons || {{}})[across.reason] || 'no split is supported'}}.`;
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
    bars.innerHTML = groupBars(DATA.work, 'chapters of this book', 'Style groups among the chapters')
      + barChart(groupsOf(DATA.book_groups).sizes, `books of ${{DATA.language_name}}`,
                 'Where this book sits', `<p class="small">Groups fitted over every book of the
                 language. The marked row is this book's.</p>`, mineAcross());
    const mk = document.createElement('div'); side.appendChild(mk); markerList(DATA.work, mk);
    mk.insertAdjacentHTML('afterbegin', '<p class="small">Each point is a chapter of this book.</p>');
    return;
  }}
  const ch = DATA.chapters[chapter];
  const many = groupsOf(ch).k > 1;
  list.innerHTML = `<h2>Chapter ${{ch.label}} — verses</h2>` + groupSummary(ch, 'verses') + groupLegend(ch)
    + `<button class="row" id="up"><span class="t">← back to chapters</span><span class="n"></span></button>`
    + `<div class="vhead"><span>verse</span><span>${{many ? 'group' : ''}}</span><span>text</span></div>`
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
  bars.innerHTML = groupBars(ch, `verses of chapter ${{ch.label}}`, 'Style groups among the verses');
  const mk = document.createElement('div'); side.appendChild(mk);
  if (verse !== null) {{
    const v = ch.v[verse], xy = vxyOf(v);
    mk.innerHTML = `<h2>${{ch.label}}:${{v[0]}}</h2>
      ${{speakerTag(v[5])}}
      <div class="vtext ${{RTL ? 'rtl' : ''}}">${{v[1]}}</div>
      ${{many ? `<div class="mk"><span>style group in this chapter</span>
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
    }, ensure_ascii=False), encoding="utf-8")
    return out
