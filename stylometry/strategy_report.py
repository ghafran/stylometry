"""Run each strategy on its own, all together, and all-but-one; then render a page you can steer.

Three questions a reader should be able to ask of any stylometric result, and this answers each:

* **What can this strategy see by itself?** Every family is run alone, so its solo accuracy is visible
  rather than buried in a combined number.
* **What do they see together?** All of them at once.
* **What is lost without this one?** All-but-one for each, which is the only way to see a family's
  marginal contribution. A family with a good solo score and no marginal contribution is duplicating
  what others already measure.

Every run uses the same units and the same whole-work holdout, so the numbers are comparable.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .strategies import BY_KEY, REGISTRY, build_matrix, describe
from .supervised import classify


def _score(X, labels, works, seed=0) -> dict:
    out = classify(X, labels, works, kind="svm", seed=seed)
    return {"accuracy": round(out["accuracy"], 4), "n": out["n"],
            "n_features": int(X.shape[1])}


def sweep(docs, texts, labels, works, language="grc", pos=None, seed=0) -> dict:
    """Solo, combined and leave-one-out scores for every available strategy."""
    available = [s.key for s in REGISTRY if s.available_for(language)]
    counts = Counter(labels)
    result = {
        "language": language, "n_units": len(docs), "n_works": len(set(works)),
        "n_labels": len(counts),
        "majority_baseline": round(max(counts.values()) / len(labels), 4),
        "chance": round(1 / len(counts), 4),
        "strategies": describe(language),
        "solo": {}, "leave_one_out": {}, "combined": None,
        "unavailable": [s.key for s in REGISTRY if not s.available_for(language)],
    }

    for key in available:
        try:
            X, _, _ = build_matrix([key], docs, texts, language, pos, seed)
        except ValueError:
            result["solo"][key] = {"accuracy": None, "reason": "produced no features"}
            continue
        result["solo"][key] = _score(X, labels, works, seed)

    X_all, names, spans = build_matrix(available, docs, texts, language, pos, seed)
    result["combined"] = _score(X_all, labels, works, seed)
    result["combined"]["strategies"] = available

    for key in available:
        rest = [k for k in available if k != key]
        if not rest:
            continue
        try:
            X, _, _ = build_matrix(rest, docs, texts, language, pos, seed)
        except ValueError:
            continue
        without = _score(X, labels, works, seed)
        result["leave_one_out"][key] = {
            **without,
            "cost_of_removing": round(result["combined"]["accuracy"] - without["accuracy"], 4),
        }
    return result


STATUS_STYLE = {
    "measured": ("#E3EADF", "#33502A"),
    "approximated": ("#F5E9D7", "#6B4E1E"),
    "partial": ("#F5E9D7", "#6B4E1E"),
    "not authorial": ("#E2E7ED", "#2F4A63"),
    "unavailable": ("#EFE0DC", "#6E2C18"),
}


def render_html(result: dict, title: str = "Strategy explorer") -> str:
    """A page where each strategy can be selected and read on its own."""
    payload = json.dumps(result, ensure_ascii=False)
    rows = []
    for s in result["strategies"]:
        bg, fg = STATUS_STYLE.get(s["status"], STATUS_STYLE["measured"])
        rows.append(
            f'<button class="strat" data-key="{s["key"]}" role="listitem">'
            f'<span class="pill" style="background:{bg};color:{fg}">{s["status"]}</span>'
            f'<span class="nm">{s["name"]}</span>'
            f'<span class="ms">{s["measures"]}</span>'
            + (f'<span class="nt">{s["note"]}</span>' if s["note"] else "")
            + '<span class="acc" data-acc></span></button>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--paper:#FAF9F5;--ink:#141413;--muted:#56554F;--faint:#807E76;--line:#DCD9CE;--clay:#A8482A;--card:#fff;--panel:#F3F1E9}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--paper:#16161A;--ink:#EDEBE4;--muted:#B4B1A7;--faint:#93918A;--line:#33322D;--clay:#E0906F;--card:#1F1F23;--panel:#232329}}}}
:root[data-theme="dark"]{{--paper:#16161A;--ink:#EDEBE4;--muted:#B4B1A7;--faint:#93918A;--line:#33322D;--clay:#E0906F;--card:#1F1F23;--panel:#232329}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:'IBM Plex Sans',system-ui,sans-serif;line-height:1.55}}
.wrap{{max-width:1180px;margin:0 auto;padding:44px 16px 72px}}
h1{{font-family:'Spectral',Georgia,serif;font-size:clamp(27px,4.4vw,40px);font-weight:600;margin:0 0 8px;letter-spacing:-.015em}}
.lede{{color:var(--muted);max-width:70ch;margin:0 0 6px}}
.mono{{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums}}
.cols{{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:26px;margin-top:26px}}
@media (max-width:860px){{.cols{{grid-template-columns:1fr}}}}
.modes{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}}
.mode{{font:500 13px/1 'IBM Plex Sans',sans-serif;padding:9px 14px;border:1px solid var(--line);border-radius:999px;background:var(--card);color:var(--ink);cursor:pointer}}
.mode[aria-pressed="true"]{{background:var(--ink);color:var(--paper);border-color:var(--ink)}}
.strat{{display:grid;grid-template-columns:auto 1fr auto;gap:4px 10px;width:100%;text-align:left;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:13px 15px;margin-bottom:9px;cursor:pointer;font:inherit;color:inherit}}
.strat:hover{{border-color:var(--clay)}}
.strat[aria-current="true"]{{border-color:var(--clay);box-shadow:inset 3px 0 0 var(--clay)}}
.pill{{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.06em;text-transform:uppercase;padding:3px 7px;border-radius:4px;white-space:nowrap;align-self:start}}
.nm{{font-weight:600;font-size:15px}}
.ms{{grid-column:2;font-size:13px;color:var(--muted)}}
.nt{{grid-column:2;font-size:12.5px;color:var(--faint);border-left:2px solid var(--line);padding-left:9px;margin-top:4px}}
.acc{{grid-row:1;grid-column:3;font-family:'IBM Plex Mono',monospace;font-size:15px;font-weight:500;color:var(--clay);white-space:nowrap}}
.dim{{opacity:.5}}
.panel{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:22px 24px;position:sticky;top:20px}}
.big{{font-family:'Spectral',Georgia,serif;font-size:60px;font-weight:600;line-height:1;margin:6px 0 2px}}
.bar{{height:9px;background:var(--panel);border-radius:999px;overflow:hidden;margin:16px 0 6px}}
.bar i{{display:block;height:100%;background:var(--clay)}}
.kv{{display:flex;justify-content:space-between;font-size:13.5px;padding:7px 0;border-top:1px solid var(--line)}}
.kv span:last-child{{font-family:'IBM Plex Mono',monospace}}
.note{{background:var(--panel);border-left:3px solid var(--clay);border-radius:0 8px 8px 0;padding:12px 15px;font-size:13px;color:var(--muted);margin-top:16px}}
footer{{margin-top:40px;padding-top:18px;border-top:1px solid var(--line);color:var(--faint);font-size:12.5px}}
</style></head><body><div class="wrap">
<h1>{title}</h1>
<p class="lede">Every strategy run on its own, all together, and all-but-one — same passages, same whole-work holdout, so the numbers compare. Choose a mode, then a strategy.</p>
<p class="lede mono" style="font-size:13px">{result['n_units']} passages · {result['n_works']} works · {result['n_labels']} labels · majority baseline {result['majority_baseline']:.1%} · chance {result['chance']:.1%}</p>

<div class="cols">
 <div>
  <div class="modes" role="group" aria-label="What to show">
    <button class="mode" data-mode="solo" aria-pressed="true">Strategy alone</button>
    <button class="mode" data-mode="loo" aria-pressed="false">What is lost without it</button>
  </div>
  <div role="list">{''.join(rows)}</div>
 </div>
 <div><div class="panel" id="panel"></div></div>
</div>

<footer>Accuracy is leave-one-work-out attribution with a linear SVM. A strategy scoring near the majority baseline has not failed to be implemented — it has failed to distinguish these labels, which is itself a result.</footer>
</div>
<script>
const DATA = {payload};
let mode = 'solo', current = null;
const fmt = v => v === null || v === undefined ? '—' : (v * 100).toFixed(1) + '%';

function accFor(key) {{
  const solo = DATA.solo[key];
  if (mode === 'solo') return solo && solo.accuracy !== null ? solo.accuracy : null;
  const loo = DATA.leave_one_out[key];
  return loo ? loo.cost_of_removing : null;
}}

function paintList() {{
  document.querySelectorAll('.strat').forEach(el => {{
    const key = el.dataset.key;
    const v = accFor(key);
    const cell = el.querySelector('[data-acc]');
    const unavailable = DATA.unavailable.includes(key);
    el.classList.toggle('dim', unavailable);
    el.setAttribute('aria-current', key === current ? 'true' : 'false');
    if (unavailable) {{ cell.textContent = 'n/a'; return; }}
    cell.textContent = mode === 'solo' ? fmt(v)
      : (v === null ? '—' : (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + ' pts');
  }});
}}

function paintPanel() {{
  const p = document.getElementById('panel');
  if (!current) {{
    const c = DATA.combined;
    p.innerHTML = `<div class="mono" style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint)">All strategies together</div>
      <div class="big">${{fmt(c.accuracy)}}</div>
      <div style="color:var(--muted);font-size:13.5px">attribution accuracy over ${{c.n}} passages</div>
      <div class="bar"><i style="width:${{(c.accuracy * 100).toFixed(1)}}%"></i></div>
      <div class="kv"><span>features</span><span>${{c.n_features}}</span></div>
      <div class="kv"><span>majority baseline</span><span>${{fmt(DATA.majority_baseline)}}</span></div>
      <div class="kv"><span>chance</span><span>${{fmt(DATA.chance)}}</span></div>
      <div class="note">Select a strategy on the left to see what it can do alone, or switch mode to see what is lost when it is removed.</div>`;
    return;
  }}
  const s = DATA.strategies.find(x => x.key === current);
  const solo = DATA.solo[current] || {{}};
  const loo = DATA.leave_one_out[current];
  const unavailable = DATA.unavailable.includes(current);
  const acc = solo.accuracy;
  p.innerHTML = `<div class="mono" style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint)">${{s.status}}</div>
    <div style="font-family:Spectral,Georgia,serif;font-size:23px;font-weight:600;margin:4px 0 6px">${{s.name}}</div>
    <div style="color:var(--muted);font-size:13.5px">${{s.measures}}</div>
    ${{unavailable
      ? `<div class="note"><strong>Not available for this corpus.</strong> ${{s.note || ''}}</div>`
      : `<div class="big">${{fmt(acc)}}</div>
         <div style="color:var(--muted);font-size:13.5px">on its own, over ${{solo.n || 0}} passages</div>
         <div class="bar"><i style="width:${{acc ? (acc * 100).toFixed(1) : 0}}%"></i></div>
         <div class="kv"><span>features in this block</span><span>${{solo.n_features ?? '—'}}</span></div>
         <div class="kv"><span>majority baseline</span><span>${{fmt(DATA.majority_baseline)}}</span></div>
         <div class="kv"><span>all strategies together</span><span>${{fmt(DATA.combined.accuracy)}}</span></div>
         ${{loo ? `<div class="kv"><span>without this strategy</span><span>${{fmt(loo.accuracy)}}</span></div>
         <div class="kv"><span>cost of removing it</span><span>${{(loo.cost_of_removing >= 0 ? '+' : '') + (loo.cost_of_removing * 100).toFixed(1)}} pts</span></div>` : ''}}
         ${{s.note ? `<div class="note">${{s.note}}</div>` : ''}}`}}`;
}}

document.querySelectorAll('.mode').forEach(b => b.addEventListener('click', () => {{
  mode = b.dataset.mode;
  document.querySelectorAll('.mode').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
  paintList();
}}));
document.querySelectorAll('.strat').forEach(el => el.addEventListener('click', () => {{
  current = current === el.dataset.key ? null : el.dataset.key;
  paintList(); paintPanel();
}}));
paintList(); paintPanel();
</script></body></html>"""


def save(result: dict, out_dir: str | Path, title: str = "Strategy explorer") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "strategies.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "strategies.html").write_text(render_html(result, title), encoding="utf-8")
    return out
