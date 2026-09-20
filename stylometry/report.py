"""Self-contained, dependency-free exploration of inferred authorship results."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

_VERSE_FIELDS = (
    "id", "language", "collection", "book", "book_title", "chapter", "verse",
    "text", "author_id", "status", "evidence_tokens", "passage_id",
    "distance_margin", "reference_author", "token_count", "has_gap",
)
_ROLLUP_FIELDS = (
    "level", "language", "collection", "book", "book_title", "chapter", "verse", "id",
    "verse_count", "assigned_verse_count", "insufficient_verse_count", "low_evidence_verse_count",
    "author_count", "dominant_author", "author_ids", "author_counts",
)
_UI_VERSE_FIELDS = _VERSE_FIELDS + ("source_reference",)


def _html_payload(result: dict) -> dict:
    """Retain visible results without duplicating every verse as a rollup.

    A shared field list removes repeated property names from the largest table.
    This is a display-only projection: report.json and CSV exports remain full.
    """
    payload = {key: result[key] for key in (
        "schema_version", "config", "languages", "authors", "benchmark",
        "discovery_validation", "source_coverage",
    ) if key in result}
    payload["verse_fields"] = list(_UI_VERSE_FIELDS)
    payload["verse_rows"] = [
        [row.get(field) for field in _UI_VERSE_FIELDS]
        for row in result.get("verses", [])
    ]
    payload["rollups"] = [
        row for row in result.get("rollups", []) if row.get("level") != "verse"
    ]
    return payload


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    # Text is exported verbatim so that references and original text stay intact.
    return value


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def write_report(result: dict, out_dir: Path) -> Path:
    """Write a portable report and machine-readable exports; return index.html.

    JSON is streamed to avoid creating additional full-corpus strings in memory.
    The HTML embeds a compact display projection directly and works from a local
    file without a web server. Full scientific results remain in report.json.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    with (out_dir / "report.json").open("w", encoding="utf-8") as stream:
        for chunk in encoder.iterencode(result):
            stream.write(chunk)
        stream.write("\n")
    _write_csv(out_dir / "verses.csv", _VERSE_FIELDS, result.get("verses", []))
    _write_csv(out_dir / "rollups.csv", _ROLLUP_FIELDS, result.get("rollups", []))
    return write_html_report(result, out_dir)


def write_html_report(result: dict, out_dir: Path) -> Path:
    """Refresh only index.html, preserving existing scientific export files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path = out_dir / "index.html"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(_HTML_START)
        for chunk in encoder.iterencode(_html_payload(result)):
            stream.write(chunk.replace("&", "\\u0026").replace("<", "\\u003c")
                         .replace(">", "\\u003e").replace("\u2028", "\\u2028")
                         .replace("\u2029", "\\u2029"))
        stream.write(_HTML_END)
    return path


_HTML_START = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Authorship Atlas · Text analysis</title>
<style>
:root{color-scheme:light;--ink:#172d35;--muted:#60747b;--line:#dce5e6;--bg:#f3f6f5;--paper:#fff;--teal:#076d69;--soft:#e2f2ee;--amber:#866415}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,input,select{font:inherit}button,select,a{touch-action:manipulation}button{cursor:pointer}button:disabled{cursor:default;opacity:.45}button:focus-visible,a:focus-visible,select:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid #3b9db7;outline-offset:3px}header{padding:32px 42px 26px;border-bottom:1px solid var(--line);background:var(--paper);display:flex;align-items:center;justify-content:space-between;gap:24px}.eyebrow{font-size:11px;font-weight:750;letter-spacing:.18em;text-transform:uppercase;color:var(--teal)}h1{font-size:32px;font-weight:650;letter-spacing:-1.2px;line-height:1.2;margin:7px 0}h2{font-size:21px;letter-spacing:-.5px;margin:0 0 8px}h3{font-size:15px;margin:0 0 8px}p{margin:0 0 12px}.muted{color:var(--muted)}.downloads{display:flex;gap:8px;flex-wrap:wrap}.button,.downloads a{border:1px solid var(--line);background:white;color:var(--ink);border-radius:8px;padding:8px 12px;text-decoration:none;white-space:nowrap}.button:hover,.downloads a:hover{background:#f0f6f4;border-color:#abcac2}.layout{display:grid;grid-template-columns:245px minmax(0,1fr);min-height:calc(100vh - 143px)}aside{padding:28px 22px;border-right:1px solid var(--line);background:#f9fbfa}aside h2{font-size:14px;margin-bottom:16px}label{display:block;font-size:12px;font-weight:600;margin-bottom:6px;color:var(--muted)}select,input[type=search]{width:100%;padding:9px 10px;border:1px solid #c9d6d7;border-radius:7px;background:white;color:var(--ink);min-width:0}.field{margin-bottom:17px}aside .button{width:100%}.scope-note{font-size:12px;margin-top:20px;color:var(--muted)}main{min-width:0;padding:28px 34px 48px;max-width:1800px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin-bottom:18px}.metric{border:1px solid var(--line);background:var(--paper);border-radius:12px;padding:17px 19px}.metric .value{font-size:29px;letter-spacing:-1px;line-height:1.2;margin:5px 0}.metric .label{font-size:12px;color:var(--muted)}.metric .hint{font-size:11px;color:var(--muted)}.notice{background:#e8f1ef;border-left:3px solid #5b9e90;padding:12px 15px;border-radius:0 7px 7px 0;font-size:12px;color:#385c58;margin:0 0 23px}.tabs{display:flex;gap:23px;border-bottom:1px solid #cbdada;margin-bottom:22px;overflow:auto}.tab{background:none;border:0;color:var(--muted);padding:10px 0 13px;white-space:nowrap;border-bottom:3px solid transparent;font-weight:600}.tab[aria-selected=true]{color:var(--teal);border-bottom-color:var(--teal)}.panel[hidden]{display:none}.panel-heading{display:flex;align-items:center;justify-content:space-between;gap:15px;margin-bottom:17px}.panel-heading p{margin:0;font-size:12px}.count{font-size:12px;color:var(--muted)}.toolbar{display:flex;align-items:end;gap:12px;margin-bottom:17px}.toolbar .search{flex:1}.toolbar .size{width:100px}.panelbox{border:1px solid var(--line);background:white;border-radius:11px;overflow:hidden}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;text-align:left;font-size:13px}th{padding:12px 16px;font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);background:#f9fbfa;border-bottom:1px solid var(--line);white-space:nowrap}td{padding:14px 16px;border-bottom:1px solid #e8eeee;vertical-align:top}tbody tr:last-child td{border-bottom:0}.text-table .ref{min-width:185px;width:24%}.text-table .tag{min-width:170px;width:19%}.text-table .content{min-width:260px}.ref-title{font-weight:600}.ref-meta{font-size:11px;color:var(--muted);margin-top:3px;overflow-wrap:anywhere}.badge{display:inline-flex;border-radius:5px;padding:3px 7px;font-size:11px;color:#205e52;background:#e8f4ef;white-space:nowrap}.badge.low{background:#faf1d9;color:#7b5d19}.badge.none{background:#edf0f1;color:#647078}.author-link{color:var(--teal);border:0;background:none;padding:0;font:inherit;font-weight:600;text-align:left;overflow-wrap:anywhere}.author-link:hover{text-decoration:underline}.tiny{font-size:11px;color:var(--muted);margin-top:5px}.verse-text{font-family:Georgia,"Times New Roman",serif;font-size:15px;line-height:1.65;white-space:pre-wrap;overflow-wrap:anywhere}.excerpt{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}.verse-detail{margin-top:7px}.verse-detail summary{font-size:11px;color:var(--teal);cursor:pointer}.verse-detail[open] .full{margin-top:8px}.truth{font-family:inherit;font-size:11px;border-top:1px dashed var(--line);padding-top:7px;margin-top:9px;color:var(--muted)}.pagination{display:flex;align-items:center;justify-content:space-between;padding:13px 16px;background:#fcfdfc;gap:12px}.page-buttons{display:flex;gap:8px}.empty{text-align:center;padding:54px 20px;color:var(--muted)}.empty strong{display:block;font-size:17px;color:var(--ink);margin-bottom:7px}.hierarchy-level{width:165px}.trail{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted);flex-wrap:wrap;margin-bottom:14px}.trail button{background:none;border:0;color:var(--teal);padding:2px;font-size:12px}.author-pills{display:flex;gap:5px;flex-wrap:wrap}.pill{font-size:11px;padding:2px 6px;background:var(--soft);border-radius:4px}.language-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:15px}.language-card,.method-card{border:1px solid var(--line);border-radius:10px;background:white;padding:20px}.language-card .number{font-size:32px;line-height:1.2;letter-spacing:-1px;margin:10px 0 3px}.language-card details{font-size:12px;color:var(--muted);margin-top:12px}.language-card summary{cursor:pointer}.method-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}.method-card p{font-size:13px;color:var(--muted)}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f4;padding:15px;border-radius:7px;font:12px/1.6 ui-monospace,monospace;color:#34515a;max-height:480px;overflow:auto}.benchmark-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:17px 0}.benchmark-value{font-size:27px;line-height:1.4}.footer-note{font-size:11px;color:var(--muted);margin-top:22px}noscript{display:block;padding:30px}.loading{padding:40px;color:var(--muted)}@media(max-width:1100px){header{padding:25px}.layout{grid-template-columns:205px minmax(0,1fr)}main{padding:25px 20px}.metrics{grid-template-columns:repeat(2,1fr)}.method-grid{grid-template-columns:1fr}}@media(max-width:740px){header{display:block;padding:22px}.downloads{margin-top:17px}.layout{display:block}aside{border-right:0;border-bottom:1px solid var(--line);padding:18px;display:grid;grid-template-columns:1fr 1fr;gap:10px 14px}aside h2,aside .scope-note{grid-column:1/-1;margin:0}.field{margin:0}aside .button{align-self:end}main{padding:20px 16px}.metric{padding:12px}.metric .value{font-size:25px}.tabs{gap:17px}.panel-heading{align-items:start}.toolbar{flex-wrap:wrap}.pagination{flex-wrap:wrap}.method-grid{grid-template-columns:1fr}h1{font-size:28px}}

.contribution-toolbar{display:flex;justify-content:space-between;align-items:end;gap:20px;margin:18px 0}.contribution-toolbar .measure{width:190px;flex:none}.contribution-scope{font-weight:600;color:var(--teal);margin-top:6px}.contribution-card{border:1px solid var(--line);background:var(--paper);border-radius:12px;padding:22px;margin-bottom:20px}.contribution-heading{display:flex;justify-content:space-between;align-items:baseline;gap:16px;margin-bottom:14px}.contribution-heading h3{font-size:18px;margin:0}.contribution-axis,.contribution-row{display:grid;grid-template-columns:130px minmax(100px,1fr) 170px;gap:14px;align-items:center}.contribution-axis{font-size:11px;color:var(--muted);margin-bottom:8px}.contribution-ticks{display:flex;justify-content:space-between}.contribution-axis>span:last-child{text-align:right}.contribution-bars{max-height:570px;overflow-y:auto;scrollbar-gutter:stable}.contribution-row{padding:9px 0;min-height:43px}.contribution-row.selected{background:#eaf5f1;border-radius:5px}.contribution-row.dimmed{opacity:.5}.contribution-label{font-size:12px;overflow-wrap:anywhere}.contribution-track{height:18px;background:#eef3f2;border-radius:4px;position:relative;overflow:hidden;background-image:linear-gradient(to right,transparent calc(100% - 1px),#d8e3e0 0);background-size:25% 100%}.contribution-fill{height:100%;border-radius:4px;min-width:0}.contribution-value{font-size:12px;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.contribution-value .share{color:var(--muted);display:inline-block;min-width:55px;margin-left:10px}.contribution-note{font-size:12px;color:var(--muted);margin:10px 0 0}.contribution-empty{padding:30px 10px;color:var(--muted)}
@media(max-width:740px){.contribution-toolbar{align-items:start;flex-direction:column}.contribution-card{padding:16px 12px}.contribution-axis,.contribution-row{grid-template-columns:90px minmax(65px,1fr) 110px;gap:8px}.contribution-value{font-size:11px}.contribution-value .share{min-width:43px;margin-left:4px}.contribution-heading{align-items:start;flex-direction:column;gap:5px}}
@media(max-width:400px){.contribution-axis,.contribution-row{grid-template-columns:75px minmax(0,1fr) 90px;gap:6px}.contribution-value .share{display:block;margin-left:0}.contribution-ticks span:nth-child(even){display:none}}
.trail[hidden]{display:none}
.map-toolbar{display:flex;gap:18px;align-items:end;flex-wrap:wrap;margin:15px 0}.map-toolbar>div{width:190px}.map-toolbar p{margin:0 0 8px;flex:1;min-width:180px;font-size:12px}.map-explanation{font-size:12px;color:var(--muted)}.map-row{border:1px solid var(--line);border-radius:10px;background:white;padding:16px 18px;margin:12px 0}.map-row-heading{display:flex;justify-content:space-between;gap:14px;align-items:start}.map-title{font-size:15px}.map-amount{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;font-size:13px}.map-stack{display:flex;width:100%;height:30px;overflow:hidden;border:0;padding:0;background:#e8eeed;border-radius:5px;margin:13px 0 9px;cursor:pointer}.map-stack:hover{box-shadow:0 0 0 2px #79aba0}.map-segment{height:100%;min-width:0;transition:opacity .15s}.map-segment.dimmed{opacity:.25}.map-preview{font-size:11px;color:var(--muted);line-height:1.7;overflow-wrap:anywhere}.map-preview .swatch{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px}.map-preview>span{margin-right:12px;display:inline-block}.map-breakdown{font-size:12px;margin-top:8px}.map-breakdown>summary{cursor:pointer;color:var(--teal)}.map-breakdown table{margin-top:10px;font-size:12px}.map-breakdown th,.map-breakdown td{padding:8px}.map-totals{margin-top:22px;border-top:1px solid var(--line);padding-top:18px}.map-totals>summary{cursor:pointer;font-weight:600;color:var(--teal)}.map-totals .contribution-scope{margin-top:14px}.map-leaf{padding:22px;background:white;border:1px solid var(--line);border-radius:10px;margin:16px 0}.map-leaf .verse-text{margin:18px 0}.map-zero{font-size:12px;color:var(--muted);margin:12px 0}.map-rows:empty{display:none}#map-children[hidden]{display:none}@media(max-width:740px){.map-toolbar{gap:12px}.map-toolbar>div{flex:1;min-width:120px}.map-row{padding:13px 12px}.map-row-heading{flex-wrap:wrap}.map-amount{text-align:left}.map-leaf{padding:15px}#panel-hierarchy .panel-heading{flex-direction:column;align-items:stretch}}
.works-picker{max-width:360px;margin:18px 0}.works-overview{border:1px solid var(--line);border-radius:11px;background:white;padding:20px;margin-bottom:20px}.works-overview h3{font-size:19px}.works-counts{display:flex;gap:24px;flex-wrap:wrap;margin-top:14px}.works-counts strong{display:block;font-size:23px;font-weight:600}.works-counts span{font-size:12px;color:var(--muted)}.works-branch{border:1px solid var(--line);border-radius:8px;background:white;margin:10px 0;overflow:hidden}.works-branch>summary{cursor:pointer;padding:13px 16px;color:var(--ink);font-weight:600;overflow-wrap:anywhere}.works-branch>summary:hover{background:#f0f6f4}.works-branch>summary .tiny{display:inline;font-weight:400;margin-left:12px}.works-children{padding:0 14px 8px 20px}.works-chapter>.works-children{padding:0}.works-chapter .text-table .ref{min-width:160px}.works-chapter .text-table .tag{min-width:140px}.works-caption{font-size:12px;color:var(--muted);margin:0 0 10px}.works-more{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 16px;flex-wrap:wrap}@media(max-width:740px){.works-children{padding:0 7px 5px 9px}.works-branch>summary{padding:11px}.works-branch>summary .tiny{display:block;margin-left:0}.works-counts{gap:15px}}
</style>
</head>
<body>
<header><div><div class="eyebrow">Stylometry workspace</div><h1>Authorship Atlas</h1><p class="muted">Explore the voices behind the text.</p></div><nav class="downloads" aria-label="Download analysis"><a href="verses.csv" download>Verse tags ↓</a><a href="rollups.csv" download>Rollups ↓</a><a href="report.json" download>Full analysis ↓</a></nav></header>
<div class="layout">
<aside aria-label="Filter the corpus"><h2>Explore your corpus</h2>
<div class="field"><label for="language">Language</label><select id="language"></select></div>
<div class="field"><label for="collection">Collection</label><select id="collection"></select></div>
<div class="field"><label for="book">Book</label><select id="book"></select></div>
<div class="field"><label for="chapter">Chapter</label><select id="chapter"></select></div>
<div class="field"><label for="author">Inferred author</label><select id="author"></select></div>
<div class="field"><label for="status">Evidence status</label><select id="status"><option value="">All evidence</option value="assigned">Assigned</option><option value="low_evidence">Low evidence</option><option value="insufficient_text">Insufficient text</option></select></div>
<button type="button" class="button" id="reset">Reset filters</button><p class="scope-note">These filters apply to Text explorer, Authorship map, Author groups, and By author. Opening a result keeps your filters. Broaden them here to explore more text.</p>
</aside>
<main><noscript>This report needs JavaScript for interactive exploration. Download the CSV files or report.json to read every result.</noscript>
<div class="metrics" id="metrics" aria-live="polite"></div>
<p class="muted" id="source-scope" hidden></p>
<div class="notice">These are <strong>estimated style groups</strong>, not verified identities. Tags are inferred from passages and inherited by their verses or paragraphs. Low evidence and insufficient text remain visible. Reference authors are used only for validation.</div>
<div id="estimate-warnings" aria-live="polite"></div>
<nav class="tabs" role="tablist" aria-label="Analysis views"><button type="button" class="tab" id="tab-text" role="tab" aria-controls="panel-text" aria-selected="true" data-tab="text">Text explorer</button><button type="button" class="tab" id="tab-hierarchy" role="tab" aria-controls="panel-hierarchy" aria-selected="false" data-tab="hierarchy">Authorship map</button><button type="button" class="tab" id="tab-authors" role="tab" aria-controls="panel-authors" aria-selected="false" data-tab="authors">Author groups</button><button type="button" class="tab" id="tab-works" role="tab" aria-controls="panel-works" aria-selected="false" data-tab="works">By author</button><button type="button" class="tab" id="tab-benchmark" role="tab" aria-controls="panel-benchmark" aria-selected="false" data-tab="benchmark">English validation</button><button type="button" class="tab" id="tab-method" role="tab" aria-controls="panel-method" aria-selected="false" data-tab="method">About the analysis</button></nav>
<div id="selection-summary" class="trail" aria-live="polite"></div>
<section class="panel" id="panel-text" role="tabpanel" aria-labelledby="tab-text"><div class="panel-heading"><div><h2>Every text, an attribution</h2><p class="muted">Inspect verse tags, supporting evidence, and matching voices.</p></div></div><div class="toolbar"><div class="search"><label for="search">Search text, title, reference, or inferred author</label><input id="search" type="search" placeholder="Search the selected corpus…"></div><div class="size"><label for="page-size">Per page</label><select id="page-size"><option>25</option><option selected>50</option><option>100</option></select></div></div><div class="panelbox"><div class="table-wrap"><table class="text-table"><thead><tr><th>Text reference</th><th>Inferred author</th><th>Verse / paragraph</th></tr></thead><tbody id="verse-rows"></tbody></table></div><div id="verse-empty" hidden class="empty"><strong>No matching text</strong>Try a broader filter or a different search.</div><div class="pagination"><span class="count" id="verse-count" aria-live="polite"></span><div class="page-buttons"><button type="button" class="button" id="previous">← Previous</button><button type="button" class="button" id="next">Next →</button></div></div></div></section>
<section class="panel" id="panel-hierarchy" role="tabpanel" aria-labelledby="tab-hierarchy" hidden>
<div class="panel-heading"><div><h2>Authorship map</h2><p class="muted">See who contributed at every level. Select a bar to explore its books, chapters, and verses.</p></div><button type="button" class="button" id="map-open-text">Open matching text</button></div>
<nav id="trail" class="trail" aria-label="Map location"></nav>
<div class="map-toolbar"><div><label for="level">Group by</label><select id="level"><option value="language">Language</option><option value="collection">Collection</option><option value="book">Book</option><option value="chapter">Chapter</option><option value="verse">Verse / paragraph</option></select></div><div><label for="contribution-measure">Measure contribution by</label><select id="contribution-measure"><option value="words">Words</option><option value="verses">Verses / paragraphs</option></select></div><p id="map-summary" class="muted" aria-live="polite"></p></div>
<p class="map-explanation">Each colored segment is an inferred author's share of that row. Counts include unassigned text and honor all corpus filters and search.</p>
<div id="map-highlight" class="trail" hidden></div>
<div id="map-detail"></div>
<div id="map-children"><h3 id="map-level-title"></h3><div id="rollup-rows" class="map-rows"></div><div id="rollup-empty" hidden class="empty"><strong>No matching text</strong>Broaden the corpus filters or clear the search.</div><div class="pagination"><span class="count" id="rollup-count"></span><div class="page-buttons"><button type="button" class="button" id="rollup-previous">← Previous</button><button type="button" class="button" id="rollup-next">Next →</button></div></div></div>
<details class="map-totals"><summary>Author totals for this selection</summary><p class="contribution-scope" id="contribution-scope"></p><div id="contribution-charts" aria-live="polite"></div></details>
<p class="footer-note">Open an author breakdown to see exact amounts and percentages. Words count each verse's own text once. Graph navigation stays within the sidebar filters; breadcrumbs return to earlier graph levels.</p>
</section>
<section class="panel" id="panel-authors" role="tabpanel" aria-labelledby="tab-authors" hidden><div class="panel-heading"><div><h2>Follow an inferred voice</h2><p class="muted">Author groups represented in your filtered corpus. Select a group to see its matching text.</p></div></div><div class="panelbox"><div class="table-wrap"><table><thead><tr><th>Inferred author</th><th>Language</th><th>Verses / paragraphs</th><th>Books</th><th>Collections</th><th>Words</th><th>Explore</th></tr></thead><tbody id="author-rows"></tbody></table></div><div id="author-empty" hidden class="empty"><strong>No inferred author groups</strong>No assigned text matches the current filters.</div><div class="pagination"><span class="count" id="author-count"></span><div class="page-buttons"><button type="button" class="button" id="author-previous">← Previous</button><button type="button" class="button" id="author-next">Next →</button></div></div></div><p class="footer-note">Counts use only matching text in the current corpus filters and search. Words count each verse’s own text once.</p></section>
<section class="panel" id="panel-works" role="tabpanel" aria-labelledby="tab-works" hidden>
<div class="panel-heading"><div><h2>Text by author</h2><p class="muted">Follow one inferred author through every collection, book, chapter, and verse in your filtered corpus.</p></div></div>
<div class="works-picker"><label for="works-author">Browse author</label><select id="works-author"></select></div>
<div id="works-overview" aria-live="polite"></div><div id="works-tree"></div>
<p class="footer-note">Open a collection, then a book and chapter to read its assigned verses or paragraphs. All corpus filters and search remain active. Clear filters in the sidebar to see an author's wider coverage.</p>
</section>
<section class="panel" id="panel-benchmark" role="tabpanel" aria-labelledby="tab-benchmark" hidden><h2>The English reference test</h2><p class="muted">Compare inferred groups with the known English author labels. The reference corpus has 13 expected authors when complete.</p><div id="benchmark-content"></div></section>
<section class="panel" id="panel-method" role="tabpanel" aria-labelledby="tab-method" hidden><h2>Estimates, with evidence</h2><p class="muted">A separate authorship model is fitted for each language.</p><div id="source-coverage"></div><div id="language-cards" class="language-grid"></div><div class="method-grid"><article class="method-card"><h3>How to read a tag</h3><p>An author tag identifies an inferred writing-style group. It does not establish a historical author’s identity. The same tag in one language points to text assigned to the same group, including across collections.</p><p>Short verses and English paragraphs borrow evidence from their surrounding passage. A passage can contain several writers; its inherited tag can miss a change within that passage.</p></article><article class="method-card"><h3>How to read uncertainty</h3><p>“Low evidence” marks an attribution that needs caution. “Insufficient text” has no supported attribution. A distance margin compares a passage’s distance to its assigned group’s center with its nearest alternative center; it is not a probability of authorship.</p><p>Genre, translation, period, editing, and shared quotations can produce style differences or similarities. Estimated group counts therefore need external validation.</p></article></div><details class="method-card"><summary>Analysis configuration</summary><pre id="configuration"></pre></details></section>
<p class="footer-note">Portable report · Works offline · Use the CSV exports for complete verse-level and hierarchy results.</p>
</main></div>
<script type="application/json" id="report-data">'''

_HTML_END = r'''</script>
<script>
'use strict';
const data = JSON.parse(document.getElementById('report-data').textContent);
document.getElementById('report-data').remove();
const verses = (data.verse_rows || []).map(values => {const row={};data.verse_fields.forEach((field,index)=>{row[field]=values[index];});return row;});
delete data.verse_rows;
const authors = data.authors || [], languages = data.languages || [];
const $ = id => document.getElementById(id);
const count = value => new Intl.NumberFormat().format(value || 0);
const str = value => value == null ? '' : String(value);
const state = {language:'', collection:'', book:'', chapter:'', author:'', status:'', query:'', focusId:'', page:0, rollupPage:0, authorPage:0, size:50, level:'language', tab:'text'};
let filtered = verses, filteredRollups = [], filteredAuthors = [];
let worksAuthorKey='', worksSource=null;
let mapSource=null,mapPath=[],mapRows=[],mapHighlight=null;
const mapPageSize=12;
function el(tag, text, cls) { const node = document.createElement(tag); if(text != null) node.textContent = str(text); if(cls) node.className = cls; return node; }
function unique(rows, key) { return Array.from(new Set(rows.map(row => str(row[key])).filter(Boolean))).sort((a,b) => a.localeCompare(b,undefined,{numeric:true})); }
function selectOptions(id, values, label, display) { const select = $(id); select.replaceChildren(); const all = el('option', label); all.value=''; select.append(all); if(state[id] && !values.includes(state[id]))values=[...values,state[id]]; for(const value of values){ const option=el('option',display ? display(value) : value); option.value=value; select.append(option); } select.value=state[id]; }
function scope(row, through='chapter') { const keys=['language','collection','book','chapter']; for(const key of keys){ if(state[key] && str(row[key])!==state[key]) return false; if(key===through) break; } return true; }
function selectCorpus(rows,selection){
 const q=str(selection.query).trim().toLocaleLowerCase();
 return rows.filter(row=>['language','collection','book','chapter'].every(key=>!selection[key] || str(row[key])===str(selection[key]))
  && (!selection.author || str(row.author_id)===str(selection.author))
  && (!selection.status || row.status===selection.status)
  && (!selection.focusId || str(row.id)===str(selection.focusId))
  && (!q || [row.text,row.book_title,row.book,row.id,row.author_id].some(value=>str(value).toLocaleLowerCase().includes(q))));
}
function authorSelection(selection,author,language){
 return {...selection,language:selection.language || str(language),author:str(author),focusId:''};
}
function drillSelection(selection,row){
 const next={...selection,focusId:row.level==='verse'?str(row.id):''};
 for(const key of ['language','collection','book','chapter'])if(!next[key] && row[key]!=null)next[key]=str(row[key]);
 const levels=['language','collection','book','chapter','verse'];
 next.level=levels[levels.indexOf(row.level)+1] || 'verse';
 return next;
}
function hierarchySelection(selection,level){
 return {...selection,level,focusId:''};
}
function refreshOptions(){
 selectOptions('language',unique(verses,'language'),'All languages');
 const inLanguage=verses.filter(row => !state.language || str(row.language)===state.language);
 selectOptions('collection',unique(inLanguage,'collection'),'All collections');
 const inCollection=inLanguage.filter(row => !state.collection || str(row.collection)===state.collection);
 const titles=new Map(inCollection.map(row=>[str(row.book),str(row.book_title || row.book)]));
 selectOptions('book',unique(inCollection,'book'),'All books',value => titles.get(value) || value);
 const inBook=inCollection.filter(row=> !state.book || str(row.book)===state.book);
 selectOptions('chapter',unique(inBook,'chapter'),'All chapters');
 selectOptions('author',unique(inBook.filter(row=>(!state.chapter || str(row.chapter)===state.chapter)&&(!state.status || row.status===state.status)),'author_id'),'All inferred authors');
 $('status').value=state.status;
}
function authorButton(author,language){ if(!author) return el('span','Unassigned','muted'); const button=el('button',author,'author-link'); button.type='button'; button.title='Find text assigned to this group within the current corpus filters'; button.addEventListener('click',()=>{Object.assign(state,authorSelection(state,author,language));refreshOptions();resetPages();refresh();activate('text');}); return button; }
function metric(value,label,hint){const node=el('div',null,'metric');node.append(el('div',label,'label'),el('div',value,'value'),el('div',hint,'hint'));return node;}
function contributionSummary(rows,selection,measure='words'){
 const groups=new Map();
 for(const row of rows){
  if(['language','collection','book','chapter'].some(key=>selection[key] && String(row[key])!==String(selection[key])))continue;
  const language=String(row.language);
  if(!groups.has(language))groups.set(language,{language,total:0,units:0,missingWords:0,authors:new Map()});
  const group=groups.get(language),author=row.author_id || null;
  const knownWords=typeof row.token_count==='number' && Number.isFinite(row.token_count) && row.token_count>=0;
  const amount=measure==='verses'?1:knownWords?row.token_count:0;
  if(!group.authors.has(author))group.authors.set(author,{author,amount:0,units:0,lowEvidenceUnits:0});
  const entry=group.authors.get(author);
  entry.amount+=amount;entry.units++;entry.lowEvidenceUnits+=row.status==='low_evidence'?1:0;
  group.total+=amount;group.units++;group.missingWords+=knownWords?0:1;
 }
 return Array.from(groups.values()).sort((a,b)=>a.language.localeCompare(b.language)).map(group=>({
  ...group,authors:Array.from(group.authors.values()).sort((a,b)=>{
   if(a.author===null)return b.author===null?0:1;
   if(b.author===null)return -1;
   return b.amount-a.amount || a.author.localeCompare(b.author);
  }).map(entry=>({...entry,share:group.total?entry.amount/group.total:0}))
 }));
}
function contributionColor(language,author){
 if(!author)return '#899b9e';
 let hash=0;for(const letter of language+':'+author)hash=((hash<<5)-hash+letter.charCodeAt(0))|0;
 return 'hsl('+((Math.abs(hash)*137.508)%360).toFixed(1)+' 43% 42%)';
}
function renderContributions(){
 const target=$('contribution-charts'),measure=$('contribution-measure').value;
 const groups=contributionSummary(mapRows,{},measure),unit=measure==='words'?'words':'verses / paragraphs';
 const names={eng:'English',grc:'Greek',hbo:'Hebrew',arb:'Arabic'};
 $('contribution-scope').textContent=mapPath.length?mapPath.map(mapTitle).join(' › '):'Current corpus selection';
 target.replaceChildren();
 if(!groups.length){target.append(el('div','No text in this selection. Choose another language, collection, book or chapter.','panelbox contribution-empty'));return;}
 for(const group of groups){
  const card=el('article',null,'contribution-card'),heading=el('div',null,'contribution-heading');
  heading.append(el('h3',names[group.language] || group.language),el('span',count(group.total)+' '+unit+' in selection','muted'));
  card.append(heading);
  if(measure==='words' && group.missingWords){
   card.append(el('p',count(group.missingWords)+' text units have no word counts. Switch to verses / paragraphs for complete contribution shares.','notice'));
   target.append(card);continue;
  }
  if(!group.total){card.append(el('div','No words to chart in this selection. Switch to verses / paragraphs to include empty text units.','contribution-empty'));target.append(card);continue;}
  const axis=el('div',null,'contribution-axis'),ticks=el('div',null,'contribution-ticks');
  for(const value of ['0%','25%','50%','75%','100%'])ticks.append(el('span',value));
  axis.append(el('span','Inferred author'),ticks,el('span','Amount · share'));card.append(axis);
  const bars=el('div',null,'contribution-bars');bars.setAttribute('role','list');
  bars.setAttribute('aria-label',(names[group.language] || group.language)+' author contributions by '+unit);
  for(const entry of group.authors){
   const selected=mapHighlight?.author===entry.author && mapHighlight?.language===group.language;
   const row=el('div',null,'contribution-row'+(selected?' selected':mapHighlight?' dimmed':''));
   row.setAttribute('role','listitem');row.dataset.author=entry.author || '';row.dataset.language=group.language;
   const percentage=(entry.share*100).toFixed(1)+'%',label=entry.author || 'Unassigned';
   row.setAttribute('aria-label',label+': '+count(entry.amount)+' '+unit+', '+percentage+' of selected '+group.language+' text');
   const authorLabel=el('div',null,'contribution-label');
   if(entry.author){
    const button=el('button',entry.author,'author-link');button.type='button';
    button.setAttribute('aria-pressed',String(selected));button.title='Highlight this author without changing the selected text';
    button.addEventListener('click',()=>{mapHighlight=selected?null:{author:entry.author,language:group.language};renderRollups();});
    authorLabel.append(button);
   }else authorLabel.append(el('span','Unassigned','muted'));
   const track=el('div',null,'contribution-track'),fill=el('div',null,'contribution-fill');
   fill.style.width=(entry.share*100)+'%';fill.style.backgroundColor=contributionColor(group.language,entry.author);
   track.setAttribute('aria-hidden','true');track.append(fill);
   const value=el('div',null,'contribution-value');value.append(el('span',count(entry.amount)),el('span',percentage,'share'));
   row.append(authorLabel,track,value);bars.append(row);
  }
  card.append(bars,el('p','Each bar is a share of all '+unit+' in the current selection for this language.','contribution-note'));
  target.append(card);
 }
}
function renderMetrics(){ const groups=new Set(filtered.filter(row=>row.author_id).map(row=>str(row.language)+'\u0000'+row.author_id));const tagged=filtered.filter(row=>row.author_id).length; const insufficient=filtered.filter(row=>row.status==='insufficient_text').length; const langCount=new Set(filtered.map(row=>row.language)).size;$('metrics').replaceChildren(metric(count(filtered.length),'Verses / paragraphs','In the current text selection'),metric(count(groups.size),'Estimated authors',langCount>1?'Language-specific groups, summed':'Groups represented in this selection'),metric(count(tagged),'Tagged text units',filtered.length ? (100*tagged/filtered.length).toFixed(1)+'% of selected text' : 'No text in this selection'),metric(count(insufficient),'Insufficient text','Text without enough supporting evidence')); }
function reference(row){const english=/^(en|eng|english)$/i.test(str(row.language));return (row.book_title || row.book || row.collection || row.language || 'Text')+' · '+(row.chapter == null ? '' : str(row.chapter)+':')+(english?'¶ ':'')+str(row.verse);}
function verseRow(row){const tr=el('tr');const ref=el('td',null,'ref');ref.append(el('div',reference(row),'ref-title'),el('div',str(row.language)+' / '+str(row.collection),'ref-meta'),el('div',row.id,'ref-meta'));const tag=el('td',null,'tag');tag.append(authorButton(row.author_id,row.language));const status=row.status || (row.author_id?'assigned':'insufficient_text');tag.append(el('div',null,'tiny'));tag.lastChild.append(el('span',status==='low_evidence'?'Low evidence':status==='insufficient_text'?'Insufficient text':'Assigned','badge '+(status==='low_evidence'?'low':status==='insufficient_text'?'none':'')));tag.append(el('div',count(row.evidence_tokens)+' passage tokens','tiny'));if(row.distance_margin!=null)tag.append(el('div','Distance margin: '+Number(row.distance_margin).toFixed(3),'tiny'));const content=el('td',null,'content');const preview=el('div',row.text || '','verse-text excerpt');preview.dir='auto';content.append(preview);const details=el('details',null,'verse-detail');details.append(el('summary','Full text & evidence'));const full=el('div',row.text || '','verse-text full');full.dir='auto';details.append(full,el('div','Passage: '+(row.passage_id || 'No passage assigned'),'tiny'));if(row.reference_author)details.append(el('div','Reference author (validation only): '+row.reference_author,'truth'));content.append(details);tr.append(ref,tag,content);return tr;}
function pageInfo(total,page,size){return total ? count(page*size+1)+'–'+count(Math.min((page+1)*size,total))+' of '+count(total) : '0 results';}
function renderVerses(){ const start=state.page*state.size;const fragment=document.createDocumentFragment();filtered.slice(start,start+state.size).forEach(row=>fragment.append(verseRow(row)));$('verse-rows').replaceChildren(fragment);$('verse-empty').hidden=filtered.length>0;$('verse-count').textContent=pageInfo(filtered.length,state.page,state.size);$('previous').disabled=state.page===0;$('next').disabled=(state.page+1)*state.size>=filtered.length;}
function resetPages(){state.page=0;state.rollupPage=0;state.authorPage=0;}
function renderEstimateWarnings(){
 const target=$('estimate-warnings');target.replaceChildren();
 const limited=[],unstable=[],unestablished=[];
 for(const language of languages){if(state.language && str(language.language)!==state.language)continue;const selection=language.selection || {};if(selection.at_search_limit)limited.push(language.language);if(selection.stable===false)unstable.push(language.language);if(selection.stable==null && language.estimated_authors)unestablished.push(language.language);}
 const notes=[];
 if(limited.length)notes.push('Author-count search limit reached: '+limited.join(', ')+'. Counts remain unresolved.');
 if(unstable.length)notes.push('Unstable group assignments: '+unstable.join(', ')+'. Treat tags as uncertain.');
 if(unestablished.length)notes.push('Estimate stability not established: '+unestablished.join(', ')+'. More evidence is needed.');
 if(notes.length)target.append(el('p',notes.join(' '),'notice'));
}
function refresh(){filtered=selectCorpus(verses,state);renderMetrics();renderEstimateWarnings();renderVerses();renderRollups();renderAuthors();renderSelection();renderAuthorWorks();}
function renderSelection(){
 const summary=$('selection-summary');summary.replaceChildren();
 if(state.query){summary.append(el('span','Search: '+state.query));const clear=el('button','Clear search');clear.type='button';clear.addEventListener('click',()=>{clearTimeout(searchTimer);state.query='';$('search').value='';resetPages();refresh();});summary.append(clear);}
 if(state.focusId){summary.append(el('span','Text: '+state.focusId));const clear=el('button','Show all matching text');clear.type='button';clear.addEventListener('click',()=>{state.focusId='';resetPages();refresh();});summary.append(clear);}
 summary.hidden=!['text','hierarchy','authors','works'].includes(state.tab) || !summary.childNodes.length;
}
function activate(tab){state.tab=tab;for(const button of document.querySelectorAll('[data-tab]')){const selected=button.dataset.tab===tab;button.setAttribute('aria-selected',str(selected));button.tabIndex=selected?0:-1;$('panel-'+button.dataset.tab).hidden=!selected;}renderSelection();renderAuthorWorks();if(tab==='hierarchy')renderRollups();}
function drill(row){Object.assign(state,drillSelection(state,row));$('level').value=state.level;refreshOptions();resetPages();refresh();if(row.level==='verse')activate('text');}
function rollupTitle(row){if(row.level==='language')return row.language;if(row.level==='collection')return row.collection;if(row.level==='book'){const found=bookTitles.get(str(row.language)+'\u0000'+str(row.collection)+'\u0000'+str(row.book));return found || row.book;}if(row.level==='chapter')return 'Chapter '+str(row.chapter);return reference(row);}
const bookTitles=new Map(verses.map(row=>[str(row.language)+'\u0000'+str(row.collection)+'\u0000'+str(row.book),row.book_title || row.book]));
function verseRollup(row){const assigned=Boolean(row.author_id);return {level:'verse',language:row.language,collection:row.collection,book:row.book,book_title:row.book_title,chapter:row.chapter,verse:row.verse,id:row.id,verse_count:1,assigned_verse_count:assigned?1:0,insufficient_verse_count:assigned?0:1,low_evidence_verse_count:row.status==='low_evidence'?1:0,author_count:assigned?1:0,dominant_author:row.author_id,author_ids:assigned?[row.author_id]:[],author_counts:assigned?{[row.author_id]:1}:{}};}
function rollupSummary(rows,level){
 if(level==='verse')return rows.map(verseRollup);
 const keys=['language','collection','book','chapter'].slice(0,['language','collection','book','chapter'].indexOf(level)+1),groups=new Map();
 for(const verse of rows){
  const key=JSON.stringify(keys.map(key=>str(verse[key])));
  if(!groups.has(key)){
   const group={level,language:null,collection:null,book:null,chapter:null,verse_count:0,assigned_verse_count:0,insufficient_verse_count:0,low_evidence_verse_count:0,counts:new Map()};
   for(const field of keys)group[field]=verse[field];
   groups.set(key,group);
  }
  const group=groups.get(key);group.verse_count++;
  if(verse.author_id){group.assigned_verse_count++;group.counts.set(verse.author_id,(group.counts.get(verse.author_id) || 0)+1);}
  else group.insufficient_verse_count++;
  if(verse.status==='low_evidence')group.low_evidence_verse_count++;
 }
 return Array.from(groups.values(),group=>{
  const {counts,...row}=group,entries=Array.from(counts).sort((a,b)=>b[1]-a[1] || str(a[0]).localeCompare(str(b[0])));
  return {...row,author_count:entries.length,author_ids:entries.map(([id])=>id),author_counts:Object.fromEntries(entries),dominant_author:entries[0]?.[0] || null};
 });
}
function authorSummary(rows){
 const groups=new Map();
 for(const row of rows){
  if(!row.author_id)continue;
  const key=JSON.stringify([str(row.language),str(row.author_id)]);
  if(!groups.has(key))groups.set(key,{language:row.language,author_id:row.author_id,verse_count:0,token_count:0,books:new Set(),collections:new Set()});
  const group=groups.get(key);group.verse_count++;
  group.token_count+=typeof row.token_count==='number' && Number.isFinite(row.token_count)?row.token_count:0;
  group.books.add(JSON.stringify([str(row.collection),str(row.book)]));group.collections.add(str(row.collection));
 }
 return Array.from(groups.values(),({books,collections,...group})=>({...group,book_count:books.size,collection_count:collections.size})).sort((a,b)=>str(a.language).localeCompare(str(b.language)) || b.verse_count-a.verse_count || str(a.author_id).localeCompare(str(b.author_id)));
}
function mapStartLevel(selection){
 return ['language','collection','book','chapter'].find(key=>selection[key]==null || str(selection[key])==='') || 'verse';
}
function mapScope(rows,path){
 const last=path[path.length-1];if(!last)return rows;
 const selection={};for(const key of ['language','collection','book','chapter'])if(last[key]!=null)selection[key]=str(last[key]);
 if(last.level==='verse')selection.focusId=str(last.id);
 return selectCorpus(rows,selection);
}
function mapBreakdown(rows,level,measure='words'){
 const levels=['language','collection','book','chapter','verse'],rank=levels.indexOf(level),groups=new Map();
 if(rank<0)return [];
 const keys=levels.slice(0,Math.min(rank+1,4));
 for(const row of rows){
  const identity=keys.map(key=>str(row[key]));if(level==='verse')identity.push(str(row.id));
  const key=JSON.stringify(identity);
  if(!groups.has(key)){
   const group={level,language:null,collection:null,book:null,chapter:null,verse:null,id:null,book_title:row.book_title,verse_count:0,token_count:0,missingWords:0,total:0,authors:new Map()};
   for(const field of keys)group[field]=row[field];
   if(level==='verse'){group.verse=row.verse;group.id=row.id;}
   groups.set(key,group);
  }
  const group=groups.get(key),author=row.author_id || null,knownWords=typeof row.token_count==='number' && Number.isFinite(row.token_count) && row.token_count>=0;
  const words=knownWords?row.token_count:0,amount=measure==='verses'?1:words;
  group.verse_count++;group.token_count+=words;group.total+=amount;if(!knownWords)group.missingWords++;
  if(!group.authors.has(author))group.authors.set(author,{author,amount:0,units:0});
  const entry=group.authors.get(author);entry.amount+=amount;entry.units++;
 }
 return Array.from(groups.values(),group=>({...group,authors:Array.from(group.authors.values()).sort((a,b)=>a.author===null?1:b.author===null?-1:b.amount-a.amount || str(a.author).localeCompare(str(b.author))).map(entry=>({...entry,share:group.total?entry.amount/group.total:0}))}));
}
function mapTitle(row){return row.level==='language'?({eng:'English',grc:'Greek',hbo:'Hebrew',arb:'Arabic'}[row.language] || row.language):rollupTitle(row);}
function mapGroupingPath(path,level){
 const levels=['language','collection','book','chapter','verse'],rank=levels.indexOf(level);
 const cut=path.findIndex(row=>levels.indexOf(row.level)>=rank);
 return cut<0?path.slice():path.slice(0,cut);
}
function showHierarchy(level){mapPath=mapGroupingPath(mapPath,level);mapHighlight=null;state.level=level;state.rollupPage=0;renderRollups();}
function drillMap(row){mapPath.push(row);mapHighlight=null;const levels=['language','collection','book','chapter','verse'];state.level=levels[levels.indexOf(row.level)+1] || 'verse';state.rollupPage=0;renderRollups();}
function backMap(depth){mapPath=mapPath.slice(0,depth);mapHighlight=null;const levels=['language','collection','book','chapter','verse'];state.level=depth?levels[Math.min(levels.indexOf(mapPath[depth-1].level)+1,4)]:mapStartLevel(state);state.rollupPage=0;renderRollups();}
function openMapText(){
 const selected=mapPath[mapPath.length-1],focusId=state.focusId;
 if(mapHighlight)Object.assign(state,authorSelection(state,mapHighlight.author,mapHighlight.language));
 if(selected)Object.assign(state,drillSelection(state,selected));
 if(focusId)state.focusId=focusId;
 refreshOptions();resetPages();refresh();activate('text');
}
function mapAuthorTable(group,unit){
 const details=el('details',null,'map-breakdown'),summary=el('summary','Show author breakdown'),table=el('table'),head=el('thead'),tr=el('tr'),body=el('tbody');
 for(const label of ['Inferred author',unit,'Share'])tr.append(el('th',label));head.append(tr);
 for(const entry of group.authors){const row=el('tr');row.append(el('td',entry.author || 'Unassigned'),el('td',count(entry.amount)),el('td',(entry.share*100).toFixed(1)+'%'));body.append(row);}
 table.append(head,body);details.append(summary,table);return details;
}
function renderMapLeaf(){
 const target=$('map-detail');target.replaceChildren();
 if(mapPath[mapPath.length-1]?.level!=='verse')return false;
 const row=mapRows[0];if(!row)return false;
 const card=el('article',null,'map-leaf');card.append(el('h3',reference(row)),el('div',row.id,'tiny'));
 const status=row.status==='low_evidence'?'Low evidence':row.author_id?'Assigned':'Insufficient text';
 card.append(el('span',(row.author_id || 'Unassigned')+' · '+status,'badge '+(row.status==='low_evidence'?'low':row.author_id?'':'none')));
 const text=el('div',row.text || '(Empty text)','verse-text');text.dir='auto';card.append(text,el('p',count(row.token_count)+' words · '+count(row.evidence_tokens)+' supporting passage tokens','tiny'));
 if(row.reference_author)card.append(el('p','Reference author (validation only): '+row.reference_author,'truth'));
 target.append(card);return true;
}
function renderRollups(){
 if(state.tab!=='hierarchy')return;
 if(mapSource!==filtered){mapSource=filtered;mapPath=[];mapHighlight=null;state.rollupPage=0;state.level=mapStartLevel(state);}
 mapRows=mapScope(filtered,mapPath);$('level').value=state.level;
 const measure=$('contribution-measure').value,unit=measure==='words'?'words':'verses / paragraphs';
 const totals=mapBreakdown(mapRows,'language',measure),total=totals.reduce((n,row)=>n+row.total,0),missing=totals.some(row=>row.missingWords);
 $('map-summary').textContent=count(mapRows.length)+' matching text units · '+(measure==='words' && missing?'word counts incomplete':count(total)+' '+unit);
 $('map-open-text').disabled=!mapRows.length;
 const trail=$('trail');trail.replaceChildren();const root=el('button','Current corpus');root.type='button';root.addEventListener('click',()=>backMap(0));trail.append(root);
 mapPath.forEach((row,index)=>{trail.append(el('span','›'));const button=el('button',mapTitle(row));button.type='button';button.addEventListener('click',()=>backMap(index+1));trail.append(button);});
 const highlight=$('map-highlight');highlight.replaceChildren();highlight.hidden=!mapHighlight;
 if(mapHighlight){highlight.append(el('span','Highlighting '+mapHighlight.author+' · totals include every author'));const clear=el('button','Clear highlight');clear.type='button';clear.addEventListener('click',()=>{mapHighlight=null;renderRollups();});highlight.append(clear);}
 const leaf=renderMapLeaf();$('map-children').hidden=leaf;
 if(!leaf){
  filteredRollups=state.level==='verse'?mapRows:mapBreakdown(mapRows,state.level,measure);
  const start=state.rollupPage*mapPageSize,slice=filteredRollups.slice(start,start+mapPageSize),groups=state.level==='verse'?mapBreakdown(slice,'verse',measure):slice;
  const fragment=document.createDocumentFragment();
  $('map-level-title').textContent=({language:'Languages',collection:'Collections',book:'Books',chapter:'Chapters',verse:'Verses / paragraphs'})[state.level];
  for(const group of groups){
   const row=el('article',null,'map-row'),heading=el('div',null,'map-row-heading'),title=el('div'),button=el('button',mapTitle(group),'author-link map-title');button.type='button';button.addEventListener('click',()=>drillMap(group));
   title.append(button,el('div',[group.language,group.collection,group.book,group.level==='verse'?group.chapter:null].filter(value=>value!=null).join(' / '),'tiny'));
   const amount=el('div',measure==='words' && group.missingWords?'Word counts incomplete':count(group.total)+' '+unit,'map-amount');amount.append(el('div',count(group.authors.filter(entry=>entry.author).length)+' inferred authors · '+count(group.verse_count)+' text units','tiny'));heading.append(title,amount);row.append(heading);
   if(measure==='words' && group.missingWords)row.append(el('p','Switch to verses / paragraphs to see complete shares.','map-zero'));
   else{
    const stack=el('button',null,'map-stack');stack.type='button';stack.setAttribute('aria-label',(group.level==='verse'?'Read ':'Explore ')+mapTitle(group)+': '+group.authors.map(entry=>(entry.author || 'Unassigned')+' '+count(entry.amount)+' '+unit+' ('+(entry.share*100).toFixed(1)+'%)').join(', '));stack.addEventListener('click',()=>drillMap(group));
    for(const entry of group.authors){const segment=el('span',null,'map-segment'+(mapHighlight && (mapHighlight.author!==entry.author || mapHighlight.language!==group.language)?' dimmed':''));segment.style.width=(entry.share*100)+'%';segment.style.backgroundColor=contributionColor(group.language,entry.author);segment.title=(entry.author || 'Unassigned')+': '+count(entry.amount)+' '+unit+' · '+(entry.share*100).toFixed(1)+'%';segment.setAttribute('aria-hidden','true');stack.append(segment);}
    row.append(stack);if(!group.total)row.append(el('p','No words in this row. Select it to inspect the text, or switch to verse counts.','map-zero'));
    const preview=el('div',null,'map-preview');for(const entry of group.authors.slice(0,3)){const label=el('span'),swatch=el('span',null,'swatch');swatch.style.backgroundColor=contributionColor(group.language,entry.author);label.append(swatch,(entry.author || 'Unassigned')+' '+(entry.share*100).toFixed(1)+'%');preview.append(label);}if(group.authors.length>3)preview.append(el('span','+'+(group.authors.length-3)+' more'));row.append(preview,mapAuthorTable(group,unit));
   }
   fragment.append(row);
  }
  $('rollup-rows').replaceChildren(fragment);$('rollup-empty').hidden=filteredRollups.length>0;$('rollup-count').textContent=pageInfo(filteredRollups.length,state.rollupPage,mapPageSize);
  $('rollup-previous').disabled=state.rollupPage===0;$('rollup-next').disabled=(state.rollupPage+1)*mapPageSize>=filteredRollups.length;
 }
 renderContributions();
}
function renderAuthors(){filteredAuthors=authorSummary(filtered);const start=state.authorPage*state.size,fragment=document.createDocumentFragment();for(const row of filteredAuthors.slice(start,start+state.size)){const tr=el('tr'),tag=el('td');tag.append(authorButton(row.author_id,row.language));tr.append(tag,el('td',row.language),el('td',count(row.verse_count)),el('td',count(row.book_count)),el('td',count(row.collection_count)),el('td',count(row.token_count)));const explore=el('td'),button=el('button','Browse works','button');button.type='button';button.setAttribute('aria-label','Browse works by '+row.author_id+' in '+row.language);button.addEventListener('click',()=>{worksAuthorKey=JSON.stringify([str(row.language),str(row.author_id)]);worksSource=null;activate('works');});explore.append(button);tr.append(explore);fragment.append(tr);}$('author-rows').replaceChildren(fragment);$('author-empty').hidden=filteredAuthors.length>0;$('author-count').textContent=pageInfo(filteredAuthors.length,state.authorPage,state.size);$('author-previous').disabled=state.authorPage===0;$('author-next').disabled=(state.authorPage+1)*state.size>=filteredAuthors.length;}
function authorWorks(rows,language,author){
 const result={language,author_id:author,verse_count:0,token_count:0,collection_count:0,book_count:0,chapter_count:0,collections:[]},collections=new Map();
 for(const row of rows){
  if(!row.author_id || str(row.language)!==str(language) || str(row.author_id)!==str(author))continue;
  const collectionKey=str(row.collection),bookKey=str(row.book),chapterKey=str(row.chapter);
  if(!collections.has(collectionKey))collections.set(collectionKey,{collection:row.collection,verse_count:0,token_count:0,book_count:0,chapter_count:0,books:new Map()});
  const collection=collections.get(collectionKey);
  if(!collection.books.has(bookKey))collection.books.set(bookKey,{book:row.book,book_title:row.book_title || row.book,verse_count:0,token_count:0,chapter_count:0,chapters:new Map()});
  const book=collection.books.get(bookKey);
  if(!book.chapters.has(chapterKey))book.chapters.set(chapterKey,{chapter:row.chapter,verse_count:0,token_count:0,verses:[]});
  const chapter=book.chapters.get(chapterKey),words=typeof row.token_count==='number' && Number.isFinite(row.token_count)?row.token_count:0;
  for(const group of [result,collection,book,chapter]){group.verse_count++;group.token_count+=words;}
  chapter.verses.push(row);
 }
 const natural=(a,b)=>str(a).localeCompare(str(b),undefined,{numeric:true});
 result.collections=Array.from(collections.values()).sort((a,b)=>natural(a.collection,b.collection)).map(collection=>{
  collection.books=Array.from(collection.books.values()).sort((a,b)=>natural(a.book_title,b.book_title)).map(book=>{
   book.chapters=Array.from(book.chapters.values()).sort((a,b)=>natural(a.chapter,b.chapter));
   book.chapter_count=book.chapters.length;collection.chapter_count+=book.chapter_count;
   return book;
  });
  collection.book_count=collection.books.length;result.book_count+=collection.book_count;result.chapter_count+=collection.chapter_count;
  return collection;
 });
 result.collection_count=result.collections.length;
 return result;
}
function worksBranch(title,description,kind,build){
 const details=el('details',null,'works-branch works-'+kind),summary=el('summary');
 summary.append(el('span',title),el('span',description,'tiny'));details.append(summary);
 let loaded=false;details.addEventListener('toggle',()=>{
  if(!details.open || loaded)return;loaded=true;
  const children=el('div',null,'works-children');build(children);details.append(children);
 });
 return details;
}
function renderWorksChapter(container,chapter){
 const wrap=el('div',null,'table-wrap'),table=el('table',null,'text-table'),head=el('thead'),heading=el('tr'),body=el('tbody');
 for(const label of ['Text reference','Inferred author','Verse / paragraph'])heading.append(el('th',label));
 head.append(heading);table.append(head,body);wrap.append(table);
 const footer=el('div',null,'works-more'),shown=el('span',null,'count'),more=el('button','Show more verses / paragraphs','button');more.type='button';
 let visible=0;function appendPage(){
  const next=Math.min(visible+50,chapter.verses.length),fragment=document.createDocumentFragment();
  for(const row of chapter.verses.slice(visible,next))fragment.append(verseRow(row));
  body.append(fragment);visible=next;shown.textContent='Showing '+count(visible)+' of '+count(chapter.verses.length)+' verses / paragraphs';more.hidden=visible>=chapter.verses.length;
 }
 more.addEventListener('click',appendPage);footer.append(shown,more);container.append(wrap,footer);appendPage();
}
function renderAuthorWorks(){
 if(state.tab!=='works' || worksSource===filtered)return;
 worksSource=filtered;
 const picker=$('works-author'),overview=$('works-overview'),tree=$('works-tree');picker.replaceChildren();overview.replaceChildren();tree.replaceChildren();
 const choices=filteredAuthors.map(row=>({row,key:JSON.stringify([str(row.language),str(row.author_id)])}));
 if(!choices.length){
  worksAuthorKey='';const option=el('option','No authors in this selection');option.value='';picker.append(option);picker.disabled=true;
  const empty=el('div',null,'panelbox empty');empty.append(el('strong','No assigned text in this selection'),'Try broadening your corpus filters or clearing the search.');overview.append(empty);return;
 }
 picker.disabled=false;
 if(!choices.some(choice=>choice.key===worksAuthorKey))worksAuthorKey=choices[0].key;
 for(const {row,key} of choices){const option=el('option',row.author_id+' · '+row.language+' · '+count(row.verse_count)+' text units');option.value=key;picker.append(option);}
 picker.value=worksAuthorKey;
 const {row:selected}=choices.find(choice=>choice.key===worksAuthorKey),works=authorWorks(filtered,selected.language,selected.author_id);
 const card=el('article',null,'works-overview'),counts=el('div',null,'works-counts');card.append(el('h3',works.author_id),el('p','Language: '+works.language+' · Text assigned to this inferred author within your current filters.','works-caption'));
 for(const [label,value] of [['Collections',works.collection_count],['Books',works.book_count],['Chapters',works.chapter_count],['Verses / paragraphs',works.verse_count],['Words',works.token_count]]){
  const item=el('div');item.append(el('strong',count(value)),el('span',label));counts.append(item);
 }
 card.append(counts);overview.append(card);
 for(const collection of works.collections){
  tree.append(worksBranch(collection.collection || 'Collection not specified',count(collection.book_count)+' books · '+count(collection.chapter_count)+' chapters · '+count(collection.verse_count)+' verses / paragraphs','collection',children=>{
   for(const book of collection.books){
    children.append(worksBranch(book.book_title || book.book || 'Book not specified',count(book.chapter_count)+' chapters · '+count(book.verse_count)+' verses / paragraphs','book',chapters=>{
     for(const chapter of book.chapters)chapters.append(worksBranch(chapter.chapter==null?'Chapter not specified':'Chapter '+chapter.chapter,count(chapter.verse_count)+' verses / paragraphs · '+count(chapter.token_count)+' words','chapter',verses=>renderWorksChapter(verses,chapter)));
    }));
   }
  }));
 }
}
function humanize(key){return key.replace(/_/g,' ').replace(/\b\w/g,letter=>letter.toUpperCase());}
function renderDiscovery(container){
 const validation=data.discovery_validation;
 if(!validation || !validation.known_author_count)return;
 container.append(el('h3','Blind discovery versus reference authors'));
 const grid=el('div',null,'benchmark-grid');
 const decimal=value=>value==null?'Unavailable':Number(value).toFixed(3);
 const error=validation.count_error;
 grid.append(metric(count(validation.discovered_groups_on_labelled_text),'Discovered groups','On text with reference labels'),metric(error==null?'Unavailable':(error>0?'+':'')+error,'Author count difference','Compared with '+count(validation.known_author_count)+' loaded reference authors'),metric(decimal(validation.adjusted_rand_index),'Adjusted Rand index','1 = perfect group agreement; 0 ≈ chance'),metric(decimal(validation.normalized_mutual_information),'Normalized mutual information','1 = perfect group agreement; 0 = none'));
 container.append(grid,el('p',validation.interpretation || 'Reference authors are checked after discovery. These are clustering agreement metrics, not held-out prediction accuracy.','notice'));
 container.append(el('p',count(validation.evaluated_verses)+' labeled text units evaluated · '+(100*(validation.coverage || 0)).toFixed(1)+'% coverage. Nearby verses inherit the same passage tag.','muted'));
 const groups=Object.entries(validation.group_reference_counts || {});
 if(groups.length){const detail=el('details',null,'method-card');detail.style.marginBottom='20px';detail.append(el('summary','Compare inferred groups with reference authors'));const wrap=el('div',null,'table-wrap'),table=el('table'),head=el('thead'),tr=el('tr');for(const label of ['Inferred group','Reference-author composition'])tr.append(el('th',label));head.append(tr);const body=el('tbody');for(const [author,counts] of groups){const row=el('tr'),tag=el('td'),composition=el('td');tag.append(authorButton(author,(authors.find(item=>item.author_id===author) || {}).language || 'eng'));composition.textContent=Object.entries(counts).sort((a,b)=>b[1]-a[1]).map(([name,n])=>name+': '+count(n)).join(' · ');row.append(tag,composition);body.append(row);}table.append(head,body);wrap.append(table);detail.append(wrap);container.append(detail);}
}
function renderBenchmark(){
 const container=$('benchmark-content'),benchmark=data.benchmark;
 renderDiscovery(container);
 if(!benchmark || Object.keys(benchmark).length===0){const note=el('div',null,'empty');note.append(el('strong',data.discovery_validation && data.discovery_validation.known_author_count ? 'Held-out attribution test not run' : 'No English validation results'),'No held-out attribution test is available. Run the analysis with a labeled English collection to compare its inferred groups with known authors.');container.append(note);return;}
 const percent=value=>value==null?'Unavailable':(100*value).toFixed(1)+'%';
 const grid=el('div',null,'benchmark-grid');
 const english=languages.filter(row=>/^(en|eng|english)$/i.test(str(row.language)));
 const metrics=[['Expected authors',benchmark.expected_author_count,'Known reference corpus'],['Reference authors loaded',benchmark.known_author_count,'Coverage in this analysis'],['Estimated style groups',english.length?english.reduce((n,row)=>n+(row.estimated_authors || 0),0):null,'Unsupervised English discovery'],['Authors evaluated',benchmark.evaluated_author_count,'Eligible for held-out book testing']];
 for(const [label,value,hint] of metrics)grid.append(metric(value==null?'Unavailable':count(value),label,hint));
 container.append(grid);
 const completeness=benchmark.source_completeness || {};
 if((completeness.missing_authors || []).length)container.append(el('p','Missing reference authors: '+completeness.missing_authors.join(', ')+'. The 13-author reference set is incomplete.','notice'));
 if(benchmark.reason)container.append(el('p',benchmark.reason,'notice'));
 if(benchmark.status==='evaluated'){
  container.append(el('h3','Held-out book attribution'));
  const scores=el('div',null,'benchmark-grid');
  scores.append(metric(percent(benchmark.accuracy),'Passage accuracy','Correct labels on held-out passages'),metric(percent(benchmark.balanced_accuracy),'Balanced passage accuracy','Equal weight for each evaluated author'),metric(percent(benchmark.book_accuracy),'Book accuracy','Correct labels after aggregating passages'),metric(percent(benchmark.book_balanced_accuracy),'Balanced book accuracy','Equal weight for each evaluated author'));
  container.append(scores);
  if(benchmark.split)container.append(el('p',count((benchmark.split.train_books || []).length)+' training books · '+count((benchmark.split.test_books || []).length)+' held-out books · '+count(benchmark.split.test_passages)+' test passages. Whole books are held out from training.','muted'));
 }
 container.append(el('p',benchmark.caveat || 'Reference labels evaluate the result; they are separate from inferred author tags. A matching author count alone does not mean the attributions are correct.','notice'));
 if((benchmark.per_author || []).length){
  container.append(el('h3','Results by reference author'));
  const wrap=el('div',null,'panelbox table-wrap'),table=el('table'),head=el('thead'),headrow=el('tr'),body=el('tbody');
  for(const label of ['Reference author','Train books','Test books','Test passages','Accuracy'])headrow.append(el('th',label));head.append(headrow);
  for(const row of benchmark.per_author){const tr=el('tr');for(const value of [row.author,count(Array.isArray(row.train_books)?row.train_books.length:row.train_books),count(Array.isArray(row.test_books)?row.test_books.length:row.test_books),count(row.test_passages),percent(row.accuracy)])tr.append(el('td',value));body.append(tr);}
  table.append(head,body);wrap.append(table);container.append(wrap);
 }
 const details=el('details',null,'method-card');details.style.marginTop='18px';details.append(el('summary','Complete validation results'),el('pre',JSON.stringify(benchmark,null,2)));container.append(details);
}
function renderSourceCoverage(){const coverage=data.source_coverage;if(!coverage)return;const values=[['Parsed units',coverage.parsed_units],['Primary source units',coverage.primary_units],['Units in this analysis',coverage.analyzed_units],['Alternate witness units',coverage.alternate_witness_units]];const present=values.filter(([,value])=>value!=null);$('source-scope').hidden=false;$('source-scope').textContent=present.map(([label,value])=>count(value)+' '+label.toLowerCase()).join(' · ');const details=el('details',null,'method-card');details.style.marginBottom='18px';details.append(el('summary','Corpus scope and source policy'));if(coverage.source_policy)details.append(el('p',coverage.source_policy));for(const [label,value] of present)details.append(el('p',label+': '+count(value)));if(coverage.alternate_witness_units)details.append(el('p','Alternate manuscript witnesses are preserved in the raw data and excluded from the primary corpus analyzed here.'));$('source-coverage').append(details);}
function renderMethod(){const container=$('language-cards');for(const language of languages){const card=el('article',null,'language-card');card.append(el('h3',language.language),el('div',count(language.estimated_authors),'number'),el('div','estimated author groups','muted'),el('div',count(language.verse_count)+' text units · '+count(language.passage_count)+' passages','tiny'));if(language.selection){if(language.selection.at_search_limit)card.append(el('p','Search limit reached: author count unresolved.','notice'));if(language.selection.stable===false)card.append(el('p','Unstable across repeat fits: attribution is uncertain.','notice'));const details=el('details');details.append(el('summary','Why this estimate?'));if(language.selection.reason)details.append(el('p',language.selection.reason));details.append(el('pre',JSON.stringify(language.selection,null,2)));card.append(details);}container.append(card);}if(!languages.length)container.append(el('p','No language estimates are available.','empty'));$('configuration').textContent=JSON.stringify(data.config || {},null,2);}
function resetAll(){for(const key of ['language','collection','book','chapter','author','status','query','focusId'])state[key]='';state.level='language';$('level').value='language';$('search').value='';refreshOptions();resetPages();refresh();}
for(const key of ['language','collection','book','chapter','author','status'])$(key).addEventListener('change',()=>{state[key]=$(key).value;state.focusId='';const keys=['language','collection','book','chapter'];if(keys.includes(key)){for(const child of keys.slice(keys.indexOf(key)+1))state[child]='';if(key==='language')state.author='';}refreshOptions();resetPages();refresh();});
let searchTimer;$('search').addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{state.query=$('search').value;state.focusId='';resetPages();refresh();},200);});
$('contribution-measure').addEventListener('change',()=>{state.rollupPage=0;renderRollups();});
$('map-open-text').addEventListener('click',openMapText);
$('works-author').addEventListener('change',()=>{worksAuthorKey=$('works-author').value;worksSource=null;renderAuthorWorks();});
$('page-size').addEventListener('change',()=>{state.size=Number($('page-size').value);resetPages();refresh();});$('reset').addEventListener('click',resetAll);$('previous').addEventListener('click',()=>{state.page--;renderVerses();});$('next').addEventListener('click',()=>{state.page++;renderVerses();});$('level').addEventListener('change',()=>showHierarchy($('level').value));$('rollup-previous').addEventListener('click',()=>{state.rollupPage--;renderRollups();});$('rollup-next').addEventListener('click',()=>{state.rollupPage++;renderRollups();});$('author-previous').addEventListener('click',()=>{state.authorPage--;renderAuthors();});$('author-next').addEventListener('click',()=>{state.authorPage++;renderAuthors();});for(const button of document.querySelectorAll('[data-tab]')){button.addEventListener('click',()=>activate(button.dataset.tab));button.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const tabs=Array.from(document.querySelectorAll('[data-tab]'));const current=tabs.indexOf(button);const index=event.key==='Home'?0:event.key==='End'?tabs.length-1:(current+(event.key==='ArrowRight'?1:tabs.length-1))%tabs.length;activate(tabs[index].dataset.tab);tabs[index].focus();});}
refreshOptions();renderBenchmark();renderSourceCoverage();renderMethod();refresh();activate('text');
</script>
</body>
</html>
'''
