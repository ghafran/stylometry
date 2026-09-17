"""Rendering switches unit labels without rewriting quoted source text or schemas."""
from __future__ import annotations

import csv
import html
import json

import pytest

from stylometry.cluster import run
from stylometry.html import build_root_index, build_site
from stylometry.report import render
from test_passage_cli import source


@pytest.mark.parametrize("passage_mode", [False, True])
def test_rendering_uses_summary_units_and_keeps_quoted_text(tmp_path, passage_mode):
    out = tmp_path / "grc"
    quoted = 'verse verses <quoted> remain unchanged'
    records = [source(i, 500, text=quoted) for i in range(1, 4)]
    summary = run(records, None, out, k=1, window=0, alpha=0)
    if passage_mode:
        summary.update(input_unit="pooled_token_passage", passage_tokens=500,
                       source_mapping_file="passages.json", input_verses=12)
        (out / "passages.json").write_text(json.dumps({"passages": [], "exclusions": []}))
        (out / "summary.json").write_text(json.dumps(summary))
    original_assignments = (out / "verse_assignments.csv").read_bytes()
    original_authors = (out / "authors.json").read_bytes()
    report = render(out)
    site = build_site(out)
    dashboard = (site / "index.html").read_text()
    group = (site / "authors" / "A1.html").read_text()
    work = (site / "works" / "W.html").read_text()
    root = build_root_index(tmp_path).read_text()
    unit, units = ("passage", "passages") if passage_mode else ("verse", "verses")
    assert f"# Exploratory {unit}-level style analysis" in report
    assert f"| style group | {units} |" in report
    assert f"## Individual outlier {units}" in report
    assert f'<th class="num">{units}</th>' in dashboard
    assert f"{units.capitalize()} assigned to this style group" in group
    assert f"{unit.capitalize()} by {unit}" in work
    assert f"3 {units} ·" in root
    assert quoted in report
    assert html.escape(quoted) in group and html.escape(quoted) in work
    assert "not identified authors" in report and "not identified authors" in dashboard
    assert "no supported split, not one proven author" in report
    assert (out / "verse_assignments.csv").read_bytes() == original_assignments
    assert (out / "authors.json").read_bytes() == original_authors
    assert "author" in next(csv.reader(original_assignments.decode().splitlines()))
    if passage_mode:
        assert "3 non-overlapping 500-token passages" in report
        assert "without additional smoothing" in report
        assert "[Source mappings and exclusions](passages.json)" in report
        assert "not the authorship of individual source verses" in report
        assert 'href="../passages.json"' in dashboard
        assert 'href="../../passages.json"' in group and 'href="../../passages.json"' in work
        assert "Passage-level style groups" in dashboard
        assert "No additional smoothing was applied" in dashboard
        assert "smoothed over a ±0-verse" not in report
        assert "per-verse labels inherit" not in report
    else:
        assert "3 verse units" in report
        assert "Source mappings and exclusions" not in report + dashboard + group + work
