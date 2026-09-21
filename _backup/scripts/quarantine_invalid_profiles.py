#!/usr/bin/env python
"""Move unusable profile rows out of a profiles file so the rest stays loadable.

``load_profiles`` raises on the first row it cannot validate, so one bad record makes a whole file
unusable.  Regenerating the row in place is not an option for a legacy file: the loader also rejects a
file that mixes profiling configurations, and a freshly generated record carries provenance that the
legacy rows lack.  This moves the offending rows to ``<name>_rejected.jsonl`` instead, with the reason
attached, so nothing is thrown away and the remaining measurements load again.

    uv run python scripts/quarantine_invalid_profiles.py data/processed/profiles_deepseek_flash.jsonl
    uv run python scripts/quarantine_invalid_profiles.py --dry-run data/processed/*.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stylometry.ai.profile import coerce_profile  # noqa: E402

TAG_RULE = "style_tags must be 1-8 lowercase ASCII snake_case tags"


def reason_for(rec: dict) -> str:
    """Name the first rule the record breaks, for the quarantine file."""
    if not isinstance(rec.get("id"), str) or not rec["id"].strip():
        return "missing or empty id"
    tags = rec.get("style_tags")
    if not isinstance(tags, list) or not 1 <= len(tags) <= 8:
        return TAG_RULE
    if any(not isinstance(t, str) or not t.isascii() for t in tags):
        return f"{TAG_RULE} (non-ASCII: {[t for t in tags if isinstance(t, str) and not t.isascii()]})"
    return "failed validation; see coerce_profile"


def quarantine(path: Path, dry_run: bool = False) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    keep: list[str] = []
    rejected: list[dict] = []
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        if coerce_profile(rec, allow_legacy="provenance" not in rec) is None:
            rejected.append({"source_file": path.name, "source_line": line_no,
                             "reason": reason_for(rec), "record": rec})
        else:
            keep.append(line)
    if rejected and not dry_run:
        out = path.with_name(path.stem + "_rejected.jsonl")
        with out.open("a", encoding="utf-8") as fh:
            for r in rejected:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        path.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
    return len(keep), len(rejected)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="report without moving anything")
    args = ap.parse_args(argv)
    total = 0
    for path in args.paths:
        if path.stem.endswith(("_runs", "_errors", "_batches", "_rejected")) or not path.exists():
            continue
        kept, rejected = quarantine(path, args.dry_run)
        total += rejected
        verb = "would move" if args.dry_run else "moved"
        print(f"{path.name}: {kept} kept, {verb} {rejected}"
              + (f" -> {path.stem}_rejected.jsonl" if rejected and not args.dry_run else ""))
    return 1 if (args.dry_run and total) else 0


if __name__ == "__main__":
    raise SystemExit(main())
