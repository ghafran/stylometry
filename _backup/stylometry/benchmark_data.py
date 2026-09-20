"""Reproducible, explicitly licensed reference-text acquisition.

Only the small manifest is committed. Source files are immutable, checksum checked,
and cached under data/raw/benchmarks. Network access always requires ``download``.
Catalog author labels are evaluation assumptions, not proof of ancient authorship.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from lxml import etree

from .lang import tokenize

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "benchmarks/greek_manifest.json"
DEFAULT_CACHE = Path("data/raw/benchmarks")
PARSER_VERSION = "tei-greek-v1"
MAX_SOURCE_BYTES = 20_000_000
_NS = {"t": "http://www.tei-c.org/ns/1.0"}
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
_XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
_OMIT = {
    "teiHeader", "front", "back", "note", "head", "speaker", "label",
    "bibl", "listBibl", "castList", "stage", "rdg", "del", "fw",
    # Explicit citations/documentary quotations are not independent author prose.
    # TEI q/direct speech remains: it includes the author's constructed dialogue.
    "quote", "cit",
}
_BLOCK = {"body", "div", "p", "ab", "l", "lg", "sp", "list", "item"}


def extract_tei_greek(source: bytes) -> str:
    """Extract one Greek edition body without editorial/translation leakage.

    Preserve inline word boundaries and accepted corrections; drop critical notes,
    rejected readings/deletions, explicit external quotations, and speaker labels.
    A selected source edition is still an editorial interpretation of manuscripts.
    """
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    root = etree.fromstring(source, parser=parser)
    if any(isinstance(node, etree._Entity) for node in root.iter()):
        raise ValueError("TEI entity references are unsupported")
    editions = root.xpath(
        ".//t:body/t:div[@type='edition' and @xml:lang='grc']", namespaces=_NS
    )
    if len(editions) != 1:
        raise ValueError("expected exactly one Greek TEI edition in the body")
    edition = editions[0]
    ids = {node.get(_XML_ID) for node in edition.iter() if node.get(_XML_ID)}
    active_deletions: set[str] = set()
    chunks: list[str] = []

    def emit(value: str | None, blocked: bool) -> None:
        if value and not blocked and not active_deletions:
            chunks.append(value)

    def visit(node: etree._Element, blocked: bool = False) -> None:
        if not isinstance(node.tag, str):  # comments and processing instructions
            return
        tag = etree.QName(node).localname
        anchor = node.get(_XML_ID)
        if anchor:
            active_deletions.discard(anchor)
        if tag == "delSpan":
            target = node.get("spanTo", "").removeprefix("#")
            if target not in ids:
                raise ValueError("TEI deletion span has no matching anchor")
            active_deletions.add(target)
            return
        blocked = blocked or tag in _OMIT or node.get(_XML_LANG, "grc") not in {"grc", "el"}
        if tag in _BLOCK:
            emit("\n", blocked)
        if tag in {"choice", "app"} and not blocked:
            priorities = ["lem"] if tag == "app" else ["corr", "reg", "expan", "orig", "sic", "abbr"]
            selected = next((child for name in priorities for child in node
                             if isinstance(child.tag, str) and etree.QName(child).localname == name), None)
            if selected is None:
                raise ValueError(f"TEI {tag} has no supported preferred reading")
            visit(selected, blocked)
        else:
            emit(node.text, blocked)
            for child in node:
                visit(child, blocked)
                emit(child.tail, blocked)
        if tag in _BLOCK:
            emit("\n", blocked)

    visit(edition)
    if active_deletions:
        raise ValueError("unclosed TEI deletion span")
    text = re.sub(r"\s+", " ", "".join(chunks)).strip()
    if not tokenize(text, "grc"):
        raise ValueError("TEI edition contains no Greek prose")
    return text


def read_manifest(manifest_path: str | Path) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("works"), list):
        raise ValueError("unsupported benchmark manifest schema")
    if not manifest["works"]:
        raise ValueError("benchmark manifest has no works")
    seen_sources: set[str] = set()
    seen_works: set[str] = set()
    seen_hashes: set[str] = set()
    required = {"source_id", "work_id", "author", "work", "language", "genre", "topic",
                "url", "sha256", "parser", "license", "license_url", "source_repository",
                "source_commit", "source_path", "n_tokens"}
    for work in manifest["works"]:
        if not isinstance(work, dict) or required - work.keys():
            raise ValueError("benchmark work is missing required provenance fields")
        if any(not isinstance(work[key], str) or not work[key].strip()
               for key in required - {"n_tokens"}):
            raise ValueError("benchmark provenance fields must be nonempty strings")
        source_id = work["source_id"]
        if not isinstance(source_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", source_id):
            raise ValueError("unsafe benchmark source_id")
        if source_id in seen_sources:
            raise ValueError("duplicate benchmark source_id")
        seen_sources.add(source_id)
        if not re.fullmatch(r"[0-9a-f]{64}", work["sha256"]):
            raise ValueError("invalid source checksum")
        if work.get("text_sha256") is not None and (
            not isinstance(work["text_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", work["text_sha256"])
        ):
            raise ValueError("invalid normalized text checksum")
        if not re.fullmatch(r"[0-9a-f]{40}", work["source_commit"]):
            raise ValueError("source revision must be an immutable 40-character commit")
        url = urlparse(work["url"])
        if url.scheme != "https" or not url.netloc or work["source_commit"] not in url.path.split("/"):
            raise ValueError("source URL must use HTTPS and contain its pinned commit")
        if type(work["n_tokens"]) is not int or work["n_tokens"] < 1:
            raise ValueError("expected token count must be positive")
        if work.get("role", "attribution") not in {"attribution", "witness_control"}:
            raise ValueError("unsupported benchmark work role")
        if work.get("role", "attribution") == "attribution":
            if work["work_id"] in seen_works or work["sha256"] in seen_hashes:
                raise ValueError("duplicate work or edition would leak across attribution splits")
            seen_works.add(work["work_id"])
            seen_hashes.add(work["sha256"])
    return manifest


def _source_bytes(work: dict, cache_dir: Path, download: bool) -> bytes:
    target = cache_dir / (work["source_id"] + ".source")
    if target.exists():
        if target.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError(f"source too large: {target}")
        source = target.read_bytes()
    else:
        if not download:
            raise FileNotFoundError(
                f"Benchmark source not cached: {target}. Run the benchmark-data acquisition "
                "with --download (python -m stylometry.benchmark_data --download)."
            )
        request = Request(work["url"], headers={"User-Agent": "stylometry-reproducible-benchmark/1"})
        with urlopen(request, timeout=60) as response:
            source = response.read(MAX_SOURCE_BYTES + 1)
        if len(source) > MAX_SOURCE_BYTES:
            raise ValueError(f"source exceeds {MAX_SOURCE_BYTES} bytes: {work['source_id']}")
    actual = hashlib.sha256(source).hexdigest()
    if actual != work["sha256"]:
        raise ValueError(f"source checksum mismatch for {work['source_id']}: expected {work['sha256']}, got {actual}")
    if not target.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as temporary:
            temporary.write(source)
            temporary_path = Path(temporary.name)
        temporary_path.replace(target)
    return source


def load_benchmark(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    cache_dir: str | Path = DEFAULT_CACHE,
    download: bool = False,
    *,
    include_controls: bool = False,
) -> list[dict]:
    """Return complete, normalized works; never silently download or skip failures.

    Witness controls are available explicitly and must never be used as independent
    attribution works. Text is represented by space-separated language tokens so
    punctuation/editor labels cannot provide source-edition identification cues.
    """
    manifest_path = Path(manifest_path)
    manifest = read_manifest(manifest_path)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    works = []
    for entry in manifest["works"]:
        if entry.get("role", "attribution") != "attribution" and not include_controls:
            continue
        source = _source_bytes(entry, Path(cache_dir), download)
        if entry["parser"] == PARSER_VERSION:
            if entry["language"] != "grc":
                raise ValueError("Greek TEI parser requires language='grc'")
            extracted = extract_tei_greek(source)
        elif entry["parser"] == "plain-text-v1":
            extracted = source.decode("utf-8")
        elif entry["parser"] in {"benyehuda-html-v1", "openiti-mARkdown-v1"}:
            from .benchmark_additional import extract_additional_text
            extracted = extract_additional_text(source, entry["parser"], entry["language"])
        else:
            raise ValueError(f"unsupported benchmark parser: {entry['parser']}")
        tokens = tokenize(extracted, entry["language"])
        text_bare = " ".join(tokens)
        text_sha = hashlib.sha256(text_bare.encode("utf-8")).hexdigest()
        if len(tokens) != entry["n_tokens"]:
            raise ValueError(f"parser token count drift for {entry['source_id']}")
        if entry.get("text_sha256") and text_sha != entry["text_sha256"]:
            raise ValueError(f"normalized text checksum mismatch for {entry['source_id']}")
        works.append({
            **entry, "text_bare": text_bare, "n_tokens": len(tokens),
            "provenance": {
                "corpus_id": manifest.get("corpus_id"), "manifest_sha256": manifest_sha,
                "source_url": entry["url"], "source_sha256": entry["sha256"],
                "source_commit": entry["source_commit"], "parser": entry["parser"],
                "text_sha256": text_sha, "license": entry["license"],
                "license_url": entry["license_url"], "edition": entry.get("edition"),
                "attribution_caveat": manifest.get("attribution_caveat"),
            },
        })
    return works


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--download", action="store_true", help="Fetch missing pinned sources")
    parser.add_argument("--include-controls", action="store_true")
    args = parser.parse_args()
    works = load_benchmark(args.manifest, args.cache_dir, args.download, include_controls=args.include_controls)
    print(json.dumps({"manifest": str(args.manifest), "works": len(works),
                      "authors": len({work["author"] for work in works}),
                      "tokens": sum(work["n_tokens"] for work in works)}, indent=2))


if __name__ == "__main__":
    main()
