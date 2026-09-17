"""Conservative text extraction for non-Greek reference corpora.

These corpus adapters do not provide authorship labels or validate transfer to
scripture. Labels and that limitation belong to the reviewed source manifest.
"""
from __future__ import annotations

import re

from lxml import html


def extract_additional_text(source: bytes, parser: str, language: str) -> str:
    """Return prose without publisher metadata, headings or editorial notes.

    Ben-Yehuda's dump wraps an HTML document in another HTML document. Parsing
    it as forgiving HTML is intentional; parsing it as well-formed XML would
    reject the source. Some real prose is not wrapped in paragraphs and must
    survive. The attribution footer starts at a marked horizontal rule.
    We also reject unexpected footer text inside the retained body
    so a future edition cannot silently contribute publisher boilerplate.
    """
    if parser != "benyehuda-html-v1":
        raise ValueError(f"unsupported additional benchmark parser: {parser!r}")
    if language != "hbo":
        raise ValueError("benyehuda-html-v1 requires the Hebrew normalizer (hbo)")
    source_text = source.decode("utf-8")
    source_text = re.split(r"<hr\s*/?>\s*את הטקסט\[ים\] לעיל הפיקו", source_text, maxsplit=1)[0]
    document = html.fromstring(source_text)
    for element in list(document.xpath(
        "//head|//title|//meta|//script|//style|//h1|//h2|//h3|//h4|//h5|//h6"
        "|//*[contains(concat(' ', normalize-space(@class), ' '), ' footnotes ')]"
        "|//a[contains(concat(' ', normalize-space(@class), ' '), ' footnote ')]"
        "|//a[contains(concat(' ', normalize-space(@class), ' '), ' reversefootnote ')]"
    )):
        element.drop_tree()
    # Explicit block separators preserve paragraph boundaries without inventing
    # spaces inside inline emphasis, links, or combining Hebrew text.
    for element in document.xpath("//p|//div|//blockquote|//br"):
        element.text = "\n\n" + (element.text or "")
        element.tail = "\n\n" + (element.tail or "")
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", "".join(document.itertext())):
        value = re.sub(r"\s+", " ", paragraph).strip()
        if not value:
            continue
        if "את הטקסט[ים] לעיל הפיקו" in value or "benyehuda.org/read/" in value:
            raise ValueError("unexpected publisher footer in a prose paragraph")
        paragraphs.append(value)
    if not paragraphs:
        raise ValueError("Ben-Yehuda source contains no prose")
    return "\n\n".join(paragraphs)
