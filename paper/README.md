# Stylometry research article

The editable manuscript is `stylometry_research_article.docx`; its text source is
`stylometry_research_article.md`. The article contains approximately 7,750 words
from abstract through conclusion, including table text and headings, plus a
reproducibility statement and eleven references. Word counts depend on treatment
of hyphens and numerals; the audit records its counting convention.

The PDF is saved in `output/pdf/stylometry_research_article.pdf` and linked from
the explorer and validation pages. Regenerating the HTML retains the link when
the PDF is present in that output folder.

The paper examines the saved Stylometry 0.2.0 output as available on 22 September
2026. It reports existing experiments and descriptive calculations, without
claiming that new model fits or proposed follow-up experiments were performed.

`audit_results.py` verifies key counts and generates `results_audit.json`,
including input-file checksums and collection summaries. It uses only the Python
standard library and can be run from the repository root with
`python paper/audit_results.py`.

`build_manuscript.py` converts the Markdown source into Word using python-docx.
Rendering intermediates are excluded from version control in `.qa/`.
