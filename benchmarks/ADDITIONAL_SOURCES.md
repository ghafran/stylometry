# Modern Hebrew reference corpus

`additional_manifest.json` contains 16 complete prose works, four each by Yosef
Hayyim Brenner, Uri Nissan Gnessin, Micha Josef Berdyczewski and Isaiah Bershadsky.
The attribution labels come from Project Ben-Yehuda's catalogue, including its
author authority identifiers. The selected records have no translator or foreign
original-language entry. They are modern original Hebrew fiction, not biblical
texts assigned to presumed authors.

The corpus has 107,754 normalized tokens; individual works have at least 3,000.
Selection happened before observing any classification results. It is a small
convenience sample of modern social and psychological fiction. Genre was matched
broadly; detailed topics have not been independently annotated. This does not
make author identity independent of all subject, period or editorial differences.

The source is the [Project Ben-Yehuda public-domain dump](https://github.com/projectbenyehuda/public_domain_dump)
at commit `5e4277fead7b565f32cc4b352abd3565023a77d8`. The manifest pins
individual download URLs, source SHA-256 values, normalized-text SHA-256 values,
token counts and source editions. The catalogue itself is also checksum-pinned.
Raw source files are an ignored local cache, not redistributed in this repository.
The project's [license declaration](https://github.com/projectbenyehuda/public_domain_dump/blob/5e4277fead7b565f32cc4b352abd3565023a77d8/LICENSE)
states that its text data are public domain. Credit: Project Ben-Yehuda volunteers.

The HTML adapter excludes title and author metadata, headings, explicitly marked
footnotes and footnote references. It preserves prose outside paragraph markup,
excluding the publisher's marked footer. It retains names and quotations that occur within the work;
it does not attempt to identify every biblical quotation or editorial intervention.
Other editions remain necessary to test editorial confounding.

The manifest uses `language: hbo` solely to select the application's existing
Hebrew-script normalizer. `source_language: heb` records the real source language:
modern Hebrew. Success on this corpus would not validate Biblical Hebrew,
Aramaic, historical hand counts or named scriptural attributions. Modern Hebrew
and Biblical Hebrew are separated by major changes in language, literary practice,
and textual transmission.

# Arabic coverage gap

No Arabic attribution corpus was included in this first dataset. This is an
explicit missing validation area, not a passing test or evidence of reliability.

Two primary-source routes were investigated:

* [OpenITI](https://openiti.org/documentation/) provides reproducible Arabic text
  repositories with author/work metadata and a CC BY-NC-SA 4.0 corpus license.
  Its documentation distinguishes annotation quality and editorial cleanup.
  A review of available adab and theological works found that an expedient sample
  would mix authors with compilers, quoted authors, genres or confessional subject
  matter. Those labels require a textual specialist's review before they can be
  used as ground truth for an attribution benchmark. This is a curation issue,
  not a claim that a suitable OpenITI sample is impossible.
* Hindawi's current [Safahat catalogue](https://www.safahat.org/) identifies
  authors and the rights status of individual books. Its pages explicitly label
  the texts of Gibran's [The Broken Wings](https://www.safahat.org/books/46094724/),
  al-Rafi'i's [Rose Leaves](https://www.safahat.org/books/19618179/), and Rihani's
  [The Book of Repentance](https://www.safahat.org/books/85302613/) as public-domain
  texts. Both ordinary publisher-page requests and public EPUB download requests
  returned HTTP 403 during acquisition. The files were not acquired, so no data
  checksums, extraction guarantees or evaluation claims are made for them.

A follow-up Arabic sample should use several separately published, originally
Arabic prose works for each writer, with matched topics, independently checked
authorship, and no translation of originally English Gibran or Rihani works.
Publisher introductions and collected pieces by other writers must be excluded.
Modern Arabic performance would still not establish validity for Quranic Arabic.
