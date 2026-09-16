"""Corpus builders: each turns a raw source into a list of verse records with a common schema.

Verse record fields
-------------------
id            unique key, e.g. ``MARK.1.1`` or ``AJOHN.3.2``
source        ``sinaiticus`` | ``first1kgreek`` | ``apostolic_fathers``
work          short work code (``MARK``, ``AJOHN``, ``1CLEM`` ...)
work_title    human-readable title
collection    ``LXX`` | ``NT`` | ``noncanonical``
canon         ``OT`` | ``OT-deutero`` | ``NT`` | ``none``
group         traditional authorship grouping, used *only* to validate the clustering
chapter       chapter / section number (string)
verse         verse / unit number (string)
ref           human reference (``Mark 1:1``)
text          normalised, accented text (what the model reads)
text_bare     lower-case, diacritic-free text (what the statistics read)
n_tokens      number of Greek word tokens
copyist       Sinaiticus scribe (A, B, D) at the point where the verse starts, else null
supplied_frac fraction of words editorially supplied (Sinaiticus only)
has_gap       True if the manuscript has an unreadable gap inside the verse
duplicate_of  work code of the manuscript copy when this is a second edition of the same text
order         document order within the source
"""
