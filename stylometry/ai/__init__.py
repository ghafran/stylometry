"""Archived: the model-based half of the project.

The analysis no longer uses a model. Style is measured from the text alone - function-word rates,
suffixes, character and word n-grams, vocabulary richness, length, rhythm, collocations, adjacency -
and every method in the main package runs without an API key, a network connection or a bill.

This package keeps what was built when profiling was part of the pipeline: ``profile`` produces the
blinded six-scale style profiles, and ``compare`` weighs profiling models against each other on
accuracy, stability and cost. Nothing in the analysis path imports either. The profiles already paid
for remain readable, and the clustering still accepts them when handed them explicitly.

See README.md in this folder for why the split was made and what the profiles cost.
"""
