# Archived: the model-based code

The analysis is lexical. Nothing here is imported by the measurement path, and every result in
`output/` can be reproduced with no API key, no network and no spend.

## What is in here

- **`profile.py`** — blinded AI style profiling. The model receives only the text of a passage and an
  anonymous `unit_0001` identifier; no title, reference, book or witness. It returns six 0–1 scales
  (register, semitic interference, hypotaxis, lexical richness, rhetorical polish, emotional
  intensity), five categorical fields and device tags. Backends: Anthropic SDK, Batch API, CLI,
  DeepSeek and OpenAI-compatible endpoints.
- **`compare.py`** — weighs profiling models against each other on agreement, cluster replication and
  cost per verse.

## Why it was separated

Two reasons, in order of weight.

**It is not what makes the analysis work.** Measured on the same passages and the same whole-work
holdout, Burrows's Delta reached 58.6% on Codex Sinaiticus where the AI-profile method reached 30.9%,
and an SVM over the lexical panel reached 70.2%. The strongest single result in the project — Paul at
94% — came from word frequencies alone.

**It cannot be audited the way counting can.** A function-word rate is reproducible by anyone with the
text. A model's rating is reproducible only against that model, at that version, with that prompt, and
the model may recognise the passage and import received scholarly opinion about it rather than reading
its style. Every AI finding in this project had to be checked against a model-free control before it
could be trusted — the Meccan/Medinan gradient against verse length, the Septuagint result against
verse-initial καί. Where the control existed the AI agreed with it, which is reassuring and also means
the control was sufficient on its own.

## What the profiles cost, and what remains

About $5.75 of DeepSeek profiling produced 6,236 Qur'an units and 20,139 Codex Sinaiticus units, held
in `data/processed/profiles_quran.jsonl` and `profiles_sinaiticus.jsonl`. Those files are kept. The
clustering still accepts profiles when handed them explicitly, so an AI-backed run remains possible:

    stylometry cluster --with-ai --profiles data/processed/profiles_sinaiticus.jsonl ...

Without that flag the pipeline is lexical, which is now the default everywhere.

## What the AI blocks contributed when they were used

Three feature blocks: `ai:` (six scales), `cat:` (one-hot categorical fields) and `tag:` (device tags,
binary, kept when a tag appears in at least three units). On Codex Sinaiticus they took the run from 2
style groups to 7, and ARI against scholarly groupings from 0.04 to 0.20 — a real gain, bought with
money and unreproducibility.
