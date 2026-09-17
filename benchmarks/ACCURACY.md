# The 95% accuracy goal: where it stands

Run `stylometry accuracy-gate` to reproduce. Targets live in `accuracy_targets.json`; the gate exits
non-zero until they are met and writes `accuracy_gate.json` beside the benchmark report.

## What the target can mean

Scripture has no author labels, so accuracy cannot be measured on it. It is measured on the reference
benchmark, where the author of every work is known and whole works are held out: ancient Greek by six
catalogued authors, modern Hebrew by four.

Accuracy alone is not a sufficient target. A rule that accepts every candidate scores well among the
authors it already knows while being useless on a text whose author is absent, which is the situation
every scriptural question actually poses. The gate therefore requires three things at once: 95% author
accuracy, at most 5% of unseen authors wrongly accepted, and at least 95% of genuine matches accepted.

## Measured, preregistered protocol

Greek, 1,000-token passages, the strongest configuration:

| metric | target | measured | 95% interval | status |
|---|---|---:|---|---|
| author accuracy | ≥95% | 85.0% | 77.1% to 91.7% | failed |
| unseen authors wrongly accepted | ≤5% | 90.5% | 83.8% to 96.3% | failed |
| genuine matches accepted | ≥95% | 96.8% | 94.1% to 99.0% | inconclusive |

**The gate fails.** The accuracy shortfall is ten points. The rejection failure is far larger: the rule
meant to say "none of these authors" says it almost never.

## Does more text per sample close the gap?

An exploratory run at 2,000 and 4,000 tokens per passage. Overriding the protocol makes a run
exploratory by construction, so it cannot pass the gate however good the numbers look.

| language | passage | author accuracy | unseen accepted | authors / works |
|---|---:|---:|---:|---:|
| Greek | 2,000 | 90.4% | 78.7% | 6 / 18 |
| Greek | 4,000 | 98.3% | 46.2% | 4 / 12 |
| Hebrew | 2,000 | 72.7% | 78.1% | 4 / 16 |
| Hebrew | 4,000 | 55.6% | 81.5% | 4 / 12 |

Greek accuracy does reach the target at 4,000 tokens, and it would be easy to stop reading there. Two
things forbid it.

The first is that the 4,000-token run is a smaller and easier problem. Works too short to yield a
4,000-token sample drop out, taking two of the six authors with them, so the measurement is made
against four candidates rather than six. The benchmark's own coverage floor of eight authors and sixty
works is not close to being met. Hebrew, whose works are shorter, degrades instead of improving, which
is what a sample-size effect looks like.

The second is that rejection does not improve nearly enough. Even at 4,000 tokens the method accepts
46% of authors it has never seen. Attribution that cannot decline is not attribution.

## What this means for the project

The unit that carries enough signal for reliable attribution is thousands of words, not a verse of ten
to forty. The per-verse style groups this repository produces are exploratory descriptions of style;
nothing measured here supports reading them as identified authors, and the gate exists to keep that
distinction from eroding.

Closing the remaining gap is a research problem, not a configuration change. The directions worth
trying, in the order their evidence suggests: a calibrated rejection rule, since that is the larger
failure by far; a reference corpus with more authors, since coverage now limits every conclusion; and
a fitted verifier instead of a nearest-centroid rule.
