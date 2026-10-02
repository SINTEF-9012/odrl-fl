# Working on this project

Read `README.md` for the science — it is long and it carries every finding, every
discarded run and the reason it was discarded. This file holds what the README
and the code cannot: how the work is done here, where it stands, and the framing
decisions that live in no file.

## What this is

A composition algebra over ODRL usage policies, applied to federated learning.
Composing contributor policies is **anti-monotone** — permissions intersect while
prohibitions and duties accumulate — so admitting a contributor can never widen
what you may lawfully do with the resulting model. Thirteen increments explore
what follows from that.

## How to work here

**One coherent increment per commit.** The commit body explains what was found
and, more importantly, what turned out to be wrong. Commit messages here average
~460 words and that is deliberate — they are the project's lab notebook.

**The standing gate before any commit:**

```bash
ruff check .          # clean
python -m pytest -q   # green (544 at the time of writing)
python experiments/make_figures.py
```

Figures are rendered from the committed CSVs, never from a fresh experiment, so a
figure can never disagree with the numbers the repository claims. The README is
updated in the same commit as the result it describes.

## The honesty constraints

These have shaped this project more than any technical decision, and they are
invisible from the code. They exist because each was learned by getting it wrong.

**A negative result is the result.** Do not tune until the numbers improve. The
estimators in increment 13 failed; that is reported as a finding, not hidden.

**Never select on the data you report on.** If a formula is chosen by looking at
a lattice, fidelity measured on that same lattice is measuring the tuning.

**Look at figures, do not trust that they parsed.** Checking visually caught two
rendering bugs and five figures whose notes had been silently truncated — SVG
does not wrap text and the browser reports no error.

**Verify before claiming.** Several confident intermediate conclusions in this
project were wrong and were caught only by checking. Two worth knowing:

- "max accuracy peaks at four contributors" was a sampling artefact — 94 subsets
  of size four against one of size eight, and a maximum over more draws is larger
  by construction
- "federated Cox loses to centralised" was an artefact of an unreproducible
  train/test split, and reversed once the split was pinned

**State what an experiment cannot settle.** Several results here are
inconclusive, and the evaluation scripts detect and announce that themselves
(saturated federations, noise floors exceeding the effect) rather than leaving a
reader to infer it.

## Where things stand

**Settled, and metric-independent.** Anything combinatorial: it follows from
composing declared terms and never touches a model. Permission erosion, the
non-transitivity of compatibility, the Apriori pruning, and the headline that a
policy-blind consortium may use its own model 20% of the time against 100%.

**Not settled.** Anything about model quality. The TCGA cohort is too small —
40 events in the held-out fold, a ±0.10 interval on any C-index — and the
synthetic task has no data-quality dimension at all.

## Open items, with status

**The bundle is not signed.** A forger who rewrites the declarations *and* the
agreement consistently produces a bundle that verifies, and a test asserts
exactly that. Tolerable for a study; disqualifying if the work is positioned as a
licensing mechanism (see below). Closing it needs signatures over the digest and
a key-distribution story; C2PA is the obvious prior art.

**Federated Cox optimises the wrong objective.** The partial likelihood is not a
sum over clients — its risk set spans every patient still at risk — so averaging
local gradients converges correctly to the wrong function. Andreux et al. (2020)
reformulate to discrete time. This matters beyond the current numbers: the size
of the mismatch grows with how fragmented the risk sets are, so it would bias any
study that varies consortium size, which is the main axis of interest.

**Regret is inflated by the winner's curse.** Increment 13 measures regret
against the single best-performing consortium observed. Small consortia vary far
more than large ones against a flat ceiling, so the argmax over a thousand
candidates is usually a small lucky draw rather than a good consortium. Fixing it
needs replicate training runs *within* each federation, roughly five times the
compute. Until then the published figures are an upper bound on estimator error.

**Monotonicity holds only under the standard FL assumptions.** "More contributors
is better" was measured under uniform data quality and a shared conditional —
`task.py` varies which labels each client holds, never what the right answer is.
Introduce quality variation or concept shift and it breaks.

## Framing decisions made in conversation, recorded nowhere else

**The strongest positioning is as a proposal for licensing derivative ML models**,
not as a study of permission erosion in FL. What this project has that nothing
else does is a composition *semantics* with a proven property, verification that
recomputes rather than compares, and recursion — a bundle's agreement composes
back in when the model is reused, so rights can only narrow and a restricted
model cannot be laundered into an unrestricted one. Most of the field offers
formats; this offers a calculus. The unsigned manifest is the blocker for that
framing.

**Most of this is not FL-specific, and claiming otherwise will not survive
review.** The algebra, erosion, selection and the measured cohesion all apply to
any multi-party derivative — a trusted research environment, a data lake, a
consortium agreement. Only increments 6, 7 and 9 are genuinely federated. The
honest framing: *governance of derivative artefacts under composed licences, with
FL as the setting where the machinery becomes executable rather than the thing
that makes it novel.*

**Centralising does not dissolve the problem**, which is the usual objection.
Pool twenty datasets and the resulting model is still a derivative of twenty sets
of terms. The erosion curve is identical; only the compute location changed.

**Declared permissions are self-penalising; declared quality is self-serving.**
Claiming more restrictions than you have only shrinks what you may do, which is
why declarations can be taken at face value and why "contributors declare, the
server does not assume" is sound. That trust model does **not** transfer to data
quality, where the incentive runs the other way. Quality must be measured, not
declared.

**The estimators are a negative result. Do not claim them as a contribution.**
All three collapse to "take the largest feasible consortium"; `divergence`'s
apparent advantage was an artefact of a five-member cap and vanished on the full
lattice; `gradient`, the novel one, anti-correlates with accuracy.

**Related work to position against:** SPDX and SBOM (mature licence composition,
but for software, not data or models); OpenRAIL (restrictions that bind
derivatives, deployed, but contractual rather than computed); GA4GH ADA-M and DUO
(machine-readable data conditions, but one dataset against one request, with no
composition); UCON and the IDS usage-control line (the theory of obligations
persisting after access); CycloneDX AI-BOM (inventories model ingredients without
deriving resulting rights). The one-line framing: *software has SPDX, data has
DUO, and neither tells you what you may do with a model derived from many sources
at once.*

**ODRL Profile: AI Vocabulary** (W3C Community Group draft, 30 September 2026,
`https://w3c.github.io/odrl/ai-vocab/`) standardises `data-mining`,
`ai-training`, `gen-ai`, `ai-augment`, `search`, `index` in a subsumption
hierarchy. It is an *opt-out* vocabulary — publisher against crawler — and it has
**no composition semantics and no vocabulary for the derivative model**. It
therefore strengthens the motivation rather than threatening it: a live standards
effort stopping exactly where this work starts. Worth citing, and worth adopting
`ai-training` as an action with the same `fidelity: exact | interpretive`
discipline already applied to DUO. Not worth restructuring around — it is a
two-day-old draft and `aggregate`/`derive` capture a distinction it lacks.

## The next action

**The partial-training analysis**, which needs no training at all. Every one of
the 5115 runs in `results/groundtruth.csv` stored its full 20-round curve in the
`curve` column. So:

> For R = 1, 2, 3, 5, 10, rank the feasible candidates by their round-R accuracy,
> take the top, and measure its regret against the round-20 truth.

That tests the two-step framework directly — shortlist by policy, then run short
exploratory federations — and yields a cost-fidelity curve. If a few rounds
recover a near-optimal consortium, that is the proof of concept the paper needs
and the one thing that would make step 2 claimable. If not, that is the finding.

Note the winner's-curse caveat applies to absolute regret but **not** to relative
comparisons between strategies, since the bias hits all of them equally.

## Setting up on a new machine

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"    # POSIX: .venv/bin/pip
python -m pytest -q
```

TCGA tests skip cleanly when the cohort is absent; `odrl_fl.tcga.fetch()`
retrieves it (136 KB from the GDC, deliberately not vendored — it is not ours to
redistribute).

**Everything expensive is already committed.** The three-hour ground-truth
lattice (`results/groundtruth.csv`) and the twenty-minute EGA corpus
(`vocab/ega_corpus.json`) travel with the repository and need no recomputation.
`python experiments/make_figures.py` regenerates all seven figures from the
committed CSVs without re-running a single experiment, which is the quickest
check that everything arrived.

Requires Python >= 3.11. Flower's simulation backend needs Ray, which ships no
Windows wheel for Python 3.13, so that path works on Linux but not on Windows
with 3.13 — the in-process fast path in `odrl_fl/fedavg.py` is unaffected and is
what every experiment uses.
