# Policy-Carrying Models

**ODRL usage policies that compose through federated aggregation.**

A trained federated model is a *derivative* of every contributor's data. If
those contributors granted their data under different conditions, what may
lawfully be done with the resulting model? ODRL defines no composition
semantics for derived assets, and FL makes the question unavoidable, because
aggregation is the primitive.

This project defines that composition, and builds the machinery to enforce it
inside a federated training run.

> Status: **increment 4 complete — the go/no-go passed.** Permission erosion is
> real, large, and driven by policy diversity rather than federation size.
> Increments 5–11 proceed with one revision to scope, below.

## The core claim

Composing contributor policies gives a capability set that is **anti-monotone**
in the contributor set:

```
S ⊆ S'   ⟹   Cap(S') ⊆ Cap(S)
```

Permissions intersect, prohibitions and duties accumulate, constraints conjoin.
So **admitting a contributor can never widen what you may do with the model, no
matter how permissive that contributor is** — it can only add data. Removing one
can only widen it.

Two consequences:

- **Permission erosion.** The more institutions you federate, the fewer things
  you are allowed to do with the result. This runs directly against the "more
  participants is better" instinct.
- **A trade-off with structure.** Capability is anti-monotone; data utility is
  (roughly) monotone increasing. Selection is choosing a point on that frontier.
  Anti-monotonicity also gives Apriori-style pruning: if a subset already lacks
  a required capability, every superset lacks it too.

## The algebra, and what it already shows

```bash
python experiments/compose_demo.py
```

Composing every subset of the five example contributor policies:

```
Mean capability by federation size:
  1 contributor(s)  3.00  ##################
  2 contributor(s)  2.30  ##############
  3 contributor(s)  1.80  ###########
  4 contributor(s)  1.40  ########
  5 contributor(s)  1.00  ######
```

Note `p3`, the national registry: it holds the most valuable data and grants
only `aggregate`, so **any** subset containing it collapses to that alone — the
federation could not even run inference with the resulting model.

Composition rules, all pushing the same direction:

| Element | Rule |
|---|---|
| Permissions | intersect — an action needs everyone's grant |
| Prohibitions | union — any single prohibition binds the derivative |
| Duties | union, merged per action — the strictest wins |
| Constraints | conjoin — set meet, or interval intersection |
| Conflict | deny overrides |

Capability is lost three distinct ways, and `CompositionResult.lost` separates
them so increment 4 can ablate their contributions: somebody withheld the
grant, somebody prohibited it, or the grantors' constraints have no common
ground.

### Subsumption is load-bearing, not decorative

`spatial isAnyOf {EU}` and `spatial isAnyOf {DE}` are compatible — Germany is in
the EU — and their meet is `{DE}`. Under a flat vocabulary they read as a
conflict and the capability is lost. Compare:

```bash
python experiments/compose_demo.py --flat
```

`p1+p4` keeps `use` with the taxonomy and loses it without. Measured erosion
would otherwise be partly an artifact of omitting the hierarchy rather than a
property of the policies, which would undermine increment 4's headline result.

## Grounding: GA4GH Data Use Ontology

```bash
python experiments/corpus_demo.py
```

The corpus is not invented. Code sets are sampled from **real data use
conditions** published on EGA policy records, fetched once and committed to
`vocab/duo_frequencies.json` with provenance:

```
404 of 3000 policies carry DUO codes -- 13.5% adoption
```

That number is a finding, not an inconvenience: structured, machine-readable
data use conditions are the exception in practice. So is the fact that
`DUO:0000005` (GRU-CC) and `DUO:0000014` (RU) account for 132 code occurrences
despite having been **retired from the ontology** — real governance metadata
drifts from the standard, and a tool that meets real data has to decide what to
do with obsolete terms.

Erosion over a corpus of 40 DUO-derived contributors:

```
  size  mean capability  retains any
     1             4.00        100%
     5             3.65         91%
    10             2.45         61%
    20             1.30         32%
    40             0.00          0%
```

Measured over *sampled* subsets rather than a prefix: individual subsets are
bimodal, since a single incompatible contributor collapses a whole subset to
nothing.

### Not every DUO code fits, and that is part of the result

`vocab/duo.json` classifies all 28 observed codes:

| Class | n | Meaning |
|---|---|---|
| `permission` | 8 | sets the purpose the data may serve (GRU, HMB, DS) |
| `restriction` | 8 | a constraint or prohibition the algebra folds (NCU, GS, TS, NMDS) |
| `externally-adjudicated` | 4 | composes syntactically; an authority decides its truth out of band (IRB, US, PS, IS) |
| `deprecated` | 3 | obsolete, still present in real annotations |
| `duty` | 2 | an obligation rather than a restriction (PUB, RTN) |
| `structural` | 2 | an ontology category, not a usable condition |
| `out-of-fragment` | 1 | **COL** — cannot be expressed at all |

**COL is the interesting one.** "Collaboration required" means the data
provider must be a participant, so its satisfaction depends on the very
contributor set being composed. That is self-referential — precisely the class
the fragment excludes. A widely-used standard containing a construct the
algebra cannot take is worth reporting rather than hiding.

Every code carries a `fidelity` field (`exact` or `interpretive`), so the
places where DUO and ODRL do not line up cleanly are visible.

## The result: permission erosion is real

```bash
python experiments/run_erosion.py
```

40 DUO-derived contributors, 60 subset draws per size, 5 corpus seeds,
95% bootstrap intervals (bootstrap because the statistic is bimodal — a subset
either keeps its capability or loses all of it):

```
  size          capability  retains        purpose retained  duties
     1    4.00 [4.00,4.00]    100%     0.587 [0.551,0.624]    0.20
     3    3.57 [3.44,3.71]     89%     0.295 [0.266,0.324]    0.62
     5    3.07 [2.87,3.25]     77%     0.195 [0.170,0.220]    0.85
    10    1.79 [1.56,2.01]     45%     0.072 [0.059,0.087]    1.10
    20    0.28 [0.17,0.40]      7%     0.006 [0.003,0.009]    1.39
    30    0.01 [0.00,0.04]      0%     0.000 [0.000,0.001]    1.56
```

Note the two columns moving in opposite directions. **Capability falls while
duties accumulate** — permissions intersect and obligations union — so a growing
federation simultaneously loses what it may do and gains what it must do.

At `heterogeneity = 0` the same curve is perfectly flat at 4.00 / 100% for every
size. **Erosion is a property of policy diversity, not of federation size.**

### Go/no-go verdict

| Criterion | | Result |
|---|---|---|
| non-trivial | PASS | capability 4.00 → 1.79 at 10 contributors (55% drop), intervals disjoint |
| non-degenerate | PASS | 45% of 10-contributor subsets retain capability |
| attributable | **FAIL** | purpose=98%, spatial=2% |
| diversity-driven | PASS | 0% drop homogeneous vs 55% diverse |

Criteria were fixed **before** the study ran, and the blocking/advisory split
was fixed with them: a trivial or degenerate result stops the project, a
surprise about mechanism calls for rescoping. Both blocking criteria passed.

**The `attributable` failure is a finding, not a defect,** and the criterion was
mis-specified rather than the data disappointing. It expected several
mechanisms; the data says erosion is almost entirely **incompatible research
purposes**. That follows from the vocabulary: DUO expresses *limitations*, so
every contributor grants the same actions and a withheld grant cannot occur,
while its only prohibitions are constrained and refine a permission rather than
removing it. One mechanism at action level is a property of DUO, not an
artifact of the algebra.

The revised claim is sharper than the one planned: *if you want to federate,
purpose compatibility is the thing to negotiate.* That also tells increment 5's
selection search what to prioritise.

**Robustness.** Re-running with only `use` granted — the alternative reading of
what DUO licenses — gives 59% drop and 41% retention against 55% and 45%, with
the same attribution. The conclusions do not rest on the `DEFAULT_ACTIONS`
assumption.

### How much does this depend on *who holds which policy*?

```bash
python experiments/run_sensitivity.py
```

The main threat to validity. EGA supplies real code frequencies and real
co-occurrence, but nothing about which institution holds which conditions, so
the corpus must assume something. Seven assignment models, same curve:

| assignment model | drop at 10 | retains any | effect |
|---|---|---|---|
| uniform | 76% | 24% | large |
| **empirical** (baseline) | 64% | 36% | large |
| clustered-0.3 | 49% | 51% | moderate |
| **purpose-matched** | 42% | 58% | moderate |
| marginal | 32% | 68% | small |
| clustered-0.6 | 17% | 83% | small |
| clustered-0.9 | 5% | 95% | small |

**Erosion occurs under every model — the direction is robust. The magnitude is
not**, ranging from 5% to 76%. Stated plainly: had `clustered-0.6` been chosen
as the baseline, the pre-registered go/no-go criterion would not have been met.
Any figure quoted in the paper must name its assignment model.

Cohesion is what drives it. Erosion decays smoothly as consortium members come
to resemble one another and largely disappears once most are alike — which is
the honest answer to "but real consortia are thematically coherent."

**The load-bearing row is `purpose-matched`.** A consortium that has *already*
self-selected for compatible research purposes still loses 42% of its
capability, because **compatibility is not transitive**: two members can each be
compatible with the anchor and incompatible with one another
(`research ⊓ ds-cancer` and `research ⊓ ds-neurological` are both fine;
`ds-cancer ⊓ ds-neurological` is empty). Self-selection is necessary but not
sufficient — which is precisely the argument for selecting explicitly, and why
selection is a set-level search rather than a filter.

## Selection: recovering a usable consortium

Erosion says a federation assembled without regard to policy usually cannot
lawfully use the model it produces. Selection turns that into a search: among
all subsets of a pool, find those whose composed policy satisfies a stated
requirement, and among those, the ones offering the most data.

**It is a search, not a filter.** Because compatibility is not transitive,
candidates cannot be vetted one at a time against a reference policy —
feasibility is a property of the set. Two contributors can each satisfy the
requirement alone and grant nothing together
(`tests/test_selection.py::test_two_viable_contributors_can_be_jointly_infeasible`).

**It is tractable anyway.** Anti-monotonicity makes feasibility *downward
closed*: every subset of a feasible set is feasible, so a contributor that fails
alone can be dropped from the pool outright, and a (k+1)-set need only be
considered when all its k-subsets are feasible. That is exactly Apriori's
candidate generation. The pruning is sound only on the legal side — utility is
empirical and not reliably monotone, so pruning on it could discard the optimum.

A test asserts the search returns **exactly** the maximal sets an exhaustive
scan finds, across several seeds. Pruning must not cost completeness.

### A request asks for containment, not overlap

Getting this wrong quietly voids the whole exercise, and it did at first. A
request constraint was tested with `taxonomy.meet` — non-empty intersection — so
an agreement permitting only `biomedical-research` answered *yes* to "may we use
this for research". A consortium would believe it held a right it did not have.

It also made the purpose knob a no-op: because `research` overlaps nearly every
DUO-derived purpose in the corpus, runs with and without a purpose constraint
produced byte-identical search statistics. The fix is `taxonomy.difference` —
every term the request names must already be permitted. On the request side
`isAnyOf {a, b}` reads as "either branch may be taken", so both must be granted:
you cannot know in advance which one gets taken, and a consortium that is only
*sometimes* lawful is not a viable consortium.

### Request tightness is what makes selection hard

Pool of 16, 5 corpus seeds, requiring `use` and `aggregate`:

| request | viable alone | subsets evaluated | search space avoided | policy-aware yield | policy-blind yield |
|---|---|---|---|---|---|
| actions only | 16.0/16 | 25,634 | 60.9% | 100% | 20% |
| `purpose = ds-cancer` | 12.8/16 | 10,652 | 83.8% | 100% | 0% |
| `purpose = research` | 9.0/16 | 889 | **98.6%** | 100% | 0% |

**Deployable utility** — data drawn from a consortium actually permitted to use
it — is the only comparison that counts. A policy-blind consortium scores its
full data volume when it happens to be feasible and zero otherwise:

| request | policy-aware | policy-blind |
|---|---|---|
| actions only | 114.2 | 21.4 |
| `purpose = ds-cancer` | 95.9 | 0.0 |
| `purpose = research` | 83.1 | 0.0 |

Read the two ends of the pruning column together rather than quoting the
favourable one. **Apriori pays off precisely where selection is hard**, and
degenerates toward exhaustive enumeration where the requirement is so permissive
that almost any consortium would have satisfied it anyway. That is a property of
the problem, not a deficiency of the algorithm: the feasible family really is
that large when the constraint is weak — 14,463 feasible sets out of 65,535 at
seed 0 under the unconstrained request.

Two caveats worth stating plainly. The 100% policy-aware yield is **structural,
not empirical**: the search returns maximal feasible sets, so it succeeds
whenever any contributor is viable alone. The interesting column is the
policy-blind one next to it. And finding maximal sets by enumerating the whole
feasible family is exponential in that family's size; `max_size` caps it, but no
better algorithm is implemented.

## Increment roadmap

Ordered by **risk of invalidation**, not by build order — the expensive parts
sit behind cheap gates. Increments 0–5 involve no federated learning at all.

| # | Increment | Status |
|---|---|---|
| 0 | Vanilla Flower baseline | **done** |
| 1 | ODRL data model + JSON-LD subset parser | **done** |
| 2 | Composition algebra + property tests for anti-monotonicity | **done** |
| 3 | GA4GH **DUO** → ODRL mapping, policy corpus | **done** |
| 4 | **Permission-erosion study** ← first result, **go/no-go** | **done — PROCEED** |
| 5 | Selection with Apriori pruning; utility proxy | **done** |
| 6 | Policy-aware `Strategy`; selection at `start()` | |
| 7 | Enforcement hooks — duties dispatched and verified | |
| 8 | Policy-carrying model export + provenance log | |
| 9 | Periodic re-evaluation (expiry, count, duty violation) | |
| 10 | Real medical dataset + experiment harness | |
| 11 | Results and figures | |

Increment 4 is the decision point. If capability erosion turns out to be either
trivial (almost nothing erodes) or degenerate (everything collapses at two
contributors), there is no paper, and finding that out costs a few sessions
rather than a few months.

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate     # POSIX: . .venv/bin/activate
```

```bash
pip install -e ".[dev]"
```

```bash
python experiments/run_fedavg.py --clients 6 --rounds 5
```

The three studies, in the order they were run:

```bash
python experiments/run_erosion.py
```

```bash
python experiments/run_sensitivity.py
```

```bash
python experiments/run_selection.py
```

```bash
pytest
```

## Two execution paths, on purpose

| | Path | Used for |
|---|---|---|
| **Fast** | `odrl_fl/fedavg.py` | experiment sweeps — in-process, deterministic, no transport |
| **Real** | `odrl_fl/{server,client}_app.py` | the Flower integration, where policy enforcement hooks will live |

The evaluation compares *subsets* of contributors, and the selection problem
searches a powerset — potentially hundreds of runs. Standing up a federation per
subset costs tens of seconds of orchestration for a model that trains in under
one. So sweeps use the in-process loop, and Flower carries the systems
contribution.

Both report the **same metric**: accuracy of the global model on the shared
held-out test set. Establishing that they agree numerically is an increment-6
task; today the Flower path is verified as "it runs and learns", which it does
(0.195 → 0.553 over five rounds).

Running the Flower path needs the deployment engine, because **Flower's
simulation backend requires Ray, and Ray ships no Windows wheels for Python
3.13**. Register `[superlink.odrl-local] address = "127.0.0.1:9093"` in
`~/.flwr/config.toml`, then start `flower-superlink`, one `flower-supernode` per
client, and `flwr run . odrl-local --stream`. On Linux (ds01), `flwr run` with a
simulation federation should work directly.

## Design notes

**Synthetic data, for now.** `odrl_fl/task.py` generates tabular data with a
Dirichlet label skew whose `alpha` dials heterogeneity from near-IID to
pathological. A fixed real dataset gives no such dial, and heterogeneity is an
independent variable in the evaluation. `load_partition()` is the seam where a
real federated medical dataset swaps in at increment 10 — Fed-TCGA-BRCA is the
current candidate, because its dbGaP controlled-access heritage means real DUO
codes can be cited for it rather than only sampled.

Two properties everything downstream depends on, both pinned by tests:

**A client's partition is keyed on its own id**, so client 3 holds the same data
whether the federation has 4 members or 40. Otherwise "the model trained on
{A,C}" and "the model trained on {A,B,C}" would differ for two reasons at once.

**Models are always scored on a shared, balanced, held-out test set**, never on
the contributors' own splits. This was a real bug during increment 0: scoring
each subset on its own data made a *single* skewed client out-score the full
federation, because it was tested only on the two classes it happened to hold.

The task is tuned (`CLASS_SEPARATION`) to leave headroom, since a task that
saturates makes every subset look identical and the evaluation vacuous:

| alpha | 1 contributor | 6 contributors |
|---|---|---|
| 1.0 (default) | 0.500 | **0.965** |
| 100 (IID) | 0.996 | 0.996 |

The IID row is the control — under IID one client already sees the whole
distribution, so federating adds nothing. The utility signal comes from label
skew, which is also the realistic case: institutions specialise.

## Related work by the same authors

Independent line of work on aggregator fault tolerance
(`SINTEF-9012/raft_flower`, TAAS 2026): RAFT-based self-recovery when the
aggregator fails. Cited as related work, not built upon — there is no consensus
requirement here.
