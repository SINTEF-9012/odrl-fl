# Policy-Carrying Models

**ODRL usage policies that compose through federated aggregation.**

A trained federated model is a *derivative* of every contributor's data. If
those contributors granted their data under different conditions, what may
lawfully be done with the resulting model? ODRL defines no composition
semantics for derived assets, and FL makes the question unavoidable, because
aggregation is the primitive.

This project defines that composition, and builds the machinery to enforce it
inside a federated training run.

> Status: **increment 0** — vanilla Flower baseline. No ODRL yet.

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

## Increment roadmap

Ordered by **risk of invalidation**, not by build order — the expensive parts
sit behind cheap gates. Increments 0–5 involve no federated learning at all.

| # | Increment | Status |
|---|---|---|
| 0 | Vanilla Flower baseline | **done** |
| 1 | ODRL data model + JSON-LD subset parser | |
| 2 | Composition algebra + property tests for anti-monotonicity | |
| 3 | GA4GH **DUO** → ODRL mapping, policy corpus | |
| 4 | **Permission-erosion study** ← first result, **go/no-go** | |
| 5 | Selection with Apriori pruning; utility proxy | |
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
