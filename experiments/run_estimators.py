"""Do training-free estimators preserve the Pareto frontier?

`run_groundtruth.py` trains every subset of a ten-contributor pool, so the true
frontier over (accuracy, capability, duties) is known exactly rather than
sampled. This scores the same subsets with estimators that never train anything
and asks how much of that frontier survives.

The framing matters. Asking "how accurately does the estimator predict accuracy"
sets a bar selection does not need to clear: the search never uses the number,
only the ordering, and mostly only the part of the ordering that decides
non-dominance. So the metrics are

    Kendall tau   does it rank the candidates correctly at all
    recall        how much of the true frontier the estimator's frontier keeps
    precision     how much of what it proposes really is non-dominated
    regret        accuracy actually lost by deploying its pick

Regret is the one a practitioner cares about, and it can be small even when the
rank correlation is mediocre -- being wrong about the middle of the ordering
costs nothing if the top is right.

**Every seed is a separate federation and is scored on its own.** In `run_fedavg`
the seed reaches `load_partition` and regenerates the dataset, so it decides
which institution holds which data rather than merely how the weights start.
Pooling accuracies across seeds would average over different worlds and answer
"which contributor *index* is best on average", which nobody has ever needed to
know. The question worth asking is whether the estimator picks well *within* one
federation, so each seed runs end to end and the seeds then serve as
replications -- which is also where the uncertainty comes from, instead of from a
noise floor that was largely partition variance wearing a disguise.

    python experiments/run_estimators.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from odrl_fl.corpus import generate_corpus
from odrl_fl.erosion import bootstrap_ci
from odrl_fl.estimators import ESTIMATORS
from odrl_fl.odrl import compose

RESULTS = Path(__file__).resolve().parent.parent / "results"
GROUNDTRUTH = RESULTS / "groundtruth.csv"
OUT = RESULTS / "estimators.csv"


def kendall_tau(a: list[float], b: list[float]) -> float:
    """Rank correlation, ties handled (tau-b). Implemented rather than imported.

    The project has no scipy dependency and this is the only statistic that
    would need one. O(n^2) over a few hundred items is a moment's work.
    """
    n = len(a)
    concordant = discordant = ties_a = ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                ties_a += 1
                ties_b += 1
            elif da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    denom = np.sqrt((total + ties_a) * (total + ties_b))
    return float((concordant - discordant) / denom) if denom > 0 else 0.0


def pareto_front(points: list[tuple[float, frozenset, int]]) -> set[int]:
    """Indices of the non-dominated points over (utility, capability, duties).

    Same dominance rule as `selection.Candidate.dominates`: at least as good
    everywhere, strictly better somewhere, with duties counting against.
    """
    front = set()
    for i, (ui, ci, di) in enumerate(points):
        dominated = False
        for j, (uj, cj, dj) in enumerate(points):
            if i == j:
                continue
            at_least = uj >= ui and cj >= ci and dj <= di
            strictly = uj > ui or cj > ci or dj < di
            if at_least and strictly:
                dominated = True
                break
        if not dominated:
            front.add(i)
    return front


def load_groundtruth(
    path: Path,
) -> tuple[dict[tuple[int, ...], dict[int, float]], dict]:
    """Subset -> {seed: accuracy}, plus the settings it was trained under.

    Keyed by seed rather than collapsed into a list, because each seed is a
    different federation and the evaluation must never mix them.

    The settings come back rather than being assumed, because the estimators have
    to be built at the same heterogeneity the models were trained at. Scoring an
    estimator built at alpha 1.0 against models trained at alpha 0.3 compares two
    different worlds, and nothing about the output would look wrong.
    """
    truth: dict[tuple[int, ...], dict[int, float]] = {}
    settings: dict[str, set] = {"alpha": set(), "rounds": set()}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            members = tuple(int(m) for m in row["members"].split("-"))
            truth.setdefault(members, {})[int(row["seed"])] = float(row["accuracy"])
            if row.get("alpha"):
                settings["alpha"].add(float(row["alpha"]))
            if row.get("rounds"):
                settings["rounds"].add(int(row["rounds"]))
    for key, values in settings.items():
        if len(values) > 1:
            raise ValueError(
                f"ground truth mixes {key} values {sorted(values)}; "
                "it must be one setting throughout"
            )
    return truth, {k: (v.pop() if v else None) for k, v in settings.items()}


def evaluate_seed(
    seed: int,
    pool: int,
    alpha: float,
    truth: dict[tuple[int, ...], dict[int, float]],
) -> dict[str, dict] | None:
    """Score every estimator inside one federation.

    The corpus is regenerated at this seed as well, so a "world" means both who
    holds which data and who holds which terms -- which is what differs between
    one real federation and the next.

    `None` when fewer than two consortia are feasible: there is nothing to rank,
    and a frontier of one would report a vacuous 100%.
    """
    corpus = generate_corpus(pool, seed=seed)
    subsets = [s for s in sorted(truth, key=lambda s: (len(s), s)) if seed in truth[s]]

    legal = {}
    for members in subsets:
        result = compose([corpus[i].policy for i in members])
        agreement = result.agreement
        duties = sum(len(p.duties) for p in agreement.permissions)
        duties += len(agreement.obligations)
        legal[members] = (result.capability, duties)

    # Only feasible consortia belong on a frontier. An impermissible one has zero
    # *deployable* utility however well it trains -- the rule increment 5
    # established -- so leaving it in would let a model nobody may use dominate
    # one they may.
    feasible = [s for s in subsets if legal[s][0]]
    if len(feasible) < 2:
        return None

    accuracy = [truth[s][seed] for s in feasible]
    truth_points = [
        (accuracy[i], legal[s][0], legal[s][1]) for i, s in enumerate(feasible)
    ]
    true_front = pareto_front(truth_points)
    best_index = max(range(len(feasible)), key=lambda i: accuracy[i])

    # How much accuracy separates the frontier's own points. When the task has a
    # ceiling, every frontier point sits in the saturated region and this spread
    # collapses -- at which point *which* points are non-dominated is decided by
    # noise, and recall and precision stop measuring anything about the
    # estimator. Recorded so the reader is told rather than left to infer it.
    front_accuracy = [accuracy[i] for i in true_front]
    front_spread = max(front_accuracy) - min(front_accuracy)

    out: dict[str, dict] = {}
    for name, factory in ESTIMATORS.items():
        estimate = factory(pool, alpha=alpha, seed=seed)
        scores = [estimate([corpus[i] for i in members]) for members in feasible]
        est_points = [
            (scores[i], legal[s][0], legal[s][1]) for i, s in enumerate(feasible)
        ]
        est_front = pareto_front(est_points)
        overlap = true_front & est_front
        pick = max(est_front, key=lambda i: est_points[i][0])
        order = sorted(range(len(feasible)), key=lambda i: -scores[i])

        out[name] = {
            "pick_size": len(feasible[pick]),
            "best_size": len(feasible[best_index]),
            "tau": kendall_tau(scores, accuracy),
            "recall": len(overlap) / len(true_front),
            "precision": len(overlap) / len(est_front),
            "regret": accuracy[best_index] - accuracy[pick],
            "rank_of_best": order.index(best_index) + 1,
            "n_feasible": len(feasible),
            "front_size": len(true_front),
            "front_spread": front_spread,
            "ceiling": max(accuracy),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimator frontier fidelity")
    parser.add_argument("--pool", type=int, default=10)
    parser.add_argument("--truth", default=str(GROUNDTRUTH))
    args = parser.parse_args(argv)

    truth_path = Path(args.truth)
    if not truth_path.exists():
        print(f"No ground truth at {truth_path}.")
        print("Run: python experiments/run_groundtruth.py")
        return 1

    truth, settings = load_groundtruth(truth_path)
    alpha = settings["alpha"]
    if alpha is None:
        print("Ground truth predates the alpha/rounds columns and cannot be")
        print("matched to an estimator setting. Re-run run_groundtruth.py.")
        return 1

    # The lattice may be capped at a maximum consortium size, so completeness
    # is "every subset up to the largest size present", not "all 2^n - 1".
    # Inferring the bound from the data rather than taking it on trust means a
    # run that stopped halfway through a size still reports as incomplete.
    max_size = max(len(s) for s in truth)
    expected = sum(comb(args.pool, k) for k in range(1, max_size + 1))
    if len(truth) < expected:
        print(f"Ground truth is incomplete: {len(truth)} of {expected} subsets "
              f"up to size {max_size}.")
        print("Re-run experiments/run_groundtruth.py -- it resumes.")
        return 1
    if max_size < args.pool:
        print(f"  NOTE: capped at consortia of {max_size}; this says nothing "
              f"about larger ones\n")

    seeds = sorted({seed for runs in truth.values() for seed in runs})
    print(f"Estimator fidelity, pool {args.pool}")
    print(f"  trained at alpha {alpha}, {settings['rounds']} rounds")
    print(f"  {len(seeds)} seeds, each a separate federation, scored separately\n")

    per_seed: dict[str, list[dict]] = {name: [] for name in ESTIMATORS}
    for seed in seeds:
        result = evaluate_seed(seed, args.pool, alpha, truth)
        if result is None:
            print(f"  seed {seed}: fewer than two feasible consortia, skipped")
            continue
        for name, metrics in result.items():
            per_seed[name].append(metrics)

    if not any(per_seed.values()):
        print("No federation produced a usable comparison.")
        return 1

    sample = next(v for v in per_seed.values() if v)
    print(f"  feasible per federation  {np.mean([m['n_feasible'] for m in sample]):.0f}")
    print(f"  true frontier size       {np.mean([m['front_size'] for m in sample]):.1f}")
    print(f"  federations scored       {len(sample)}")

    # Per federation, not averaged. Averaging the spread hides the thing that
    # matters most here: some federations are saturated, where every consortium
    # performs alike and no estimator can be wrong, and some discriminate, where
    # the choice is worth tens of accuracy points. A mean over the two describes
    # neither, and reports a comfortable number for a study that is really two
    # different studies stapled together.
    spreads = [m["front_spread"] for m in sample]
    saturated_seeds = [i for i, v in enumerate(spreads) if v < 0.01]

    print("\n  per federation")
    print(f"    {'seed':>5}  {'feasible':>8}  {'frontier':>8}  {'spread':>8}  "
          f"{'ceiling':>8}")
    for i, m in enumerate(sample):
        mark = "  (saturated)" if m["front_spread"] < 0.01 else ""
        print(f"    {i:>5}  {m['n_feasible']:>8}  {m['front_size']:>8}  "
              f"{m['front_spread']:>8.4f}  {m['ceiling']:>8.4f}{mark}")

    if saturated_seeds:
        print(f"\n  {len(saturated_seeds)} of {len(spreads)} federations are "
              "saturated: every consortium performs")
        print("  alike, so no estimator can be wrong there and recall and "
              "precision")
        print("  measure nothing. The discriminating federations carry the "
              "result.")
    print()

    # Regret is measured against the single best-performing consortium observed,
    # and that maximum is itself a draw. Small consortia are far more variable
    # than large ones -- the standard deviation of accuracy roughly halves from
    # one contributor to ten -- while the ceiling is flat, so the argmax over
    # hundreds of candidates is usually a small consortium that got lucky rather
    # than a good one. An estimator that picks the choice with the best
    # *expected* accuracy is then charged for not having guessed the lucky draw.
    # Printed so the asymmetry is visible instead of buried in the number.
    best_sizes = [m["best_size"] for m in sample]
    print(f"\n  the best observed consortium had {np.mean(best_sizes):.1f} members "
          f"on average")
    print("  (sizes: " + ", ".join(str(b) for b in best_sizes) + ")")
    print("  Small consortia vary more and the ceiling is flat, so the argmax is")
    print("  often a lucky small draw. Regret against it overstates estimator "
          "error.\n")

    print(f"  {'estimator':>12}  {'picks':>6}  {'tau':>7}  {'recall':>7}  "
          f"{'precision':>9}  {'regret (95% CI)':>26}")
    rows = []
    for name, metrics in per_seed.items():
        if not metrics:
            continue
        regrets = [m["regret"] for m in metrics]
        lo, hi = bootstrap_ci(regrets)
        tau = float(np.mean([m["tau"] for m in metrics]))
        recall = float(np.mean([m["recall"] for m in metrics]))
        precision = float(np.mean([m["precision"] for m in metrics]))
        regret = float(np.mean(regrets))

        pick_size = float(np.mean([m["pick_size"] for m in metrics]))
        print(f"  {name:>12}  {pick_size:>6.1f}  {tau:>7.3f}  {recall:>7.1%}  "
              f"{precision:>9.1%}  {regret:>9.4f} [{lo:.4f}, {hi:.4f}]")
        rows.append({
            "estimator": name,
            "kendall_tau": round(tau, 4),
            "frontier_recall": round(recall, 4),
            "frontier_precision": round(precision, 4),
            "regret": round(regret, 4),
            "regret_lo": round(lo, 4),
            "regret_hi": round(hi, 4),
            "rank_of_best": round(
                float(np.mean([m["rank_of_best"] for m in metrics])), 1
            ),
            "n_feasible": round(float(np.mean([m["n_feasible"] for m in metrics])), 1),
            "true_frontier_size": round(
                float(np.mean([m["front_size"] for m in metrics])), 2
            ),
            "front_spread": round(
                float(np.mean([m["front_spread"] for m in metrics])), 5
            ),
            "pick_size": round(float(np.mean([m["pick_size"] for m in metrics])), 1),
            "best_size": round(float(np.mean([m["best_size"] for m in metrics])), 1),
            "seeds": len(metrics),
        })

    RESULTS.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {OUT}")

    # An interval straddling zero means the federations disagree about whether
    # the estimator costs anything, which is not the same as it being free.
    print("\nReading\n")
    if saturated_seeds:
        print(f"  Frontier recall and precision are not interpretable: "
              f"{len(saturated_seeds)} of")
        print(f"  {len(spreads)} federations are saturated, so non-dominance is "
              "settled by noise")
        print("  rather than by the estimator. Regret is the only usable number,")
        print("  and it is an upper bound: it charges the estimator for failing to")
        print("  guess a lucky small draw it had no way to identify.\n")
    costly = [r for r in rows if r["regret_lo"] > 0.0]
    unclear = [r for r in rows if r["regret_lo"] <= 0.0]
    if costly:
        worst = max(costly, key=lambda r: r["regret"])
        print("  Costs real accuracy, interval excludes zero: "
              f"{', '.join(r['estimator'] for r in costly)}.")
        print(f"  Worst is {worst['estimator']}, at {worst['regret']:.4f} accuracy.")
    if unclear:
        print(f"  Not separable from zero: {', '.join(r['estimator'] for r in unclear)}.")
        print("  Either free, or the federations disagree -- more seeds would say")
        print("  which, and until then neither reading should be quoted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
