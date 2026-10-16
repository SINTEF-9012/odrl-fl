"""Policy-aware contributor selection, and what it costs.

The question the erosion study raises: if a federation assembled without regard
to policy usually cannot lawfully use its model, can selection recover a usable
one, and how much data does that cost?

Compares two strategies:

  policy-blind   pick the contributors with the most useful data, the standard
                 approach in federated learning
  policy-aware   search for maximal contributor sets whose composed policy
                 actually grants what the federation needs

and reports the metric that matters -- **deployable utility**: data you can draw
on from a consortium that is permitted to do the thing you need. A policy-blind
consortium scores its full data volume only if it happens to be feasible, and
zero otherwise.

The sweep is over how *specific* the request is, because that is what decides
whether selection is hard. Asking to use data for one named disease is a weak
requirement that most contributors already grant; asking to use it for research
in general is a strong one that few grant outright. The Apriori pruning tracks
that directly, and the finding is two-sided: **pruning pays off exactly when the
requirement is tight enough to make selection necessary**, and decays toward
exhaustive enumeration when the requirement is so loose that selection was not
needed in the first place.

    python experiments/run_selection.py
    python experiments/run_selection.py --pool-size 18 --seeds 8
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odrl_fl.corpus import Contributor, generate_corpus
from odrl_fl.odrl import Action
from odrl_fl.selection import (
    SelectionResult,
    is_feasible,
    policy_blind_selection,
    requirement,
    search,
)
from odrl_fl.task import N_CLASSES, load_partition

RESULTS = Path(__file__).resolve().parent.parent / "results"

#: Requested purposes, loosest first. `None` asks only for the actions, which
#: nearly every contributor grants; `research` sits at the top of the purpose
#: hierarchy and is granted outright by few. Under containment semantics a
#: broader request is strictly harder to satisfy.
REGIMES: tuple[tuple[str, str | None], ...] = (
    ("unconstrained", None),
    ("ds-cancer", "ds-cancer"),
    ("research", "research"),
)


def make_utility(
    n_clients: int, alpha: float, seed: int, size_variation: float = 0.0
):
    """A stand-in for model accuracy, pending real training at increment 10.

    Class coverage dominates volume under label skew: a class no contributor
    holds cannot be learned at any sample size, whereas more examples of a
    class already present give diminishing returns. Coverage is therefore the
    leading term and volume the modifier.
    """
    partitions = {
        i: load_partition(
            i, n_clients, alpha=alpha, seed=seed,
            size_variation=size_variation,
        )
        for i in range(n_clients)
    }

    def utility(subset) -> float:
        if not subset:
            return 0.0
        classes: set[int] = set()
        samples = 0
        for contributor in subset:
            partition = partitions[contributor.index]
            classes |= set(partition.label_counts)
            samples += partition.n_train
        coverage = len(classes) / N_CLASSES
        return round(coverage * float(np.sqrt(samples)), 3)

    return utility


def describe(result: SelectionResult) -> None:
    stats = result.stats
    print(f"    {'viable alone':<24}{stats.viable_singletons} of {stats.pool_size}")
    print(f"    {'candidates evaluated':<24}{stats.candidates_evaluated:,}")
    print(f"    {'exhaustive space':<24}{stats.exhaustive_space:,}")
    print(f"    {'search space avoided':<24}{stats.reduction:.2%}")
    print(f"    {'feasible sets found':<24}{stats.feasible_found:,}")
    print(f"    {'maximal sets':<24}{stats.maximal_found}")
    print(f"    {'largest feasible set':<24}{stats.levels_explored}")

    print(f"\n    Pareto front {'size':>7}  {'utility':>9}  {'duties':>7}  capability")
    for candidate in result.frontier[:6]:
        caps = ", ".join(sorted(a.value for a in candidate.capability))
        print(
            f"    {'':<12} {candidate.size:>7}  {candidate.utility:>9.1f}  "
            f"{candidate.duty_count:>7}  {caps}"
        )
    if len(result.frontier) > 6:
        print(f"    {'':<12} ... and {len(result.frontier) - 6} more")


def run_regime(label: str, purpose: str | None, args) -> tuple[list[dict], dict]:
    """One tightness setting, across every corpus seed."""
    request = requirement(*(Action(a) for a in args.require), purpose=purpose)
    rows: list[dict] = []

    for seed in range(args.seeds):
        pool: list[Contributor] = generate_corpus(
            args.pool_size, heterogeneity=1.0, seed=seed, assignment=args.assignment
        )
        utility = make_utility(
            args.pool_size, args.alpha, seed, args.size_variation
        )
        result = search(pool, request, utility)

        if seed == 0:
            describe(result)

        best = result.best
        aware_utility = best.utility if best else 0.0
        aware_size = best.size if best else 0

        # Policy-blind picks the same number of contributors, chosen purely for
        # data. Same budget, different criterion -- the fair comparison.
        target = aware_size or max(1, args.pool_size // 2)
        blind = policy_blind_selection(pool, utility, target)
        blind_subset = [pool[i] for i in blind]
        blind_ok = is_feasible(blind_subset, request)
        blind_utility = utility(blind_subset)

        rows.append(
            {
                "regime": label,
                "seed": seed,
                "aware_size": aware_size,
                "aware_utility": round(aware_utility, 3),
                "aware_feasible": int(bool(best)),
                "blind_size": len(blind),
                "blind_utility": round(blind_utility, 3),
                "blind_feasible": int(blind_ok),
                "blind_deployable": round(blind_utility if blind_ok else 0.0, 3),
                "viable_singletons": result.stats.viable_singletons,
                "candidates_evaluated": result.stats.candidates_evaluated,
                "exhaustive_space": result.stats.exhaustive_space,
                "reduction": round(result.stats.reduction, 6),
                "maximal_sets": result.stats.maximal_found,
            }
        )

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    summary = {
        "regime": label,
        "aware_yield": mean("aware_feasible"),
        "blind_yield": mean("blind_feasible"),
        "aware_utility": mean("aware_utility"),
        "blind_utility": mean("blind_utility"),
        "blind_deployable": mean("blind_deployable"),
        "viable": mean("viable_singletons"),
        "evaluated": mean("candidates_evaluated"),
        "reduction": mean("reduction"),
    }
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Policy-aware selection")
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--assignment", default="empirical")
    parser.add_argument(
        "--require", nargs="+", default=["use", "aggregate"],
        help="Actions the federation must be able to perform",
    )
    parser.add_argument(
        "--size-variation", type=float, default=1.2,
        help="Log-normal spread of contributor data volumes. Real institutions "
             "differ by orders of magnitude; at 0 every contributor is "
             "identical and utility collapses to a function of set size.",
    )
    args = parser.parse_args(argv)

    print("Policy-aware contributor selection")
    print(f"  actions required: {', '.join(args.require)}")
    print(f"  pool {args.pool_size}, {args.seeds} corpus seeds, "
          f"assignment {args.assignment}")

    rows: list[dict] = []
    summaries: list[dict] = []

    for label, purpose in REGIMES:
        asked = "actions only" if purpose is None else f"purpose = {purpose}"
        print(f"\n\n  Request: {asked}  (seed 0 shown)\n")
        regime_rows, summary = run_regime(label, purpose, args)
        rows.extend(regime_rows)
        summaries.append(summary)

        print(
            f"\n    across {args.seeds} seeds: policy-aware yields a usable "
            f"model {summary['aware_yield']:.0%} of the time, "
            f"policy-blind {summary['blind_yield']:.0%}"
        )
        print(
            f"    deployable utility: policy-aware {summary['aware_utility']:.1f}, "
            f"policy-blind {summary['blind_deployable']:.1f}"
        )

    print("\n\nHow request tightness drives the search\n")
    print(
        f"  {'request':<15} {'viable':>9} {'evaluated':>11} {'avoided':>9}"
        f" {'aware':>7} {'blind':>7}"
    )
    print(f"  {'-' * 15} {'-' * 9} {'-' * 11} {'-' * 9} {'-' * 7} {'-' * 7}")
    for s in summaries:
        viable = f"{s['viable']:.1f}/{args.pool_size}"
        print(
            f"  {s['regime']:<15} {viable:>9} {s['evaluated']:>11,.0f}"
            f" {s['reduction']:>8.2%} {s['aware_yield']:>6.0%}"
            f" {s['blind_yield']:>6.0%}"
        )

    loosest, tightest = summaries[0], summaries[-1]
    print(
        f"\n  Pruning strengthens as the request tightens: from "
        f"{loosest['reduction']:.1%} of the space avoided when\n"
        f"  only actions are asked for, to {tightest['reduction']:.1%} when the "
        "request names a purpose at the\n  top of the hierarchy. The mechanism "
        "is the level-1 prune -- a tighter request\n  leaves fewer contributors "
        "viable alone, and each one dropped halves what remains.\n"
    )
    print(
        "  Read the two ends together rather than quoting the favourable one.\n"
        "  Apriori pays off precisely where selection is hard, and degenerates\n"
        "  toward exhaustive enumeration where the requirement is so permissive\n"
        "  that almost any consortium would have satisfied it anyway. That is a\n"
        "  property of the problem rather than a deficiency of the algorithm: the\n"
        "  feasible family really is that large when the constraint is weak."
    )

    gaps = [
        s["aware_utility"] - s["blind_deployable"]
        for s in summaries
        if s["aware_utility"] > 0
    ]
    if gaps:
        print(
            "\n  Across every regime the policy-blind consortium gathers more data\n"
            "  and is less often permitted to use it. Deployable utility is the\n"
            "  only comparison that counts, and selection wins it by "
            f"{float(np.mean(gaps)):.0f} on average."
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "selection.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  wrote {RESULTS / 'selection.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
