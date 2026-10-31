"""How far anti-monotone pruning gets, as the pool grows.

The paper claims that the search over subsets is affordable because the family
of admissible groups is downward closed, so a candidate can be discarded
without being composed whenever one of its subsets was already found
inadmissible. Until now that claim was measured at a single pool of sixteen.
This measures it as the pool grows, which is the question a reader asks of a
scalability claim.

Nothing trains. The search is a computation over declared terms, so this runs
in seconds at pool sizes where training every subset would take weeks. That
separation is the point: the expensive stage is the one the filter makes
smaller.

    python experiments/run_pruning.py
    python experiments/run_pruning.py --max-pool 22 --seeds 10
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odrl_fl.corpus import generate_corpus
from odrl_fl.selection import requirement, search

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "pruning.csv"

FIELDNAMES = [
    "pool", "regime", "seeds",
    "evaluated_mean", "evaluated_max", "generated_mean",
    "exhaustive", "reduction_mean", "reduction_min",
    "seconds_mean", "seconds_max", "admitted_mean",
]

def subset_size(subset) -> float:
    """Utility as the number of members, which is deliberately trivial.

    Pruning is driven by feasibility and never by utility: a candidate is
    discarded because one of its subsets could not grant the requirement, and
    no utility function changes that. Using a trivial one keeps this experiment
    a pure computation over declared terms, with no data generated and nothing
    trained, and it makes the result independent of how utility is estimated.
    """
    return float(len(subset))


#: The same three requirements the selection experiment uses, so the two
#: results are about the same thing. A requirement naming no purpose is granted
#: by nearly everyone and prunes least, which makes it the worst case.
REGIMES = {
    "unconstrained": requirement("use", "aggregate"),
    "ds-cancer": requirement("use", "aggregate", purpose="ds-cancer"),
    "research": requirement("use", "aggregate", purpose="research"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pruning against pool size")
    parser.add_argument("--min-pool", type=int, default=4)
    parser.add_argument("--max-pool", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5)
    #: A single search that runs longer than this stops the sweep for that
    #: regime. Without a stop, a regime that prunes nothing would sit in a
    #: powerset of two million subsets and the rest of the table would never
    #: be produced.
    parser.add_argument("--budget-seconds", type=float, default=120.0)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)

    print("Anti-monotone pruning against pool size")
    print(f"  pools {args.min_pool}..{args.max_pool}, {args.seeds} corpora each")
    print(f"  stopping a regime when one search exceeds "
          f"{args.budget_seconds:.0f}s\n")
    print(f"{'pool':>5} {'regime':>14} {'evaluated':>11} {'exhaustive':>12} "
          f"{'pruned':>8} {'seconds':>8} {'admitted':>9}")

    rows: list[dict] = []
    stopped: set[str] = set()

    for pool in range(args.min_pool, args.max_pool + 1):
        for name, request in REGIMES.items():
            if name in stopped:
                continue
            evaluated, generated, reductions, times, admitted = [], [], [], [], []
            for seed in range(args.seeds):
                corpus = generate_corpus(pool, seed=seed)
                started = perf_counter()
                result = search(corpus, request, subset_size)
                elapsed = perf_counter() - started
                stats = result.stats
                evaluated.append(stats.candidates_evaluated)
                generated.append(stats.candidates_generated)
                reductions.append(stats.reduction)
                times.append(elapsed)
                best = result.best
                admitted.append(len(best.members) if best else 0)
                if elapsed > args.budget_seconds:
                    break

            exhaustive = 2**pool - 1
            rows.append({
                "pool": pool,
                "regime": name,
                "seeds": len(times),
                "evaluated_mean": round(statistics.fmean(evaluated), 1),
                "evaluated_max": max(evaluated),
                "generated_mean": round(statistics.fmean(generated), 1),
                "exhaustive": exhaustive,
                "reduction_mean": round(statistics.fmean(reductions), 6),
                "reduction_min": round(min(reductions), 6),
                "seconds_mean": round(statistics.fmean(times), 4),
                "seconds_max": round(max(times), 4),
                "admitted_mean": round(statistics.fmean(admitted), 1),
            })
            print(f"{pool:>5} {name:>14} {statistics.fmean(evaluated):>11.0f} "
                  f"{exhaustive:>12} {statistics.fmean(reductions):>7.1%} "
                  f"{statistics.fmean(times):>8.3f} "
                  f"{statistics.fmean(admitted):>9.1f}")

            if max(times) > args.budget_seconds:
                print(f"      ^ {name} exceeded the budget at pool {pool}; "
                      f"not continuing it")
                stopped.add(name)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {out}")

    # The headline the paper needs: how the evaluated count grows against the
    # powerset. Reported as a ratio at the largest pool reached, per regime.
    print("\nAt the largest pool reached:")
    for name in REGIMES:
        done = [r for r in rows if r["regime"] == name]
        if not done:
            continue
        last = done[-1]
        print(f"  {name:>14}  pool {last['pool']:>2}: "
              f"{last['evaluated_mean']:.0f} of {last['exhaustive']} subsets "
              f"({last['reduction_mean']:.2%} pruned), "
              f"{last['seconds_mean']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
