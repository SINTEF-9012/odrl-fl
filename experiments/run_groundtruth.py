"""Train every subset of a small pool, so estimators can be checked against truth.

Increment 13 asks whether a training-free estimator preserves the Pareto
frontier. Answering that needs a frontier known exactly, not approximately, so
this trains **all 1023 non-empty subsets** of a ten-contributor pool rather than
sampling them. A complete lattice means a claim like "this estimator misses two
frontier points" is a fact rather than an artefact of which subsets happened to
be drawn.

Ten is chosen because 2^10 is affordable and 2^16 is not: at roughly 0.65 s per
run at the mean subset size, the full lattice across five seeds is about an hour.
The same study at a pool of sixteen would be three weeks.

Resumable on purpose. An hour of compute should not be lost to one interruption,
so completed (subset, seed) pairs are skipped on restart.

    python experiments/run_groundtruth.py
    python experiments/run_groundtruth.py --pool 10 --seeds 5
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odrl_fl.fedavg import run_fedavg

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "groundtruth.csv"

#: `alpha` and `rounds` are recorded per row, not assumed. An evaluation
#: that scored estimators built at a different heterogeneity than the models
#: were trained under would be comparing two different worlds and would look
#: perfectly healthy while doing it.
FIELDNAMES = [
    "members", "size", "seed", "alpha", "rounds",
    "accuracy", "best_accuracy", "curve",
]


def all_subsets(pool: int, max_size: int | None = None) -> list[tuple[int, ...]]:
    """Every non-empty subset up to `max_size`, smallest first.

    Ordered by size so an interrupted run leaves a *usable* prefix rather than a
    random sample: every subset up to some size is present, which is still an
    exhaustive lattice over a stated bound.

    `max_size` exists because the cost of this study is dominated by the large
    subsets -- training a ten-member consortium costs ten times a singleton, and
    sizes six and up are two thirds of the total work for a pool of ten. Capping
    buys a complete answer about small consortia instead of a partial one about
    all of them.
    """
    limit = pool if max_size is None else min(max_size, pool)
    out: list[tuple[int, ...]] = []
    for size in range(1, limit + 1):
        out.extend(combinations(range(pool), size))
    return out


def load_done(path: Path) -> set[tuple[str, int]]:
    """Which (members, seed) pairs are already recorded."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        return {(row["members"], int(row["seed"])) for row in csv.DictReader(fh)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exhaustive subset ground truth")
    parser.add_argument("--pool", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--max-size", type=int, default=None,
        help="largest consortium to train; omit for the full lattice",
    )
    # 20 rounds because the task has not converged at 5 -- accuracy was
    # still climbing +0.068 per round, so a 5-round number measures how
    # fast a consortium starts rather than how good its model is.
    parser.add_argument("--rounds", type=int, default=20)
    # alpha 0.3, not 1.0. At 1.0 every consortium converges to ~0.98 and
    # the spread across them is 0.005 -- there is no utility axis left to
    # estimate. Real skew is also the realistic case: institutions
    # specialise, and a consortium missing a class cannot learn it.
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(path)
    subsets = all_subsets(args.pool, args.max_size)
    total = len(subsets) * args.seeds

    bound = (f" (consortia of up to {args.max_size})" if args.max_size else "")
    print(f"Exhaustive ground truth over a pool of {args.pool}{bound}")
    print(f"  {len(subsets)} subsets x {args.seeds} seeds = {total} runs")
    print(f"  alpha {args.alpha}, {args.rounds} rounds")
    if done:
        print(f"  {len(done)} already recorded, resuming")

    fresh = not path.exists()
    started = time.perf_counter()
    completed = 0

    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if fresh:
            writer.writeheader()

        for members in subsets:
            key = "-".join(str(m) for m in members)
            for seed in range(args.seeds):
                if (key, seed) in done:
                    continue
                history = run_fedavg(
                    members,
                    n_clients=args.pool,
                    rounds=args.rounds,
                    alpha=args.alpha,
                    seed=seed,
                )
                writer.writerow(
                    {
                        "members": key,
                        "size": len(members),
                        "seed": seed,
                        "alpha": args.alpha,
                        "rounds": args.rounds,
                        "accuracy": round(history.final_accuracy, 6),
                        "best_accuracy": round(history.best_accuracy, 6),
                        # The whole curve, so "would R rounds have sufficed?"
                        # can be answered later without retraining anything.
                        "curve": " ".join(f"{a:.6f}" for a in history.accuracy),
                    }
                )
                fh.flush()
                completed += 1

                if completed % 100 == 0:
                    elapsed = time.perf_counter() - started
                    rate = completed / elapsed
                    remaining = (total - len(done) - completed) / max(rate, 1e-9)
                    print(f"  {completed} runs, {rate:.1f}/s, "
                          f"~{remaining / 60:.0f} min left", flush=True)

    elapsed = time.perf_counter() - started
    print(f"\n{completed} runs in {elapsed / 60:.1f} min")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
