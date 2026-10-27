"""What a short exploratory run buys, measured against the exhaustive truth.

The three estimators of increment 13 all failed: `coverage` and `divergence`
collapse to "take the largest feasible set" and `gradient` anti-correlates with
accuracy. So the second stage of selection has to train something. This asks how
little.

Nothing here trains. Every one of the runs in `results/groundtruth.csv` recorded
its whole twenty-round curve, so "would R rounds have sufficed?" is a question
about a file. The feasible set is computed with the same `generate_corpus` and
`compose` calls that `experiments/run_estimators.py` uses, so the two experiments
are talking about the same candidates.

Read the output with the module docstring of `odrl_fl.shortlist` beside it: a
short run separates bad consortia from good ones, and does not reliably separate
good from best. The `tau` column is what says so.

    python experiments/run_shortlist.py
    python experiments/run_shortlist.py --rounds 1 2 3 5 8 10 --k 1 3 5
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odrl_fl.corpus import generate_corpus
from odrl_fl.odrl import compose
from odrl_fl.shortlist import shortlist, sweep

RESULTS = Path(__file__).resolve().parent.parent / "results"
TRUTH = RESULTS / "groundtruth.csv"
OUT = RESULTS / "shortlist.csv"

#: The paired comparison against the full budget is written here and not only
#: printed. The paper's central claim is a row of this file, and a claim that
#: lives in terminal output can disagree with the repository without anyone
#: noticing -- the same reason figures here are rendered from committed CSVs
#: rather than from a fresh run.
FIELDNAMES = [
    "explore_rounds", "k", "regret", "regret_lo", "regret_hi",
    "fraction", "rounds", "tau", "rank_of_best", "n_feasible",
    "holdout", "trials", "seeds",
    # Against the full budget, paired by federation.
    "diff_mean", "diff_lo", "diff_hi", "diff_median",
    "worse", "better", "sign_p", "margin", "verdict",
    # The two policy-only baselines, repeated on every row because they are
    # constant for a campaign and a reader of one row should not have to look
    # elsewhere for what it is being compared against.
    "baseline_largest", "baseline_random",
]


def load_curves(path: Path) -> dict[int, dict[tuple[int, ...], list[list[float]]]]:
    """Curves per federation per consortium, one list per replicate.

    Keyed `seed -> members -> [curve, ...]`. Rows written before `trial`
    existed carry no such column and are read as `trial=0`, which is what they
    were trained under.
    """
    out: dict[int, dict[tuple[int, ...], list[list[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            members = tuple(int(m) for m in row["members"].split("-"))
            curve = [float(x) for x in row["curve"].split()]
            out[int(row["seed"])][members].append(curve)
    return {s: dict(v) for s, v in out.items()}


def feasible_members(pool: int, seed: int, candidates) -> list[tuple[int, ...]]:
    """The candidates the policy filter admits, by the same path as increment 13.

    An impermissible consortium has zero *deployable* utility however well it
    trains, so it is removed before anything is ranked rather than ranked and
    then discarded.
    """
    corpus = generate_corpus(pool, seed=seed)
    return [
        m for m in candidates
        if compose([corpus[i].policy for i in m]).capability
    ]


def kendall_tau(a: list[float], b: list[float]) -> float:
    """Rank agreement, counted directly.

    Quadratic, which is why the caller subsamples. The point of reporting it is
    that regret can collapse while tau stays middling -- which is exactly what
    happens here, and is the difference between "finds the best consortium" and
    "finds a good one".
    """
    concordant = discordant = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def sign_test(worse: int, better: int) -> float:
    """Two-sided exact sign test on paired differences, ties discarded.

    No assumption about the shape of the differences, and none about their
    magnitude either -- which turns out to be the problem. Across these
    federations the direction of the difference is close to evenly split at
    every exploratory budget, including one round, while the *mean* difference
    at one round is eighteen times the full budget's entire regret. The penalty
    lives in a few federations where a short run fails badly, and a test that
    counts only directions cannot see it.

    Kept and reported because the balance of directions is itself the finding:
    a short run is not systematically worse, it occasionally fails badly, and
    those are different risks for a deployment to carry. `bootstrap_mean` is
    the test that bears on how much it costs.

    Returns 1.0 when every pair is a tie: absence of information, not
    agreement.
    """
    n = worse + better
    if n == 0:
        return 1.0
    extreme = min(worse, better)
    tail = sum(math.comb(n, i) for i in range(extreme + 1))
    return min(1.0, 2 * tail / 2**n)


def bootstrap_mean(
    values: Sequence[float], draws: int = 20000, seed: int = 0
) -> tuple[float, float]:
    """Percentile interval for the mean of paired differences.

    Resampling rather than a t interval, because regret is bounded below by
    zero, most federations sit near that bound, and the few that do not carry
    the whole effect. That distribution is strongly skewed and a symmetric
    interval around the mean would misstate it in both directions.

    An interval excluding zero is evidence that the shorter budget costs
    accuracy on average. One spanning zero is not evidence that it does not --
    with twenty federations and a skewed quantity, it may only mean the study
    cannot tell.
    """
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    means = sorted(
        statistics.fmean(rng.choices(values, k=n)) for _ in range(draws)
    )
    return (means[int(0.025 * draws)], means[int(0.975 * draws)])


def paired_against_full(
    per_seed: Mapping[int, Sequence],
    rounds: Sequence[int],
    shortlist_sizes: Sequence[int],
    full: int,
) -> dict[tuple[int, int], dict]:
    """Compare every exploratory budget against the full one, federation by
    federation.

    The three verdicts are not interchangeable and the distinction is the whole
    point. `worse` means the interval on the mean difference excludes zero.
    `equivalent` means it lies entirely inside plus or minus the full budget's
    own regret, so the shorter budget differs from the full one by less than
    the full one differs from the truth. `cannot tell` means it contains zero
    but reaches past that margin, which is a statement about the study's power
    and not about the budget -- at seven rounds the interval reaches forty
    times the margin, and calling that equivalence would be the usual abuse of
    a non-significant result.
    """
    out: dict[tuple[int, int], dict] = {}
    for k in shortlist_sizes:
        baseline = {}
        for seed, points in per_seed.items():
            match = [p.regret for p in points
                     if p.explore_rounds == full and p.k == k]
            if match:
                baseline[seed] = match[0]
        if not baseline:
            continue
        margin = statistics.fmean(baseline.values())

        for r in rounds:
            diffs = []
            for seed, base in baseline.items():
                match = [p.regret for p in per_seed[seed]
                         if p.explore_rounds == r and p.k == k]
                if match:
                    diffs.append(match[0] - base)
            if not diffs:
                continue
            lo, hi = bootstrap_mean(diffs)
            worse = sum(1 for d in diffs if d > 0)
            better = sum(1 for d in diffs if d < 0)
            if r == full:
                verdict = "baseline"
            elif lo > 0 or hi < 0:
                verdict = "worse"
            elif -margin <= lo and hi <= margin:
                verdict = "equivalent"
            else:
                verdict = "cannot tell"
            out[(r, k)] = {
                "diff_mean": round(statistics.fmean(diffs), 6),
                "diff_lo": round(lo, 6),
                "diff_hi": round(hi, 6),
                "diff_median": round(statistics.median(diffs), 6),
                "worse": worse,
                "better": better,
                "sign_p": round(sign_test(worse, better), 4),
                "margin": round(margin, 6),
                "verdict": verdict,
            }
    return out


def split_replicates(
    runs: dict[tuple[int, ...], list[list[float]]],
) -> tuple[dict, dict, bool]:
    """Select on the first replicate, score on the mean of the rest.

    This is what removes the winner's curse. An argmax over hundreds of noisy
    candidates is biased upward, because the winner is partly whichever
    candidate drew the kindest initialisation, so measuring regret against it
    flatters the selector. Choosing from one replicate and scoring on others is
    a held-out split and removes the bias.

    With a single replicate there is nothing to hold out. The two coincide, the
    returned flag is False, and every regret derived from it is an upper bound
    rather than a measurement -- recorded in the output, not left to the reader.
    """
    n = min(len(v) for v in runs.values())
    select = {m: v[0] for m, v in runs.items()}
    if n < 2:
        return select, {m: v[0][-1] for m, v in runs.items()}, False
    score = {
        m: statistics.fmean(curve[-1] for curve in v[1:n])
        for m, v in runs.items()
    }
    return select, score, True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cost-fidelity of short runs")
    parser.add_argument("--pool", type=int, default=10)
    parser.add_argument("--truth", default=str(TRUTH))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument(
        "--rounds", type=int, nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20],
        help="exploratory budgets to try",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    #: tau is quadratic in the number of candidates. 128 is enough to see
    #: whether rank agreement is high or middling, which is all it is for.
    parser.add_argument("--tau-sample", type=int, default=128)
    args = parser.parse_args(argv)

    truth_path = Path(args.truth)
    if not truth_path.exists():
        print(f"no ground truth at {truth_path}", file=sys.stderr)
        return 1

    by_seed = load_curves(truth_path)
    print(f"Cost-fidelity of a two-stage selection, pool {args.pool}")
    print(f"  {len(by_seed)} federations from {truth_path.name}")

    # Composing 1023 policies per seed is the slow part of this script, so it
    # happens once and everything below reads what it produced.
    per_seed: dict[int, list] = {}
    scored: dict[int, tuple[dict, dict]] = {}
    n_feasible: list[int] = []
    trials: list[int] = []
    holdout_flags: list[bool] = []

    for seed, runs in sorted(by_seed.items()):
        feasible = feasible_members(args.pool, seed, runs)
        if len(feasible) < 2:
            print(f"  seed {seed}: fewer than two feasible, skipped")
            continue
        kept = {m: runs[m] for m in feasible}
        select, score, holdout = split_replicates(kept)

        scored[seed] = (select, score)
        per_seed[seed] = sweep(select, score, args.rounds, args.k)
        n_feasible.append(len(feasible))
        trials.append(min(len(v) for v in kept.values()))
        holdout_flags.append(holdout)

        pruned = 100 * (1 - len(feasible) / len(runs))
        print(f"  seed {seed}: {len(feasible):>4} of {len(runs)} feasible "
              f"({pruned:.1f}% pruned), {trials[-1]} replicate(s)")

    if not per_seed:
        print("nothing feasible anywhere", file=sys.stderr)
        return 1

    # Rank agreement and the rank of the true best, per exploratory budget.
    # Reported because regret alone would let a reader conclude that a short
    # run orders the candidates correctly, which it does not: regret collapses
    # at round 8 while tau is still near 0.6 and the best candidate is around
    # rank 20. The shortlist works because the top of the field has flattened,
    # not because the ordering has become right.
    diagnostics: dict[int, tuple[float, float]] = {}
    for r in args.rounds:
        taus, ranks = [], []
        for select, score in scored.values():
            sample = sorted(select)[:: max(1, len(select) // args.tau_sample)]
            taus.append(kendall_tau(
                [select[m][r - 1] for m in sample], [score[m] for m in sample]
            ))
            best = max(score, key=lambda m: score[m])
            order = shortlist({m: select[m][r - 1] for m in score}, len(score))
            ranks.append(order.index(best) + 1)
        diagnostics[r] = (statistics.fmean(taus), statistics.fmean(ranks))

    # Baselines first: the regret of taking the largest admissible set, which
    # is where every training-free estimator ends up, and of drawing an
    # admissible candidate at random. Both go into every row.
    largest, random_pick = [], []
    for _, score in scored.values():
        best = max(score.values())
        biggest = max(len(m) for m in score)
        largest.append(best - max(v for m, v in score.items()
                                  if len(m) == biggest))
        random_pick.append(best - statistics.fmean(score.values()))
    base_largest = round(statistics.fmean(largest), 6)
    base_random = round(statistics.fmean(random_pick), 6)

    # The paired comparison against the full budget, computed before the rows
    # are written so that it lands in the CSV rather than only on screen.
    full = max(args.rounds)
    paired = paired_against_full(per_seed, args.rounds, args.k, full)

    rows = []
    for r in args.rounds:
        for k in args.k:
            points = [
                p for seed in per_seed for p in per_seed[seed]
                if p.explore_rounds == r and p.k == k
            ]
            if not points:
                continue
            regrets = [p.regret for p in points]
            tau, rank = diagnostics[r]
            rows.append({
                **paired.get((r, k), {}),
                "explore_rounds": r,
                "k": k,
                "regret": round(statistics.fmean(regrets), 6),
                "regret_lo": round(min(regrets), 6),
                "regret_hi": round(max(regrets), 6),
                "fraction": round(statistics.fmean(p.fraction for p in points), 4),
                "rounds": round(statistics.fmean(p.rounds for p in points)),
                "tau": round(tau, 4),
                "rank_of_best": round(rank, 1),
                "n_feasible": round(statistics.fmean(n_feasible), 1),
                "holdout": all(holdout_flags),
                "trials": min(trials),
                "seeds": len(per_seed),
                "baseline_largest": base_largest,
                "baseline_random": base_random,
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'R':>3} {'k':>3} {'regret':>8} {'cost':>7} {'tau':>7} "
          f"{'rank of best':>13}")
    for row in rows:
        print(f"{row['explore_rounds']:>3} {row['k']:>3} {row['regret']:>8.4f} "
              f"{row['fraction']:>6.1%} {row['tau']:>7.3f} "
              f"{row['rank_of_best']:>13.0f}")

    # Baselines. Without them a regret of 0.0005 means nothing: the reader
    # cannot tell whether the task is simply flat.
    print("\nBaselines, same federations and the same feasible sets:")
    print(f"  largest feasible set, where every estimator lands  "
          f"{base_largest:.4f}")
    print(f"  a feasible candidate drawn at random               "
          f"{base_random:.4f}")

    # Is a short run as good as the full budget, or merely close to it?
    #
    # The mean regrets differ in the fourth decimal place, which no eye should
    # be trusted to read. The comparison is paired -- every exploratory budget
    # is evaluated on the same federations -- so the per-federation differences
    # are the thing to look at, and a sign test on them needs no assumption
    # about their distribution.
    full = max(args.rounds)
    print(f"\nAgainst the full budget of {full} rounds, paired by federation")
    print(f"({len(per_seed)} federations). The bootstrap interval is on the")
    print("mean difference; an interval excluding zero means the shorter")
    print("budget costs accuracy. The median and the direction counts are")
    print("there because the two disagree: the penalty sits in a few")
    print("federations rather than in a shift of the whole distribution.\n")
    print(f"{'R':>3} {'k':>3} {'mean diff':>10} {'95% interval':>20} "
          f"{'median':>9} {'w/b':>7} {'sign p':>7} {'cost':>6} {'verdict':>12}")
    for k in args.k:
        for r in args.rounds:
            row = next((x for x in rows
                        if x["explore_rounds"] == r and x["k"] == k), None)
            if row is None or row.get("verdict") in (None, "baseline"):
                continue
            mark = "*" if row["verdict"] == "worse" else " "
            print(f"{r:>3} {k:>3} {row['diff_mean']:>+10.5f} "
                  f"[{row['diff_lo']:>+8.5f},{row['diff_hi']:>+8.5f}]{mark} "
                  f"{row['diff_median']:>+9.5f} "
                  f"{row['worse']:>3}/{row['better']:<3} "
                  f"{row['sign_p']:>7.3f} {row['fraction']:>5.1%} "
                  f"{row['verdict']:>12}")
        margin = next((x["margin"] for x in rows if x["k"] == k
                       and x.get("margin") is not None), None)
        if margin is not None:
            print(f"      equivalence margin for k={k}: the full budget's own "
                  f"regret, {margin:.5f}")
            print()
    print("* the interval on the mean excludes zero: measurably worse.")
    print("'equivalent' means the whole interval lies inside +-margin.")
    print("'cannot tell' means it contains zero but reaches past the margin,")
    print("which is a statement about this study rather than about the budget.")

    if not all(holdout_flags):
        print("\nOne replicate per consortium, so selection and scoring share a")
        print("training run. Every regret above is an upper bound on the error,")
        print("not a measurement of it. Re-run the ground truth with --trials 5")
        print("to close this.")

    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
