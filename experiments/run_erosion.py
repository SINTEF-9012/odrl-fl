"""Run the permission-erosion study and report the go/no-go verdict.

The decision point. Four criteria were fixed before the study was run, so the
verdict is not fitted to whatever the numbers turned out to be:

  1. non-trivial      capability at 10 contributors well below the baseline
  2. non-degenerate   a workable fraction of subsets still retain capability
  3. attributable     more than one loss mechanism contributes materially
  4. diversity-driven homogeneous corpora must erode far less than diverse ones

Writes `results/erosion.csv` and `results/erosion.json`.

    python experiments/run_erosion.py
    python experiments/run_erosion.py --draws 120 --seeds 8
    python experiments/run_erosion.py --actions use aggregate   # sensitivity
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odrl_fl.erosion import (
    DEFAULT_HETEROGENEITIES,
    DEFAULT_SIZES,
    evaluate_criteria,
    run_study,
)
from odrl_fl.odrl import Action

RESULTS = Path(__file__).resolve().parent.parent / "results"


def print_curve(summaries, heterogeneity: float) -> None:
    rows = [s for s in summaries if s.heterogeneity == heterogeneity]
    if not rows:
        return
    print(f"\n  heterogeneity = {heterogeneity}")
    print(
        f"    {'size':>4}  {'capability':>18}  {'retains':>7}  "
        f"{'purpose retained':>22}  {'duties':>6}"
    )
    print(f"    {'-' * 4}  {'-' * 18}  {'-' * 7}  {'-' * 22}  {'-' * 6}")
    for row in rows:
        cap = f"{row.capability_mean:.2f} [{row.capability_ci[0]:.2f},{row.capability_ci[1]:.2f}]"
        pur = (
            f"{row.purpose_retention:.3f} "
            f"[{row.purpose_retention_ci[0]:.3f},{row.purpose_retention_ci[1]:.3f}]"
        )
        print(
            f"    {row.size:>4}  {cap:>18}  {row.retains_any:>6.0%}  "
            f"{pur:>22}  {row.duty_mean:>6.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Permission-erosion study")
    parser.add_argument("--pool-size", type=int, default=40)
    parser.add_argument("--draws", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--actions", nargs="+", default=None,
        help="Restrict the granted action set (sensitivity check on the "
             "modelling assumption that DUO-silent actions are all granted)",
    )
    args = parser.parse_args(argv)

    actions = [Action(a) for a in args.actions] if args.actions else None

    print("Permission-erosion study")
    print(f"  pool {args.pool_size} contributors, {args.draws} draws per size, "
          f"{args.seeds} corpus seeds")
    if actions:
        print(f"  actions restricted to: {', '.join(a.value for a in actions)}")

    summaries = run_study(
        pool_size=args.pool_size,
        sizes=DEFAULT_SIZES,
        heterogeneities=DEFAULT_HETEROGENEITIES,
        corpus_seeds=tuple(range(args.seeds)),
        draws=args.draws,
        actions=actions,
    )

    for heterogeneity in (1.0, 0.5, 0.0):
        print_curve(summaries, heterogeneity)

    print("\n\nWhy capability is lost (heterogeneity = 1.0)\n")
    totals: dict[str, int] = {}
    operands: dict[str, int] = {}
    for summary in summaries:
        if summary.heterogeneity == 1.0:
            for reason, count in summary.loss_reasons.items():
                totals[reason] = totals.get(reason, 0) + count
            for operand, count in summary.conflict_operands.items():
                operands[operand] = operands.get(operand, 0) + count

    grand = sum(totals.values()) or 1
    print("  by mechanism")
    for reason, count in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"    {reason:<22} {count / grand:>5.1%}")
    print(
        "    DUO is a limitation vocabulary: every contributor grants the same\n"
        "    actions, so a withheld grant cannot arise, and its only prohibitions\n"
        "    are constrained, refining a permission rather than removing it.\n"
        "    One mechanism at this level is a property of the vocabulary."
    )

    total_operands = sum(operands.values()) or 1
    print("\n  by condition that could not be reconciled")
    for operand, count in sorted(operands.items(), key=lambda kv: -kv[1]):
        bar = "#" * round(40 * count / total_operands)
        print(f"    {operand:<22} {count / total_operands:>5.1%}  {bar}")

    print("\n\nGo/no-go criteria, fixed before the study\n")
    criteria = evaluate_criteria(summaries)
    for criterion in criteria:
        mark = "PASS" if criterion.passed else "FAIL"
        kind = "" if criterion.blocking else "   (advisory)"
        print(f"  [{mark}] {criterion.name:<18} {criterion.detail}{kind}")

    blocking_failures = [c for c in criteria if c.blocking and not c.passed]
    advisory_failures = [c for c in criteria if not c.blocking and not c.passed]
    verdict = not blocking_failures

    if not verdict:
        print("\n  VERDICT: STOP -- a blocking criterion failed")
    elif advisory_failures:
        print("\n  VERDICT: PROCEED, with the scope revised.")
        print(
            "  No blocking criterion failed. These were pre-registered as calling\n"
            "  for rescoping rather than abandonment:"
        )
        for criterion in advisory_failures:
            print(f"    - {criterion.name}: {criterion.detail}")
    else:
        print("\n  VERDICT: PROCEED")

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = [s.as_row() for s in summaries]
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (RESULTS / "erosion.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (RESULTS / "erosion.json").write_text(
        json.dumps(
            {
                "config": {
                    "pool_size": args.pool_size,
                    "draws": args.draws,
                    "seeds": args.seeds,
                    "actions": [a.value for a in actions] if actions else "default",
                },
                "criteria": [
                    {"name": c.name, "passed": c.passed, "detail": c.detail}
                    for c in criteria
                ],
                "verdict": "proceed" if verdict else "do-not-proceed",
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n  wrote {RESULTS / 'erosion.csv'} and erosion.json")
    return 0 if verdict else 2


if __name__ == "__main__":
    raise SystemExit(main())
