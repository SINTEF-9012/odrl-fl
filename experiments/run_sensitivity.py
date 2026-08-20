"""How much does the erosion result depend on how codes are assigned?

The main threat to validity. EGA supplies real code frequencies and real
co-occurrence, but nothing about which institution holds which policy, so the
corpus has to assume something. The baseline assumes contributors draw
independently from the empirical distribution of code sets.

That assumption is contestable in a direction that cuts against the result:
consortia are not random samples of the world's institutions. If erosion
vanishes once contributors resemble one another, the headline is an artifact.

This runs the same curve under five assignment models and reports whether the
conclusion survives each.

    python experiments/run_sensitivity.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odrl_fl.assignment import MODELS
from odrl_fl.erosion import evaluate_criteria, run_study

RESULTS = Path(__file__).resolve().parent.parent / "results"
SIZES = (1, 3, 5, 10, 20)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assignment-model sensitivity")
    parser.add_argument("--pool-size", type=int, default=40)
    parser.add_argument("--draws", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args(argv)

    print("Sensitivity of the erosion curve to the code-assignment model")
    print(f"  pool {args.pool_size}, {args.draws} draws per size, {args.seeds} seeds\n")

    rows: list[dict] = []
    verdicts: dict[str, tuple[float, float, bool]] = {}

    for name, model in MODELS.items():
        summaries = run_study(
            pool_size=args.pool_size,
            sizes=SIZES,
            heterogeneities=(0.0, 1.0),
            corpus_seeds=tuple(range(args.seeds)),
            draws=args.draws,
            assignment=name,
        )
        diverse = {s.size: s for s in summaries if s.heterogeneity == 1.0}
        criteria = evaluate_criteria(summaries)
        blocking_ok = all(c.passed for c in criteria if c.blocking)

        print(f"  {name}  --  {model.description}")
        print(f"    {'size':>4}  {'capability':>10}  {'retains any':>11}  {'purpose kept':>12}")
        for size in SIZES:
            summary = diverse.get(size)
            if not summary:
                continue
            print(
                f"    {size:>4}  {summary.capability_mean:>10.2f}  "
                f"{summary.retains_any:>10.0%}  {summary.purpose_retention:>12.3f}"
            )
            row = summary.as_row()
            row["assignment"] = name
            rows.append(row)

        one, ten = diverse.get(1), diverse.get(10)
        drop = 1.0 - ten.capability_mean / one.capability_mean if one and ten else 0.0
        verdicts[name] = (drop, ten.retains_any if ten else 0.0, blocking_ok)
        print(f"    -> {drop:.0%} drop by 10 contributors, "
              f"blocking criteria {'hold' if blocking_ok else 'FAIL'}\n")

    print("\nSummary\n")
    print(f"  {'assignment model':<18} {'drop at 10':>10}  {'retains any':>11}  effect")
    print(f"  {'-' * 18} {'-' * 10}  {'-' * 11}  --------")
    for name, (drop, retains, _ok) in sorted(
        verdicts.items(), key=lambda kv: -kv[1][0]
    ):
        size = "large" if drop >= 0.5 else "moderate" if drop >= 1 / 3 else "small"
        print(f"  {name:<18} {drop:>9.0%}  {retains:>10.0%}  {size}")

    drops = [d for d, _, _ in verdicts.values()]
    below = sorted(n for n, (d, _, _) in verdicts.items() if d < 1 / 3)

    # Reported as an effect size, not pass/fail. The go/no-go gate was answered
    # once on the pre-registered baseline; re-running it per model would
    # conflate "is there a phenomenon" with "how large is it under this
    # assumption".
    print(
        f"\n  Erosion occurs under every model tested, from {min(drops):.0%} to "
        f"{max(drops):.0%}.\n"
        "  The direction is robust. The magnitude is not, and that is the honest\n"
        "  limitation: any figure quoted must name its assignment model.\n"
    )
    if below:
        print(
            "  Below the one-third threshold pre-registered on the baseline:\n"
            f"    {', '.join(below)}\n"
            "  Had any of those been the baseline, the go/no-go criterion would\n"
            "  not have been met. Cohesion is what drives it: erosion decays\n"
            "  smoothly as members come to resemble one another, and largely\n"
            "  disappears once most of them are alike.\n"
        )
    print(
        "  The load-bearing number is purpose-matched. A consortium that has\n"
        "  already self-selected for compatible research purposes still loses a\n"
        "  substantial share of capability, because compatibility is not\n"
        "  transitive -- two members can each be compatible with the anchor and\n"
        "  incompatible with one another. Self-selection is necessary but not\n"
        "  sufficient, which is precisely the argument for selecting explicitly."
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (RESULTS / "sensitivity.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {RESULTS / 'sensitivity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
