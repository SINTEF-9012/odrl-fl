"""What the erosion curve looks like once the assignment stops being assumed.

`run_sensitivity.py` answers "how much does the result depend on how codes are
assigned?" by running the curve under seven assumed models, and the answer is
uncomfortable: erosion at ten contributors ranges from 5% to 76%, and under
`clustered-0.6` the pre-registered go/no-go criterion would not have been met.

This narrows that range by measuring the assignment instead of assuming it. EGA
publishes which Data Access Committee issues each policy, so how alike two
institutions' conditions actually are is observable. The measured cohesion
brackets the plausible range, and the curve is re-run at both ends of it.

The bracket is a bracket and not a point on purpose. A DAC is one institution,
so its internal agreement is the *most* homophilous grouping obtainable -- an
upper bound. Unrelated DACs are the least -- a lower bound. Real consortia sit
between, and this reports the interval rather than picking a spot inside it.

    python experiments/run_measured.py
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odrl_fl.assignment import ASSUMED, register_clustered
from odrl_fl.ega import CACHE, by_dac, cohesion_bracket, load, provenance
from odrl_fl.erosion import run_study

RESULTS = Path(__file__).resolve().parent.parent / "results"
SIZES = (1, 3, 5, 10, 20)


def drop_at(summaries, size: int = 10) -> tuple[float, float]:
    """Capability drop and retention at `size`, on the diverse arm."""
    diverse = {s.size: s for s in summaries if s.heterogeneity == 1.0}
    one, target = diverse.get(1), diverse.get(size)
    if not one or not target or not one.capability_mean:
        return (float("nan"), float("nan"))
    return (1.0 - target.capability_mean / one.capability_mean, target.retains_any)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Erosion at measured cohesion")
    parser.add_argument("--pool-size", type=int, default=40)
    parser.add_argument("--draws", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args(argv)

    if not CACHE.exists():
        print(f"No EGA corpus at {CACHE}.")
        print("Run: python -m odrl_fl.ega --fetch")
        return 1

    corpus = load()
    prov = provenance()
    groups = by_dac(corpus)
    bracket = cohesion_bracket(corpus)

    print("Measured assignment structure, from EGA data access committees\n")
    print(f"  fetched                    {prov['fetched']}")
    print(f"  policies seen              {prov['policies_seen']}")
    print(f"  carrying DUO conditions    {prov['policies_annotated']} "
          f"({prov['adoption_rate']:.1%} adoption)")
    print(f"  distinct DACs              {len(groups)}")
    print(f"  DACs with 2+ policies      {bracket['n_dacs_multi']}")
    print(f"  with a sample volume       {prov['contributors_with_volume']}\n")

    print("  chance coincidence p       "
          f"{bracket['p_emp']:.4f}")
    print("  agreement, different DACs  "
          f"{bracket['agreement_between']:.4f}   (floor)")
    print("  agreement, same DAC        "
          f"{bracket['agreement_within']:.4f}   "
          f"[{bracket['agreement_within_lo']:.4f}, "
          f"{bracket['agreement_within_hi']:.4f}]   (ceiling)")
    print()
    lower = bracket["cohesion_lower"]
    upper = bracket["cohesion_upper"]
    print(f"  cohesion bracket           {lower:.3f} .. {upper:.3f}")
    print(f"    upper bound interval     [{bracket['cohesion_upper_lo']:.3f}, "
          f"{bracket['cohesion_upper_hi']:.3f}]\n")

    # The curve at each end of the bracket, plus the corpus that makes no
    # parametric assumption at all, plus the old baseline for reference.
    conditions = [
        ("empirical", "the assumed baseline, for reference"),
        (register_clustered(lower), "at the measured floor"),
        (register_clustered(upper), "at the measured ceiling"),
        ("measured", "real code sets from the real institutions"),
    ]

    rows: list[dict] = []
    verdicts: dict[str, tuple[float, float]] = {}

    print("Erosion curve under each\n")
    for name, why in conditions:
        summaries = run_study(
            pool_size=args.pool_size,
            sizes=SIZES,
            heterogeneities=(0.0, 1.0),
            corpus_seeds=tuple(range(args.seeds)),
            draws=args.draws,
            assignment=name,
        )
        diverse = {s.size: s for s in summaries if s.heterogeneity == 1.0}

        print(f"  {name}  --  {why}")
        print(f"    {'size':>4}  {'capability':>10}  {'retains any':>11}  "
              f"{'purpose kept':>12}")
        for size in SIZES:
            summary = diverse.get(size)
            if not summary:
                continue
            print(f"    {size:>4}  {summary.capability_mean:>10.2f}  "
                  f"{summary.retains_any:>10.0%}  "
                  f"{summary.purpose_retention:>12.3f}")
            row = summary.as_row()
            row["condition"] = name
            row["note"] = why
            rows.append(row)

        drop, retains = drop_at(summaries)
        verdicts[name] = (drop, retains)
        print(f"    -> {drop:.0%} drop by 10 contributors\n")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "measured.csv"
    if rows:
        # Conflict columns only appear on rows where a conflict occurred, so
        # the header is the union across rows rather than the first row's keys.
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out}")

    # What this did and did not settle.
    assumed = [
        drop_at(
            run_study(
                pool_size=args.pool_size,
                sizes=(1, 10),
                heterogeneities=(1.0,),
                corpus_seeds=tuple(range(args.seeds)),
                draws=args.draws,
                assignment=name,
            )
        )[0]
        for name in ASSUMED
    ]
    assumed = [d for d in assumed if not math.isnan(d)]
    measured_drops = [
        verdicts[n][0] for n, _ in conditions[1:] if not math.isnan(verdicts[n][0])
    ]

    print("\nWhat the measurement bought\n")
    if assumed:
        print(f"  assumed models spanned     {min(assumed):.0%} .. {max(assumed):.0%}")
    if measured_drops:
        print(f"  measured bracket spans     {min(measured_drops):.0%} .. "
              f"{max(measured_drops):.0%}")
    if assumed and measured_drops:
        before = max(assumed) - min(assumed)
        after = max(measured_drops) - min(measured_drops)
        if after < before:
            print(f"  the range narrows from {before:.0%} wide to {after:.0%} wide.")
        else:
            print(f"  the range does NOT narrow ({before:.0%} -> {after:.0%} wide). "
                  "Measuring the assignment did not resolve the threat.")
    print(f"\n  Caveat that does not go away: the ceiling rests on "
          f"{bracket['n_dacs_multi']} institutions,")
    print("  the only ones in EGA publishing more than one machine-readable policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
