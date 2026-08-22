"""Render every figure from the committed result CSVs.

Figures are built from `results/*.csv` rather than by re-running the
experiments, so this stays fast and, more importantly, so a figure can never
disagree with the numbers the repository actually claims. If a CSV is missing,
the figure it feeds is skipped with a note naming the experiment that produces
it, instead of the script dying halfway through.

    python experiments/make_figures.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odrl_fl.figures import Axis, Bar, Series, grouped_bar_chart, line_chart, save

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def read(name: str) -> list[dict] | None:
    path = RESULTS / name
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else default


def fig_erosion(rows: list[dict]) -> str:
    """Capability retention against federation size, one line per heterogeneity.

    The headline result: composing more contributors monotonically loses
    permissions, and how fast depends on how unlike each other their terms are.
    """
    by_het: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_het[row["heterogeneity"]].append(row)

    series = []
    for het in sorted(by_het, key=float):
        group = sorted(by_het[het], key=lambda r: int(r["size"]))
        series.append(
            Series(
                label=f"h = {float(het):g}",
                xs=[int(r["size"]) for r in group],
                ys=[fnum(r, "retains_any") for r in group],
                lo=[fnum(r, "retains_any_lo") for r in group],
                hi=[fnum(r, "retains_any_hi") for r in group],
            )
        )

    return line_chart(
        series,
        title="Permission erosion under composition",
        subtitle=(
            "Share of federations retaining any usable permission, "
            "by number of contributors"
        ),
        x=Axis(label="contributors in the federation", lo=1),
        y=Axis(label="retains any permission", lo=0, hi=1, fmt="{:.0%}"),
        legend_title="heterogeneity",
        notes=(
            "Bands are 95% bootstrap intervals over 300 draws. "
            "h = 0 is an identical-policy control."
        ),
    )


def fig_purpose(rows: list[dict]) -> str:
    """Purpose retention -- the sharper collapse, and the more consequential one.

    A federation can keep the *action* it needs and still lose the *purpose* it
    needed it for, which is the failure that actually blocks a study.
    """
    by_het: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_het[row["heterogeneity"]].append(row)

    series = []
    for het in sorted(by_het, key=float):
        group = sorted(by_het[het], key=lambda r: int(r["size"]))
        series.append(
            Series(
                label=f"h = {float(het):g}",
                xs=[int(r["size"]) for r in group],
                ys=[fnum(r, "purpose_retention") for r in group],
                lo=[fnum(r, "purpose_retention_lo") for r in group],
                hi=[fnum(r, "purpose_retention_hi") for r in group],
            )
        )

    return line_chart(
        series,
        title="Purpose retention collapses faster than action capability",
        subtitle="Share of federations still permitted the requested purpose",
        x=Axis(label="contributors in the federation", lo=1),
        y=Axis(label="purpose retained", lo=0, hi=1, fmt="{:.0%}"),
        legend_title="heterogeneity",
        notes=(
            "Losing the purpose blocks the study even when the action survives."
        ),
    )


def fig_sensitivity(rows: list[dict]) -> str:
    """Does the result depend on how policies are paired with institutions?

    The pairing is the least defensible part of the setup -- it is modelled, not
    observed -- so the honest check is whether the conclusion moves when the
    pairing model changes. Seven models, same shape.
    """
    by_assignment: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_assignment[row["assignment"]].append(row)

    series = []
    for name in sorted(by_assignment):
        group = sorted(by_assignment[name], key=lambda r: int(r["size"]))
        series.append(
            Series(
                label=name,
                xs=[int(r["size"]) for r in group],
                ys=[fnum(r, "purpose_retention") for r in group],
            )
        )

    return line_chart(
        series,
        title="The collapse does not depend on how policies are assigned",
        subtitle=(
            "Purpose retention under seven models of which institution "
            "holds which terms"
        ),
        x=Axis(label="contributors in the federation", lo=1),
        y=Axis(label="purpose retained", lo=0, hi=1, fmt="{:.0%}"),
        legend_title="assignment",
        notes=(
            "Pairing policies with institutions is modelled, not observed -- "
            "so the check is whether the conclusion survives changing it."
        ),
    )


def fig_selection(rows: list[dict]) -> str:
    """What policy-aware selection buys: a model you are allowed to use.

    Blind utility is the higher bar and the misleading one. Deployable utility
    scores an impermissible model at zero, which is what it is worth.
    """
    regimes: list[str] = []
    for row in rows:
        if row["regime"] not in regimes:
            regimes.append(row["regime"])

    bars = []
    sublabels: dict[str, str] = {}
    for regime in regimes:
        group = [r for r in rows if r["regime"] == regime]
        n = len(group)
        aware = sum(fnum(r, "aware_utility") for r in group) / n
        blind = sum(fnum(r, "blind_utility") for r in group) / n
        deployable = sum(fnum(r, "blind_deployable") for r in group) / n
        feasible = sum(fnum(r, "blind_feasible") for r in group) / n
        bars += [
            Bar(regime, "policy-aware", aware, annotation=f"{aware:.0f}"),
            Bar(regime, "policy-blind, raw", blind, annotation=f"{blind:.0f}"),
            Bar(regime, "policy-blind, deployable", deployable,
                annotation=f"{deployable:.0f}"),
        ]
        sublabels[regime] = f"blind usable {feasible:.0%} of the time"

    return grouped_bar_chart(
        bars,
        title="Policy-blind selection wins on paper and loses in deployment",
        subtitle="Mean utility by request regime, synthetic corpus",
        y=Axis(label="utility", fmt="{:.0f}"),
        sublabels=sublabels,
        notes=(
            "Deployable utility scores an impermissible federation at zero -- "
            "the raw bar is what ignoring policy appears to buy, the third is "
            "what it is actually worth."
        ),
    )


def fig_tcga(rows: list[dict]) -> str:
    """The real-data figure, drawn so it cannot overclaim.

    The bars differ by less than the measurement error, and the dashed rule says
    so. What separates the conditions is the deployable bar, which is driven by
    feasibility rather than by model quality.
    """
    n = len(rows)
    aware = sum(fnum(r, "aware_cindex") for r in rows) / n
    blind = sum(fnum(r, "blind_cindex") for r in rows) / n
    deployable = sum(fnum(r, "blind_deployable") for r in rows) / n
    feasible = sum(fnum(r, "blind_feasible") for r in rows) / n
    everyone = sum(fnum(r, "everyone_cindex") for r in rows) / n

    bars = [
        Bar("whole federation", "C-index as measured", everyone,
            annotation=f"{everyone:.3f}"),
        Bar("policy-aware", "C-index as measured", aware, annotation=f"{aware:.3f}"),
        Bar("policy-blind", "C-index as measured", blind, annotation=f"{blind:.3f}"),
        Bar("policy-blind", "deployable", deployable, annotation=f"{deployable:.3f}"),
    ]

    return grouped_bar_chart(
        bars,
        title="TCGA-BRCA: what real data settles, and what it does not",
        subtitle=(
            "16 real institutions, 1096 patients, Cox model scored by "
            "Harrell's C-index"
        ),
        y=Axis(label="C-index", lo=0, hi=0.8, fmt="{:.1f}"),
        reference=(0.5, "chance (C-index 0.5)"),
        sublabels={
            "policy-aware": "usable 100% of the time",
            "policy-blind": f"usable {feasible:.0%} of the time",
        },
        notes=(
            "The C-index bars differ by ~0.01 against a bootstrap interval of "
            "+/-0.11, so they are not separable. The deployable bar is, because "
            "feasibility does not depend on a metric."
        ),
    )


FIGURE_SPECS = (
    ("erosion.csv", "erosion.svg", fig_erosion, "run_erosion.py"),
    ("erosion.csv", "purpose-retention.svg", fig_purpose, "run_erosion.py"),
    ("sensitivity.csv", "sensitivity.svg", fig_sensitivity, "run_sensitivity.py"),
    ("selection.csv", "selection.svg", fig_selection, "run_selection.py"),
    ("tcga.csv", "tcga.svg", fig_tcga, "run_tcga.py"),
)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    written = skipped = 0

    for source, target, build, producer in FIGURE_SPECS:
        rows = read(source)
        if not rows:
            print(f"  skip  {target:<24} needs results/{source} "
                  f"-- run experiments/{producer}")
            skipped += 1
            continue
        path = save(build(rows), FIGURES / target)
        print(f"  wrote {target:<24} {path.stat().st_size / 1024:>6.1f} KB")
        written += 1

    print(f"\n  {written} figure(s) in {FIGURES}"
          + (f", {skipped} skipped" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
