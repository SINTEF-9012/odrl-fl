"""Describe a generated corpus and preview the erosion curve.

Increment 4 measures this properly. This is the first look at the phenomenon
over a population grounded in real data use conditions rather than five
hand-written policies.

    python experiments/corpus_demo.py
    python experiments/corpus_demo.py --contributors 60 --heterogeneity 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odrl_fl.corpus import (
    corpus_stats,
    frequencies,
    generate_corpus,
    policies_of,
)
from odrl_fl.odrl import compose
from odrl_fl.odrl.duo import describe, duo_table


def erosion_curve(corpus, sizes, draws: int, seed: int = 0):
    """Mean capability over randomly drawn subsets of each size.

    Sampled rather than read off a prefix: individual subsets are bimodal --
    one incompatible contributor collapses a whole subset to nothing -- so a
    single subset says little about the population.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        if size > len(corpus):
            continue
        totals, retained = [], 0
        for _ in range(draws):
            picked = rng.choice(len(corpus), size=size, replace=False)
            result = compose(policies_of([corpus[i] for i in picked]))
            totals.append(len(result.capability))
            retained += bool(result.capability)
        rows.append((size, float(np.mean(totals)), retained / draws))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Describe a generated corpus")
    parser.add_argument("--contributors", type=int, default=40)
    parser.add_argument("--heterogeneity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=int, default=80)
    args = parser.parse_args(argv)

    provenance = frequencies()["_provenance"]
    print("Source of the code distribution")
    print(f"  {provenance['source']} ({provenance['fetched']})")
    print(
        f"  {provenance['policies_annotated']} of {provenance['policies_seen']} "
        f"policies carry DUO codes -- {provenance['adoption_rate']:.1%} adoption"
    )
    print("  Structured data use conditions are the exception, not the rule.\n")

    table = duo_table()
    by_class: dict[str, int] = {}
    for entry in table.values():
        by_class[entry["class"]] = by_class.get(entry["class"], 0) + 1
    print("Mapping coverage")
    for cls, count in sorted(by_class.items()):
        print(f"  {cls:<24} {count:>2}")
    out_of_fragment = [c for c, e in table.items() if e["class"] == "out-of-fragment"]
    for code in out_of_fragment:
        print(f"\n  Not expressible: {describe(code)}")
        print(f"    {table[code]['odrl']['reason']}")

    corpus = generate_corpus(
        args.contributors, heterogeneity=args.heterogeneity, seed=args.seed
    )
    stats = corpus_stats(corpus)
    print(f"\nCorpus: {stats['contributors']} contributors, "
          f"heterogeneity {args.heterogeneity}")
    print(f"  distinct code sets     {stats['distinct_code_sets']}")
    print(f"  code diversity         {stats['code_diversity']:.3f}")
    print(f"  mean codes each        {stats['mean_codes_per_contributor']}")
    print(f"  mean capability alone  {stats['mean_individual_capability']}")
    if stats["unsupported_codes"]:
        print(f"  codes not expressible  {', '.join(stats['unsupported_codes'])}")

    print("\nErosion over randomly drawn subsets\n")
    print(f"  {'size':>4}  {'mean capability':>15}  {'retains any':>11}")
    print(f"  {'-' * 4}  {'-' * 15}  {'-' * 11}")
    sizes = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 40]
    for size, mean, retained in erosion_curve(corpus, sizes, args.draws, args.seed):
        bar = "#" * round(mean * 5)
        print(f"  {size:>4}  {mean:>15.2f}  {retained:>10.0%}  {bar}")

    print(
        "\nCapability falls as contributors are added, and the fall is driven by\n"
        "policy diversity rather than size: rerun with --heterogeneity 0 and the\n"
        "curve is flat, because identical policies compose to what any one of\n"
        "them already permitted."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
