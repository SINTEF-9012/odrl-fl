"""Compose every subset of the example policies and print what survives.

A qualitative look at capability erosion, one increment before measuring it
properly. Increment 4 replaces this with a study over a generated corpus; this
is here to make the phenomenon visible on five hand-written policies that can
be read and checked by eye.

    python experiments/compose_demo.py
    python experiments/compose_demo.py --flat     # without the taxonomy
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odrl_fl.odrl import FLAT, capability, compose, default_taxonomy, load_policies

POLICY_DIR = Path(__file__).resolve().parent.parent / "policies"


def short(uid: str) -> str:
    return uid.split(":")[-1][:2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compose all policy subsets")
    parser.add_argument(
        "--flat", action="store_true",
        help="Disable the subsumption taxonomy, so {EU} and {DE} read as a "
             "conflict. Shows how much apparent erosion is an artifact of "
             "omitting the hierarchy.",
    )
    args = parser.parse_args(argv)
    taxonomy = FLAT if args.flat else default_taxonomy()

    policies = sorted(load_policies(POLICY_DIR), key=lambda p: p.uid)
    print(f"{len(policies)} contributor offers, individually:\n")
    for policy in policies:
        caps = sorted(a.value for a in capability(policy))
        print(f"  {short(policy.uid)}  {', '.join(caps)}")

    print("\nComposed, by subset size:\n")
    print(f"  {'subset':<14} {'n':>2}  {'capability':<34} lost")
    print(f"  {'-' * 14} {'-' * 2}  {'-' * 34} {'-' * 30}")

    totals: dict[int, list[int]] = {}
    for size in range(1, len(policies) + 1):
        for group in combinations(policies, size):
            result = compose(group, taxonomy=taxonomy)
            caps = sorted(a.value for a in result.capability)
            lost = ", ".join(
                f"{a.value}({r.value.split('_')[0]})"
                for a, r in sorted(result.lost.items(), key=lambda kv: kv[0].value)
            )
            label = "+".join(short(p.uid) for p in group)
            print(f"  {label:<14} {size:>2}  {', '.join(caps) or '(none)':<34} {lost}")
            totals.setdefault(size, []).append(len(caps))

    print("\nMean capability by federation size:\n")
    for size, counts in sorted(totals.items()):
        mean = sum(counts) / len(counts)
        bar = "#" * round(mean * 6)
        print(f"  {size} contributor(s)  {mean:>4.2f}  {bar}")

    print(
        "\nCapability is anti-monotone: it never rises as contributors are added.\n"
        "Note p3 -- the national registry, the most valuable dataset -- grants\n"
        "only `aggregate`, so any subset containing it collapses to that alone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
