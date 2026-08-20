"""Generating a population of contributor policies.

Increment 2's algebra had five hand-written policies to chew on. The erosion
study needs a *population*, and if that population were invented the headline
result would be unanswerable: a reviewer asks why erosion looks like that
rather than some other way, and "we chose it" is not an answer.

So code sets are sampled from `vocab/duo_frequencies.json`, derived from real
data use conditions published on EGA policy records. The threat to validity
shrinks from "invented" to "sampled from a documented source", and the source
is committed rather than fetched live, so the corpus stays reproducible.

`heterogeneity` is the dial the evaluation sweeps. At 0.0 every contributor
carries the single most common code set, so composition should barely erode; at
1.0 code sets are drawn independently from the empirical distribution. The
expectation is that erosion is a function of policy diversity, and that dial is
what tests it.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from odrl_fl.odrl import Policy, capability, compose
from odrl_fl.odrl.duo import DEFAULT_ACTIONS, translate

FREQUENCIES_PATH = (
    Path(__file__).resolve().parent.parent / "vocab" / "duo_frequencies.json"
)

#: Values for the parameters DUO codes leave open. Drawing regions from a pool
#: spanning more than one jurisdiction matters: `spatial` conflicts between
#: contributors are one of the mechanisms that erodes capability, and a corpus
#: where everyone says "EU" would never exhibit it.
REGIONS = ("EU", "DE", "FR", "US", "UK")
EXPIRIES = ("2027-06-30", "2028-12-31", "2030-01-01")
MORATORIA = ("2026-06-01", "2027-01-01")


@dataclass(frozen=True)
class Contributor:
    """One generated party: its DUO codes and the Offer they translate to."""

    index: int
    codes: tuple[str, ...]
    policy: Policy
    unsupported: tuple[str, ...]

    @property
    def uid(self) -> str:
        return self.policy.uid


@lru_cache(maxsize=1)
def frequencies() -> dict[str, Any]:
    return json.loads(FREQUENCIES_PATH.read_text(encoding="utf-8"))


def combinations() -> list[tuple[tuple[str, ...], int]]:
    """Observed code sets with their counts, most common first."""
    return [
        (tuple(entry["codes"]), int(entry["count"]))
        for entry in frequencies()["combinations"]
    ]


def modal_combination() -> tuple[str, ...]:
    return combinations()[0][0]


def sample_codes(rng: np.random.Generator, heterogeneity: float) -> tuple[str, ...]:
    """Draw one contributor's code set.

    With probability `1 - heterogeneity` the modal set is used, which is what
    makes the low end of the dial homogeneous; otherwise a set is drawn in
    proportion to how often it occurs in the source data.
    """
    if heterogeneity <= 0.0 or rng.random() >= heterogeneity:
        return modal_combination()

    observed = combinations()
    weights = np.array([count for _, count in observed], dtype=float)
    weights /= weights.sum()
    return observed[int(rng.choice(len(observed), p=weights))][0]


def generate_corpus(
    n: int,
    *,
    heterogeneity: float = 1.0,
    seed: int = 0,
    actions: Sequence[Any] = DEFAULT_ACTIONS,
) -> list[Contributor]:
    """Generate `n` contributor policies.

    Fully determined by `seed`. The erosion study compares subsets of this
    corpus, so it cannot shift underneath the comparison.
    """
    if n < 1:
        raise ValueError("a corpus needs at least one contributor")
    if not 0.0 <= heterogeneity <= 1.0:
        raise ValueError("heterogeneity must be in [0, 1]")

    contributors: list[Contributor] = []
    for index in range(n):
        # Keyed per contributor, so contributor 3 is the same policy whether
        # the corpus has 10 members or 200 -- the same property the data
        # partitions have, and for the same reason.
        rng = np.random.default_rng([seed, index])
        codes = sample_codes(rng, heterogeneity)
        params = {
            "region": REGIONS[int(rng.integers(len(REGIONS)))],
            "expiry": EXPIRIES[int(rng.integers(len(EXPIRIES)))],
            "moratorium_until": MORATORIA[int(rng.integers(len(MORATORIA)))],
        }
        policy, unsupported = translate(
            codes,
            uid=f"urn:policy:contributor-{index:03d}",
            assigner=f"urn:party:contributor-{index:03d}",
            params=params,
            actions=actions,
        )
        contributors.append(
            Contributor(
                index=index,
                codes=tuple(codes),
                policy=policy,
                unsupported=unsupported,
            )
        )
    return contributors


def policies_of(contributors: Sequence[Contributor]) -> list[Policy]:
    return [c.policy for c in contributors]


# --- Description -----------------------------------------------------------


def code_diversity(contributors: Sequence[Contributor]) -> float:
    """Fraction of contributor pairs whose code sets differ.

    The operational reading of `heterogeneity`: 0.0 when every contributor
    carries the same conditions, approaching 1.0 as they diverge.
    """
    if len(contributors) < 2:
        return 0.0
    sets = [frozenset(c.codes) for c in contributors]
    pairs = differing = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            pairs += 1
            differing += sets[i] != sets[j]
    return differing / pairs


def corpus_stats(contributors: Sequence[Contributor]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for contributor in contributors:
        counts.update(contributor.codes)

    individual = [len(capability(c.policy)) for c in contributors]
    composed = compose(policies_of(contributors))

    return {
        "contributors": len(contributors),
        "distinct_code_sets": len({frozenset(c.codes) for c in contributors}),
        "code_diversity": round(code_diversity(contributors), 4),
        "mean_codes_per_contributor": round(
            sum(len(c.codes) for c in contributors) / len(contributors), 2
        ),
        "mean_individual_capability": round(sum(individual) / len(individual), 3),
        "composed_capability": sorted(a.value for a in composed.capability),
        "lost": {a.value: r.value for a, r in composed.lost.items()},
        "unsupported_codes": sorted(
            {code for c in contributors for code in c.unsupported}
        ),
        "top_codes": counts.most_common(8),
    }
