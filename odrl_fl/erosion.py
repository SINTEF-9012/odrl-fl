"""Measuring how composition erodes capability as a federation grows.

The decision point for the whole project: is permission erosion a phenomenon
worth reporting, or an artifact?

Two lessons from earlier increments shape the method.

**Subsets are bimodal.** A single incompatible contributor collapses a whole
subset to nothing, so an arbitrary triple may score zero while the mean at that
size is near four. Everything here is measured over *sampled* subsets, never a
prefix, and summarised with bootstrap intervals rather than a normal
approximation -- with a bimodal statistic, a mean plus or minus a standard
error would describe a shape the data does not have.

**Counting actions is too coarse.** Every contributor in the DUO-derived corpus
grants the same four actions with the same base constraints, so the actions
live or die together and `capability_count` behaves almost like a bit.
`purpose_retention` measures the surviving *breadth* of the purpose space
instead, which narrows continuously: `{research}` covers many leaf purposes,
`{disease-specific-research}` covers one. That is where the gradual signal is.

`duty_count` is reported alongside because it moves the other way. Duties union
while permissions intersect, so a growing federation simultaneously loses
capability and accumulates obligation -- a sharper statement of the trade-off
than capability alone.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from odrl_fl.corpus import Contributor, generate_corpus, policies_of
from odrl_fl.odrl import (
    Action,
    LeftOperand,
    Operator,
    Policy,
    Taxonomy,
    compose,
    default_taxonomy,
)

#: Federation sizes the study reports on.
DEFAULT_SIZES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30)

#: Policy-diversity levels for the criterion that erosion tracks heterogeneity
#: rather than mere size.
DEFAULT_HETEROGENEITIES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


#: Leaf-universe cache. Keyed on identity and holding the taxonomy itself, so a
#: collected object cannot have its id reused under a stale entry. `lru_cache`
#: is unavailable here: a Taxonomy carries a dict and is not hashable.
_UNIVERSE_CACHE: dict[int, tuple[Taxonomy, frozenset[str]]] = {}


def purpose_universe(taxonomy: Taxonomy | None = None) -> frozenset[str]:
    """Every leaf purpose term the taxonomy can express.

    Leaves rather than all terms: an interior node like `research` is a
    shorthand for the leaves beneath it, so counting both would double-count
    the same breadth.
    """
    tax = taxonomy or default_taxonomy()
    cached = _UNIVERSE_CACHE.get(id(tax))
    if cached is not None and cached[0] is tax:
        return cached[1]

    edges = tax.broader.get("purpose", {})
    terms = set(edges) | {narrow for values in edges.values() for narrow in values}
    leaves = frozenset(term for term in terms if not edges.get(term))
    _UNIVERSE_CACHE[id(tax)] = (tax, leaves)
    return leaves


def breadth(values: frozenset[str], taxonomy: Taxonomy | None = None) -> int:
    """How many leaf purposes a value set covers."""
    tax = taxonomy or default_taxonomy()
    leaves = purpose_universe(tax)
    covered: set[str] = set()
    for value in values:
        covered |= tax.descendants("purpose", value) & leaves
        if value in leaves:
            covered.add(value)
    return len(covered)


def purpose_retention(policy: Policy, taxonomy: Taxonomy | None = None) -> float:
    """Fraction of the purpose space the policy still permits, in [0, 1].

    Unconstrained purpose means the whole space; no surviving permission means
    none of it. Averaged over surviving permissions, since different actions may
    be constrained differently.
    """
    tax = taxonomy or default_taxonomy()
    total = len(purpose_universe(tax))
    if not total:
        return 0.0
    if not policy.permissions:
        return 0.0

    scores = []
    for permission in policy.permissions:
        positive = [
            c
            for c in permission.constraints
            if c.left_operand is LeftOperand.PURPOSE
            and c.operator is Operator.IS_ANY_OF
        ]
        if not positive:
            scores.append(1.0)
            continue
        allowed = frozenset().union(*(c.right_operand for c in positive))
        scores.append(breadth(allowed, tax) / total)
    return float(np.mean(scores))


@dataclass(frozen=True)
class Measurement:
    """One composed subset."""

    size: int
    heterogeneity: float
    seed: int
    capability_count: int
    retains_any: bool
    purpose_retention: float
    duty_count: int
    loss_reasons: tuple[str, ...]
    #: Operands that could not be reconciled. At action level nearly every
    #: loss in a DUO-derived population is a conflict, so this is where the
    #: informative variation lives.
    conflict_operands: tuple[str, ...]


def measure(
    subset: Sequence[Contributor],
    *,
    heterogeneity: float,
    seed: int,
    taxonomy: Taxonomy | None = None,
) -> Measurement:
    result = compose(policies_of(subset), taxonomy=taxonomy)
    agreement = result.agreement
    return Measurement(
        size=len(subset),
        heterogeneity=heterogeneity,
        seed=seed,
        capability_count=len(result.capability),
        retains_any=bool(result.capability),
        purpose_retention=purpose_retention(agreement, taxonomy),
        duty_count=len(agreement.obligations),
        loss_reasons=tuple(reason.value for reason in result.lost.values()),
        conflict_operands=tuple(result.conflicts.values()),
    )


def sample_subsets(
    corpus: Sequence[Contributor],
    size: int,
    draws: int,
    rng: np.random.Generator,
) -> list[list[Contributor]]:
    """Draw `draws` random subsets of the given size, without replacement."""
    if size > len(corpus):
        return []
    return [
        [corpus[i] for i in rng.choice(len(corpus), size=size, replace=False)]
        for _ in range(draws)
    ]


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean.

    Bootstrap rather than a normal approximation because the underlying
    statistic is bimodal: subsets either keep their capability or lose all of
    it, so a symmetric interval around the mean would describe a distribution
    the data does not have.
    """
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (float(values[0]), float(values[0]))

    generator = rng or np.random.default_rng(0)
    data = np.asarray(values, dtype=float)
    means = data[
        generator.integers(0, len(data), size=(resamples, len(data)))
    ].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    )


@dataclass
class Summary:
    """Aggregated results for one (size, heterogeneity) cell."""

    size: int
    heterogeneity: float
    n: int
    capability_mean: float
    capability_ci: tuple[float, float]
    retains_any: float
    retains_any_ci: tuple[float, float]
    purpose_retention: float
    purpose_retention_ci: tuple[float, float]
    duty_mean: float
    loss_reasons: dict[str, int] = field(default_factory=dict)
    conflict_operands: dict[str, int] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "heterogeneity": self.heterogeneity,
            "n": self.n,
            "capability_mean": round(self.capability_mean, 4),
            "capability_lo": round(self.capability_ci[0], 4),
            "capability_hi": round(self.capability_ci[1], 4),
            "retains_any": round(self.retains_any, 4),
            "retains_any_lo": round(self.retains_any_ci[0], 4),
            "retains_any_hi": round(self.retains_any_ci[1], 4),
            "purpose_retention": round(self.purpose_retention, 4),
            "purpose_retention_lo": round(self.purpose_retention_ci[0], 4),
            "purpose_retention_hi": round(self.purpose_retention_ci[1], 4),
            "duty_mean": round(self.duty_mean, 4),
            **{
                f"loss_{reason}": count
                for reason, count in sorted(self.loss_reasons.items())
            },
            **{
                f"conflict_{operand}": count
                for operand, count in sorted(self.conflict_operands.items())
            },
        }


def summarise(
    measurements: Sequence[Measurement],
    *,
    size: int,
    heterogeneity: float,
    rng: np.random.Generator | None = None,
) -> Summary:
    caps = [float(m.capability_count) for m in measurements]
    keeps = [float(m.retains_any) for m in measurements]
    purposes = [m.purpose_retention for m in measurements]
    duties = [float(m.duty_count) for m in measurements]

    reasons: Counter[str] = Counter()
    operands: Counter[str] = Counter()
    for m in measurements:
        reasons.update(m.loss_reasons)
        operands.update(m.conflict_operands)

    return Summary(
        size=size,
        heterogeneity=heterogeneity,
        n=len(measurements),
        capability_mean=float(np.mean(caps)) if caps else 0.0,
        capability_ci=bootstrap_ci(caps, rng=rng),
        retains_any=float(np.mean(keeps)) if keeps else 0.0,
        retains_any_ci=bootstrap_ci(keeps, rng=rng),
        purpose_retention=float(np.mean(purposes)) if purposes else 0.0,
        purpose_retention_ci=bootstrap_ci(purposes, rng=rng),
        duty_mean=float(np.mean(duties)) if duties else 0.0,
        loss_reasons=dict(reasons),
        conflict_operands=dict(operands),
    )


def run_study(
    *,
    pool_size: int = 40,
    sizes: Sequence[int] = DEFAULT_SIZES,
    heterogeneities: Sequence[float] = DEFAULT_HETEROGENEITIES,
    corpus_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    draws: int = 60,
    actions: Sequence[Action] | None = None,
    taxonomy: Taxonomy | None = None,
) -> list[Summary]:
    """Sweep federation size against policy heterogeneity.

    Repeated across several corpus seeds so the result is not one realisation
    of the sampling; measurements from every seed are pooled before summarising.
    """
    from odrl_fl.odrl.duo import DEFAULT_ACTIONS

    action_set = tuple(actions or DEFAULT_ACTIONS)
    summaries: list[Summary] = []

    for heterogeneity in heterogeneities:
        pooled: dict[int, list[Measurement]] = {size: [] for size in sizes}
        for seed in corpus_seeds:
            corpus = generate_corpus(
                pool_size, heterogeneity=heterogeneity, seed=seed, actions=action_set
            )
            rng = np.random.default_rng([seed, 0xE205])
            for size in sizes:
                for subset in sample_subsets(corpus, size, draws, rng):
                    pooled[size].append(
                        measure(
                            subset,
                            heterogeneity=heterogeneity,
                            seed=seed,
                            taxonomy=taxonomy,
                        )
                    )
        boot = np.random.default_rng(0xB007)
        for size in sizes:
            if pooled[size]:
                summaries.append(
                    summarise(
                        pooled[size],
                        size=size,
                        heterogeneity=heterogeneity,
                        rng=boot,
                    )
                )
    return summaries


# --- Go/no-go ---------------------------------------------------------------


@dataclass(frozen=True)
class Criterion:
    """One pre-registered go/no-go criterion.

    `blocking` encodes the distinction fixed before the study: a trivial or
    degenerate result stops the project, whereas a surprise about *which*
    mechanism drives erosion, or how strongly diversity matters, calls for
    rescoping rather than abandonment. Keeping that in the code stops the
    threshold being renegotiated once the numbers are in.
    """

    name: str
    passed: bool
    detail: str
    blocking: bool = True


def evaluate_criteria(summaries: Sequence[Summary]) -> list[Criterion]:
    """Check the four criteria fixed before the study was run.

    Stated in advance precisely so the verdict is not fitted to whatever the
    numbers turned out to be.
    """
    diverse = {s.size: s for s in summaries if s.heterogeneity == 1.0}
    homogeneous = {s.size: s for s in summaries if s.heterogeneity == 0.0}
    criteria: list[Criterion] = []

    one, ten = diverse.get(1), diverse.get(10)
    if one and ten:
        drop = 1.0 - (ten.capability_mean / one.capability_mean) if one.capability_mean else 0.0
        separated = ten.capability_ci[1] < one.capability_ci[0]
        criteria.append(
            Criterion(
                "non-trivial",  # blocking
                drop >= 1 / 3 and separated,
                f"capability {one.capability_mean:.2f} -> {ten.capability_mean:.2f} "
                f"({drop:.0%} drop), intervals "
                f"{'disjoint' if separated else 'overlapping'}",
            )
        )

    if ten:
        criteria.append(
            Criterion(
                "non-degenerate",
                ten.retains_any > 0.20,
                f"{ten.retains_any:.0%} of 10-contributor subsets retain capability",
            )
        )

    # Attribution is asked at operand granularity, not by LossReason. DUO is a
    # limitation vocabulary: every contributor grants the same actions, so
    # NOT_GRANTED_BY_ALL cannot fire, and DUO's only prohibitions are
    # constrained, so they refine a permission rather than removing it. At
    # action level there is therefore exactly one mechanism by construction,
    # and the informative question is which *condition* the contributors could
    # not reconcile.
    operands: Counter[str] = Counter()
    for summary in summaries:
        if summary.heterogeneity == 1.0:
            operands.update(summary.conflict_operands)
    grand = sum(operands.values())
    material = [o for o, c in operands.items() if grand and c / grand > 0.10]
    criteria.append(
        Criterion(
            "attributable",
            len(material) >= 2,
            ", ".join(f"{o}={c / grand:.0%}" for o, c in operands.most_common())
            or "no conflicts recorded",
            blocking=False,
        )
    )

    hom_ten, hom_one = homogeneous.get(10), homogeneous.get(1)
    if hom_ten and hom_one and one and ten:
        hom_drop = (
            1.0 - (hom_ten.capability_mean / hom_one.capability_mean)
            if hom_one.capability_mean
            else 0.0
        )
        div_drop = (
            1.0 - (ten.capability_mean / one.capability_mean)
            if one.capability_mean
            else 0.0
        )
        criteria.append(
            Criterion(
                "diversity-driven",
                div_drop > hom_drop + 0.20,
                f"drop at 10 contributors: {hom_drop:.0%} homogeneous vs "
                f"{div_drop:.0%} diverse",
                blocking=False,
            )
        )
    return criteria
