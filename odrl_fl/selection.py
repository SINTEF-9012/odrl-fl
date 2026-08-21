"""Choosing a consortium: which contributors, given what you need to do.

The erosion study established that a federation assembled without regard to
policy usually cannot lawfully use the model it produces. This module turns
that into a search: among all subsets of a pool, find those whose composed
policy satisfies a stated requirement, and among *those*, the ones offering the
most data.

**Why this is a search and not a filter.** Purpose compatibility is not
transitive. Two contributors can each be compatible with a reference policy and
incompatible with one another -- `research` meets `ds-cancer` and `research`
meets `ds-neurological`, but the two diseases do not meet. So candidates cannot
be vetted one at a time against a template; feasibility is a property of the
set. The sensitivity study measured the consequence: consortia that had already
self-selected on purpose still lost 42% of their capability.

**Why the search is tractable anyway.** Capability is anti-monotone, so
feasibility is *downward closed*: every subset of a feasible set is feasible,
and every superset of an infeasible set is infeasible. Two consequences do the
work:

  - a contributor that fails the requirement *alone* can be dropped from the
    pool outright, since every set containing it fails too;
  - a candidate of size k+1 need only be considered when all its k-subsets are
    feasible, which is exactly Apriori's candidate generation.

Note where the leverage comes from. Utility is only *approximately* monotone --
it is empirical, and non-monotone under extreme skew -- so pruning on it would
risk discarding the optimum. All the sound pruning comes from the legal side.

**Why maximal sets are the answer.** Utility rises with contributors while
capability falls. Since feasibility is the constraint rather than an objective,
there is never a reason to stop short of a maximal feasible set: adding a
contributor that keeps you feasible cannot reduce utility. The Pareto front is
therefore drawn from the maximal feasible sets.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations as iter_combinations

from odrl_fl.corpus import Contributor
from odrl_fl.odrl import (
    Action,
    Constraint,
    LeftOperand,
    Operator,
    Permission,
    Policy,
    PolicyType,
    compose,
    satisfies,
)

#: Scores a contributor set. Higher is better.
UtilityFn = Callable[[Sequence[Contributor]], float]


def requirement(
    *actions: Action,
    purpose: str | None = None,
    uid: str = "urn:request:federation",
) -> Policy:
    """Build a Request: what the federation needs to be able to do.

    `purpose` matters more than it looks. A request naming only actions is
    nearly always satisfiable, because every contributor grants the same
    actions and only their conditions differ -- so the search finds almost
    everything feasible and has nothing to prune. A realistic request says what
    the data is *for*, and that is the constraint the whole erosion result
    turns on.
    """
    constraints = frozenset()
    if purpose is not None:
        constraints = frozenset(
            {Constraint(LeftOperand.PURPOSE, Operator.IS_ANY_OF, frozenset({purpose}))}
        )
    return Policy(
        uid=uid,
        type=PolicyType.REQUEST,
        permissions=frozenset(
            Permission(action=action, constraints=constraints) for action in actions
        ),
    )


@dataclass(frozen=True)
class SearchStats:
    """What the search cost, and what the pruning saved."""

    pool_size: int
    viable_singletons: int
    candidates_generated: int
    candidates_evaluated: int
    feasible_found: int
    maximal_found: int
    levels_explored: int

    @property
    def exhaustive_space(self) -> int:
        """Subsets a brute-force search would have to consider."""
        return 2**self.pool_size - 1

    @property
    def reduction(self) -> float:
        """Fraction of the exhaustive space never touched."""
        if not self.exhaustive_space:
            return 0.0
        return 1.0 - (self.candidates_evaluated / self.exhaustive_space)


@dataclass(frozen=True)
class Candidate:
    """A feasible contributor set with its scores."""

    members: tuple[int, ...]
    utility: float
    capability: frozenset[Action]
    duty_count: int

    @property
    def size(self) -> int:
        return len(self.members)

    def dominates(self, other: Candidate) -> bool:
        """Pareto dominance: at least as good everywhere, better somewhere.

        Duties count *against* a candidate. Two consortia can offer identical
        capability and identical data while owing very different obligations,
        which is a real distinction rather than a tiebreak.
        """
        at_least_as_good = (
            self.utility >= other.utility
            and self.capability >= other.capability
            and self.duty_count <= other.duty_count
        )
        strictly_better = (
            self.utility > other.utility
            or self.capability > other.capability
            or self.duty_count < other.duty_count
        )
        return at_least_as_good and strictly_better


@dataclass(frozen=True)
class SelectionResult:
    frontier: tuple[Candidate, ...]
    stats: SearchStats
    maximal: tuple[tuple[int, ...], ...] = field(default_factory=tuple)

    @property
    def best(self) -> Candidate | None:
        """Highest-utility point on the frontier.

        A convenience, not a recommendation: the frontier exists because the
        objectives are incommensurable, and picking among them is a human
        decision.
        """
        return max(self.frontier, key=lambda c: c.utility, default=None)


def is_feasible(
    subset: Sequence[Contributor], request: Policy, taxonomy=None
) -> bool:
    """Does this set's composed policy grant everything the request asks for?"""
    if not subset:
        return False
    agreement = compose([c.policy for c in subset], taxonomy=taxonomy).agreement
    return satisfies(agreement, request, taxonomy=taxonomy)


def search(
    pool: Sequence[Contributor],
    request: Policy,
    utility: UtilityFn,
    *,
    max_size: int | None = None,
    taxonomy=None,
) -> SelectionResult:
    """Apriori search for maximal feasible contributor sets.

    Returns the Pareto front over (utility, capability, duty burden), together
    with statistics on how much of the search space the pruning avoided.
    """
    candidates_generated = 0
    candidates_evaluated = 0

    # Level 1. A contributor that cannot satisfy the requirement on its own can
    # be discarded from the pool entirely: capability only shrinks as the set
    # grows, so every set containing it is infeasible too. This is the single
    # most effective prune, and it is sound only because of anti-monotonicity.
    viable: list[int] = []
    for position, contributor in enumerate(pool):
        candidates_generated += 1
        candidates_evaluated += 1
        if is_feasible([contributor], request, taxonomy):
            viable.append(position)

    feasible: dict[int, set[tuple[int, ...]]] = {1: {(v,) for v in viable}}
    limit = max_size or len(viable)
    level = 1

    while level < limit and feasible.get(level):
        previous = sorted(feasible[level])
        nxt: set[tuple[int, ...]] = set()

        for left_index, left in enumerate(previous):
            for right in previous[left_index + 1 :]:
                # Join step: merge two feasible k-sets sharing a k-1 prefix.
                if left[:-1] != right[:-1]:
                    break
                merged = (*left, right[-1])
                candidates_generated += 1

                # Prune step: every k-subset must already be feasible. Sound
                # because feasibility is downward closed, and it removes most
                # candidates before any composition is attempted.
                if any(
                    sub not in feasible[level]
                    for sub in iter_combinations(merged, level)
                ):
                    continue

                candidates_evaluated += 1
                if is_feasible([pool[i] for i in merged], request, taxonomy):
                    nxt.add(merged)

        if not nxt:
            break
        level += 1
        feasible[level] = nxt

    # Maximal sets: feasible, and not contained in any larger feasible set.
    all_feasible = [s for sets in feasible.values() for s in sets]
    by_size = sorted(all_feasible, key=len, reverse=True)
    maximal: list[tuple[int, ...]] = []
    for candidate in by_size:
        members = set(candidate)
        if not any(members < set(bigger) for bigger in maximal):
            maximal.append(candidate)

    scored: list[Candidate] = []
    for members in maximal:
        subset = [pool[i] for i in members]
        result = compose([c.policy for c in subset], taxonomy=taxonomy)
        scored.append(
            Candidate(
                members=members,
                utility=utility(subset),
                capability=result.capability,
                duty_count=len(result.agreement.obligations),
            )
        )

    stats = SearchStats(
        pool_size=len(pool),
        viable_singletons=len(viable),
        candidates_generated=candidates_generated,
        candidates_evaluated=candidates_evaluated,
        feasible_found=len(all_feasible),
        maximal_found=len(maximal),
        levels_explored=level,
    )
    return SelectionResult(
        frontier=tuple(pareto_front(scored)),
        stats=stats,
        maximal=tuple(maximal),
    )


def pareto_front(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Non-dominated candidates, largest first.

    A set rather than a single optimum, because the objectives are
    incommensurable: there is no exchange rate between a percentage point of
    data and the right to redistribute the model.
    """
    items = list(candidates)
    front = [
        candidate
        for candidate in items
        if not any(other.dominates(candidate) for other in items)
    ]
    return sorted(front, key=lambda c: (-c.utility, c.size))


# --- Baselines to compare against ------------------------------------------


def policy_blind_selection(
    pool: Sequence[Contributor], utility: UtilityFn, size: int
) -> tuple[int, ...]:
    """Pick the `size` contributors that maximise utility, ignoring policy.

    Standard practice in federated learning, and the baseline the evaluation
    needs: it produces the best model, which may well be one nobody is
    permitted to use.
    """
    ranked = sorted(
        range(len(pool)), key=lambda i: utility([pool[i]]), reverse=True
    )
    return tuple(sorted(ranked[:size]))
