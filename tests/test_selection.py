"""Tests for policy-aware contributor selection.

Two things need guarding here. The first is that the Apriori search returns
*exactly* what an exhaustive scan would: the pruning is only sound because
capability is anti-monotone, and if that ever stopped holding the search would
silently start dropping viable consortia rather than failing loudly.

The second is the property that makes a search necessary at all. Compatibility
is not transitive, so two contributors can each satisfy a requirement alone and
fail together. `test_two_viable_contributors_can_be_jointly_infeasible` is the
one to look at if the case for this module is ever questioned.
"""

from __future__ import annotations

from itertools import combinations

import pytest

from odrl_fl.corpus import Contributor, generate_corpus
from odrl_fl.odrl import (
    Action,
    Constraint,
    LeftOperand,
    Operator,
    Permission,
    Policy,
    PolicyType,
)
from odrl_fl.selection import (
    Candidate,
    is_feasible,
    pareto_front,
    policy_blind_selection,
    requirement,
    search,
)

ACTIONS = (Action.USE, Action.AGGREGATE)


def contributor(index: int, *purposes: str, actions=ACTIONS) -> Contributor:
    """A contributor permitting `actions` for `purposes` (all, if none given)."""
    constraints = frozenset()
    if purposes:
        constraints = frozenset(
            {Constraint(LeftOperand.PURPOSE, Operator.IS_ANY_OF, frozenset(purposes))}
        )
    return Contributor(
        index=index,
        codes=(),
        policy=Policy(
            uid=f"urn:test:contributor:{index}",
            type=PolicyType.OFFER,
            permissions=frozenset(
                Permission(action=a, constraints=constraints) for a in actions
            ),
        ),
        unsupported=(),
    )


def brute_force(pool, request) -> set[tuple[int, ...]]:
    """Every feasible subset, by exhaustive scan. The reference to check against."""
    return {
        combo
        for size in range(1, len(pool) + 1)
        for combo in combinations(range(len(pool)), size)
        if is_feasible([pool[i] for i in combo], request)
    }


def maximal_of(family) -> set[tuple[int, ...]]:
    return {c for c in family if not any(set(c) < set(other) for other in family)}


def count_utility(subset) -> float:
    return float(len(subset))


# ---------------------------------------------------------------------------


class TestFeasibility:
    def test_empty_set_is_not_feasible(self):
        assert not is_feasible([], requirement(*ACTIONS))

    def test_feasibility_is_downward_closed(self):
        """The property every prune in the search depends on."""
        pool = generate_corpus(10, seed=1)
        request = requirement(*ACTIONS)
        family = brute_force(pool, request)

        for combo in family:
            for size in range(1, len(combo)):
                for sub in combinations(combo, size):
                    assert sub in family, (
                        f"{sub} infeasible but its superset {combo} is feasible; "
                        "capability is not anti-monotone"
                    )

    def test_two_viable_contributors_can_be_jointly_infeasible(self):
        """Non-transitivity: why this is a search and not a filter.

        Each contributor satisfies the requirement alone. Their purposes do not
        meet, so together they grant nothing. No amount of per-candidate vetting
        against a reference policy would catch this.
        """
        cancer = contributor(0, "ds-cancer")
        neuro = contributor(1, "ds-neurological")
        request = requirement(*ACTIONS)

        assert is_feasible([cancer], request)
        assert is_feasible([neuro], request)
        assert not is_feasible([cancer, neuro], request)


class TestSearchCorrectness:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_search_finds_exactly_the_brute_force_maximal_sets(self, seed):
        """The load-bearing test: pruning must not cost completeness."""
        pool = generate_corpus(9, seed=seed)
        request = requirement(*ACTIONS)

        found = set(search(pool, request, count_utility).maximal)
        assert found == maximal_of(brute_force(pool, request))

    def test_a_contributor_infeasible_alone_never_appears(self):
        """The level-1 prune, and the reason it is sound."""
        pool = [
            contributor(0, "research"),
            contributor(1, "research"),
            contributor(2, "clinical-care"),
        ]
        request = requirement(*ACTIONS, purpose="research")

        result = search(pool, request, count_utility)
        assert result.stats.viable_singletons == 2
        assert all(2 not in members for members in result.maximal)

    def test_every_reported_set_is_actually_feasible(self):
        pool = generate_corpus(10, seed=4)
        request = requirement(*ACTIONS)

        result = search(pool, request, count_utility)
        for members in result.maximal:
            assert is_feasible([pool[i] for i in members], request)

    def test_reported_sets_are_maximal(self):
        """Nothing reported may be extendable while staying feasible."""
        pool = generate_corpus(10, seed=5)
        request = requirement(*ACTIONS)
        result = search(pool, request, count_utility)

        for members in result.maximal:
            for candidate in set(range(len(pool))) - set(members):
                extended = tuple(sorted((*members, candidate)))
                assert not is_feasible([pool[i] for i in extended], request), (
                    f"{members} is reported maximal but extends with {candidate}"
                )

    def test_max_size_caps_the_search(self):
        pool = generate_corpus(10, seed=1)
        result = search(pool, requirement(*ACTIONS), count_utility, max_size=3)
        assert all(len(members) <= 3 for members in result.maximal)

    def test_pruning_never_evaluates_more_than_the_exhaustive_space(self):
        pool = generate_corpus(10, seed=2)
        stats = search(pool, requirement(*ACTIONS), count_utility).stats
        assert stats.candidates_evaluated <= stats.exhaustive_space
        assert 0.0 <= stats.reduction <= 1.0


class TestRequestTightness:
    def test_a_broader_request_is_harder_to_satisfy(self):
        """Containment semantics, observed through the search.

        Granting all of `research` implies granting `ds-cancer`, so the family
        feasible under the broader request is contained in the narrower one's.
        """
        pool = generate_corpus(9, seed=0)
        broad = brute_force(pool, requirement(*ACTIONS, purpose="research"))
        narrow = brute_force(pool, requirement(*ACTIONS, purpose="ds-cancer"))
        assert broad <= narrow

    def test_tightening_the_request_never_grows_the_viable_pool(self):
        pool = generate_corpus(12, seed=3)
        loose = search(pool, requirement(*ACTIONS), count_utility).stats
        tight = search(
            pool, requirement(*ACTIONS, purpose="research"), count_utility
        ).stats
        assert tight.viable_singletons <= loose.viable_singletons


class TestParetoFront:
    def make(self, members, utility, caps, duties) -> Candidate:
        return Candidate(
            members=members,
            utility=utility,
            capability=frozenset(caps),
            duty_count=duties,
        )

    def test_more_utility_dominates(self):
        better = self.make((0, 1), 10.0, {Action.USE}, 0)
        worse = self.make((0,), 5.0, {Action.USE}, 0)
        assert better.dominates(worse)
        assert not worse.dominates(better)

    def test_fewer_duties_dominates_at_equal_utility(self):
        """Duties are a real axis, not a tiebreak."""
        light = self.make((0,), 10.0, {Action.USE}, 0)
        heavy = self.make((1,), 10.0, {Action.USE}, 3)
        assert light.dominates(heavy)
        assert not heavy.dominates(light)

    def test_broader_capability_dominates(self):
        broad = self.make((0,), 10.0, {Action.USE, Action.DISTRIBUTE}, 0)
        narrow = self.make((1,), 10.0, {Action.USE}, 0)
        assert broad.dominates(narrow)

    def test_incomparable_candidates_both_survive(self):
        """The reason a front exists: no exchange rate between the axes."""
        data_rich = self.make((0, 1), 20.0, {Action.USE}, 2)
        unencumbered = self.make((2,), 8.0, {Action.USE}, 0)
        assert not data_rich.dominates(unencumbered)
        assert not unencumbered.dominates(data_rich)
        assert len(pareto_front([data_rich, unencumbered])) == 2

    def test_front_excludes_dominated_candidates(self):
        best = self.make((0, 1), 20.0, {Action.USE}, 0)
        dominated = self.make((2,), 5.0, {Action.USE}, 1)
        assert pareto_front([best, dominated]) == [best]


class TestBaseline:
    def test_policy_blind_selection_can_be_infeasible(self):
        """The baseline's whole point: the best data need not be usable."""
        pool = [
            contributor(0, "ds-cancer"),
            contributor(1, "ds-neurological"),
        ]
        request = requirement(*ACTIONS)
        chosen = policy_blind_selection(pool, count_utility, 2)
        assert not is_feasible([pool[i] for i in chosen], request)

    def test_policy_blind_selection_respects_the_budget(self):
        pool = generate_corpus(10, seed=0)
        assert len(policy_blind_selection(pool, count_utility, 4)) == 4
