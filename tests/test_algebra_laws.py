"""Property tests for the composition algebra.

`test_anti_monotonicity` is the one that matters. The entire project rests on
capability shrinking as contributors are added -- the erosion result measures
it, the selection search prunes with it, and the paper's framing assumes it.
Asserting it over thousands of generated policy sets is what turns it from a
claim into a checked property.

The remaining laws establish that `compose` is a well-behaved fold:
commutative, associative and idempotent. If any of them fails, the supported
fragment needs rescoping, and finding that here rather than in increment 4 is
the whole reason this increment comes first.
"""

from __future__ import annotations

import json

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from odrl_fl.odrl import capability, compose, conjoin, policy_to_jsonld
from odrl_fl.odrl.composition import Unsatisfiable
from tests.strategies import constraint_sets, policies, policy_sets

# These run a real fold over several policies; the default deadline is tight
# for that on a loaded machine and the timing is not what is under test.
SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


class TestAntiMonotonicity:
    """Cap is order-reversing: bigger contributor set, smaller capability."""

    @SETTINGS
    @given(policy_sets(min_size=1, max_size=3), policies(uid="urn:policy:extra"))
    def test_adding_a_contributor_never_widens_capability(self, base, extra):
        before = compose(base).capability
        after = compose([*base, extra]).capability
        assert after <= before

    @SETTINGS
    @given(policy_sets(min_size=2, max_size=4))
    def test_any_subset_is_at_least_as_capable_as_the_whole(self, group):
        whole = compose(group).capability
        for i in range(len(group)):
            subset = group[:i] + group[i + 1 :]
            assert whole <= compose(subset).capability

    @SETTINGS
    @given(policy_sets(min_size=1, max_size=4))
    def test_capability_never_exceeds_the_least_permissive_member(self, group):
        """The bound implied by intersection: no contributor can be out-voted."""
        composed = compose(group).capability
        for policy in group:
            assert composed <= capability(policy)

    @SETTINGS
    @given(policy_sets(min_size=1, max_size=3), policies(uid="urn:policy:extra"))
    def test_removing_a_contributor_never_narrows_capability(self, base, extra):
        """The inverse, and why revocation can *restore* capability."""
        with_extra = compose([*base, extra]).capability
        without = compose(base).capability
        assert with_extra <= without


class TestFoldLaws:
    @SETTINGS
    @given(policy_sets(min_size=2, max_size=4))
    def test_commutative(self, group):
        forward = compose(group)
        backward = compose(list(reversed(group)))
        assert forward.agreement == backward.agreement
        assert forward.lost == backward.lost

    @SETTINGS
    @given(policy_sets(min_size=3, max_size=3))
    def test_associative_on_capability(self, group):
        """Capability composes associatively.

        Stated on capability rather than on the whole Agreement: folding a
        composed Agreement back in re-derives duties and identifiers, so the
        documents differ structurally while permitting exactly the same set.
        """
        a, b, c = group
        left = compose([compose([a, b]).agreement, c]).capability
        right = compose([a, compose([b, c]).agreement]).capability
        assert left == right == compose(group).capability

    @SETTINGS
    @given(policies())
    def test_idempotent(self, policy):
        once = compose([policy]).capability
        twice = compose([policy, policy]).capability
        assert once == twice == capability(policy)

    @SETTINGS
    @given(policies())
    def test_single_policy_composition_preserves_capability(self, policy):
        assert compose([policy]).capability == capability(policy)


class TestDeterminism:
    @SETTINGS
    @given(policy_sets(min_size=1, max_size=4))
    def test_identical_input_gives_byte_identical_output(self, group):
        """Needed for a provenance hash: the same contributors must always
        yield the same Agreement document."""
        first = json.dumps(policy_to_jsonld(compose(group).agreement), sort_keys=True)
        second = json.dumps(policy_to_jsonld(compose(group).agreement), sort_keys=True)
        assert first == second

    @SETTINGS
    @given(policy_sets(min_size=2, max_size=4))
    def test_agreement_uid_is_order_independent(self, group):
        assert compose(group).agreement.uid == compose(list(reversed(group))).agreement.uid

    @SETTINGS
    @given(policy_sets(min_size=1, max_size=3))
    def test_agreement_records_its_contributors(self, group):
        agreement = compose(group).agreement
        assert agreement.derived_from == frozenset(p.uid for p in group)


class TestConjunction:
    @SETTINGS
    @given(constraint_sets(), constraint_sets())
    def test_conjunction_is_commutative(self, left, right):
        try:
            forward = conjoin([left, right])
        except Unsatisfiable:
            try:
                conjoin([right, left])
            except Unsatisfiable:
                return
            raise AssertionError("unsatisfiable in one order only")
        assert forward == conjoin([right, left])

    @SETTINGS
    @given(constraint_sets())
    def test_conjunction_is_idempotent(self, constraints):
        try:
            once = conjoin([constraints])
        except Unsatisfiable:
            return
        assert conjoin([constraints, constraints]) == once

    @SETTINGS
    @given(constraint_sets(), constraint_sets())
    def test_conjunction_never_loosens(self, left, right):
        """A conjoined set constrains at least as many operands as either side.

        The constraint-level statement of the same anti-monotonicity: adding a
        contributor's conditions can only narrow the permitted region.
        """
        try:
            both = conjoin([left, right])
        except Unsatisfiable:
            return
        operands = {c.left_operand for c in both}
        assert operands >= {c.left_operand for c in left}
        assert operands >= {c.left_operand for c in right}

    @SETTINGS
    @given(st.lists(constraint_sets(), min_size=1, max_size=4))
    def test_conjunction_is_order_independent(self, sets):
        try:
            forward = conjoin(sets)
        except Unsatisfiable:
            return
        assert conjoin(list(reversed(sets))) == forward
