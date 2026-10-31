"""Tests for membership over time.

The class to read first is `TestTheLicenceFollowsTheData`. It pins the trap the
whole module exists to avoid: a departure *widens* what the remaining members
would permit, every step of that reasoning is correct, and acting on it would
license uses of data whose owner has just withdrawn. The distinction between the
active set and the contribution set is the fix, and
`test_the_governing_agreement_ignores_who_is_still_training` is the assertion
that keeps it.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from odrl_fl.corpus import generate_corpus
from odrl_fl.lifecycle import (
    ConsortiumCollapsed,
    Departure,
    Membership,
    active_agreement,
    capability_at,
    constraint_holds_at,
    expires_between,
    expiring,
    governing_agreement,
    in_force,
    reevaluate,
    widened_purposes,
    withdraw,
)
from odrl_fl.odrl import (
    Action,
    Constraint,
    LeftOperand,
    MalformedPolicy,
    Operator,
    Permission,
    Policy,
    PolicyType,
    capability,
)
from odrl_fl.selection import requirement

ACTIONS = (Action.USE, Action.AGGREGATE)


def offer(uid: str, *purposes: str, actions=ACTIONS) -> Policy:
    constraints = frozenset()
    if purposes:
        constraints = frozenset(
            {Constraint(LeftOperand.PURPOSE, Operator.IS_ANY_OF, frozenset(purposes))}
        )
    return Policy(
        uid=uid,
        type=PolicyType.OFFER,
        permissions=frozenset(
            Permission(action=a, constraints=constraints) for a in actions
        ),
    )


def when(text: str) -> Constraint:
    return Constraint(LeftOperand.DATETIME, Operator.LTEQ, text)


# ---------------------------------------------------------------------------


class TestTemporalEvaluation:
    def test_a_time_limit_holds_before_it_lapses(self):
        assert constraint_holds_at(when("2028-12-31"), date(2026, 1, 1))

    def test_a_time_limit_fails_after_it_lapses(self):
        assert not constraint_holds_at(when("2026-01-01"), date(2027, 1, 1))

    def test_a_moratorium_blocks_until_it_lifts(self):
        lifts = Constraint(LeftOperand.DATETIME, Operator.GTEQ, "2027-01-01")
        assert not constraint_holds_at(lifts, date(2026, 6, 1))
        assert constraint_holds_at(lifts, date(2027, 6, 1))

    def test_a_datetime_and_a_date_compare_without_raising(self):
        """DUO supplies dates; a server may hold a datetime. Must not blow up."""
        assert constraint_holds_at(
            when("2028-12-31"), datetime(2026, 1, 1, 12, 30, tzinfo=UTC)
        )

    def test_non_temporal_constraints_are_left_alone(self):
        purpose = Constraint(
            LeftOperand.PURPOSE, Operator.IS_ANY_OF, frozenset({"research"})
        )
        assert constraint_holds_at(purpose, date(2026, 1, 1))

    def test_an_unparseable_bound_cannot_be_built_in_the_first_place(self):
        """The model validates ISO-8601 at construction, so time evaluation
        never sees a malformed bound. `constraint_holds_at` still guards against
        one rather than raising deep inside a round -- but the guard is
        unreachable through the normal path, which is the better outcome.
        """
        with pytest.raises(MalformedPolicy, match="not ISO-8601"):
            when("not-a-date")

    def test_in_force_drops_only_the_lapsed_permissions(self):
        policy = expiring(offer("urn:a"), "2026-01-01", actions=[Action.USE])
        surviving = {p.action for p in in_force(policy, date(2027, 1, 1)).permissions}
        assert surviving == {Action.AGGREGATE}

    def test_in_force_is_a_no_op_when_nothing_has_lapsed(self):
        policy = offer("urn:a", "research")
        assert in_force(policy, date(2026, 1, 1)) is policy

    def test_expires_between_detects_the_change(self):
        policy = expiring(offer("urn:a"), "2026-06-01")
        assert expires_between(policy, date(2026, 1, 1), date(2027, 1, 1))
        assert not expires_between(policy, date(2026, 1, 1), date(2026, 2, 1))


class TestMembership:
    def test_departure_shrinks_active_but_not_contributed(self):
        """The invariant everything else rests on."""
        m = Membership.starting([1, 2, 3])
        after = m.depart(2, server_round=4, reason=Departure.WITHDREW)

        assert after.active == (1, 3)
        assert after.contributed == (1, 2, 3)
        assert after.departed == {2}

    def test_departing_twice_changes_nothing(self):
        m = Membership.starting([1, 2]).depart(2, 3, Departure.WITHDREW)
        assert m.depart(2, 4, Departure.EXPIRED) == m

    def test_a_departure_is_recorded_with_round_and_reason(self):
        m = Membership.starting([1, 2]).depart(2, 7, Departure.EXPIRED, "lapsed")
        assert m.departures[0].server_round == 7
        assert m.departures[0].reason is Departure.EXPIRED
        assert m.departures[0].detail == "lapsed"

    def test_a_joiner_is_added_to_both_sets(self):
        m = Membership.starting([1]).admit(2)
        assert m.active == (1, 2)
        assert m.contributed == (1, 2)


class TestReevaluation:
    def policies(self):
        return {n: offer(f"urn:p:{n}", "research") for n in (1, 2, 3)}

    def test_a_stable_consortium_is_left_alone(self):
        out = reevaluate(
            Membership.starting([1, 2, 3]),
            self.policies(),
            server_round=2,
            at=date(2026, 1, 1),
        )
        assert not out.changed
        assert out.membership.active == (1, 2, 3)

    def test_a_withdrawal_is_detected(self):
        policies = self.policies()
        policies[2] = withdraw(policies[2])
        out = reevaluate(
            Membership.starting([1, 2, 3]), policies, server_round=3,
            at=date(2026, 1, 1),
        )
        assert [r.node_id for r in out.departed] == [2]
        assert out.departed[0].reason is Departure.WITHDREW

    def test_an_expiry_is_detected_and_named_as_such(self):
        """A lapse and a withdrawal are different events, recorded differently."""
        policies = self.policies()
        policies[2] = expiring(policies[2], "2026-01-01")
        out = reevaluate(
            Membership.starting([1, 2, 3]), policies, server_round=3,
            at=date(2027, 1, 1),
        )
        assert out.departed[0].reason is Departure.EXPIRED

    def test_a_node_that_stopped_answering_is_dropped(self):
        out = reevaluate(
            Membership.starting([1, 2, 3]), self.policies(), server_round=2,
            at=date(2026, 1, 1), unreachable=[3],
        )
        assert out.departed[0].reason is Departure.UNREACHABLE

    def test_a_repeat_duty_breach_is_dropped(self):
        out = reevaluate(
            Membership.starting([1, 2, 3]), self.policies(), server_round=5,
            at=date(2026, 1, 1), breaches={1: 3}, breach_limit=2,
        )
        assert out.departed[0].node_id == 1
        assert out.departed[0].reason is Departure.DUTY_BREACH

    def test_a_single_breach_is_tolerated(self):
        out = reevaluate(
            Membership.starting([1, 2, 3]), self.policies(), server_round=5,
            at=date(2026, 1, 1), breaches={1: 1}, breach_limit=2,
        )
        assert not out.changed

    def test_terms_narrowed_below_the_request_count_as_withdrawal(self):
        policies = self.policies()
        policies[2] = offer("urn:p:2", "clinical-care")
        out = reevaluate(
            Membership.starting([1, 2, 3]), policies, server_round=4,
            at=date(2026, 1, 1), request=requirement(*ACTIONS, purpose="research"),
        )
        assert [r.node_id for r in out.departed] == [2]

    def test_everybody_leaving_is_fatal(self):
        """An empty federation must stop, not quietly train on nothing."""
        policies = {n: withdraw(p) for n, p in self.policies().items()}
        with pytest.raises(ConsortiumCollapsed):
            reevaluate(
                Membership.starting([1, 2, 3]), policies, server_round=2,
                at=date(2026, 1, 1),
            )


class TestTheLicenceFollowsTheData:
    """The trap this module exists to avoid."""

    def setup_policies(self):
        # 3 is the restrictive member: it alone confines the consortium.
        return {
            1: offer("urn:p:1", "research"),
            2: offer("urn:p:2", "research"),
            3: offer("urn:p:3", "ds-cancer"),
        }

    def test_departure_widens_what_the_remaining_members_would_permit(self):
        """Correct reasoning, from a premise that stopped holding."""
        policies = self.setup_policies()
        after = Membership.starting([1, 2, 3]).depart(3, 2, Departure.WITHDREW)

        governing, extra = widened_purposes(policies, after)
        assert governing == {"ds-cancer"}
        assert "research" in extra

    def test_the_governing_agreement_ignores_who_is_still_training(self):
        """The assertion that keeps the licence honest.

        Node 3's data is in the weights from round one. Its departure must not
        widen what the model may be used for.
        """
        policies = self.setup_policies()
        before = Membership.starting([1, 2, 3])
        after = before.depart(3, 2, Departure.WITHDREW)

        assert governing_agreement(policies, after) == governing_agreement(
            policies, before
        )

    def test_the_active_agreement_is_strictly_more_permissive(self):
        policies = self.setup_policies()
        after = Membership.starting([1, 2, 3]).depart(3, 2, Departure.WITHDREW)

        governing = governing_agreement(policies, after)
        active = active_agreement(policies, after)
        assert active != governing
        assert capability(governing) <= capability(active)

    def test_reevaluation_reports_the_widening_it_refuses_to_act_on(self):
        policies = {
            1: offer("urn:p:1", "research"),
            2: offer("urn:p:2", "research", actions=(Action.USE,)),
        }
        out = reevaluate(
            Membership.starting([1, 2]),
            {1: policies[1], 2: withdraw(policies[2])},
            server_round=3,
            at=date(2026, 1, 1),
        )
        # Node 2 only ever granted `use`, so its departure restores `aggregate`
        # to the active set -- but not to the model.
        assert Action.AGGREGATE in out.widened

    def test_the_widening_is_real_on_the_corpus(self):
        """Not an artifact of hand-built policies."""
        pool = {c.index: c.policy for c in generate_corpus(12, seed=4)}
        members = sorted(pool)[:5]
        before = Membership.starting(members)

        widened_any = False
        for node in members:
            after = before.depart(node, 2, Departure.WITHDREW)
            _governing, extra = widened_purposes(pool, after)
            assert governing_agreement(pool, after) == governing_agreement(pool, before)
            widened_any = widened_any or bool(extra)
        assert widened_any, "no departure widened; the corpus fixture has drifted"


class TestCapabilityAt:
    def test_capability_narrows_once_a_limit_lapses(self):
        policy = expiring(offer("urn:a"), "2026-01-01", actions=[Action.USE])
        assert capability_at(policy, date(2025, 1, 1)) == {
            Action.USE,
            Action.AGGREGATE,
        }
        assert capability_at(policy, date(2027, 1, 1)) == {Action.AGGREGATE}

    def test_a_fully_lapsed_policy_permits_nothing(self):
        policy = expiring(offer("urn:a"), "2026-01-01")
        assert not capability_at(policy, date(2027, 1, 1))
