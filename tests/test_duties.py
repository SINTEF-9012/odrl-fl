"""Tests for duty classification and enforcement.

The classification is the substance of increment 7, so most of these tests are
about it rather than about the mechanisms. `TestTheDomainFinding` pins the claim
the increment rests on: of the duties real DUO annotations produce, none is
dischargeable during training. If a future vocabulary change made that false,
these tests should fail and the README should be rewritten -- which is the
point of asserting it.
"""

from __future__ import annotations

import pytest

from odrl_fl.corpus import generate_corpus
from odrl_fl.duties import (
    Compliance,
    Directive,
    DutyKind,
    Ledger,
    UnmetPrecondition,
    check_gates,
    classify,
    decode_directives,
    duties_of,
    encode_directives,
    plan_duties,
    verify,
)
from odrl_fl.odrl import (
    Action,
    Constraint,
    Duty,
    LeftOperand,
    Operator,
    Permission,
    Policy,
    PolicyType,
    compose,
    load_policies,
)
from odrl_fl.odrl.duo import translate

POLICY_DIR = "policies"


def duty(action: Action, *constraints: Constraint) -> Duty:
    return Duty(action=action, constraints=frozenset(constraints))


def epsilon(value: float) -> Constraint:
    return Constraint(LeftOperand.DP_EPSILON, Operator.LTEQ, value)


def agreement(*, obligations=(), preconditions=()) -> Policy:
    return Policy(
        uid="urn:test:agreement",
        type=PolicyType.AGREEMENT,
        permissions=frozenset(
            {Permission(action=Action.USE, duties=frozenset(preconditions))}
        ),
        obligations=frozenset(obligations),
    )


# ---------------------------------------------------------------------------


class TestClassification:
    def test_consent_gates(self):
        assert classify(duty(Action.OBTAIN_CONSENT)) is DutyKind.GATE

    def test_anonymise_dispatches(self):
        assert classify(duty(Action.ANONYMIZE)) is DutyKind.DISPATCH

    @pytest.mark.parametrize(
        "action",
        [Action.INFORM, Action.TRANSFER, Action.ATTRIBUTE, Action.COMPENSATE],
    )
    def test_the_rest_can_only_be_recorded(self, action):
        assert classify(duty(action)) is DutyKind.RECORD

    def test_an_unknown_duty_is_recorded_not_ignored(self):
        """The conservative default, and the one that matters.

        A duty nobody wrote a handler for must be carried and reported, never
        treated as satisfied because the code did not recognise it.
        """
        assert classify(duty(Action.DISTRIBUTE)) is DutyKind.RECORD


class TestTheDomainFinding:
    """The claim increment 7 rests on. Asserted so it cannot rot silently."""

    def test_no_duo_duty_is_dischargeable_during_training(self):
        for seed in range(4):
            for contributor in generate_corpus(15, seed=seed):
                plan = plan_duties(contributor.policy)
                assert not plan.directives, (
                    f"{contributor.codes} produced a runtime directive; the "
                    "README's claim that DUO has no such vocabulary is stale"
                )

    def test_duo_produces_exactly_three_duties(self):
        seen = set()
        for seed in range(4):
            for contributor in generate_corpus(15, seed=seed):
                pre, post = duties_of(contributor.policy)
                seen |= {d.action for d in (*pre, *post)}
        assert seen == {Action.OBTAIN_CONSENT, Action.INFORM, Action.TRANSFER}

    def test_irb_gates_and_publication_only_records(self):
        policy, _ = translate(
            ["DUO:0000042", "DUO:0000021", "DUO:0000019"], uid="urn:t"
        )
        plan = plan_duties(policy)
        assert [d.action for d in plan.gates] == [Action.OBTAIN_CONSENT]
        assert [o.action for o in plan.outstanding] == ["inform"]
        assert not plan.directives

    def test_the_runtime_duty_comes_only_from_the_profile_fixtures(self):
        """`anonymize` with a budget exists, but not in DUO."""
        fixtures = {p.uid.split(":")[-1][:2]: p for p in load_policies(POLICY_DIR)}
        plan = plan_duties(compose([fixtures["p1"], fixtures["p3"]]).agreement)
        assert plan.directives == (Directive(action="anonymize", epsilon=1.0),)


class TestPlanning:
    def test_a_precondition_from_one_member_binds_the_consortium(self):
        with_irb, _ = translate(["DUO:0000042", "DUO:0000021"], uid="urn:a")
        without, _ = translate(["DUO:0000042"], uid="urn:b")
        plan = plan_duties(compose([with_irb, without]).agreement)
        assert [d.action for d in plan.gates] == [Action.OBTAIN_CONSENT]

    def test_outstanding_obligations_name_who_owes_them(self):
        policy = agreement(obligations=[duty(Action.INFORM)])
        plan = plan_duties(policy, owed_by=("urn:party:a", "urn:party:b"))
        assert plan.outstanding[0].owed_by == ("urn:party:a", "urn:party:b")

    def test_outstanding_obligations_are_never_marked_discharged(self):
        """A ledger of open duties must not read like a clean bill of health."""
        plan = plan_duties(agreement(obligations=[duty(Action.INFORM)]))
        assert all(not o.discharged for o in plan.outstanding)

    def test_the_strictest_budget_wins(self):
        policy = agreement(
            preconditions=[duty(Action.ANONYMIZE, epsilon(0.5), epsilon(2.0))]
        )
        assert plan_duties(policy).directives[0].epsilon == 0.5

    def test_a_budgetless_dispatch_duty_still_dispatches(self):
        policy = agreement(preconditions=[duty(Action.ANONYMIZE)])
        assert plan_duties(policy).directives == (
            Directive(action="anonymize", epsilon=None),
        )

    def test_an_agreement_with_no_duties_plans_nothing(self):
        assert plan_duties(agreement()).is_trivial


class TestGate:
    def test_evidence_clears_the_gate(self):
        cleared = check_gates(
            [duty(Action.OBTAIN_CONSENT)], {"obtainConsent": "IRB-2026-0142"}
        )
        assert cleared == {"obtainConsent": "IRB-2026-0142"}

    def test_missing_evidence_is_fatal(self):
        with pytest.raises(UnmetPrecondition, match="obtainConsent"):
            check_gates([duty(Action.OBTAIN_CONSENT)], {})

    def test_blank_evidence_does_not_count(self):
        """A configured-but-empty reference is absence, not assertion."""
        with pytest.raises(UnmetPrecondition):
            check_gates([duty(Action.OBTAIN_CONSENT)], {"obtainConsent": "   "})

    def test_evidence_for_the_wrong_duty_does_not_help(self):
        with pytest.raises(UnmetPrecondition):
            check_gates([duty(Action.OBTAIN_CONSENT)], {"anonymize": "DPIA-1"})

    def test_no_gates_needs_no_evidence(self):
        assert check_gates([], {}) == {}


class TestDirectiveWire:
    def test_directives_round_trip(self):
        directives = (
            Directive(action="anonymize", epsilon=0.5),
            Directive(action="inform"),
        )
        assert decode_directives(encode_directives(directives)) == directives

    def test_an_empty_payload_yields_nothing(self):
        assert decode_directives("") == ()
        assert decode_directives(None) == ()

    def test_a_corrupt_payload_yields_nothing_rather_than_raising(self):
        """The client must still reply, so the server can reject it for silence."""
        assert decode_directives("{not json") == ()
        assert decode_directives('{"not": "a list"}') == ()

    def test_malformed_entries_are_skipped_individually(self):
        blob = '[{"action": "anonymize", "epsilon": 1.0}, {"no-action": true}]'
        assert decode_directives(blob) == (
            Directive(action="anonymize", epsilon=1.0),
        )


class TestVerification:
    def test_a_matching_claim_passes(self):
        ok, _ = verify(
            [Directive("anonymize", 1.0)], Compliance({"anonymize": 1.0})
        )
        assert ok

    def test_a_tighter_budget_than_required_passes(self):
        ok, _ = verify(
            [Directive("anonymize", 1.0)], Compliance({"anonymize": 0.5})
        )
        assert ok

    def test_a_looser_budget_fails(self):
        ok, reason = verify(
            [Directive("anonymize", 0.5)], Compliance({"anonymize": 2.0})
        )
        assert not ok
        assert "exceeds" in reason

    def test_silence_fails(self):
        """The load-bearing case.

        A reply reporting nothing is indistinguishable on the wire from a reply
        that skipped the duty, and only one of those is safe to aggregate.
        """
        ok, reason = verify([Directive("anonymize", 1.0)], Compliance())
        assert not ok
        assert "did not report" in reason

    def test_a_null_budget_fails_when_one_was_required(self):
        ok, _ = verify(
            [Directive("anonymize", 1.0)], Compliance({"anonymize": None})
        )
        assert not ok

    def test_no_directives_always_passes(self):
        ok, _ = verify([], Compliance())
        assert ok

    def test_compliance_round_trips(self):
        claim = Compliance({"anonymize": 0.5, "inform": None})
        assert Compliance.decode(claim.encode()).applied == claim.applied

    def test_a_corrupt_claim_decodes_to_claiming_nothing(self):
        assert Compliance.decode("{bad").applied == {}
        assert Compliance.decode("[1,2]").applied == {}


class TestLedger:
    def test_rejections_are_recorded_with_round_and_reason(self):
        ledger = Ledger()
        ledger.reject(3, 11, "budget exceeded")
        assert ledger.rejected_nodes == {11}
        assert any("round 3" in line for line in ledger.summary())

    def test_the_summary_states_outstanding_duties_as_outstanding(self):
        plan = plan_duties(
            agreement(obligations=[duty(Action.INFORM)]), owed_by=("urn:party:a",)
        )
        ledger = Ledger(outstanding=plan.outstanding)
        assert any("OUTSTANDING" in line for line in ledger.summary())
        assert any("urn:party:a" in line for line in ledger.summary())

    def test_cleared_gates_are_reported_with_their_evidence(self):
        ledger = Ledger(gates_cleared={"obtainConsent": "IRB-1"})
        assert any("IRB-1" in line for line in ledger.summary())
