"""Tests for the ODRL model and parser.

Two groups matter most. **Canonicalisation**, because composition compares
constraints across policies and two policies expressing the same restriction
differently must intersect rather than conflict. And **subset enforcement**,
because a constraint that is silently dropped produces a composed capability
nobody granted -- plausible-looking numbers that are simply wrong.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from odrl_fl.odrl import (
    Action,
    Constraint,
    Duty,
    LeftOperand,
    MalformedPolicy,
    Operator,
    Permission,
    Policy,
    PolicyType,
    Prohibition,
    UnsupportedConstruct,
    load_policies,
    load_policy,
    parse_constraint,
    parse_policy,
    policy_to_jsonld,
)

POLICY_DIR = Path(__file__).resolve().parent.parent / "policies"


def c(left: str, op: str, right) -> dict:
    return {"leftOperand": left, "operator": op, "rightOperand": right}


def minimal(**overrides) -> dict:
    doc = {
        "@type": "Offer",
        "uid": "urn:policy:test",
        "permission": [{"action": "use"}],
    }
    doc.update(overrides)
    return doc


class TestCanonicalisation:
    def test_eq_becomes_is_any_of_for_set_valued_operands(self):
        """`purpose eq research` and `purpose isAnyOf [research]` are the same
        restriction, and must compare equal or composition will treat two
        identical policies as conflicting."""
        a = parse_constraint(c("purpose", "eq", "research"))
        b = parse_constraint(c("purpose", "isAnyOf", ["research"]))
        assert a == b
        assert a.operator is Operator.IS_ANY_OF
        assert a.right_operand == frozenset({"research"})

    def test_neq_becomes_is_none_of(self):
        a = parse_constraint(c("purpose", "neq", "commercial"))
        b = parse_constraint(c("purpose", "isNoneOf", ["commercial"]))
        assert a == b
        assert a.operator is Operator.IS_NONE_OF

    def test_scalar_operands_keep_their_comparison_operator(self):
        con = parse_constraint(c("dateTime", "lteq", "2027-06-30"))
        assert con.operator is Operator.LTEQ
        assert con.right_operand == "2027-06-30"

    def test_constraints_are_hashable_and_deduplicate(self):
        a = parse_constraint(c("purpose", "eq", "research"))
        b = parse_constraint(c("purpose", "isAnyOf", ["research"]))
        assert len({a, b}) == 1

    def test_set_membership_order_is_irrelevant(self):
        a = parse_constraint(c("purpose", "isAnyOf", ["research", "commercial"]))
        b = parse_constraint(c("purpose", "isAnyOf", ["commercial", "research"]))
        assert a == b

    def test_ordering_is_total_for_deterministic_output(self):
        """Composition must emit rules in a stable order regardless of the
        order they arrived in, or two nodes could serialise the same policy
        differently."""
        cons = [
            parse_constraint(c("spatial", "eq", "EU")),
            parse_constraint(c("purpose", "eq", "research")),
            parse_constraint(c("dateTime", "lteq", "2027-01-01")),
        ]
        rng = random.Random(0)
        expected = sorted(cons)
        for _ in range(20):
            shuffled = cons[:]
            rng.shuffle(shuffled)
            assert sorted(shuffled) == expected


class TestSubsetEnforcement:
    @pytest.mark.parametrize(
        "operand", ["count", "elapsedTime", "meteredTime", "event", "payAmount"]
    )
    def test_stateful_operands_are_rejected(self, operand):
        with pytest.raises(UnsupportedConstruct, match="stateful"):
            parse_constraint(c(operand, "lteq", 20))

    @pytest.mark.parametrize("operand", ["cohortSize", "recipientCount"])
    def test_self_referential_operands_are_rejected(self, operand):
        with pytest.raises(UnsupportedConstruct, match="contributor set"):
            parse_constraint(c(operand, "gteq", 5))

    @pytest.mark.parametrize("container", ["and", "or", "xone"])
    def test_logical_containers_are_rejected(self, container):
        with pytest.raises(UnsupportedConstruct, match="logical constraint"):
            parse_constraint({container: [c("purpose", "eq", "research")]})

    def test_unknown_action_is_rejected(self):
        with pytest.raises(UnsupportedConstruct, match="action"):
            parse_policy(minimal(permission=[{"action": "sell"}]))

    def test_consequence_and_remedy_are_rejected(self):
        with pytest.raises(UnsupportedConstruct, match="consequence|remedy"):
            parse_policy(
                minimal(
                    permission=[
                        {
                            "action": "use",
                            "duty": [{"action": "delete", "consequence": []}],
                        }
                    ]
                )
            )

    def test_rejection_is_loud_not_silent(self):
        """The property that protects every downstream result: an unsupported
        constraint must never be dropped, because the composed capability would
        then exceed what was granted."""
        doc = minimal(
            permission=[{"action": "use", "constraint": [c("count", "lteq", 20)]}]
        )
        with pytest.raises(UnsupportedConstruct):
            parse_policy(doc)

    def test_set_operator_on_a_scalar_operand_is_rejected(self):
        with pytest.raises(UnsupportedConstruct):
            parse_constraint(c("dateTime", "isAnyOf", ["2027-01-01"]))


class TestMalformedInput:
    def test_missing_uid(self):
        with pytest.raises(MalformedPolicy, match="uid"):
            parse_policy({"@type": "Offer", "permission": [{"action": "use"}]})

    def test_policy_with_no_rules(self):
        with pytest.raises(MalformedPolicy, match="no rules"):
            parse_policy({"uid": "urn:policy:empty", "@type": "Offer"})

    def test_rule_without_action(self):
        with pytest.raises(MalformedPolicy, match="action"):
            parse_policy(minimal(permission=[{"constraint": []}]))

    def test_constraint_without_right_operand(self):
        with pytest.raises(MalformedPolicy, match="rightOperand"):
            parse_constraint({"leftOperand": "purpose", "operator": "eq"})

    def test_constraint_without_left_operand(self):
        with pytest.raises(MalformedPolicy, match="leftOperand"):
            parse_constraint({"operator": "eq", "rightOperand": "research"})

    def test_not_an_object(self):
        with pytest.raises(MalformedPolicy):
            parse_policy("[1, 2, 3]")


class TestJsonLdPragmatics:
    def test_prefixed_and_bare_terms_are_equivalent(self):
        """Real policies write `odrl:use`, a full IRI, or bare `use`."""
        bare = parse_policy(minimal(permission=[{"action": "use"}]))
        prefixed = parse_policy(minimal(permission=[{"action": "odrl:use"}]))
        iri = parse_policy(
            minimal(permission=[{"action": "http://www.w3.org/ns/odrl/2/use"}])
        )
        assert bare.permitted_actions == prefixed.permitted_actions == iri.permitted_actions

    def test_single_object_where_a_list_is_allowed(self):
        policy = parse_policy(minimal(permission={"action": "use"}))
        assert policy.permitted_actions == {Action.USE}

    def test_id_objects_are_unwrapped(self):
        policy = parse_policy(
            minimal(assigner={"@id": "urn:party:hospital"}, target={"@id": "urn:asset:x"})
        )
        assert policy.assigner == "hospital"
        assert policy.target == "x"

    def test_typed_value_objects(self):
        con = parse_constraint(
            {
                "leftOperand": "dpEpsilon",
                "operator": "lteq",
                "rightOperand": {"@value": 1.0, "@type": "xsd:decimal"},
            }
        )
        assert con.right_operand == 1.0

    def test_round_trip_through_json_ld(self):
        original = load_policy(POLICY_DIR / "p3-national-registry.json")
        assert parse_policy(policy_to_jsonld(original)) == original

    def test_serialisation_is_stable(self):
        """Byte-stable output, so a policy can be hashed into a provenance
        record later."""
        policy = load_policy(POLICY_DIR / "p1-university-hospital.json")
        assert policy_to_jsonld(policy) == policy_to_jsonld(policy)


class TestWorkedExamplePolicies:
    def test_all_five_parse(self):
        policies = load_policies(POLICY_DIR)
        assert len(policies) == 5
        assert all(p.type is PolicyType.OFFER for p in policies)

    def test_p1_is_the_permissive_one(self):
        p1 = load_policy(POLICY_DIR / "p1-university-hospital.json")
        assert p1.permitted_actions == {
            Action.USE,
            Action.AGGREGATE,
            Action.DERIVE,
            Action.DISTRIBUTE,
        }
        assert not p1.prohibited_actions

    def test_p2_prohibits_derivation(self):
        p2 = load_policy(POLICY_DIR / "p2-private-clinic.json")
        assert Action.DERIVE in p2.prohibited_actions
        assert Action.DERIVE not in p2.permitted_actions

    def test_p3_permits_only_aggregation_and_carries_a_dp_duty(self):
        """The restrictive registry: valuable data, minimal grant. It is what
        collapses the composed capability set in the worked example."""
        p3 = load_policy(POLICY_DIR / "p3-national-registry.json")
        assert p3.permitted_actions == {Action.AGGREGATE}
        assert p3.prohibited_actions == {Action.DISTRIBUTE, Action.DERIVE}

        duties = p3.permission_for(Action.AGGREGATE).duties
        assert len(duties) == 1
        duty = next(iter(duties))
        assert duty.action is Action.ANONYMIZE
        epsilon = next(iter(duty.constraints))
        assert epsilon.left_operand is LeftOperand.DP_EPSILON
        assert epsilon.right_operand == 1.0

    def test_p5_allows_commercial_purposes(self):
        p5 = load_policy(POLICY_DIR / "p5-startup.json")
        purposes = next(iter(p5.permission_for(Action.USE).constraints))
        assert purposes.right_operand == frozenset({"research", "commercial"})


class TestModelConstruction:
    def test_policies_are_hashable_and_comparable(self):
        """Composition builds sets of rules, so the value types have to behave
        like values."""
        a = Policy(uid="urn:p", permissions=frozenset({Permission(action=Action.USE)}))
        b = Policy(uid="urn:p", permissions=frozenset({Permission(action=Action.USE)}))
        assert a == b
        assert len({a, b}) == 1

    def test_permitted_and_prohibited_action_views(self):
        policy = Policy(
            uid="urn:p",
            permissions=frozenset({Permission(action=Action.USE)}),
            prohibitions=frozenset({Prohibition(action=Action.DISTRIBUTE)}),
        )
        assert policy.permitted_actions == {Action.USE}
        assert policy.prohibited_actions == {Action.DISTRIBUTE}

    def test_permission_lookup_by_action(self):
        policy = Policy(uid="urn:p", permissions=frozenset({Permission(action=Action.USE)}))
        assert policy.permission_for(Action.USE) is not None
        assert policy.permission_for(Action.DERIVE) is None

    def test_empty_set_valued_constraint_is_rejected(self):
        with pytest.raises(MalformedPolicy):
            Constraint(LeftOperand.PURPOSE, Operator.IS_ANY_OF, frozenset())

    def test_describe_is_readable(self):
        policy = load_policy(POLICY_DIR / "p2-private-clinic.json")
        text = policy.describe()
        assert "must not derive" in text
        assert "may aggregate" in text

    def test_duty_renders_its_constraints(self):
        duty = Duty(
            action=Action.ANONYMIZE,
            constraints=frozenset(
                {Constraint(LeftOperand.DP_EPSILON, Operator.LTEQ, 1.0)}
            ),
        )
        assert "anonymize" in str(duty)
        assert "dpEpsilon" in str(duty)
