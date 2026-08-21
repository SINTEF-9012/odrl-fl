"""Unit tests for the composition algebra, and the worked example.

`TestWorkedExample` is the one to read first: it pins the three subsets that
were computed by hand during design, including the case where admitting the
most valuable dataset collapses three capabilities to one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from odrl_fl.odrl import (
    FLAT,
    Action,
    Constraint,
    Duty,
    LeftOperand,
    LossReason,
    Operator,
    Permission,
    Policy,
    PolicyType,
    Prohibition,
    Taxonomy,
    Unsatisfiable,
    UnsupportedConstruct,
    capability,
    compose,
    conjoin,
    default_taxonomy,
    load_policies,
    merge_duties,
    satisfies,
)

POLICY_DIR = Path(__file__).resolve().parent.parent / "policies"


@pytest.fixture(scope="module")
def worked():
    """The five example Offers, keyed p1..p5."""
    return {p.uid.split(":")[-1][:2]: p for p in load_policies(POLICY_DIR)}


def purpose(*values, negated: bool = False) -> Constraint:
    op = Operator.IS_NONE_OF if negated else Operator.IS_ANY_OF
    return Constraint(LeftOperand.PURPOSE, op, frozenset(values))


def spatial(*values) -> Constraint:
    return Constraint(LeftOperand.SPATIAL, Operator.IS_ANY_OF, frozenset(values))


def offer(uid: str, permissions=(), prohibitions=(), obligations=()) -> Policy:
    return Policy(
        uid=uid,
        type=PolicyType.OFFER,
        permissions=frozenset(permissions),
        prohibitions=frozenset(prohibitions),
        obligations=frozenset(obligations),
    )


# ---------------------------------------------------------------------------


class TestWorkedExample:
    def test_three_permissive_contributors_keep_three_capabilities(self, worked):
        result = compose([worked["p1"], worked["p2"], worked["p4"]])
        assert result.capability == {Action.USE, Action.AGGREGATE, Action.DISTRIBUTE}
        assert {d.action for d in result.agreement.obligations} == {
            Action.ATTRIBUTE,
            Action.INFORM,
        }
        # P2 prohibits derivation, and deny overrides.
        assert result.lost[Action.DERIVE] is LossReason.PROHIBITED

    def test_equal_capability_at_a_different_cost(self, worked):
        """{P1,P4,P5} matches {P1,P2,P4} on capability but owes compensation.

        Two viable subsets, same permissions, different duty burden -- the
        trade-off that makes selection (increment 5) a real choice rather than
        a lookup.
        """
        result = compose([worked["p1"], worked["p4"], worked["p5"]])
        assert result.capability == {Action.USE, Action.AGGREGATE, Action.DISTRIBUTE}
        assert Action.COMPENSATE in {d.action for d in result.agreement.obligations}

    def test_adding_the_registry_collapses_capability(self, worked):
        """The headline, in miniature.

        P3 holds the most valuable data and grants only `aggregate`. Admitting
        it takes the federation from three capabilities to one -- it cannot
        even run inference with the resulting model.
        """
        without = compose([worked["p1"], worked["p2"]]).capability
        with_registry = compose([worked["p1"], worked["p2"], worked["p3"]]).capability

        assert len(without) == 3
        assert with_registry == {Action.AGGREGATE}
        assert with_registry < without  # strict subset: anti-monotonicity biting

    def test_commercial_purpose_is_lost_to_intersection(self, worked):
        """P5 permits commercial use; P1 and P4 permit research only. The
        agreement keeps only the common ground."""
        result = compose([worked["p1"], worked["p4"], worked["p5"]])
        use = result.agreement.permission_for(Action.USE)
        purposes = next(
            c for c in use.constraints if c.left_operand is LeftOperand.PURPOSE
        )
        assert purposes.right_operand == frozenset({"research"})

    def test_without_subsumption_erosion_is_overstated(self, worked):
        """Why the taxonomy is load-bearing rather than decorative.

        P1 permits use across the EU; P4, a regional hospital, permits it in
        Germany. Those are compatible -- Germany is in the EU -- and the meet is
        {DE}. Under a flat vocabulary the two read as a conflict and `use` is
        lost, so measured erosion would be partly an artifact of omitting the
        hierarchy rather than a property of the policies.
        """
        pair = [worked["p1"], worked["p4"]]

        with_taxonomy = compose(pair)
        assert Action.USE in with_taxonomy.capability
        spatial_constraint = next(
            c
            for c in with_taxonomy.agreement.permission_for(Action.USE).constraints
            if c.left_operand is LeftOperand.SPATIAL
        )
        assert spatial_constraint.right_operand == frozenset({"DE"})

        flat = compose(pair, taxonomy=FLAT)
        assert Action.USE not in flat.capability
        assert flat.lost[Action.USE] is LossReason.CONSTRAINT_CONFLICT

    def test_agreement_is_traceable_to_its_contributors(self, worked):
        result = compose([worked["p1"], worked["p3"]])
        assert result.agreement.type is PolicyType.AGREEMENT
        assert result.agreement.derived_from == {
            worked["p1"].uid,
            worked["p3"].uid,
        }


class TestSetConjunction:
    def test_intersection(self):
        result = conjoin([{purpose("research", "education")}, {purpose("research")}])
        assert next(iter(result)).right_operand == frozenset({"research"})

    def test_subsumption_finds_common_ground(self):
        """`{EU}` and `{DE}` are compatible: Germany is in the EU.

        Without the taxonomy this reads as a conflict, which would inflate
        measured erosion.
        """
        result = conjoin([{spatial("EU")}, {spatial("DE")}])
        assert next(iter(result)).right_operand == frozenset({"DE"})

    def test_disjoint_values_are_unsatisfiable(self):
        with pytest.raises(Unsatisfiable, match="no shared value"):
            conjoin([{purpose("research")}, {purpose("commercial")}])

    def test_unrelated_regions_are_unsatisfiable(self):
        with pytest.raises(Unsatisfiable):
            conjoin([{spatial("EU")}, {spatial("US")}])

    def test_negatives_accumulate(self):
        result = conjoin(
            [
                {purpose("commercial", negated=True)},
                {purpose("education", negated=True)},
            ]
        )
        constraint = next(iter(result))
        assert constraint.operator is Operator.IS_NONE_OF
        assert constraint.right_operand == frozenset({"commercial", "education"})

    def test_negative_narrows_a_positive(self):
        result = conjoin(
            [
                {purpose("research", "commercial")},
                {purpose("commercial", negated=True)},
            ]
        )
        constraint = next(iter(result))
        assert constraint.operator is Operator.IS_ANY_OF
        assert constraint.right_operand == frozenset({"research"})

    def test_negative_can_exhaust_a_positive(self):
        with pytest.raises(Unsatisfiable, match="also excluded"):
            conjoin([{purpose("research")}, {purpose("research", negated=True)}])

    def test_exclusion_is_inherited_downwards(self):
        """Excluding `EU` excludes `DE`."""
        with pytest.raises(Unsatisfiable):
            conjoin(
                [
                    {spatial("DE")},
                    {Constraint(LeftOperand.SPATIAL, Operator.IS_NONE_OF, frozenset({"EU"}))},
                ]
            )

    def test_redundant_values_are_reduced(self):
        """`{research, clinical-research}` means `{research}`."""
        result = conjoin([{purpose("research", "clinical-research")}])
        assert next(iter(result)).right_operand == frozenset({"research"})

    def test_flat_taxonomy_falls_back_to_exact_matching(self):
        with pytest.raises(Unsatisfiable):
            conjoin([{spatial("EU")}, {spatial("DE")}], taxonomy=FLAT)


class TestIntervalConjunction:
    def test_tighter_upper_bound_wins(self):
        result = conjoin(
            [
                {Constraint(LeftOperand.DATETIME, Operator.LTEQ, "2027-06-30")},
                {Constraint(LeftOperand.DATETIME, Operator.LTEQ, "2026-01-01")},
            ]
        )
        assert next(iter(result)).right_operand == "2026-01-01"

    def test_tighter_lower_bound_wins(self):
        result = conjoin(
            [
                {Constraint(LeftOperand.DATETIME, Operator.GTEQ, "2026-01-01")},
                {Constraint(LeftOperand.DATETIME, Operator.GTEQ, "2027-06-30")},
            ]
        )
        assert next(iter(result)).right_operand == "2027-06-30"

    def test_crossed_bounds_are_unsatisfiable(self):
        with pytest.raises(Unsatisfiable, match="do not overlap"):
            conjoin(
                [
                    {Constraint(LeftOperand.DATETIME, Operator.GTEQ, "2028-01-01")},
                    {Constraint(LeftOperand.DATETIME, Operator.LTEQ, "2026-01-01")},
                ]
            )

    def test_strictest_privacy_budget_wins(self):
        """Falls out of the interval rule: eps<=1 and eps<=0.5 gives eps<=0.5."""
        result = conjoin(
            [
                {Constraint(LeftOperand.DP_EPSILON, Operator.LTEQ, 1.0)},
                {Constraint(LeftOperand.DP_EPSILON, Operator.LTEQ, 0.5)},
            ]
        )
        assert next(iter(result)).right_operand == 0.5

    def test_a_window_survives_as_two_bounds(self):
        result = conjoin(
            [
                {Constraint(LeftOperand.DATETIME, Operator.GTEQ, "2026-01-01")},
                {Constraint(LeftOperand.DATETIME, Operator.LTEQ, "2027-06-30")},
            ]
        )
        assert len(result) == 2


class TestProhibitions:
    def test_unconstrained_prohibition_removes_the_action(self):
        a = offer("urn:a", permissions=[Permission(action=Action.DERIVE)])
        b = offer(
            "urn:b",
            permissions=[Permission(action=Action.DERIVE)],
            prohibitions=[Prohibition(action=Action.DERIVE)],
        )
        result = compose([a, b])
        assert Action.DERIVE not in result.capability
        assert result.lost[Action.DERIVE] is LossReason.PROHIBITED

    def test_prohibition_is_reported_ahead_of_a_missing_grant(self):
        """Both apply; deny overrides, so the prohibition is the binding cause."""
        a = offer("urn:a", permissions=[Permission(action=Action.USE)])
        b = offer(
            "urn:b",
            permissions=[Permission(action=Action.AGGREGATE)],
            prohibitions=[Prohibition(action=Action.USE)],
        )
        assert compose([a, b]).lost[Action.USE] is LossReason.PROHIBITED

    def test_constrained_prohibition_carves_out_a_region(self):
        """DUO's NMDS -- "no methods research" -- has exactly this shape."""
        a = offer(
            "urn:a",
            permissions=[
                Permission(
                    action=Action.DERIVE,
                    constraints=frozenset({purpose("research")}),
                )
            ],
        )
        b = offer(
            "urn:b",
            permissions=[
                Permission(
                    action=Action.DERIVE,
                    constraints=frozenset({purpose("research")}),
                )
            ],
            prohibitions=[
                Prohibition(
                    action=Action.DERIVE,
                    constraints=frozenset({purpose("methods-research")}),
                )
            ],
        )
        result = compose([a, b])
        assert Action.DERIVE in result.capability
        constraints = result.agreement.permission_for(Action.DERIVE).constraints
        purposes = next(
            c for c in constraints if c.left_operand is LeftOperand.PURPOSE
        )
        # research survives; the excluded sub-purpose is not separately listed,
        # since "research except methods-research" is not expressible here.
        assert "research" in purposes.right_operand

    def test_multi_constraint_prohibition_is_rejected(self):
        a = offer("urn:a", permissions=[Permission(action=Action.DERIVE)])
        b = offer(
            "urn:b",
            permissions=[Permission(action=Action.DERIVE)],
            prohibitions=[
                Prohibition(
                    action=Action.DERIVE,
                    constraints=frozenset({purpose("research"), spatial("EU")}),
                )
            ],
        )
        with pytest.raises(UnsupportedConstruct, match="disjunction"):
            compose([a, b])


class TestDuties:
    def test_duties_accumulate(self):
        merged = merge_duties(
            [{Duty(action=Action.ATTRIBUTE)}, {Duty(action=Action.INFORM)}]
        )
        assert {d.action for d in merged} == {Action.ATTRIBUTE, Action.INFORM}

    def test_duties_on_the_same_action_merge_to_the_strictest(self):
        strict = Duty(
            action=Action.ANONYMIZE,
            constraints=frozenset(
                {Constraint(LeftOperand.DP_EPSILON, Operator.LTEQ, 0.5)}
            ),
        )
        loose = Duty(
            action=Action.ANONYMIZE,
            constraints=frozenset(
                {Constraint(LeftOperand.DP_EPSILON, Operator.LTEQ, 2.0)}
            ),
        )
        merged = merge_duties([{strict}, {loose}])
        assert len(merged) == 1
        assert next(iter(next(iter(merged)).constraints)).right_operand == 0.5


class TestCapabilityAndMatching:
    def test_capability_subtracts_prohibitions(self):
        policy = offer(
            "urn:a",
            permissions=[Permission(action=Action.USE), Permission(action=Action.DERIVE)],
            prohibitions=[Prohibition(action=Action.DERIVE)],
        )
        assert capability(policy) == {Action.USE}

    def test_satisfies_when_the_agreement_covers_the_request(self, worked):
        agreement = compose([worked["p1"], worked["p2"], worked["p4"]]).agreement
        request = Policy(
            uid="urn:request:federation",
            type=PolicyType.REQUEST,
            permissions=frozenset(
                {
                    Permission(action=Action.AGGREGATE),
                    Permission(action=Action.DISTRIBUTE),
                }
            ),
        )
        assert satisfies(agreement, request)

    def test_does_not_satisfy_a_missing_capability(self, worked):
        agreement = compose([worked["p1"], worked["p2"], worked["p3"]]).agreement
        request = Policy(
            uid="urn:request:federation",
            type=PolicyType.REQUEST,
            permissions=frozenset({Permission(action=Action.DISTRIBUTE)}),
        )
        assert not satisfies(agreement, request)

    def test_does_not_satisfy_an_incompatible_purpose(self, worked):
        agreement = compose([worked["p1"], worked["p4"]]).agreement
        commercial = Policy(
            uid="urn:request:commercial",
            type=PolicyType.REQUEST,
            permissions=frozenset(
                {
                    Permission(
                        action=Action.USE,
                        constraints=frozenset({purpose("commercial")}),
                    )
                }
            ),
        )
        assert not satisfies(agreement, commercial)

    def test_a_request_broader_than_the_grant_is_refused(self):
        """Containment, not overlap.

        An agreement permitting only biomedical research does not answer yes to
        "may we use this for research". Testing for a non-empty intersection
        instead is how a consortium ends up believing it holds a right it does
        not have -- and it silently reduces a purpose-constrained request to a
        no-op, since almost every purpose in the corpus overlaps `research`.
        """
        agreement = offer(
            "urn:narrow",
            permissions=[
                Permission(
                    action=Action.USE,
                    constraints=frozenset({purpose("biomedical-research")}),
                )
            ],
        )
        broad = Policy(
            uid="urn:request:broad",
            type=PolicyType.REQUEST,
            permissions=frozenset(
                {
                    Permission(
                        action=Action.USE,
                        constraints=frozenset({purpose("research")}),
                    )
                }
            ),
        )
        assert not satisfies(agreement, broad)

    def test_a_request_narrower_than_the_grant_is_allowed(self):
        """The other direction must still work, or nothing is ever feasible."""
        agreement = offer(
            "urn:broad",
            permissions=[
                Permission(
                    action=Action.USE,
                    constraints=frozenset({purpose("research")}),
                )
            ],
        )
        narrow = Policy(
            uid="urn:request:narrow",
            type=PolicyType.REQUEST,
            permissions=frozenset(
                {
                    Permission(
                        action=Action.USE,
                        constraints=frozenset({purpose("ds-cancer")}),
                    )
                }
            ),
        )
        assert satisfies(agreement, narrow)

    def test_a_request_touching_a_prohibited_subtree_is_refused(self):
        """Exclusion is inherited downwards, so a broad request collides with it.

        General research subsumes cancer research, so an agreement excluding
        cancer research cannot grant general research. A request for a sibling
        purpose is unaffected.
        """
        agreement = offer(
            "urn:excluding",
            permissions=[
                Permission(
                    action=Action.USE,
                    constraints=frozenset({purpose("ds-cancer", negated=True)}),
                )
            ],
        )

        def request_for(value: str) -> Policy:
            return Policy(
                uid=f"urn:request:{value}",
                type=PolicyType.REQUEST,
                permissions=frozenset(
                    {
                        Permission(
                            action=Action.USE,
                            constraints=frozenset({purpose(value)}),
                        )
                    }
                ),
            )

        assert not satisfies(agreement, request_for("research"))
        assert not satisfies(agreement, request_for("ds-cancer"))
        assert satisfies(agreement, request_for("ds-neurological"))

    def test_empty_composition_is_rejected(self):
        with pytest.raises(ValueError):
            compose([])


class TestTaxonomy:
    def test_subsumption_is_transitive(self):
        tax = default_taxonomy()
        assert tax.subsumes("spatial", "world", "DE")

    def test_a_term_subsumes_itself(self):
        assert default_taxonomy().subsumes("purpose", "research", "research")

    def test_unknown_terms_subsume_only_themselves(self):
        tax = default_taxonomy()
        assert tax.subsumes("purpose", "unheard-of", "unheard-of")
        assert not tax.subsumes("purpose", "unheard-of", "research")

    def test_meet_picks_the_more_specific_term(self):
        assert default_taxonomy().meet("spatial", {"EU"}, {"DE"}) == {"DE"}

    def test_join_drops_subsumed_terms(self):
        assert default_taxonomy().join("spatial", {"EU"}, {"DE"}) == {"EU"}

    def test_difference_inherits_exclusion(self):
        assert default_taxonomy().difference("spatial", {"DE"}, {"EU"}) == frozenset()

    def test_extension_adds_edges(self):
        """The hook increment 3 uses to layer DUO's hierarchy on top."""
        tax = default_taxonomy().extended(
            {"purpose": {"research": ["genomics-research"]}}
        )
        assert tax.subsumes("purpose", "research", "genomics-research")
        assert not default_taxonomy().subsumes("purpose", "research", "genomics-research")

    def test_flat_taxonomy_has_no_hierarchy(self):
        assert FLAT.meet("spatial", {"EU"}, {"DE"}) == frozenset()

    def test_taxonomy_from_dict_skips_comment_keys(self):
        tax = Taxonomy.from_dict({"_comment": {"x": ["y"]}, "purpose": {"a": ["b"]}})
        assert "_comment" not in tax.broader
        assert tax.subsumes("purpose", "a", "b")
