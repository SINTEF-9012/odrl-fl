"""Hypothesis strategies for generating ODRL policies.

Shared by the property tests. Generated policies stay inside the supported
fragment by construction -- the point is to explore the algebra's behaviour,
not the parser's rejection paths, which `test_odrl.py` covers directly.
"""

from __future__ import annotations

from hypothesis import strategies as st

from odrl_fl.odrl import (
    Action,
    Constraint,
    Duty,
    LeftOperand,
    Operator,
    Permission,
    Policy,
    PolicyType,
    Prohibition,
)

#: Deliberately small vocabularies. Composition is interesting when value sets
#: overlap partially; drawing from a large pool would make almost every
#: intersection empty and almost every generated case degenerate.
PURPOSES = ["research", "clinical-research", "commercial", "education"]
REGIONS = ["world", "EU", "DE", "FR", "US"]
RECIPIENTS = ["consortium", "public", "third-party"]
DATES = ["2026-01-01", "2027-06-30", "2028-12-31"]

PERMISSION_ACTIONS = [
    Action.USE,
    Action.AGGREGATE,
    Action.DERIVE,
    Action.DISTRIBUTE,
]
DUTY_ACTIONS = [Action.ATTRIBUTE, Action.INFORM, Action.COMPENSATE, Action.ANONYMIZE]


@st.composite
def set_constraints(draw, operand: LeftOperand, values: list[str]):
    chosen = draw(st.lists(st.sampled_from(values), min_size=1, max_size=3, unique=True))
    operator = draw(st.sampled_from([Operator.IS_ANY_OF, Operator.IS_NONE_OF]))
    return Constraint(operand, operator, frozenset(chosen))


@st.composite
def datetime_constraints(draw):
    return Constraint(
        LeftOperand.DATETIME,
        draw(st.sampled_from([Operator.LTEQ, Operator.GTEQ])),
        draw(st.sampled_from(DATES)),
    )


@st.composite
def epsilon_constraints(draw):
    return Constraint(
        LeftOperand.DP_EPSILON,
        Operator.LTEQ,
        draw(st.sampled_from([0.1, 0.5, 1.0, 2.0])),
    )


def constraints():
    return st.one_of(
        set_constraints(LeftOperand.PURPOSE, PURPOSES),
        set_constraints(LeftOperand.SPATIAL, REGIONS),
        set_constraints(LeftOperand.RECIPIENT, RECIPIENTS),
        datetime_constraints(),
    )


@st.composite
def constraint_sets(draw, max_size: int = 3):
    drawn = draw(st.lists(constraints(), max_size=max_size))
    # At most one constraint per operand per rule: ODRL allows several, but a
    # single rule repeating an operand is already a conjunction, and the
    # generator would mostly produce self-contradictory rules.
    by_operand: dict[LeftOperand, Constraint] = {}
    for constraint in drawn:
        by_operand.setdefault(constraint.left_operand, constraint)
    return frozenset(by_operand.values())


@st.composite
def duties(draw):
    action = draw(st.sampled_from(DUTY_ACTIONS))
    if action is Action.ANONYMIZE and draw(st.booleans()):
        return Duty(action=action, constraints=frozenset({draw(epsilon_constraints())}))
    return Duty(action=action)


@st.composite
def permissions(draw, action: Action | None = None):
    return Permission(
        action=action or draw(st.sampled_from(PERMISSION_ACTIONS)),
        constraints=draw(constraint_sets()),
        duties=frozenset(draw(st.lists(duties(), max_size=2))),
    )


@st.composite
def prohibitions(draw):
    # Only unconstrained prohibitions are generated. Constrained ones are
    # supported for a narrow shape (one positive set-valued constraint) and are
    # exercised directly in test_composition.py; generating the unsupported
    # shapes would just produce expected exceptions.
    return Prohibition(action=draw(st.sampled_from(PERMISSION_ACTIONS)))


@st.composite
def policies(draw, uid: str | None = None):
    actions = draw(
        st.lists(st.sampled_from(PERMISSION_ACTIONS), min_size=1, max_size=4, unique=True)
    )
    perms = frozenset(draw(permissions(action=a)) for a in actions)
    prohibs = frozenset(draw(st.lists(prohibitions(), max_size=1)))
    return Policy(
        uid=uid or f"urn:policy:{draw(st.integers(min_value=0, max_value=10**6))}",
        type=PolicyType.OFFER,
        permissions=perms,
        prohibitions=prohibs,
        obligations=frozenset(draw(st.lists(duties(), max_size=2))),
    )


@st.composite
def policy_sets(draw, min_size: int = 1, max_size: int = 4):
    count = draw(st.integers(min_value=min_size, max_value=max_size))
    return [draw(policies(uid=f"urn:policy:{i}")) for i in range(count)]
