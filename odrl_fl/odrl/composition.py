"""Composing contributor policies into the policy of the derived model.

A federated model is a derivative of every contributor's data, so the question
"what may be done with the model?" has to be answered from all of their
policies at once. ODRL defines no semantics for this. The algebra here is the
project's central contribution:

    permissions   intersect          an action needs everyone's grant
    prohibitions  union              any single prohibition binds the derivative
    duties        union, merged      every obligation must be discharged
    constraints   conjoin            the most restrictive bound wins
    conflict      deny overrides     safe default, and the ODRL convention

Every rule pushes the same direction, which gives the property the rest of the
project rests on:

    S ⊆ S'  ⟹  Cap(S') ⊆ Cap(S)

**Capability is anti-monotone in the contributor set.** Admitting a contributor
can never widen what the model may be used for, however permissive that
contributor is -- it can only add data. Removing one can only widen it.
`tests/test_algebra_laws.py` checks this against generated policies rather than
taking it on faith.

Capability can be lost three distinct ways, and `CompositionResult.lost`
separates them because increment 4 needs to ablate their contributions:
somebody withheld the grant, somebody prohibited it, or the grantors' own
constraints do not intersect.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum

from odrl_fl.odrl.model import (
    SET_VALUED,
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
)
from odrl_fl.odrl.taxonomy import Taxonomy, default_taxonomy


class LossReason(str, Enum):
    """Why an action that somebody granted did not survive composition."""

    #: At least one contributor never granted it.
    NOT_GRANTED_BY_ALL = "not_granted_by_all"
    #: At least one contributor prohibited it outright.
    PROHIBITED = "prohibited"
    #: Everyone granted it, but their constraints have no common ground --
    #: e.g. one permits research only and another commercial only.
    CONSTRAINT_CONFLICT = "constraint_conflict"


class Unsatisfiable(Exception):
    """No assignment satisfies a conjoined constraint set."""


@dataclass(frozen=True)
class CompositionResult:
    """The derived Agreement, plus why each candidate action did not survive."""

    agreement: Policy
    lost: dict[Action, LossReason] = field(default_factory=dict)

    @property
    def capability(self) -> frozenset[Action]:
        return capability(self.agreement)

    def describe(self) -> str:
        lines = [self.agreement.describe()]
        for action in sorted(self.lost, key=lambda a: a.value):
            lines.append(f"  x {action.value} lost: {self.lost[action].value}")
        return "\n".join(lines)


# --- Ordering for interval-domain values -----------------------------------

_LOWER_OPS = (Operator.GT, Operator.GTEQ)
_UPPER_OPS = (Operator.LT, Operator.LTEQ)


def _ordinal(constraint: Constraint) -> float | date | datetime:
    """A comparable value for an interval-domain constraint."""
    value = constraint.right_operand
    if constraint.left_operand is LeftOperand.DATETIME:
        text = str(value)
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            try:
                return date.fromisoformat(text)
            except ValueError as exc:
                raise MalformedPolicy(
                    f"dateTime '{text}' is not ISO-8601"
                ) from exc
    if isinstance(value, (int, float)):
        return float(value)
    raise MalformedPolicy(
        f"'{constraint.left_operand.value}' needs an ordered value, got {value!r}"
    )


def _same_kind(a, b) -> bool:
    """Guard against comparing a date with a datetime, which raises."""
    return isinstance(a, type(b)) or isinstance(b, type(a))


# --- Conjunction -----------------------------------------------------------


def _conjoin_set_valued(
    operand: LeftOperand, constraints: Sequence[Constraint], taxonomy: Taxonomy
) -> list[Constraint]:
    """Intersect positives, union negatives, subtract the latter from the former."""
    positive: frozenset[str] | None = None
    negative: frozenset[str] = frozenset()

    for constraint in constraints:
        values = constraint.right_operand
        assert isinstance(values, frozenset)
        if constraint.operator is Operator.IS_ANY_OF:
            # Reduce to maximal elements *before* folding. A single set can
            # already be redundant -- `{research, clinical-research}` means
            # exactly `{research}`, since the first subsumes the second -- and
            # leaving it unreduced makes conjunction non-idempotent: folding
            # once returns the redundant set, folding twice returns the reduced
            # one. Found by the property test, not by inspection.
            values = taxonomy.join(operand.value, values, ())
            positive = (
                values
                if positive is None
                else taxonomy.meet(operand.value, positive, values)
            )
            if not positive:
                raise Unsatisfiable(
                    f"no shared value for '{operand.value}' across contributors"
                )
        else:  # IS_NONE_OF
            negative = taxonomy.join(operand.value, negative, values)

    if positive is not None:
        if negative:
            positive = taxonomy.difference(operand.value, positive, negative)
            if not positive:
                raise Unsatisfiable(
                    f"every permitted value of '{operand.value}' is also excluded"
                )
        return [Constraint(operand, Operator.IS_ANY_OF, positive)]

    if negative:
        return [Constraint(operand, Operator.IS_NONE_OF, negative)]
    return []


def _conjoin_interval(
    operand: LeftOperand, constraints: Sequence[Constraint]
) -> list[Constraint]:
    """Keep the tightest lower and upper bound.

    Falls out correctly for privacy budgets: `dpEpsilon lteq 1` conjoined with
    `dpEpsilon lteq 0.5` is `lteq 0.5`, the stricter requirement.
    """
    lower: Constraint | None = None
    upper: Constraint | None = None

    for constraint in constraints:
        value = _ordinal(constraint)
        if constraint.operator is Operator.EQ:
            # Pins both ends.
            for candidate, slot in ((constraint, "lower"), (constraint, "upper")):
                current = lower if slot == "lower" else upper
                if current is None:
                    if slot == "lower":
                        lower = candidate
                    else:
                        upper = candidate
            continue

        if constraint.operator in _LOWER_OPS:
            if lower is None or value > _ordinal(lower):
                lower = constraint
        elif constraint.operator in _UPPER_OPS:
            if upper is None or value < _ordinal(upper):
                upper = constraint
        else:
            raise UnsupportedConstruct(
                f"operator '{constraint.operator.value}' is not ordered and cannot "
                f"be conjoined on '{operand.value}'"
            )

    if lower is not None and upper is not None:
        low, high = _ordinal(lower), _ordinal(upper)
        if not _same_kind(low, high):
            raise MalformedPolicy(
                f"cannot compare {low!r} with {high!r} on '{operand.value}'"
            )
        strict = Operator.GT in (lower.operator,) or Operator.LT in (upper.operator,)
        if low > high or (low == high and strict):
            raise Unsatisfiable(
                f"bounds on '{operand.value}' do not overlap: {lower} and {upper}"
            )

    return [c for c in (lower, upper) if c is not None]


def conjoin(
    constraint_sets: Iterable[Iterable[Constraint]],
    taxonomy: Taxonomy | None = None,
) -> frozenset[Constraint]:
    """Conjoin constraints from several contributors into the tightest set.

    Raises `Unsatisfiable` when no assignment satisfies them all -- which is a
    capability *lost to conflict*, distinct from one that was never granted.
    """
    taxonomy = taxonomy or default_taxonomy()

    grouped: dict[LeftOperand, list[Constraint]] = {}
    for constraints in constraint_sets:
        for constraint in constraints:
            grouped.setdefault(constraint.left_operand, []).append(constraint)

    out: list[Constraint] = []
    for operand, constraints in grouped.items():
        if operand in SET_VALUED:
            out.extend(_conjoin_set_valued(operand, constraints, taxonomy))
        else:
            out.extend(_conjoin_interval(operand, constraints))
    return frozenset(out)


def merge_duties(
    duty_sets: Iterable[Iterable[Duty]], taxonomy: Taxonomy | None = None
) -> frozenset[Duty]:
    """Union duties, conjoining the constraints of duties sharing an action.

    Two `anonymize` duties demanding eps<=1 and eps<=0.5 become one demanding
    eps<=0.5: obligations accumulate, and the strictest wins.
    """
    grouped: dict[Action, list[frozenset[Constraint]]] = {}
    for duties in duty_sets:
        for duty in duties:
            grouped.setdefault(duty.action, []).append(duty.constraints)
    return frozenset(
        Duty(action=action, constraints=conjoin(sets, taxonomy))
        for action, sets in grouped.items()
    )


# --- Prohibitions ----------------------------------------------------------


def _prohibition_refinement(
    action: Action, prohibitions: Iterable[Prohibition]
) -> list[Constraint]:
    """Turn constrained prohibitions on `action` into exclusions.

    "Must not derive for methods-research" does not forbid derivation outright;
    it carves a region out of the permission. A prohibition carrying a single
    set-valued constraint becomes an `isNoneOf` on the surviving permission --
    the shape DUO's NMDS code takes, so increment 3 depends on this.

    More than one constraint would require negating a conjunctive region, and
    the negation of a conjunction is a disjunction, which this fragment cannot
    express. Rejected rather than approximated: silently dropping half of a
    prohibition would grant capability nobody allowed.
    """
    refinements: list[Constraint] = []
    for prohibition in prohibitions:
        if prohibition.action != action or not prohibition.constraints:
            continue
        if len(prohibition.constraints) > 1:
            raise UnsupportedConstruct(
                f"prohibition on '{action.value}' carries "
                f"{len(prohibition.constraints)} constraints; negating a "
                "conjunction yields a disjunction, which this fragment cannot "
                "represent"
            )
        constraint = next(iter(prohibition.constraints))
        if constraint.left_operand not in SET_VALUED:
            raise UnsupportedConstruct(
                f"prohibition on '{action.value}' constrains "
                f"'{constraint.left_operand.value}', whose negation is not a "
                "single interval"
            )
        if constraint.operator is not Operator.IS_ANY_OF:
            raise UnsupportedConstruct(
                f"prohibition on '{action.value}' uses "
                f"'{constraint.operator.value}'; only positive set membership "
                "can be negated into an exclusion"
            )
        assert isinstance(constraint.right_operand, frozenset)
        refinements.append(
            Constraint(
                constraint.left_operand, Operator.IS_NONE_OF, constraint.right_operand
            )
        )
    return refinements


# --- Composition -----------------------------------------------------------


def _agreement_uid(policies: Sequence[Policy]) -> str:
    """Deterministic identifier derived from the contributing policies.

    Stable across runs and independent of ordering, so an Agreement can be
    referenced and hashed into a provenance record.
    """
    digest = hashlib.sha256(
        "\n".join(sorted(p.uid for p in policies)).encode("utf-8")
    ).hexdigest()
    return f"urn:agreement:{digest[:16]}"


def compose(
    policies: Iterable[Policy], *, taxonomy: Taxonomy | None = None
) -> CompositionResult:
    """Fold contributor Offers into the Agreement governing the derived model."""
    contributors = sorted(policies, key=lambda p: p.uid)
    if not contributors:
        raise ValueError("composition needs at least one policy")
    taxonomy = taxonomy or default_taxonomy()

    prohibitions = frozenset().union(*(p.prohibitions for p in contributors))
    blocked = {p.action for p in prohibitions if not p.constraints}

    # Candidates are the actions somebody granted; one nobody granted was never
    # in play and is not a loss.
    candidates = frozenset().union(*(p.permitted_actions for p in contributors))
    granted_by_all = frozenset.intersection(
        *(p.permitted_actions for p in contributors)
    )

    permissions: list[Permission] = []
    lost: dict[Action, LossReason] = {}

    for action in sorted(candidates, key=lambda a: a.value):
        # Prohibition is reported ahead of a missing grant when both apply.
        # Deny overrides, so the prohibition is the binding cause: it would
        # block the action even under a universal grant. Reporting the weaker
        # reason would make prohibitions look less consequential than they are
        # in increment 4's ablation.
        if action in blocked:
            lost[action] = LossReason.PROHIBITED
            continue
        if action not in granted_by_all:
            lost[action] = LossReason.NOT_GRANTED_BY_ALL
            continue

        grants = [p.permission_for(action) for p in contributors]
        try:
            constraints = conjoin(
                [g.constraints for g in grants if g is not None]
                + [_prohibition_refinement(action, prohibitions)],
                taxonomy,
            )
            duties = merge_duties(
                (g.duties for g in grants if g is not None), taxonomy
            )
        except Unsatisfiable:
            lost[action] = LossReason.CONSTRAINT_CONFLICT
            continue

        permissions.append(
            Permission(action=action, constraints=constraints, duties=duties)
        )

    agreement = Policy(
        uid=_agreement_uid(contributors),
        type=PolicyType.AGREEMENT,
        permissions=frozenset(permissions),
        prohibitions=prohibitions,
        obligations=merge_duties((p.obligations for p in contributors), taxonomy),
        derived_from=frozenset(p.uid for p in contributors),
    )
    return CompositionResult(agreement=agreement, lost=lost)


def capability(policy: Policy) -> frozenset[Action]:
    """What a policy actually permits, after prohibitions bite.

    The coarse metric behind the erosion curve. It counts *actions*, so it
    understates erosion: a permission narrowed from
    `purpose in {research, commercial}` to `{research}` still counts as one
    capability. Constraint tightening is measured separately from the
    Agreement, which retains its constraints.
    """
    blocked = {p.action for p in policy.prohibitions if not p.constraints}
    return frozenset(policy.permitted_actions) - blocked


# --- Matching an Agreement against a Request -------------------------------


def _constraint_allows(
    allowed: Constraint | None, required: Constraint, taxonomy: Taxonomy
) -> bool:
    if allowed is None:
        return True  # unconstrained by the agreement
    if required.left_operand in SET_VALUED:
        assert isinstance(required.right_operand, frozenset)
        assert isinstance(allowed.right_operand, frozenset)
        if allowed.operator is Operator.IS_NONE_OF:
            return bool(
                taxonomy.difference(
                    required.left_operand.value,
                    required.right_operand,
                    allowed.right_operand,
                )
            )
        return bool(
            taxonomy.meet(
                required.left_operand.value,
                allowed.right_operand,
                required.right_operand,
            )
        )
    try:
        _conjoin_interval(required.left_operand, [allowed, required])
    except Unsatisfiable:
        return False
    return True


def satisfies(
    agreement: Policy, request: Policy, *, taxonomy: Taxonomy | None = None
) -> bool:
    """Does `agreement` grant everything `request` asks for?

    The federation publishes a Request describing the actions and conditions it
    needs; a contributor subset is viable only if its Agreement covers them.
    Increment 5's search uses this as its feasibility test.
    """
    taxonomy = taxonomy or default_taxonomy()
    available = capability(agreement)

    for wanted in request.permissions:
        if wanted.action not in available:
            return False
        granted = agreement.permission_for(wanted.action)
        if granted is None:
            return False
        by_operand = {c.left_operand: c for c in granted.constraints}
        for required in wanted.constraints:
            if not _constraint_allows(
                by_operand.get(required.left_operand), required, taxonomy
            ):
                return False
    return True
