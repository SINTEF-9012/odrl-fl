"""Typed model for the ODRL subset this project composes over.

**The subset is deliberately small, and everything outside it is rejected
loudly rather than ignored.** Silently dropping a constraint would compute a
capability the contributor never granted -- the one failure mode that would
invalidate every downstream result while still producing plausible numbers.

What is in, and why it composes cleanly:

| Left operand | Domain      | Composition |
|--------------|-------------|-------------|
| `purpose`    | set of terms| set intersection (with subsumption) |
| `spatial`    | set of regions | set intersection (with containment) |
| `recipient`  | set of parties | set intersection |
| `dateTime`   | interval    | interval intersection |

What is out, and why:

- **Stateful constraints** -- `count`, `elapsedTime`, `meteredTime`, `event`,
  `payAmount`, positional operands. These accumulate over time, so they have no
  meaningful composition across parties: if A permits 20 uses and B permits 30,
  the derived model's budget is neither 20, nor 30, nor 50. Deferred until the
  algebra for the static case is settled.
- **Self-referential constraints** -- a minimum cohort size ("aggregate my data
  only with at least 5 others") is not stateful, but its satisfaction depends on
  the very contributor set being composed, which makes composition a fixpoint
  rather than a fold. Interesting, and deferred for the same reason.
- **Logical constraint containers** -- `and`, `or`, `xone`. Constraints on a
  rule are already conjunctive; disjunction would require a normal form before
  composition. Deferred.

Types are frozen and hashable throughout: composition builds new policies from
old, property-based tests generate them by the thousand, and set operations
over rules are the natural way to express intersection and union.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ODRLError(Exception):
    """Base for anything wrong with a policy."""


class UnsupportedConstruct(ODRLError):
    """A valid ODRL construct that this subset deliberately excludes.

    Distinct from `MalformedPolicy`: the input is legitimate ODRL, but outside
    the fragment whose composition semantics are defined here.
    """


class MalformedPolicy(ODRLError):
    """The document is not well-formed ODRL."""


class PolicyType(str, Enum):
    """ODRL policy subtypes.

    The lifecycle this project cares about: a contributor publishes an `OFFER`,
    the federation issues a `REQUEST`, and matching them produces an
    `AGREEMENT`. That is the Data Space negotiation pattern, so the vocabulary
    is already the right one.
    """

    SET = "Set"
    OFFER = "Offer"
    REQUEST = "Request"
    AGREEMENT = "Agreement"


class Action(str, Enum):
    """Actions, chosen for how they map onto the FL lifecycle.

    `aggregate` and `derive` are the load-bearing ones: a federated model *is*
    an aggregation, and fine-tuning it downstream *is* derivation. ODRL having
    these in its core vocabulary is why it fits this problem better than a
    generic access-control language.
    """

    USE = "use"
    AGGREGATE = "aggregate"
    DERIVE = "derive"
    DISTRIBUTE = "distribute"
    TRANSFER = "transfer"

    #: Actions that appear as *duties* rather than permissions. All ODRL core
    #: terms, and between them they cover the obligations that dominate medical
    #: data sharing: credit the source, report back, pay, protect, dispose.
    ANONYMIZE = "anonymize"
    ATTRIBUTE = "attribute"
    INFORM = "inform"
    COMPENSATE = "compensate"
    DELETE = "delete"


class LeftOperand(str, Enum):
    """Constraint left operands in the supported subset."""

    PURPOSE = "purpose"
    SPATIAL = "spatial"
    RECIPIENT = "recipient"
    DATETIME = "dateTime"
    #: Profile extension, not ODRL core. Carries the privacy budget for an
    #: `anonymize` duty ("you may use my data if you apply DP with eps <= 1"),
    #: which is the most common quantified obligation in medical data sharing.
    DP_EPSILON = "dpEpsilon"


class Operator(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    LT = "lt"
    LTEQ = "lteq"
    GT = "gt"
    GTEQ = "gteq"
    IS_ANY_OF = "isAnyOf"
    IS_NONE_OF = "isNoneOf"


#: Recognised ODRL left operands that this subset rejects, kept explicit so the
#: error can say *why* rather than "unknown term".
STATEFUL_OPERANDS = frozenset(
    {
        "count",
        "elapsedTime",
        "meteredTime",
        "timeInterval",
        "event",
        "payAmount",
        "absolutePosition",
        "relativePosition",
        "absoluteSize",
        "relativeSize",
        "deliveryChannel",
    }
)

#: Operands whose satisfaction depends on the contributor set being composed.
SELF_REFERENTIAL_OPERANDS = frozenset({"cohortSize", "recipientCount", "partyCount"})

#: Left operands whose values form a *set*, composed by intersection. The rest
#: (`dateTime`, `dpEpsilon`) are scalars ordered by comparison operators.
SET_VALUED = frozenset(
    {LeftOperand.PURPOSE, LeftOperand.SPATIAL, LeftOperand.RECIPIENT}
)


@dataclass(frozen=True, order=True)
class Constraint:
    """A single `leftOperand operator rightOperand` triple.

    Stored in a canonical form so that composition compares like with like:
    for set-valued operands `eq x` is rewritten to `isAnyOf {x}` and `neq x` to
    `isNoneOf {x}`. Without that, two policies expressing the same restriction
    differently would fail to intersect.
    """

    left_operand: LeftOperand
    operator: Operator
    right_operand: str | float | frozenset[str]

    def __post_init__(self) -> None:
        canonical_op, canonical_value = self._canonicalise()
        object.__setattr__(self, "operator", canonical_op)
        object.__setattr__(self, "right_operand", canonical_value)
        self._validate()

    def _canonicalise(self) -> tuple[Operator, str | float | frozenset[str]]:
        value = self.right_operand
        if self.left_operand not in SET_VALUED:
            return self.operator, value

        if isinstance(value, (list, tuple, set, frozenset)):
            value = frozenset(str(v) for v in value)
        else:
            value = frozenset({str(value)})

        if self.operator in (Operator.EQ, Operator.IS_ANY_OF):
            return Operator.IS_ANY_OF, value
        if self.operator in (Operator.NEQ, Operator.IS_NONE_OF):
            return Operator.IS_NONE_OF, value
        raise UnsupportedConstruct(
            f"operator '{self.operator.value}' is not defined for the set-valued "
            f"operand '{self.left_operand.value}'; use eq/neq/isAnyOf/isNoneOf"
        )

    def _validate(self) -> None:
        if self.left_operand in SET_VALUED:
            if not isinstance(self.right_operand, frozenset) or not self.right_operand:
                raise MalformedPolicy(
                    f"'{self.left_operand.value}' needs a non-empty set of values"
                )
            return

        # Checked before the value type, so that using a set operator on a
        # scalar operand reports *that* rather than the type mismatch it causes.
        if self.operator in (Operator.IS_ANY_OF, Operator.IS_NONE_OF):
            raise UnsupportedConstruct(
                f"'{self.operator.value}' is only defined for set-valued operands "
                f"({', '.join(sorted(o.value for o in SET_VALUED))}), "
                f"not '{self.left_operand.value}'"
            )

        if self.left_operand is LeftOperand.DATETIME and not isinstance(
            self.right_operand, str
        ):
            raise MalformedPolicy("dateTime needs an ISO-8601 string")
        if self.left_operand is LeftOperand.DP_EPSILON and not isinstance(
            self.right_operand, (int, float)
        ):
            raise MalformedPolicy("dpEpsilon needs a number")

    @property
    def is_set_valued(self) -> bool:
        return self.left_operand in SET_VALUED

    def __str__(self) -> str:
        if isinstance(self.right_operand, frozenset):
            value = "{" + ", ".join(sorted(self.right_operand)) + "}"
        else:
            value = str(self.right_operand)
        return f"{self.left_operand.value} {self.operator.value} {value}"


@dataclass(frozen=True, order=True)
class Duty:
    """An obligation.

    As a `Permission.duties` entry it is a *pre-duty*: it must be discharged to
    exercise the permission at all. That distinction is what gives duties an
    enforcement point in FL -- dispatched with the training config, verified
    before an update is admitted.
    """

    action: Action
    constraints: frozenset[Constraint] = field(default_factory=frozenset)

    def __str__(self) -> str:
        if not self.constraints:
            return self.action.value
        inner = " and ".join(str(c) for c in sorted(self.constraints))
        return f"{self.action.value} [{inner}]"


@dataclass(frozen=True, order=True)
class Permission:
    """An action a party may perform, subject to constraints and pre-duties."""

    action: Action
    constraints: frozenset[Constraint] = field(default_factory=frozenset)
    duties: frozenset[Duty] = field(default_factory=frozenset)

    def __str__(self) -> str:
        parts = [f"may {self.action.value}"]
        if self.constraints:
            parts.append("if " + " and ".join(str(c) for c in sorted(self.constraints)))
        if self.duties:
            parts.append("provided " + " and ".join(str(d) for d in sorted(self.duties)))
        return " ".join(parts)


@dataclass(frozen=True, order=True)
class Prohibition:
    """An action a party must not perform."""

    action: Action
    constraints: frozenset[Constraint] = field(default_factory=frozenset)

    def __str__(self) -> str:
        text = f"must not {self.action.value}"
        if self.constraints:
            text += " if " + " and ".join(str(c) for c in sorted(self.constraints))
        return text


@dataclass(frozen=True)
class Policy:
    """A complete policy.

    Rules are held as frozensets: composition is expressed as intersection and
    union over them, and ordering carries no meaning in ODRL.
    """

    uid: str
    type: PolicyType = PolicyType.OFFER
    assigner: str | None = None
    assignee: str | None = None
    target: str | None = None
    permissions: frozenset[Permission] = field(default_factory=frozenset)
    prohibitions: frozenset[Prohibition] = field(default_factory=frozenset)
    obligations: frozenset[Duty] = field(default_factory=frozenset)

    @property
    def permitted_actions(self) -> frozenset[Action]:
        return frozenset(p.action for p in self.permissions)

    @property
    def prohibited_actions(self) -> frozenset[Action]:
        return frozenset(p.action for p in self.prohibitions)

    def permission_for(self, action: Action) -> Permission | None:
        return next((p for p in self.permissions if p.action == action), None)

    def describe(self) -> str:
        """Human-readable rendering, for worked examples and debugging."""
        lines = [f"{self.type.value} {self.uid}"]
        if self.assigner:
            lines.append(f"  assigner: {self.assigner}")
        for permission in sorted(self.permissions):
            lines.append(f"  + {permission}")
        for prohibition in sorted(self.prohibitions):
            lines.append(f"  - {prohibition}")
        for duty in sorted(self.obligations):
            lines.append(f"  ! must {duty}")
        return "\n".join(lines)
