"""Membership over time: consent expires, is withdrawn, or is breached.

Every increment so far treated a federation as a decision made once. Real
consent is not like that. A time limit lapses mid-run, an institution withdraws,
a member stops evidencing a duty it agreed to. The federation has to notice, and
the response settled earlier in the project is the conservative one: **exclude
from future rounds**. Undoing a past round is machine unlearning, which is a
different research problem.

That conservatism creates the trap this module exists to avoid.

**Departure widens the remaining consortium's rights.** Capability is
anti-monotone -- `S ⊆ S' ⟹ Cap(S') ⊆ Cap(S)` -- so removing a contributor can
only *widen* what the rest may do. The restrictive member who leaves takes their
restriction with them. Recomposing over who is currently active therefore
produces a broader agreement than the one the run started under, and every step
of that reasoning is individually correct.

It is still wrong, because **the model already contains the departed member's
data**. Rounds one to k were trained on it, and no later round removes it. An
agreement composed over current membership would license uses of that data which
its owner never granted -- arrived at by correct reasoning from a premise that
quietly stopped holding.

So the licence follows the **contribution set**, not the active set:

    active        who trains in the next round; may shrink
    contributed   whose data is in the weights; only ever grows

`governing_agreement` composes over `contributed`. `active_agreement` composes
over `active` and exists only so the gap between them can be measured and
reported -- it is never the model's licence. Measured on the corpus, that gap is
real: at one seed a single departure takes purpose retention from 0.000 to
0.333, so the naive recomposition would license an entire purpose subtree that
nobody granted for data already in the model.

A holder of the model needs to know this happened, so departures are recorded
and travel in the bundle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from enum import Enum

from odrl_fl.odrl import (
    Action,
    Constraint,
    LeftOperand,
    Operator,
    Permission,
    Policy,
    capability,
    compose,
)


class Departure(Enum):
    """Why a member stopped training. Recorded, never inferred silently."""

    #: Re-declared terms that no longer satisfy what the federation needs.
    WITHDREW = "withdrew"
    #: A temporal constraint that held at the start no longer holds.
    EXPIRED = "expired"
    #: Stopped answering.
    UNREACHABLE = "unreachable"
    #: Repeatedly failed to evidence a duty it was dispatched.
    DUTY_BREACH = "duty_breach"


@dataclass(frozen=True)
class DepartureRecord:
    node_id: int
    server_round: int
    reason: Departure
    detail: str = ""


def _as_instant(value) -> date | datetime | None:
    """Parse a bound, keeping day-granular bounds as dates.

    `date.fromisoformat` is tried **first**, deliberately.
    `datetime.fromisoformat("2028-12-31")` also succeeds and yields midnight at
    the *start* of that day, which would make a time limit of 2028-12-31 expire
    before 2028-12-31 had happened. DUO's TS and MOR bounds are day-granular, so
    a date has to stay a date.
    """
    if isinstance(value, (date, datetime)):
        return value
    text = str(value)
    for parse in (date.fromisoformat, datetime.fromisoformat):
        try:
            return parse(text)
        except ValueError:
            continue
    return None


def _comparable(left, right) -> tuple:
    """Line two instants up so they can be compared without raising.

    Two mismatches are possible and both are ordinary rather than exceptional: a
    date against a datetime, and an aware datetime against a naive one. Mixing
    either raises `TypeError` in Python, which would surface as a crash midway
    through a training round.
    """
    left_dt, right_dt = isinstance(left, datetime), isinstance(right, datetime)

    # A day-granular bound wins: comparing it against a precise instant is only
    # meaningful at the coarser resolution.
    if left_dt and not right_dt:
        return left.date(), right
    if right_dt and not left_dt:
        return left, right.date()

    # Both datetimes, but one carries a timezone and the other does not. Policy
    # bounds are written without one, so a naive bound is read as UTC.
    if left_dt and right_dt and (left.tzinfo is None) != (right.tzinfo is None):
        left = left.replace(tzinfo=UTC) if left.tzinfo is None else left
        right = right.replace(tzinfo=UTC) if right.tzinfo is None else right

    return left, right


def constraint_holds_at(constraint: Constraint, at: date | datetime) -> bool:
    """Does a temporal constraint hold at this instant?

    Only `dateTime` is evaluated. Every other operand is left alone and reported
    as holding, because this function answers "has time invalidated it?" and
    nothing else -- a purpose or spatial constraint is not something an instant
    can settle.
    """
    if constraint.left_operand is not LeftOperand.DATETIME:
        return True

    bound = _as_instant(constraint.right_operand)
    if bound is None:
        return True  # unparseable bounds are somebody else's error to raise

    now, bound = _comparable(at, bound)
    match constraint.operator:
        case Operator.LTEQ:
            return now <= bound
        case Operator.LT:
            return now < bound
        case Operator.GTEQ:
            return now >= bound
        case Operator.GT:
            return now > bound
        case Operator.EQ:
            return now == bound
        case Operator.NEQ:
            return now != bound
    return True


def in_force(policy: Policy, at: date | datetime) -> Policy:
    """The policy as it stands at a given instant.

    Permissions whose temporal constraints have lapsed are dropped -- both a
    time limit that has passed and a moratorium that has not yet lifted. The
    permission disappears rather than being marked expired, so everything
    downstream that reasons about capability keeps working unchanged.
    """
    surviving = frozenset(
        permission
        for permission in policy.permissions
        if all(constraint_holds_at(c, at) for c in permission.constraints)
    )
    if surviving == policy.permissions:
        return policy
    return replace(policy, permissions=surviving)


def capability_at(policy: Policy, at: date | datetime) -> frozenset[Action]:
    return capability(in_force(policy, at))


def expires_between(
    policy: Policy, start: date | datetime, end: date | datetime
) -> bool:
    """Would this policy's capability change between two instants?"""
    return capability_at(policy, start) != capability_at(policy, end)


@dataclass(frozen=True)
class Membership:
    """Who trains next, and whose data is already in the weights.

    The distinction is the whole point. `active` may shrink as members depart;
    `contributed` only grows, because a round that has happened cannot be
    unhappened.
    """

    active: tuple[int, ...]
    contributed: tuple[int, ...]
    departures: tuple[DepartureRecord, ...] = ()

    @classmethod
    def starting(cls, admitted: Sequence[int]) -> Membership:
        members = tuple(dict.fromkeys(admitted))
        return cls(active=members, contributed=members)

    def depart(
        self, node_id: int, server_round: int, reason: Departure, detail: str = ""
    ) -> Membership:
        """Remove a member from future rounds, keeping them in the record.

        They stay in `contributed` deliberately. Their data is in the model and
        removing them from the provenance would make the licence look broader
        than the model's history supports.
        """
        if node_id not in self.active:
            return self
        return Membership(
            active=tuple(n for n in self.active if n != node_id),
            contributed=self.contributed,
            departures=(
                *self.departures,
                DepartureRecord(node_id, server_round, reason, detail),
            ),
        )

    def admit(self, node_id: int) -> Membership:
        """Add a member. Joining contributes data from the next round on."""
        if node_id in self.active:
            return self
        return Membership(
            active=(*self.active, node_id),
            contributed=tuple(dict.fromkeys((*self.contributed, node_id))),
            departures=self.departures,
        )

    @property
    def departed(self) -> frozenset[int]:
        return frozenset(self.contributed) - frozenset(self.active)

    @property
    def is_empty(self) -> bool:
        return not self.active


class ConsortiumCollapsed(Exception):
    """Departures have left nobody able to train.

    Fatal, like an infeasible consortium. Continuing with an empty federation
    would produce nothing, and continuing with members who no longer consent
    would produce something worse.
    """


def governing_agreement(
    policies: Mapping[int, Policy], membership: Membership, *, taxonomy=None
) -> Policy:
    """The model's licence: composed over everyone whose data is in it.

    **Not** over the active set. See the module docstring -- recomposing over
    current membership widens the agreement every time a restrictive member
    leaves, while their data stays in the weights.
    """
    contributors = [policies[n] for n in membership.contributed if n in policies]
    if not contributors:
        raise ConsortiumCollapsed("no contributor policies to compose")
    return compose(contributors, taxonomy=taxonomy).agreement


def active_agreement(
    policies: Mapping[int, Policy], membership: Membership, *, taxonomy=None
) -> Policy:
    """What the *remaining* members alone would permit.

    Provided for measurement and reporting only. Using it as the model's licence
    is precisely the error this module exists to prevent, which is why it is a
    separate function with this docstring rather than a flag on the one above.
    """
    remaining = [policies[n] for n in membership.active if n in policies]
    if not remaining:
        raise ConsortiumCollapsed("no active members remain")
    return compose(remaining, taxonomy=taxonomy).agreement


@dataclass(frozen=True)
class Reevaluation:
    """The outcome of checking membership at the top of a round."""

    membership: Membership
    departed: tuple[DepartureRecord, ...] = ()
    #: Actions the active set alone would now permit but the model may not use,
    #: because the departed members' data is still in the weights.
    widened: frozenset[Action] = field(default_factory=frozenset)

    @property
    def changed(self) -> bool:
        return bool(self.departed)


def reevaluate(
    membership: Membership,
    policies: Mapping[int, Policy],
    *,
    server_round: int,
    at: date | datetime,
    request: Policy | None = None,
    unreachable: Sequence[int] = (),
    breaches: Mapping[int, int] | None = None,
    breach_limit: int = 2,
    taxonomy=None,
) -> Reevaluation:
    """Check every active member and drop the ones that no longer qualify.

    Pure: no Grid, no clock of its own -- `at` is passed in so a test can place
    the federation on either side of an expiry without waiting for one.

    Four ways to leave, checked in order of how definite they are: a member that
    did not answer, one whose terms have lapsed, one whose terms no longer meet
    the request, and one that has failed its duties too often.
    """
    from odrl_fl.odrl import satisfies

    breaches = breaches or {}
    updated = membership
    records: list[DepartureRecord] = []

    for node_id in membership.active:
        policy = policies.get(node_id)

        if node_id in unreachable:
            reason, detail = Departure.UNREACHABLE, "stopped answering"
        elif policy is None:
            reason, detail = Departure.WITHDREW, "no longer declares a policy"
        elif not capability(policy):
            # Grants nothing at any time. That is a withdrawal, not a lapse --
            # worth separating, because one is a decision somebody took and the
            # other is a clock running out.
            reason, detail = Departure.WITHDREW, "revoked every permission"
        elif not capability_at(policy, at):
            reason, detail = Departure.EXPIRED, f"terms do not permit anything at {at}"
        elif request is not None and not satisfies(
            in_force(policy, at), request, taxonomy=taxonomy
        ):
            reason, detail = (
                Departure.WITHDREW,
                "re-declared terms no longer satisfy the request",
            )
        elif breaches.get(node_id, 0) >= breach_limit:
            reason, detail = (
                Departure.DUTY_BREACH,
                f"failed to evidence a dispatched duty {breaches[node_id]} times",
            )
        else:
            continue

        updated = updated.depart(node_id, server_round, reason, detail)
        records.append(DepartureRecord(node_id, server_round, reason, detail))

    if updated.is_empty:
        raise ConsortiumCollapsed(
            f"round {server_round}: every member has departed "
            f"({', '.join(r.reason.value for r in records)})"
        )

    # The gap the module exists to close: what the remaining members alone would
    # now permit, over and above what the model's contribution set allows.
    widened: frozenset[Action] = frozenset()
    if records and policies:
        governing = capability(governing_agreement(policies, updated, taxonomy=taxonomy))
        try:
            remaining = capability(
                active_agreement(policies, updated, taxonomy=taxonomy)
            )
        except ConsortiumCollapsed:
            remaining = frozenset()
        widened = remaining - governing

    return Reevaluation(membership=updated, departed=tuple(records), widened=widened)


def widened_purposes(
    policies: Mapping[int, Policy], membership: Membership, *, taxonomy=None
) -> tuple[frozenset[str], frozenset[str]]:
    """Purposes the active set would allow, against those the model may claim.

    Action-level capability is nearly binary in this corpus, so the widening a
    departure causes shows up in purpose breadth long before it shows up in the
    action set. Returns `(governing, active_only)`.
    """

    def purposes_of(policy: Policy) -> frozenset[str]:
        values: set[str] = set()
        for permission in policy.permissions:
            for constraint in permission.constraints:
                if (
                    constraint.left_operand is LeftOperand.PURPOSE
                    and constraint.operator is Operator.IS_ANY_OF
                ):
                    values |= set(constraint.right_operand)
        return frozenset(values)

    governing = purposes_of(governing_agreement(policies, membership, taxonomy=taxonomy))
    try:
        active = purposes_of(active_agreement(policies, membership, taxonomy=taxonomy))
    except ConsortiumCollapsed:
        active = frozenset()
    return governing, active - governing


def withdraw(policy: Policy, *, actions: Sequence[Action] = ()) -> Policy:
    """A policy with `actions` revoked -- what a re-declaration after a
    withdrawal looks like. Revoking everything models full withdrawal."""
    revoked = frozenset(actions) or frozenset(a.action for a in policy.permissions)
    return replace(
        policy,
        permissions=frozenset(
            p for p in policy.permissions if p.action not in revoked
        ),
    )


def expiring(policy: Policy, when: str, *, actions: Sequence[Action] = ()) -> Policy:
    """A policy whose named actions lapse after `when`. For tests and demos."""
    limit = Constraint(LeftOperand.DATETIME, Operator.LTEQ, when)
    targets = frozenset(actions) or frozenset(p.action for p in policy.permissions)
    return replace(
        policy,
        permissions=frozenset(
            Permission(
                action=p.action,
                constraints=(
                    p.constraints | {limit} if p.action in targets else p.constraints
                ),
                duties=p.duties,
            )
            for p in policy.permissions
        ),
    )
