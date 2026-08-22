"""What a federation owes, and which of it a system can actually enforce.

The composition algebra tells a consortium what it may do and what it owes. This
module is about the second half, and its main content is a distinction that the
phrase "enforcement hooks" tends to hide.

**Duties differ by when they are discharged, and that decides what a system can
do about them.** Three postures:

  `GATE`      discharged *before* the run, out of band, by somebody else --
              an ethics committee approving a protocol. The system cannot
              perform it, but it can refuse to start without evidence, which
              is the strongest posture available anywhere here.

  `DISPATCH`  discharged *during* training, by the client, on data the server
              never sees. The server can put the requirement in the training
              config and reject updates that come back without a matching
              claim of compliance.

  `RECORD`    discharged *after* the model exists -- publishing results,
              returning findings to the source registry, attribution,
              compensation. Nothing in a training run can perform these. The
              system can only carry them, unsatisfied and attributed, so that
              whoever holds the model knows what is outstanding.

**The uncomfortable finding.** Of the duties real GA4GH DUO annotations
actually produce, none is `DISPATCH`. Across the corpus there are exactly
three: `obtainConsent` from DUO:0000021 (IRB) is a `GATE`, and `inform` from
DUO:0000019 (PUB) and `transfer` from DUO:0000029 (RTN) are both `RECORD`. The
one duty a client could act on mid-training -- `anonymize` under a `dpEpsilon`
budget -- exists only in the hand-written profile fixtures, because DUO has no
vocabulary for it.

That is a claim about the domain rather than about this implementation, and it
cuts a particular way: for real data-use terms, the useful machinery is
provenance and accountability, not runtime interception. Which is why
increment 8 is the one that matters for these obligations.

**The attestation gap.** In the `DISPATCH` posture the server verifies that a
client *said* it complied. It cannot verify that the computation happened. A
client that reports an epsilon it never applied is undetectable from here, and
closing that needs secure aggregation or hardware attestation -- named as
future work, not attempted. Every claim this module makes about enforcement is
bounded by that sentence.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from odrl_fl.odrl import Action, Duty, LeftOperand, Operator, Policy

#: Key under which directives travel in the training ConfigRecord.
DIRECTIVE_KEY = "odrl-duties"

#: Key under which a client reports what it did about them.
COMPLIANCE_KEY = "odrl-compliance"


class DutyKind(Enum):
    """When a duty is discharged, which determines what can be done about it."""

    GATE = "gate"
    DISPATCH = "dispatch"
    RECORD = "record"


#: Classification of every duty action the vocabulary can produce.
#:
#: `obtainConsent` is the only GATE: ethics approval precedes the work, and no
#: amount of runtime machinery substitutes for it. `anonymize` is the only
#: DISPATCH, and note it arrives from the profile extension rather than from
#: DUO. Everything else is RECORD -- an obligation on the people holding the
#: model, discharged in the world rather than in the training loop.
KINDS: Mapping[Action, DutyKind] = {
    Action.OBTAIN_CONSENT: DutyKind.GATE,
    Action.ANONYMIZE: DutyKind.DISPATCH,
    Action.INFORM: DutyKind.RECORD,
    Action.TRANSFER: DutyKind.RECORD,
    Action.ATTRIBUTE: DutyKind.RECORD,
    Action.COMPENSATE: DutyKind.RECORD,
    Action.DELETE: DutyKind.RECORD,
}


def classify(duty: Duty) -> DutyKind:
    """Which posture applies to this duty.

    Unknown duty actions classify as `RECORD`. Deliberately the conservative
    default: an unrecognised obligation is carried and reported rather than
    quietly treated as satisfied, and it will never be silently skipped because
    nobody wrote a handler for it.
    """
    return KINDS.get(duty.action, DutyKind.RECORD)


def _epsilon_of(duty: Duty) -> float | None:
    """The tightest privacy budget a duty demands, if it names one."""
    budgets = [
        float(c.right_operand)
        for c in duty.constraints
        if c.left_operand is LeftOperand.DP_EPSILON
        and c.operator in (Operator.LTEQ, Operator.LT, Operator.EQ)
        and isinstance(c.right_operand, (int, float))
    ]
    return min(budgets) if budgets else None


@dataclass(frozen=True)
class Directive:
    """A duty translated into something a client can act on."""

    action: str
    epsilon: float | None = None

    def as_dict(self) -> dict:
        payload: dict = {"action": self.action}
        if self.epsilon is not None:
            payload["epsilon"] = self.epsilon
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping) -> Directive:
        epsilon = payload.get("epsilon")
        return cls(
            action=str(payload["action"]),
            epsilon=None if epsilon is None else float(epsilon),
        )


@dataclass(frozen=True)
class Obligation:
    """A duty the system cannot discharge, recorded so it is not lost."""

    action: str
    owed_by: tuple[str, ...] = ()
    detail: str = ""
    #: Always false in this increment. Present because the alternative -- simply
    #: omitting undischarged duties -- would make a ledger of outstanding
    #: obligations indistinguishable from a clean bill of health.
    discharged: bool = False


@dataclass(frozen=True)
class DutyPlan:
    """Everything the agreement demands, sorted by what can be done about it."""

    gates: tuple[Duty, ...] = ()
    directives: tuple[Directive, ...] = ()
    outstanding: tuple[Obligation, ...] = ()

    @property
    def is_trivial(self) -> bool:
        return not (self.gates or self.directives or self.outstanding)


def duties_of(agreement: Policy) -> tuple[tuple[Duty, ...], tuple[Duty, ...]]:
    """Split an agreement's duties into preconditions and obligations.

    Preconditions are permission-scoped (ODRL attaches a `duty` to a Permission
    to say the permission is void until it is met); obligations are
    policy-scoped and stand on their own. `compose` unions both, so a
    precondition contributed by a single member binds the whole consortium.
    """
    preconditions = {
        duty for permission in agreement.permissions for duty in permission.duties
    }
    return tuple(sorted(preconditions)), tuple(sorted(agreement.obligations))


def plan_duties(agreement: Policy, *, owed_by: Sequence[str] = ()) -> DutyPlan:
    """Sort an agreement's duties into the three postures."""
    preconditions, obligations = duties_of(agreement)

    gates: list[Duty] = []
    directives: list[Directive] = []
    outstanding: list[Obligation] = []

    for duty in (*preconditions, *obligations):
        kind = classify(duty)
        if kind is DutyKind.GATE:
            gates.append(duty)
        elif kind is DutyKind.DISPATCH:
            directives.append(
                Directive(action=duty.action.value, epsilon=_epsilon_of(duty))
            )
        else:
            outstanding.append(
                Obligation(
                    action=duty.action.value,
                    owed_by=tuple(owed_by),
                    detail=str(duty),
                )
            )

    return DutyPlan(
        gates=tuple(gates),
        directives=tuple(directives),
        outstanding=tuple(outstanding),
    )


class UnmetPrecondition(Exception):
    """A duty that must be discharged before the run has not been evidenced.

    Fatal by design, exactly like `NoFeasibleConsortium`. Training under an
    unmet precondition produces a model whose provenance is defective from the
    first round, and no later step can repair it.
    """


def check_gates(
    gates: Iterable[Duty], evidence: Mapping[str, str]
) -> dict[str, str]:
    """Confirm every gating duty has evidence. Raises if any does not.

    `evidence` maps a duty action to a reference -- an IRB approval number, a
    DPIA identifier. Deliberately thin: nothing here validates that the
    reference is real, and it is a record of who asserted what rather than a
    verification. The gate's value is that the assertion must be made
    explicitly, by a person, before the run proceeds.
    """
    satisfied: dict[str, str] = {}
    missing: list[str] = []

    for duty in gates:
        reference = evidence.get(duty.action.value, "").strip()
        if reference:
            satisfied[duty.action.value] = reference
        else:
            missing.append(duty.action.value)

    if missing:
        raise UnmetPrecondition(
            "the agreement requires "
            + ", ".join(sorted(missing))
            + " before any training may happen, and no evidence was supplied."
            + ' Set duty-evidence, e.g. "obtainConsent=IRB-2026-0142".'
        )
    return satisfied


def encode_directives(directives: Iterable[Directive]) -> str:
    """Serialise directives for the training ConfigRecord."""
    return json.dumps([d.as_dict() for d in directives], sort_keys=True)


def decode_directives(blob: str | None) -> tuple[Directive, ...]:
    """Read directives a server sent. An unreadable payload yields none.

    A client that cannot parse its instructions must not proceed as though
    there were none, so callers pair this with `report_compliance`: no claim of
    compliance means the update is rejected server-side.
    """
    if not blob:
        return ()
    try:
        payload = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return ()
    if not isinstance(payload, list):
        return ()
    out: list[Directive] = []
    for item in payload:
        try:
            out.append(Directive.from_dict(item))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(out)


@dataclass(frozen=True)
class Compliance:
    """A client's claim about what it did. A claim, not a proof."""

    applied: Mapping[str, float | None] = field(default_factory=dict)

    def encode(self) -> str:
        return json.dumps(
            {k: v for k, v in sorted(self.applied.items())}, sort_keys=True
        )

    @classmethod
    def decode(cls, blob: str | None) -> Compliance:
        if not blob:
            return cls()
        try:
            payload = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            return cls()
        if not isinstance(payload, dict):
            return cls()
        return cls(applied={str(k): v for k, v in payload.items()})


def verify(
    directives: Sequence[Directive], claim: Compliance
) -> tuple[bool, str]:
    """Does a client's claim cover every directive it was given?

    Returns `(ok, reason)`. Silence fails: a reply that reports nothing is
    treated as non-compliant rather than as nothing-to-report, because the two
    are indistinguishable on the wire and only one of them is safe.

    What this checks is arithmetic on numbers the client chose to send. See the
    attestation gap in the module docstring.
    """
    for directive in directives:
        if directive.action not in claim.applied:
            return False, f"did not report on the {directive.action} duty"
        if directive.epsilon is None:
            continue
        reported = claim.applied[directive.action]
        if reported is None:
            return False, f"reported no budget for {directive.action}"
        if float(reported) > directive.epsilon:
            return (
                False,
                (
                    f"{directive.action} budget {reported} exceeds "
                    f"the required {directive.epsilon}"
                ),
            )
    return True, "compliant"


@dataclass
class Ledger:
    """What is outstanding, and what happened during the run.

    Exists to be handed to increment 8 and attached to the exported model. A
    duty this system cannot discharge must never end up looking discharged, so
    the ledger records obligations as open and says who owes them.
    """

    outstanding: tuple[Obligation, ...] = ()
    gates_cleared: Mapping[str, str] = field(default_factory=dict)
    rejections: list[tuple[int, int, str]] = field(default_factory=list)
    #: Members that stopped training partway: (round, node, reason, detail).
    #: Their data stays in the model, so this is provenance rather than
    #: bookkeeping -- see `odrl_fl.lifecycle`.
    departures: list[tuple[int, int, str, str]] = field(default_factory=list)

    def reject(self, server_round: int, node_id: int, reason: str) -> None:
        self.rejections.append((server_round, node_id, reason))

    def depart(
        self, server_round: int, node_id: int, reason: str, detail: str = ""
    ) -> None:
        self.departures.append((server_round, node_id, reason, detail))

    @property
    def rejected_nodes(self) -> frozenset[int]:
        return frozenset(node for _, node, _ in self.rejections)

    @property
    def departed_nodes(self) -> frozenset[int]:
        return frozenset(node for _, node, _, _ in self.departures)

    def summary(self) -> list[str]:
        lines = []
        for action, reference in sorted(self.gates_cleared.items()):
            lines.append(f"{action}: cleared, evidence {reference}")
        for obligation in self.outstanding:
            owed = ", ".join(obligation.owed_by) or "the consortium"
            lines.append(f"{obligation.action}: OUTSTANDING, owed by {owed}")
        for server_round, node_id, reason in self.rejections:
            lines.append(f"round {server_round}: dropped node {node_id} -- {reason}")
        for server_round, node_id, reason, detail in self.departures:
            lines.append(
                f"round {server_round}: node {node_id} left ({reason}) -- {detail}; "
                "its data remains in the model"
            )
        return lines
