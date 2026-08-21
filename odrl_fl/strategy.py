"""Policy-aware federated learning: the federation selects its own members.

Everything up to increment 5 was an offline study. This module is where the
result becomes a system: before any training happens, the server asks every
connected node what it is permitted to do, composes the answers, searches for a
consortium whose combined terms actually grant what the federation needs, and
then trains **only on the admitted nodes**.

Stock `FedAvg` samples participants at random from the Grid. That is the
behaviour being replaced -- not because random sampling is bad statistics, but
because it takes no position on whether the resulting model may lawfully exist.

**Policies come from the contributors.** The server could look them up in the
same corpus the offline studies use, and that would be simpler, but a server
that invents its clients' consent terms demonstrates nothing. Each node
translates its own DUO codes locally and declares the result; the server only
ever sees what it was told.

Three pieces, deliberately separated:

  `Announcement` / `parse_announcement`   the wire format
  `admit`                                 the decision -- a pure function
  `PolicyAwareFedAvg`                     the strategy that honours it

The decision is pure so it can be tested without a federation, which matters
here: Flower's simulation backend needs Ray, and Ray ships no Windows wheel for
Python 3.13, so no test on this machine can start one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from logging import INFO

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
)
from flwr.common.logger import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from odrl_fl.duties import (
    COMPLIANCE_KEY,
    DIRECTIVE_KEY,
    Compliance,
    Directive,
    DutyPlan,
    Ledger,
    encode_directives,
    plan_duties,
    verify,
)
from odrl_fl.odrl import (
    Action,
    ODRLError,
    Policy,
    compose,
    parse_policy,
    policy_to_jsonld,
)
from odrl_fl.selection import SelectionResult, UtilityFn, search

#: Keys in the declaration record. Defined once and used by both ends, so the
#: wire format has a single definition rather than two that drift apart.
POLICY_KEY = "odrl-policy"
CODES_KEY = "duo-codes"
SAMPLES_KEY = "num-examples"
LABELS_KEY = "label-counts"

#: Group id for the declaration round, so it is distinguishable from training
#: rounds in Flower's logs.
DECLARATION_GROUP = "policy-declaration"


class DeclarationError(Exception):
    """A node's declaration could not be understood."""


@dataclass(frozen=True)
class Announcement:
    """What one contributor tells the server about itself.

    Satisfies `selection.PolicyBearing`, so it can be fed to the search
    directly without being converted into a corpus `Contributor` first.
    """

    node_id: int
    policy: Policy
    n_train: int = 0
    label_counts: Mapping[int, int] = field(default_factory=dict)
    codes: tuple[str, ...] = ()

    @property
    def classes(self) -> frozenset[int]:
        return frozenset(self.label_counts)


def declaration(
    policy: Policy,
    *,
    n_train: int = 0,
    label_counts: Mapping[int, int] | None = None,
    codes: Sequence[str] = (),
) -> ConfigRecord:
    """Build the record a client sends back when asked to declare itself.

    Volume and class histogram ride along because the server has to score
    candidate consortia and cannot see anyone's data. **This is a disclosure**:
    a label histogram tells the server something about a contributor's
    population, and an institution unwilling to reveal it would need a coarser
    utility signal than the one used here. Flagged rather than solved.
    """
    return ConfigRecord(
        {
            POLICY_KEY: json.dumps(policy_to_jsonld(policy)),
            LABELS_KEY: json.dumps(
                {str(k): int(v) for k, v in (label_counts or {}).items()}
            ),
            SAMPLES_KEY: int(n_train),
            CODES_KEY: ",".join(codes),
        }
    )


def parse_announcement(node_id: int, record: Mapping) -> Announcement:
    """Read a declaration record. Raises `DeclarationError` on anything odd.

    Strict on purpose. A node whose declaration cannot be parsed is excluded,
    never admitted on a default -- guessing at consent terms is the one failure
    mode this whole project exists to prevent.
    """
    try:
        blob = record[POLICY_KEY]
    except (KeyError, TypeError) as exc:
        raise DeclarationError(f"node {node_id} declared no policy") from exc

    try:
        policy = parse_policy(json.loads(blob))
    except (json.JSONDecodeError, ODRLError, TypeError) as exc:
        raise DeclarationError(
            f"node {node_id} declared a policy that could not be parsed: {exc}"
        ) from exc

    try:
        raw_labels = json.loads(record.get(LABELS_KEY) or "{}")
        labels = {int(k): int(v) for k, v in raw_labels.items()}
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DeclarationError(
            f"node {node_id} declared an unreadable label histogram: {exc}"
        ) from exc

    codes = tuple(c for c in str(record.get(CODES_KEY) or "").split(",") if c)
    return Announcement(
        node_id=node_id,
        policy=policy,
        n_train=int(record.get(SAMPLES_KEY) or 0),
        label_counts=labels,
        codes=codes,
    )


def declared_utility(n_classes: int) -> UtilityFn:
    """Score a candidate consortium from what its members declared.

    The same measure the offline sweeps use: class coverage leads, volume
    modifies. A class nobody holds cannot be learned at any sample size,
    whereas more examples of a class already present give diminishing returns.

    `n_classes` is passed in rather than imported so this module stays free of
    the learning task -- the policy machinery has no business knowing what is
    being trained.
    """

    def utility(subset: Sequence[Announcement]) -> float:
        if not subset:
            return 0.0
        classes: set[int] = set()
        samples = 0
        for announcement in subset:
            classes |= announcement.classes
            samples += announcement.n_train
        return (len(classes) / n_classes) * (samples**0.5)

    return utility


@dataclass(frozen=True)
class Admission:
    """Who is in, who is out, and what the federation may and must do."""

    admitted: tuple[int, ...]
    agreement: Policy
    capability: frozenset[Action]
    plan: DutyPlan
    excluded: Mapping[int, str]
    result: SelectionResult | None = None

    @property
    def is_empty(self) -> bool:
        return not self.admitted


class NoFeasibleConsortium(Exception):
    """No subset of the declared policies grants what the federation needs.

    Raised rather than handled. Falling back to training on everyone would
    produce exactly the unusable model this work is about, and would do it
    silently.
    """


def admit(
    announcements: Sequence[Announcement],
    request: Policy,
    utility: UtilityFn,
    *,
    taxonomy=None,
    rejected: Mapping[int, str] | None = None,
) -> Admission:
    """Choose a consortium from what the nodes declared.

    Pure: no Grid, no transport, no clock. Everything that decides who trains
    happens here, which is what makes the decision testable and auditable.
    """
    excluded: dict[int, str] = dict(rejected or {})

    if not announcements:
        raise NoFeasibleConsortium("no node declared a usable policy")

    outcome = search(announcements, request, utility, taxonomy=taxonomy)
    best = outcome.best
    if best is None:
        raise NoFeasibleConsortium(
            f"none of the {len(announcements)} declared policies compose into an "
            "agreement satisfying the request"
        )

    chosen = tuple(announcements[i].node_id for i in best.members)
    for index, announcement in enumerate(announcements):
        if index not in best.members:
            excluded[announcement.node_id] = (
                "terms do not compose with the selected consortium"
            )

    admitted_policies = [announcements[i].policy for i in best.members]
    composed = compose(admitted_policies, taxonomy=taxonomy)

    # Sort what the agreement demands by what can be done about it. Attributed
    # to the members that granted it, so an outstanding obligation names who
    # owes it rather than floating free.
    plan = plan_duties(
        composed.agreement,
        owed_by=tuple(announcements[i].policy.assigner or f"node-{announcements[i].node_id}"
                      for i in best.members),
    )

    return Admission(
        admitted=chosen,
        agreement=composed.agreement,
        capability=best.capability,
        plan=plan,
        excluded=excluded,
        result=outcome,
    )


def collect_announcements(
    grid: Grid, *, timeout: float | None = None
) -> tuple[list[Announcement], dict[int, str]]:
    """Ask every connected node to declare itself.

    Returns the announcements that parsed, and a reason for each node that did
    not answer or could not be understood.
    """
    node_ids = list(grid.get_node_ids())
    messages = [
        grid.create_message(
            content=RecordDict({"config": ConfigRecord({})}),
            message_type=MessageType.QUERY,
            dst_node_id=node_id,
            group_id=DECLARATION_GROUP,
        )
        for node_id in node_ids
    ]

    announcements: list[Announcement] = []
    rejected: dict[int, str] = {}
    answered: set[int] = set()

    for reply in grid.send_and_receive(messages, timeout=timeout):
        node_id = reply.metadata.src_node_id
        answered.add(node_id)
        if reply.has_error():
            rejected[node_id] = f"declaration failed: {reply.error}"
            continue
        try:
            record = reply.content["config"]
        except KeyError:
            rejected[node_id] = "reply carried no config record"
            continue
        try:
            announcements.append(parse_announcement(node_id, record))
        except DeclarationError as exc:
            rejected[node_id] = str(exc)

    for node_id in node_ids:
        if node_id not in answered:
            rejected[node_id] = "did not answer the declaration round"

    return announcements, rejected


class PolicyAwareFedAvg(FedAvg):
    """FedAvg restricted to an admitted consortium.

    The only behavioural change is *who gets addressed*. Aggregation, metric
    handling and the round loop are inherited untouched, so any difference in
    the resulting model is attributable to membership rather than to a modified
    learning rule -- which is the comparison the evaluation needs.

    `fraction_train` and `fraction_evaluate` no longer govern participation:
    membership is decided once, by policy, before `start()`. They are kept only
    as the on/off switch the base class treats them as.
    """

    def __init__(
        self,
        admitted: Sequence[int],
        directives: Sequence[Directive] = (),
        ledger: Ledger | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._admitted = tuple(dict.fromkeys(admitted))
        self._directives = tuple(directives)
        self._ledger = ledger if ledger is not None else Ledger()
        if not self._admitted:
            raise ValueError(
                "a policy-aware federation needs at least one admitted node; "
                "an empty admission should have raised NoFeasibleConsortium"
            )

    @property
    def admitted(self) -> tuple[int, ...]:
        return self._admitted

    @property
    def ledger(self) -> Ledger:
        return self._ledger

    def _addressed(self, record: RecordDict, message_type: str) -> list[Message]:
        return [
            Message(content=record, message_type=message_type, dst_node_id=node_id)
            for node_id in self._admitted
        ]

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if self.fraction_train == 0.0:
            return []
        config["server-round"] = server_round
        # Dispatch the duties a client can act on. Sent every round rather than
        # once at setup: a client joining late, or restarting, must not end up
        # training without its instructions.
        config[DIRECTIVE_KEY] = encode_directives(self._directives)
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._addressed(record, MessageType.TRAIN)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Drop updates that do not evidence their duties, then aggregate.

        Filters and delegates, so FedAvg's own validation and example-count
        weighting stay untouched -- a dropped update must not alter how the
        surviving ones are averaged.

        Bounded by the attestation gap: this checks what a client *reported*.
        See `odrl_fl.duties`.
        """
        kept: list[Message] = []
        for reply in replies:
            # Transport errors are FedAvg's business, not a duty question.
            if reply.has_error():
                kept.append(reply)
                continue

            ok, reason = verify(self._directives, self._claim_of(reply))
            if ok:
                kept.append(reply)
            else:
                node_id = reply.metadata.src_node_id
                self._ledger.reject(server_round, node_id, reason)
                log(INFO, "round %s: dropped node %s -- %s", server_round, node_id, reason)

        return super().aggregate_train(server_round, kept)

    def _claim_of(self, reply: Message) -> Compliance:
        """What a reply claims about its duties. Absent means claiming nothing."""
        for record in reply.content.config_records.values():
            if COMPLIANCE_KEY in record:
                return Compliance.decode(record[COMPLIANCE_KEY])
        return Compliance()

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if self.fraction_evaluate == 0.0:
            return []
        config["server-round"] = server_round
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._addressed(record, MessageType.EVALUATE)
