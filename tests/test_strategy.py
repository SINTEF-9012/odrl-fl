"""Tests for the policy-aware strategy.

Ray publishes no Windows wheel for Python 3.13, so no test here can start a
real federation. That constraint shaped the design rather than being worked
around: everything that *decides* anything is a pure function over declared
policies, and only message addressing needs a Grid at all. A `FakeGrid` covers
that last part.

The test that matters most is
`TestPolicyAwareFedAvg::test_only_admitted_nodes_are_addressed` -- if the
strategy ever addressed an excluded node, the federation would train on data it
was not permitted to use, which is the exact failure this project exists to
prevent.
"""

from __future__ import annotations

import json

import pytest
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp import Grid

from odrl_fl.corpus import generate_corpus
from odrl_fl.duties import (
    COMPLIANCE_KEY,
    DIRECTIVE_KEY,
    Compliance,
    Directive,
    decode_directives,
)
from odrl_fl.odrl import (
    Action,
    Constraint,
    LeftOperand,
    Operator,
    Permission,
    Policy,
    PolicyType,
)
from odrl_fl.selection import requirement
from odrl_fl.strategy import (
    LABELS_KEY,
    POLICY_KEY,
    SAMPLES_KEY,
    Announcement,
    DeclarationError,
    NoFeasibleConsortium,
    PolicyAwareFedAvg,
    admit,
    collect_announcements,
    declaration,
    parse_announcement,
)
from odrl_fl.task import Net

ACTIONS = (Action.USE, Action.AGGREGATE)


def offer(index: int, *purposes: str) -> Policy:
    constraints = frozenset()
    if purposes:
        constraints = frozenset(
            {Constraint(LeftOperand.PURPOSE, Operator.IS_ANY_OF, frozenset(purposes))}
        )
    return Policy(
        uid=f"urn:policy:node-{index}",
        type=PolicyType.OFFER,
        assigner=f"urn:party:node-{index}",
        permissions=frozenset(
            Permission(action=a, constraints=constraints) for a in ACTIONS
        ),
    )


def announcement(node_id: int, *purposes: str, n_train: int = 100) -> Announcement:
    return Announcement(
        node_id=node_id,
        policy=offer(node_id, *purposes),
        n_train=n_train,
        label_counts={0: n_train // 2, 1: n_train // 2},
    )


def volume(subset) -> float:
    return float(sum(a.n_train for a in subset))


class FakeGrid(Grid):
    """A Grid that answers with canned replies. No transport, no SuperLink."""

    def __init__(self, replies: dict[int, RecordDict | None]) -> None:
        self._replies = replies
        self.sent: list[Message] = []

    # -- the part the tests exercise -------------------------------------
    def get_node_ids(self):
        return list(self._replies)

    def create_message(
        self, content, message_type, dst_node_id, group_id, ttl=None
    ) -> Message:
        return Message(
            content=content, message_type=message_type, dst_node_id=dst_node_id
        )

    def send_and_receive(self, messages, *, timeout=None):
        out = []
        for message in messages:
            self.sent.append(message)
            node_id = message.metadata.dst_node_id
            content = self._replies[node_id]
            if content is None:  # node stays silent
                continue
            # `reply_to` sets src_node_id from the original destination, which
            # is exactly what a real node's reply carries.
            out.append(Message(content, reply_to=message))
        return out

    # -- unused by these tests -------------------------------------------
    def push_messages(self, messages):
        raise NotImplementedError

    def pull_messages(self, message_ids):
        raise NotImplementedError

    @property
    def run(self):
        raise NotImplementedError

    def set_run(self, run_id):
        raise NotImplementedError


# ---------------------------------------------------------------------------


class TestWireFormat:
    def test_a_declaration_round_trips(self):
        policy = offer(0, "research")
        record = declaration(
            policy, n_train=250, label_counts={0: 100, 1: 150}, codes=("DUO:0000042",)
        )
        parsed = parse_announcement(7, record)

        assert parsed.node_id == 7
        assert parsed.policy == policy
        assert parsed.n_train == 250
        assert parsed.label_counts == {0: 100, 1: 150}
        assert parsed.codes == ("DUO:0000042",)

    def test_a_real_corpus_policy_survives_the_wire(self):
        """Corpus policies are far richer than the hand-built ones above."""
        for contributor in generate_corpus(8, seed=0):
            record = declaration(contributor.policy, n_train=10)
            assert parse_announcement(0, record).policy == contributor.policy

    def test_a_missing_policy_is_rejected(self):
        with pytest.raises(DeclarationError, match="declared no policy"):
            parse_announcement(1, ConfigRecord({SAMPLES_KEY: 10}))

    def test_unparseable_policy_is_rejected(self):
        record = ConfigRecord({POLICY_KEY: "{not json", LABELS_KEY: "{}"})
        with pytest.raises(DeclarationError, match="could not be parsed"):
            parse_announcement(1, record)

    def test_a_policy_that_is_valid_json_but_not_a_policy_is_rejected(self):
        record = ConfigRecord({POLICY_KEY: json.dumps({"nope": 1}), LABELS_KEY: "{}"})
        with pytest.raises(DeclarationError, match="could not be parsed"):
            parse_announcement(1, record)

    def test_a_broken_histogram_is_rejected_rather_than_ignored(self):
        record = ConfigRecord(
            {POLICY_KEY: json.dumps({"uid": "x"}), LABELS_KEY: "[[["}
        )
        with pytest.raises(DeclarationError):
            parse_announcement(1, record)

    def test_declaration_survives_absent_optional_fields(self):
        record = declaration(offer(0))
        parsed = parse_announcement(3, record)
        assert parsed.n_train == 0
        assert parsed.codes == ()


class TestAdmission:
    def test_a_compatible_pool_is_admitted_whole(self):
        pool = [announcement(i, "research") for i in range(3)]
        admission = admit(pool, requirement(*ACTIONS), volume)

        assert set(admission.admitted) == {0, 1, 2}
        assert not admission.excluded
        assert Action.USE in admission.capability

    def test_an_incompatible_node_is_excluded_with_a_reason(self):
        """Non-transitivity in the strategy, not just the search."""
        pool = [
            announcement(0, "ds-cancer"),
            announcement(1, "ds-cancer"),
            announcement(2, "ds-neurological"),
        ]
        admission = admit(pool, requirement(*ACTIONS), volume)

        assert 2 not in admission.admitted
        assert admission.excluded[2]

    def test_nodes_that_never_answered_stay_excluded(self):
        pool = [announcement(0, "research")]
        admission = admit(
            pool,
            requirement(*ACTIONS),
            volume,
            rejected={9: "did not answer the declaration round"},
        )
        assert 9 not in admission.admitted
        assert "did not answer" in admission.excluded[9]

    def test_no_declarations_is_fatal(self):
        with pytest.raises(NoFeasibleConsortium):
            admit([], requirement(*ACTIONS), volume)

    def test_nothing_feasible_is_fatal_rather_than_a_silent_fallback(self):
        """The failure mode the whole project exists to prevent."""
        pool = [announcement(0, "clinical-care"), announcement(1, "commercial")]
        with pytest.raises(NoFeasibleConsortium):
            admit(pool, requirement(*ACTIONS, purpose="research"), volume)

    def test_the_admitted_agreement_actually_grants_the_request(self):
        from odrl_fl.odrl import satisfies

        pool = [announcement(i, "research") for i in range(4)]
        request = requirement(*ACTIONS, purpose="ds-cancer")
        admission = admit(pool, request, volume)
        assert satisfies(admission.agreement, request)

    def test_admission_prefers_more_data_among_feasible_sets(self):
        pool = [
            announcement(0, "research", n_train=10),
            announcement(1, "research", n_train=900),
        ]
        admission = admit(pool, requirement(*ACTIONS), volume)
        assert 1 in admission.admitted


class TestCollectAnnouncements:
    def test_declarations_are_gathered_from_every_node(self):
        grid = FakeGrid(
            {
                n: RecordDict({"config": declaration(offer(n, "research"), n_train=50)})
                for n in (11, 12, 13)
            }
        )
        announcements, rejected = collect_announcements(grid)

        assert sorted(a.node_id for a in announcements) == [11, 12, 13]
        assert not rejected
        assert all(m.metadata.message_type == MessageType.QUERY for m in grid.sent)

    def test_a_silent_node_is_recorded_not_assumed(self):
        grid = FakeGrid(
            {
                11: RecordDict({"config": declaration(offer(11, "research"))}),
                12: None,
            }
        )
        announcements, rejected = collect_announcements(grid)

        assert [a.node_id for a in announcements] == [11]
        assert "did not answer" in rejected[12]

    def test_a_malformed_declaration_is_recorded_not_assumed(self):
        grid = FakeGrid(
            {
                11: RecordDict({"config": ConfigRecord({POLICY_KEY: "{bad"})}),
            }
        )
        announcements, rejected = collect_announcements(grid)

        assert not announcements
        assert 11 in rejected

    def test_a_reply_without_a_config_record_is_recorded(self):
        grid = FakeGrid({11: RecordDict({"metrics": ConfigRecord({})})})
        announcements, rejected = collect_announcements(grid)
        assert not announcements
        assert 11 in rejected


class TestPolicyAwareFedAvg:
    def strategy(self, admitted=(11, 13)):
        return PolicyAwareFedAvg(admitted=admitted)

    def test_only_admitted_nodes_are_addressed(self):
        """The load-bearing guarantee of the whole increment."""
        grid = FakeGrid({n: None for n in (11, 12, 13, 14)})
        messages = list(
            self.strategy().configure_train(
                1, ArrayRecord(Net().state_dict()), ConfigRecord({}), grid
            )
        )
        assert sorted(m.metadata.dst_node_id for m in messages) == [11, 13]

    def test_evaluation_is_restricted_too(self):
        """An excluded node must not see the model at all, not even to score it."""
        grid = FakeGrid({n: None for n in (11, 12, 13, 14)})
        messages = list(
            self.strategy().configure_evaluate(
                1, ArrayRecord(Net().state_dict()), ConfigRecord({}), grid
            )
        )
        assert sorted(m.metadata.dst_node_id for m in messages) == [11, 13]

    def test_the_server_round_is_still_injected(self):
        grid = FakeGrid({11: None, 13: None})
        config = ConfigRecord({})
        list(
            self.strategy().configure_train(
                4, ArrayRecord(Net().state_dict()), config, grid
            )
        )
        assert config["server-round"] == 4

    def test_membership_ignores_how_many_nodes_are_connected(self):
        """Selection happens once, before start(); the Grid cannot widen it."""
        strategy = self.strategy(admitted=(11,))
        crowded = FakeGrid({n: None for n in range(100)})
        messages = list(
            strategy.configure_train(
                1, ArrayRecord(Net().state_dict()), ConfigRecord({}), crowded
            )
        )
        assert [m.metadata.dst_node_id for m in messages] == [11]

    def test_an_empty_admission_is_refused(self):
        with pytest.raises(ValueError, match="at least one admitted node"):
            PolicyAwareFedAvg(admitted=())

    def test_duplicate_admissions_are_collapsed(self):
        assert PolicyAwareFedAvg(admitted=(11, 11, 13)).admitted == (11, 13)


class TestPathsAgree:
    def test_our_aggregation_matches_flowers(self):
        """Promised by `odrl_fl/fedavg.py`'s docstring, discharged here.

        The sweeps run in-process and the systems contribution runs on Flower.
        If the two aggregated differently, every number measured on the fast
        path would be a claim about a different algorithm.
        """
        import torch
        from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

        from odrl_fl.fedavg import federated_average

        torch.manual_seed(0)
        states = [Net().state_dict() for _ in range(3)]
        weights = [100.0, 250.0, 50.0]

        ours = federated_average(states, weights)

        contents = [
            RecordDict(
                {
                    "arrays": ArrayRecord(state),
                    "metrics": MetricRecord({"num-examples": w}),
                }
            )
            for state, w in zip(states, weights)
        ]
        theirs = aggregate_arrayrecords(contents, "num-examples").to_torch_state_dict()

        for key in ours:
            assert torch.allclose(ours[key], theirs[key], atol=1e-6), key


class TestDutyDispatch:
    """Directives out, compliance claims back, non-compliant updates dropped."""

    def strategy(self, directives=(), ledger=None):
        return PolicyAwareFedAvg(
            admitted=(11, 12), directives=directives, ledger=ledger
        )

    def train_config(self, strategy) -> ConfigRecord:
        grid = FakeGrid({11: None, 12: None})
        config = ConfigRecord({})
        list(
            strategy.configure_train(
                1, ArrayRecord(Net().state_dict()), config, grid
            )
        )
        return config

    def reply(self, node_id: int, claim: Compliance | None, weight: float = 10.0):
        """A well-formed training reply, optionally carrying a compliance claim."""
        content = RecordDict(
            {
                "arrays": ArrayRecord(Net().state_dict()),
                "metrics": MetricRecord({"num-examples": weight}),
            }
        )
        if claim is not None:
            content["config"] = ConfigRecord({COMPLIANCE_KEY: claim.encode()})
        request = Message(
            content=RecordDict({"config": ConfigRecord({})}),
            message_type=MessageType.TRAIN,
            dst_node_id=node_id,
        )
        return Message(content, reply_to=request)

    def test_directives_are_dispatched_in_the_train_config(self):
        directives = (Directive(action="anonymize", epsilon=0.5),)
        config = self.train_config(self.strategy(directives))
        assert decode_directives(config[DIRECTIVE_KEY]) == directives

    def test_the_key_is_present_even_with_no_directives(self):
        """A client must be able to tell "nothing required" from "not told"."""
        config = self.train_config(self.strategy())
        assert DIRECTIVE_KEY in config
        assert decode_directives(config[DIRECTIVE_KEY]) == ()

    def test_a_compliant_update_is_aggregated(self):
        directives = (Directive(action="anonymize", epsilon=1.0),)
        strategy = self.strategy(directives)
        arrays, _ = strategy.aggregate_train(
            1,
            [
                self.reply(11, Compliance({"anonymize": 1.0})),
                self.reply(12, Compliance({"anonymize": 0.5})),
            ],
        )
        assert arrays is not None
        assert not strategy.ledger.rejections

    def test_a_non_compliant_update_is_dropped_and_recorded(self):
        """The enforcement, such as it is: the update does not reach the average."""
        directives = (Directive(action="anonymize", epsilon=0.5),)
        strategy = self.strategy(directives)
        arrays, _ = strategy.aggregate_train(
            1,
            [
                self.reply(11, Compliance({"anonymize": 0.5})),
                self.reply(12, Compliance({"anonymize": 9.0})),
            ],
        )
        assert arrays is not None  # the compliant one still aggregates
        assert strategy.ledger.rejected_nodes == {12}
        assert "exceeds" in strategy.ledger.rejections[0][2]

    def test_a_silent_reply_is_dropped(self):
        """Omitting the claim must not be a way to skip the duty."""
        strategy = self.strategy((Directive(action="anonymize", epsilon=1.0),))
        strategy.aggregate_train(
            2, [self.reply(11, Compliance({"anonymize": 1.0})), self.reply(12, None)]
        )
        assert strategy.ledger.rejected_nodes == {12}

    def test_nothing_is_dropped_when_nothing_was_required(self):
        strategy = self.strategy()
        strategy.aggregate_train(1, [self.reply(11, None), self.reply(12, None)])
        assert not strategy.ledger.rejections

    def test_the_ledger_records_which_round_dropped_a_node(self):
        strategy = self.strategy((Directive(action="anonymize", epsilon=0.1),))
        strategy.aggregate_train(7, [self.reply(11, Compliance({"anonymize": 5.0}))])
        assert strategy.ledger.rejections[0][0] == 7

    def test_dropping_every_update_aggregates_nothing(self):
        """A round in which nobody complied must not silently produce a model."""
        strategy = self.strategy((Directive(action="anonymize", epsilon=0.1),))
        arrays, _ = strategy.aggregate_train(1, [self.reply(11, None)])
        assert arrays is None
