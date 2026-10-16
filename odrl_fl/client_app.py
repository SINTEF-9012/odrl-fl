"""Flower ClientApp: declaration, local training, and evaluation.

Three entry points now. `declare` answers the server's policy round: the node
states what it permits, in ODRL, and the server composes those answers to
decide who trains. Declaring is the client's job rather than the server's on
purpose -- a server that looked its clients' consent terms up in a table would
be assuming precisely what the system is supposed to establish.

`train_fn` also discharges the duties the agreement dispatched, and reports
what it did. The server drops updates whose claim does not cover the directives
they were given, so silence is not a way to skip a duty. It remains a *claim*:
nothing here proves the computation happened, and closing that gap needs secure
aggregation or attestation. See `odrl_fl.duties`.
"""

from __future__ import annotations

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp

from odrl_fl.corpus import generate_corpus
from odrl_fl.duties import (
    COMPLIANCE_KEY,
    DIRECTIVE_KEY,
    Compliance,
    decode_directives,
)
from odrl_fl.odrl import Action
from odrl_fl.odrl.duo import translate
from odrl_fl.strategy import declaration
from odrl_fl.task import (
    Net,
    evaluate,
    load_partition,
    privatise,
    to_loaders,
    train,
)

app = ClientApp()


def _setup(context: Context):
    """Build this client's model and loaders from its node config."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    alpha = float(context.run_config.get("alpha", 1.0))
    seed = int(context.run_config.get("seed", 0))

    partition = load_partition(partition_id, num_partitions, alpha=alpha, seed=seed)
    trainloader, testloader = to_loaders(partition)
    return Net(), trainloader, testloader, partition


@app.query()
def declare(msg: Message, context: Context) -> Message:
    """State this node's data-use terms, in ODRL.

    Codes come from `duo-codes` in the node's own config when it sets one, and
    otherwise from the shared corpus keyed on partition id -- which keeps the
    Flower path comparable with the offline sweeps, where the same corpus
    supplies the same policies. In a real deployment they would come from the
    institution's data access committee, and nothing above this line would
    change.

    The reply also carries data volume and the local class histogram, because
    the server has to score candidate consortia without seeing any data. That
    is a genuine disclosure -- see `strategy.declaration`.
    """
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    seed = int(context.run_config.get("seed", 0))

    configured = str(context.node_config.get("duo-codes") or "").strip()
    if configured:
        codes = tuple(c.strip() for c in configured.split(",") if c.strip())
        policy, _unsupported = translate(
            codes,
            uid=f"urn:policy:node-{partition_id}",
            assigner=f"urn:party:node-{partition_id}",
            target=f"urn:asset:partition-{partition_id}",
            strict=False,
        )
    else:
        contributor = generate_corpus(num_partitions, seed=seed)[partition_id]
        codes, policy = contributor.codes, contributor.policy

    partition = load_partition(
        partition_id,
        num_partitions,
        alpha=float(context.run_config.get("alpha", 1.0)),
        seed=seed,
    )

    reply = RecordDict(
        {
            "config": declaration(
                policy,
                n_train=partition.n_train,
                label_counts=partition.label_counts,
                codes=codes,
            )
        }
    )
    return Message(reply, reply_to=msg)


@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    net, trainloader, _, partition = _setup(context)
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    config = msg.content["config"]
    epochs = int(config.get("local-epochs", 1))
    loss = train(net, trainloader, epochs=epochs)

    # Discharge whatever duties the agreement dispatched, and say what was
    # done. The server drops updates that arrive without a matching claim, so
    # staying silent is not a way to skip this.
    state = net.state_dict()
    applied: dict[str, float | None] = {}
    for directive in decode_directives(config.get(DIRECTIVE_KEY)):
        if directive.action == Action.ANONYMIZE.value and directive.epsilon:
            state = privatise(state, directive.epsilon)
            applied[directive.action] = directive.epsilon
        else:
            # A duty this client cannot act on locally. Reported as
            # unaddressed rather than omitted -- the server has to be able to
            # tell "nothing to do" from "did not answer".
            applied[directive.action] = None

    reply = RecordDict(
        {
            "arrays": ArrayRecord(state),
            "metrics": MetricRecord(
                {
                    "train_loss": loss,
                    # FedAvg weights the average by this key.
                    "num-examples": partition.n_train,
                }
            ),
            "config": ConfigRecord(
                {COMPLIANCE_KEY: Compliance(applied=applied).encode()}
            ),
        }
    )
    return Message(reply, reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    net, _, testloader, partition = _setup(context)
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    loss, accuracy = evaluate(net, testloader)

    reply = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    "eval_loss": loss,
                    "accuracy": accuracy,
                    "num-examples": len(partition.y_test),
                }
            )
        }
    )
    return Message(reply, reply_to=msg)
