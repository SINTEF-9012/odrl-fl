"""Flower ClientApp: local training and evaluation.

Plain and policy-free at increment 0. Policy *duties* will later be dispatched
into `configure_train`'s ConfigRecord and executed here (differential privacy,
metric reporting), then verified server-side in `aggregate_train` -- so the
shape of this file matters more than its current content: whatever a duty
requires a client to do has to be executable at exactly these two entry points.
"""

from __future__ import annotations

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from odrl_fl.task import Net, evaluate, load_partition, to_loaders, train

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


@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    net, trainloader, _, partition = _setup(context)
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    epochs = int(msg.content["config"].get("local-epochs", 1))
    loss = train(net, trainloader, epochs=epochs)

    reply = RecordDict(
        {
            "arrays": ArrayRecord(net.state_dict()),
            "metrics": MetricRecord(
                {
                    "train_loss": loss,
                    # FedAvg weights the average by this key.
                    "num-examples": partition.n_train,
                }
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
