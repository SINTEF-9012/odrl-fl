"""Flower ServerApp: vanilla FedAvg.

The baseline. Later increments replace `FedAvg` with a policy-aware strategy
that (a) selects contributors from the composed policy rather than sampling
them, (b) dispatches duties into the training config, (c) verifies those duties
before admitting an update, and (d) attaches the composed policy to the exported
model. None of that exists yet -- this is the thing those changes are measured
against.
"""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from odrl_fl.task import Net, evaluate, global_test_loader

app = ServerApp()


def make_central_evaluate(seed: int):
    """Score the global model centrally on the shared held-out test set.

    Deliberately the *same* metric `odrl_fl.fedavg` reports, so the Flower path
    and the fast experiment path are directly comparable. Federated evaluation
    on clients' own splits is not comparable across contributor subsets --
    each subset would be scored on a different distribution.
    """
    testloader = global_test_loader(seed=seed)

    def central_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        net = Net()
        net.load_state_dict(arrays.to_torch_state_dict())
        loss, accuracy = evaluate(net, testloader)
        return MetricRecord({"central_loss": loss, "central_accuracy": accuracy})

    return central_evaluate


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = int(context.run_config.get("num-server-rounds", 5))
    local_epochs = int(context.run_config.get("local-epochs", 1))
    fraction_train = float(context.run_config.get("fraction-train", 1.0))
    min_nodes = int(context.run_config.get("min-nodes", 2))

    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_train,
        min_train_nodes=min_nodes,
        min_evaluate_nodes=min_nodes,
        min_available_nodes=min_nodes,
    )

    seed = int(context.run_config.get("seed", 0))
    torch.manual_seed(seed)

    strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(Net().state_dict()),
        num_rounds=num_rounds,
        train_config=ConfigRecord({"local-epochs": local_epochs}),
        evaluate_config=ConfigRecord({}),
        evaluate_fn=make_central_evaluate(seed),
    )
