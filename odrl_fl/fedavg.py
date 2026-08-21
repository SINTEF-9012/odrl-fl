"""A minimal in-process FedAvg loop, used for experiment sweeps.

**Why this exists alongside the Flower apps.**

The evaluation compares *subsets* of contributors -- potentially hundreds of
them, since the selection problem searches a powerset. Standing up a full
federation per subset costs tens of seconds in orchestration for a model that
trains in under one, and adds failure modes that have nothing to do with the
research question.

So there are two paths, deliberately:

- **this module** drives the sweeps: in-process, deterministic, no transport,
  no orchestration;
- **`server_app.py` / `client_app.py`** are the real Flower integration, where
  the policy enforcement hooks live and where the systems contribution is
  demonstrated.

Both report the *same* metric -- accuracy of the global model on the shared
held-out test set -- so the two paths are directly comparable. Establishing
that they agree numerically is a task for increment 6, when the policy-aware
strategy lands; until then the Flower path is verified only as "it runs and
learns", which it does.

A second, practical reason this module exists: Flower's simulation backend
requires Ray, and Ray publishes no Windows wheels for Python 3.13, so
`run_simulation` is unavailable during development on this platform. The
deployment engine works, but costs a SuperLink plus N SuperNode processes per
run -- untenable for a sweep over a powerset.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch

from odrl_fl.task import (
    Net,
    evaluate,
    global_test_loader,
    load_partition,
    to_loaders,
    train,
)


@dataclass
class History:
    """Per-round training record."""

    contributors: list[int]
    accuracy: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)

    @property
    def final_accuracy(self) -> float:
        return self.accuracy[-1] if self.accuracy else 0.0

    @property
    def best_accuracy(self) -> float:
        return max(self.accuracy) if self.accuracy else 0.0


def federated_average(
    state_dicts: Sequence[dict[str, torch.Tensor]], weights: Sequence[float]
) -> dict[str, torch.Tensor]:
    """Weighted parameter average -- the FedAvg aggregation step.

    Weights are normalised here, so callers can pass raw example counts.
    """
    if not state_dicts:
        raise ValueError("nothing to aggregate")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")

    normalised = [w / total for w in weights]
    out: dict[str, torch.Tensor] = OrderedDict()
    for key in state_dicts[0]:
        stacked = torch.stack(
            [sd[key].float() * w for sd, w in zip(state_dicts, normalised)]
        )
        out[key] = stacked.sum(dim=0).to(state_dicts[0][key].dtype)
    return out


def run_fedavg(
    contributors: Iterable[int],
    n_clients: int,
    rounds: int = 5,
    local_epochs: int = 1,
    alpha: float = 1.0,
    seed: int = 0,
    size_variation: float = 0.0,
) -> History:
    """Train a federated model over `contributors` and report per-round accuracy.

    `n_clients` is the size of the *population*, not of the selected subset:
    partitions are keyed on client id, so client 3 holds the same data whether
    or not clients 1 and 2 were selected. That is what makes comparing subsets
    meaningful.

    `size_variation` must match whatever produced the partitions being
    reasoned about elsewhere. The policy-aware path has contributors
    *declare* their data volume, and a declaration that does not match the
    data actually trained on is worse than no declaration at all.

    Evaluation uses the **shared held-out global test set**, never the
    contributors' own splits. Scoring a subset on its own data makes utility
    incomparable across subsets: a single client holding two of four classes
    would be tested only on those two and would out-score the full federation.
    """
    contributors = sorted(set(contributors))
    if not contributors:
        raise ValueError("at least one contributor is required")

    # Seed torch as well as numpy: initial parameters must not vary between
    # runs, or subset comparisons pick up initialisation noise.
    torch.manual_seed(seed)

    partitions = {
        cid: load_partition(
            cid, n_clients, alpha=alpha, seed=seed, size_variation=size_variation
        )
        for cid in contributors
    }
    trainloaders = {cid: to_loaders(part)[0] for cid, part in partitions.items()}
    testloader = global_test_loader(seed=seed)

    global_model = Net()
    history = History(contributors=list(contributors))

    for _ in range(rounds):
        state_dicts, weights = [], []
        for cid in contributors:
            local = Net()
            local.load_state_dict(global_model.state_dict())
            train(local, trainloaders[cid], epochs=local_epochs)
            state_dicts.append(local.state_dict())
            weights.append(float(partitions[cid].n_train))

        global_model.load_state_dict(federated_average(state_dicts, weights))

        loss, accuracy = evaluate(global_model, testloader)
        history.loss.append(loss)
        history.accuracy.append(accuracy)

    return history
