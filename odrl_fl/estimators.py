"""Estimating what a consortium's model would be worth, without training it.

Selection searches a powerset, and the legal side of that search is cheap:
composition is exact, needs no data and no training, and under a tight request
it removes most of the space. The utility side is the expensive half, and if
scoring a candidate means training it then the search is affordable only for
pools small enough not to need one.

So the question here is deliberately narrower than "can we predict accuracy":

    does a training-free estimator preserve the Pareto *frontier*?

Selection never needs the absolute number. It needs the ranking to be right, and
specifically it needs to agree about which candidates are non-dominated. That is
a much weaker requirement, and whether the cheap signals meet it is what
`experiments/run_estimators.py` measures against an exhaustively trained lattice.

Every estimator here is a **factory**: it precomputes once over the pool and
returns a `UtilityFn` that answers per-subset queries cheaply. That shape is what
makes a powerset search viable -- O(n) work up front, O(1) or O(|subset|) per
query, rather than anything per-subset that touches data.

Three are implemented, and the first is the incumbent on purpose. Without the
proxy the project already used sitting in the same table, "the new estimator
works" would have nothing to be false against.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np


#: Anything the estimators can identify a contributor by. The corpus and the
#: Flower path both carry `index`; nothing else is needed.
class Indexed(Protocol):
    @property
    def index(self) -> int: ...


UtilityFn = Callable[[Sequence[Indexed]], float]


def _pooled_counts(
    subset: Sequence[Indexed],
    label_counts: dict[int, dict[int, int]],
    n_classes: int,
) -> np.ndarray:
    counts = np.zeros(n_classes, dtype=float)
    for contributor in subset:
        for label, count in label_counts[contributor.index].items():
            if 0 <= label < n_classes:
                counts[label] += count
    return counts


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, so the range is a clean [0, 1].

    Symmetric and bounded, unlike KL, which matters because the comparison runs
    in both directions across thousands of subsets and an unbounded score would
    let one pathological candidate dominate any aggregate.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return 1.0
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(np.clip(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0, 1.0))


def coherence(gradients: np.ndarray) -> float:
    """How much a set of client updates agree: ``‖mean gᵢ‖ / mean ‖gᵢ‖``.

    One when every client pulls the same way, approaching zero when they cancel.
    This is almost literally the question federated averaging asks -- the server
    averages these vectors, and an average of opposed vectors is a small step in
    a direction none of the clients wanted.

    Cheap in exactly the way the search needs: the per-client gradients are
    computed once over the pool, and any subset's score is then a couple of
    vector operations over rows that are already in memory.
    """
    if len(gradients) == 0:
        return 0.0
    norms = np.linalg.norm(gradients, axis=1)
    mean_norm = float(norms.mean())
    if mean_norm <= 0:
        return 0.0
    return float(np.linalg.norm(gradients.mean(axis=0)) / mean_norm)


def make_coverage_utility(
    n_clients: int,
    alpha: float = 1.0,
    seed: int = 0,
    size_variation: float = 0.0,
) -> UtilityFn:
    """The incumbent: class coverage scaled by the square root of volume.

    Lifted unchanged from `experiments/run_selection.py`, where it was
    introduced as "a stand-in for model accuracy, pending real training". It is
    here so that every later claim has a baseline to beat, and because it is
    itself a training-free estimator and deserves to be scored as one.
    """
    from odrl_fl.task import N_CLASSES, load_partition

    partitions = {
        i: load_partition(
            i, n_clients, alpha=alpha, seed=seed, size_variation=size_variation
        )
        for i in range(n_clients)
    }

    def utility(subset: Sequence[Indexed]) -> float:
        if not subset:
            return 0.0
        classes: set[int] = set()
        samples = 0
        for contributor in subset:
            partition = partitions[contributor.index]
            classes |= set(partition.label_counts)
            samples += partition.n_train
        return float(len(classes) / N_CLASSES * np.sqrt(samples))

    return utility


def make_divergence_utility(
    n_clients: int,
    alpha: float = 1.0,
    seed: int = 0,
    size_variation: float = 0.0,
) -> UtilityFn:
    """Volume discounted by how far the pooled labels sit from the target.

    The established non-IID signal: federated accuracy degrades with the
    distance between the clients' pooled label distribution and the population
    the model is scored on (Zhao et al., 2018). The evaluation set here is
    balanced by construction, so the target is uniform.

    `(1 - JS) * sqrt(samples)` keeps the same shape as the incumbent -- volume
    with diminishing returns, scaled by a quality term in [0, 1] -- so the two
    differ in what they measure rather than in how they are scaled.
    """
    from odrl_fl.task import N_CLASSES, load_partition

    partitions = {
        i: load_partition(
            i, n_clients, alpha=alpha, seed=seed, size_variation=size_variation
        )
        for i in range(n_clients)
    }
    label_counts = {i: p.label_counts for i, p in partitions.items()}
    target = np.full(N_CLASSES, 1.0 / N_CLASSES)

    def utility(subset: Sequence[Indexed]) -> float:
        if not subset:
            return 0.0
        counts = _pooled_counts(subset, label_counts, N_CLASSES)
        samples = counts.sum()
        if samples <= 0:
            return 0.0
        return float((1.0 - js_divergence(counts, target)) * np.sqrt(samples))

    return utility


def client_gradients(
    n_clients: int,
    alpha: float = 1.0,
    seed: int = 0,
    size_variation: float = 0.0,
) -> np.ndarray:
    """One flattened gradient per client, taken at the shared initialisation.

    The initialisation is the same one `run_fedavg` starts from, seeded
    identically, because the quantity of interest is whether these clients would
    pull the *same* model in compatible directions. Gradients measured at
    different starting points would not be comparable.

    Cost is one forward and backward pass per client over its own data -- O(n)
    for the whole pool, paid once, after which any of the 2^n subsets can be
    scored without touching data again.
    """
    import torch

    from odrl_fl.task import Net, load_partition

    torch.manual_seed(seed)
    model = Net()
    criterion = torch.nn.CrossEntropyLoss()

    rows = []
    for client in range(n_clients):
        partition = load_partition(
            client, n_clients, alpha=alpha, seed=seed, size_variation=size_variation
        )
        model.zero_grad(set_to_none=True)
        x = torch.tensor(partition.x_train, dtype=torch.float32)
        y = torch.tensor(partition.y_train, dtype=torch.long)
        criterion(model(x), y).backward()
        rows.append(
            torch.cat([
                p.grad.detach().flatten()
                for p in model.parameters()
                if p.grad is not None
            ]).numpy()
        )
    model.zero_grad(set_to_none=True)
    return np.vstack(rows)


def make_gradient_utility(
    n_clients: int,
    alpha: float = 1.0,
    seed: int = 0,
    size_variation: float = 0.0,
    gradients: np.ndarray | None = None,
) -> UtilityFn:
    """Volume scaled by how well the clients' updates agree.

    Same shape again -- `coherence * sqrt(samples)` -- so it is comparable with
    the other two and differs only in the quality term. `gradients` can be
    passed in when several estimators share one precompute.
    """
    from odrl_fl.task import load_partition

    if gradients is None:
        gradients = client_gradients(
            n_clients, alpha=alpha, seed=seed, size_variation=size_variation
        )
    volumes = {
        i: load_partition(
            i, n_clients, alpha=alpha, seed=seed, size_variation=size_variation
        ).n_train
        for i in range(n_clients)
    }

    def utility(subset: Sequence[Indexed]) -> float:
        if not subset:
            return 0.0
        rows = gradients[[c.index for c in subset]]
        samples = sum(volumes[c.index] for c in subset)
        return float(coherence(rows) * np.sqrt(samples))

    return utility


#: Name to factory, for the evaluation to iterate. The incumbent leads because
#: every other row is measured against it.
ESTIMATORS: dict[str, Callable[..., UtilityFn]] = {
    "coverage": make_coverage_utility,
    "divergence": make_divergence_utility,
    "gradient": make_gradient_utility,
}
