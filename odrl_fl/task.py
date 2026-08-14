"""Model, data, and partitioning.

Increment 0 uses **synthetic tabular data**. The only job of the dataset here is
to make the pipeline runnable in seconds so later increments have a working
baseline to attach to.

The choice is deliberate rather than lazy. Later work sweeps data heterogeneity
as an independent variable, and a fixed real dataset gives no dial to turn:
`alpha` below parameterises a Dirichlet label skew, from near-IID (`alpha` large)
to pathological (`alpha` small). A real federated medical dataset --
Fed-TCGA-BRCA is the current candidate, since its dbGaP controlled-access
heritage means real Data Use Ontology codes can be cited for it -- swaps in
behind `load_partition` without anything else changing.

`load_partition` is therefore the seam. Everything downstream addresses data
only through it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#: Defaults for the synthetic problem. Small on purpose: the pipeline should
#: run end to end in seconds, so that iterating on policy logic is never gated
#: on training time.
N_FEATURES = 20
N_CLASSES = 4
SAMPLES_PER_CLIENT = 500

#: Spread of the class centroids relative to unit within-class noise.
#:
#: Tuned so the task is neither trivial nor hopeless: a single skewed client
#: reaches roughly 0.5-0.6 on the balanced global test set while the full
#: federation reaches ~0.9. Without that headroom there is no utility axis to
#: trade capability against -- a task that saturates at 1.0 after one round
#: makes every contributor subset look identical.
CLASS_SEPARATION = 0.9

#: Size of the held-out global test set (see `make_global_test`).
GLOBAL_TEST_SIZE = 2000


class Net(nn.Module):
    """A small MLP. Deliberately unremarkable -- the contribution is not here."""

    def __init__(self, n_features: int = N_FEATURES, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class Partition:
    """One client's local data."""

    client_id: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.y_train)

    @property
    def label_counts(self) -> dict[int, int]:
        """Class histogram. Used later to reason about what a contributor adds."""
        values, counts = np.unique(self.y_train, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}


def class_means(
    seed: int = 0, n_classes: int = N_CLASSES, n_features: int = N_FEATURES
) -> np.ndarray:
    """Class centroids for a given seed.

    Keyed on `seed` alone and on a fixed sub-stream, so clients and the global
    test set are drawn from the *same* distribution no matter how many clients
    exist. Everything else depends on this being stable.
    """
    return np.random.default_rng([seed, 0xC1A5]).normal(
        loc=0.0, scale=CLASS_SEPARATION, size=(n_classes, n_features)
    )


def make_global_test(
    seed: int = 0,
    n_samples: int = GLOBAL_TEST_SIZE,
    n_features: int = N_FEATURES,
    n_classes: int = N_CLASSES,
) -> tuple[np.ndarray, np.ndarray]:
    """A balanced, held-out test set shared by every experimental condition.

    Models trained on different contributor subsets **must** be scored on the
    same distribution, or "utility" silently changes meaning between subsets: a
    single client holding two of four classes would otherwise be evaluated only
    on those two easy classes and score *higher* than the full federation.

    Balanced across classes on purpose -- it stands for the population the model
    is ultimately meant to serve, not for any contributor's local skew.
    """
    rng = np.random.default_rng([seed, 0x7E57])
    means = class_means(seed, n_classes, n_features)

    per_class = n_samples // n_classes
    labels = np.repeat(np.arange(n_classes), per_class)
    features = means[labels] + rng.normal(
        loc=0.0, scale=1.0, size=(len(labels), n_features)
    )
    return features.astype(np.float32), labels.astype(np.int64)


def make_dataset(
    n_clients: int,
    alpha: float = 1.0,
    seed: int = 0,
    samples_per_client: int = SAMPLES_PER_CLIENT,
    n_features: int = N_FEATURES,
    n_classes: int = N_CLASSES,
) -> list[Partition]:
    """Generate a federated synthetic dataset with Dirichlet label skew.

    `alpha` controls heterogeneity, the standard non-IID mechanism in the FL
    literature: small values concentrate each client on a few classes, large
    values approach a uniform (IID) split.

    Fully determined by `seed`. Later increments compare *subsets* of
    contributors, which is only meaningful if a given client always holds the
    same data regardless of who else is in the federation -- so nothing here may
    depend on `n_clients` beyond the number of partitions produced.
    """
    means = class_means(seed, n_classes, n_features)

    partitions: list[Partition] = []
    for client_id in range(n_clients):
        # Per-client generator keyed on the client id: client 3's data is the
        # same whether the federation has 4 members or 40.
        client_rng = np.random.default_rng([seed, client_id])

        proportions = client_rng.dirichlet(np.full(n_classes, alpha))
        labels = client_rng.choice(n_classes, size=samples_per_client, p=proportions)

        features = means[labels] + client_rng.normal(
            loc=0.0, scale=1.0, size=(samples_per_client, n_features)
        )

        split = int(0.8 * samples_per_client)
        partitions.append(
            Partition(
                client_id=client_id,
                x_train=features[:split].astype(np.float32),
                y_train=labels[:split].astype(np.int64),
                x_test=features[split:].astype(np.float32),
                y_test=labels[split:].astype(np.int64),
            )
        )
    return partitions


def load_partition(
    client_id: int, n_clients: int, alpha: float = 1.0, seed: int = 0
) -> Partition:
    """Return one client's data. **The seam for swapping in a real dataset.**"""
    return make_dataset(n_clients=n_clients, alpha=alpha, seed=seed)[client_id]


def global_test_loader(seed: int = 0, batch_size: int = 64) -> DataLoader:
    """Loader over the shared held-out test set."""
    x, y = make_global_test(seed=seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size
    )


def to_loaders(
    partition: Partition, batch_size: int = 32
) -> tuple[DataLoader, DataLoader]:
    train = TensorDataset(
        torch.from_numpy(partition.x_train), torch.from_numpy(partition.y_train)
    )
    test = TensorDataset(
        torch.from_numpy(partition.x_test), torch.from_numpy(partition.y_test)
    )
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size),
    )


def train(net: Net, loader: DataLoader, epochs: int = 1, lr: float = 0.01) -> float:
    """Train locally. Returns mean loss per batch."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.to(DEVICE).train()

    total, batches = 0.0, 0
    for _ in range(epochs):
        for features, labels in loader:
            optimizer.zero_grad()
            loss = criterion(net(features.to(DEVICE)), labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
    return total / max(1, batches)


def evaluate(net: Net, loader: DataLoader) -> tuple[float, float]:
    """Evaluate. Returns `(loss, accuracy)`."""
    criterion = nn.CrossEntropyLoss()
    net.to(DEVICE).eval()

    loss, correct, seen = 0.0, 0, 0
    with torch.no_grad():
        for features, labels in loader:
            outputs = net(features.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += float(criterion(outputs, labels).item())
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
            seen += len(labels)
    return loss, correct / max(1, seen)
