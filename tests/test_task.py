"""Tests for the model and the data seam.

The determinism properties here are not incidental. Later increments compare
*subsets* of contributors -- "what would the model be if we federated {A, C, E}
instead of {A, B, C}?" -- and that comparison is only meaningful if a given
client holds identical data across those runs. These tests pin that.
"""

from __future__ import annotations

import numpy as np
import torch

from odrl_fl.task import (
    N_CLASSES,
    N_FEATURES,
    Net,
    class_means,
    evaluate,
    load_partition,
    make_dataset,
    make_global_test,
    to_loaders,
    train,
)


class TestModel:
    def test_output_shape(self):
        net = Net()
        out = net(torch.randn(8, N_FEATURES))
        assert out.shape == (8, N_CLASSES)

    def test_state_dict_round_trips(self):
        a, b = Net(), Net()
        b.load_state_dict(a.state_dict())
        for pa, pb in zip(a.parameters(), b.parameters()):
            assert torch.equal(pa, pb)


class TestPartitioning:
    def test_shapes_and_dtypes(self):
        parts = make_dataset(n_clients=4, seed=0)
        assert len(parts) == 4
        for p in parts:
            assert p.x_train.shape[1] == N_FEATURES
            assert p.x_train.dtype == np.float32
            assert p.y_train.dtype == np.int64
            assert p.n_train > 0
            assert len(p.x_test) > 0

    def test_deterministic_given_a_seed(self):
        a = make_dataset(n_clients=4, seed=7)
        b = make_dataset(n_clients=4, seed=7)
        for pa, pb in zip(a, b):
            assert np.array_equal(pa.x_train, pb.x_train)
            assert np.array_equal(pa.y_train, pb.y_train)

    def test_different_seeds_differ(self):
        a = make_dataset(n_clients=4, seed=1)
        b = make_dataset(n_clients=4, seed=2)
        assert not np.array_equal(a[0].x_train, b[0].x_train)

    def test_a_client_holds_the_same_data_regardless_of_federation_size(self):
        """The property that makes subset comparison valid.

        Client 2's data must not change because other clients joined or left --
        otherwise "the model trained on {A,C}" and "the model trained on
        {A,B,C}" would differ for two reasons at once, and the contribution of
        policy-driven selection could not be isolated.
        """
        small = load_partition(2, n_clients=4, seed=3)
        large = load_partition(2, n_clients=40, seed=3)
        assert np.array_equal(small.x_train, large.x_train)
        assert np.array_equal(small.y_train, large.y_train)

    def test_clients_hold_different_data(self):
        parts = make_dataset(n_clients=4, seed=0)
        assert not np.array_equal(parts[0].x_train, parts[1].x_train)

    def test_alpha_controls_heterogeneity(self):
        """Small alpha should concentrate clients on fewer classes.

        Compared on mean label entropy across clients, which is the quantity
        the heterogeneity sweep will vary.
        """

        def mean_entropy(alpha: float) -> float:
            entropies = []
            for p in make_dataset(n_clients=12, alpha=alpha, seed=11):
                counts = np.array(list(p.label_counts.values()), dtype=float)
                probs = counts / counts.sum()
                entropies.append(-(probs * np.log(probs)).sum())
            return float(np.mean(entropies))

        assert mean_entropy(0.05) < mean_entropy(100.0)

    def test_label_counts_sum_to_training_size(self):
        p = load_partition(0, n_clients=3, seed=0)
        assert sum(p.label_counts.values()) == p.n_train


class TestGlobalTestSet:
    def test_is_balanced_across_classes(self):
        """It stands for the population the model serves, not any contributor's
        local skew, so it must not favour whichever classes happen to be
        well represented among the selected contributors."""
        _, y = make_global_test(seed=0)
        counts = np.bincount(y, minlength=N_CLASSES)
        assert counts.min() == counts.max()

    def test_identical_for_every_subset_of_contributors(self):
        """Models trained on different subsets have to be scored on the same
        distribution, or utility is not comparable between them."""
        a = make_global_test(seed=4)
        b = make_global_test(seed=4)
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])

    def test_drawn_from_the_same_distribution_as_the_clients(self):
        """Same class centroids as the partitions, otherwise the held-out set
        measures something the clients were never trained for."""
        means = class_means(seed=0)
        x, y = make_global_test(seed=0)
        for cls in range(N_CLASSES):
            centroid = x[y == cls].mean(axis=0)
            assert np.linalg.norm(centroid - means[cls]) < 0.5

    def test_different_seeds_give_different_sets(self):
        assert not np.array_equal(make_global_test(seed=1)[0], make_global_test(seed=2)[0])


class TestTrainingLoop:
    def test_training_reduces_loss_and_learns(self):
        part = load_partition(0, n_clients=2, alpha=100.0, seed=0)
        trainloader, testloader = to_loaders(part)
        net = Net()

        _, acc_before = evaluate(net, testloader)
        for _ in range(15):
            train(net, trainloader, epochs=1)
        _, acc_after = evaluate(net, testloader)

        assert acc_after > acc_before
        # The synthetic classes are well separated; anything near chance means
        # the pipeline is broken rather than merely undertrained.
        assert acc_after > 0.5

    def test_evaluate_returns_sane_values(self):
        part = load_partition(0, n_clients=2, seed=0)
        _, testloader = to_loaders(part)
        loss, acc = evaluate(Net(), testloader)
        assert loss >= 0.0
        assert 0.0 <= acc <= 1.0
