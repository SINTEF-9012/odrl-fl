"""Tests for the in-process FedAvg loop.

Determinism and subset independence matter most here: the evaluation compares
models trained on different contributor subsets, and that comparison is only
sound if the difference comes from the data rather than from run-to-run noise.
"""

from __future__ import annotations

import pytest
import torch

from odrl_fl.fedavg import History, federated_average, run_fedavg
from odrl_fl.task import Net


class TestAggregation:
    def test_average_of_identical_models_is_unchanged(self):
        net = Net()
        sd = net.state_dict()
        out = federated_average([sd, sd, sd], [1.0, 1.0, 1.0])
        for key in sd:
            assert torch.allclose(out[key], sd[key])

    def test_equal_weights_give_the_arithmetic_mean(self):
        a, b = Net().state_dict(), Net().state_dict()
        out = federated_average([a, b], [1.0, 1.0])
        for key in a:
            assert torch.allclose(out[key], (a[key] + b[key]) / 2, atol=1e-6)

    def test_weights_are_normalised(self):
        """Callers pass raw example counts, not fractions."""
        a, b = Net().state_dict(), Net().state_dict()
        assert all(
            torch.allclose(x, y, atol=1e-6)
            for x, y in zip(
                federated_average([a, b], [400.0, 400.0]).values(),
                federated_average([a, b], [0.5, 0.5]).values(),
            )
        )

    def test_a_dominant_weight_dominates_the_result(self):
        a, b = Net().state_dict(), Net().state_dict()
        out = federated_average([a, b], [999.0, 1.0])
        for key in a:
            assert torch.allclose(out[key], a[key], atol=1e-2)

    def test_rejects_empty_and_degenerate_input(self):
        with pytest.raises(ValueError):
            federated_average([], [])
        with pytest.raises(ValueError):
            federated_average([Net().state_dict()], [0.0])


class TestTrainingRun:
    def test_it_learns(self):
        history = run_fedavg(
            contributors=[0, 1, 2], n_clients=4, rounds=8, alpha=1.0, seed=0
        )
        assert len(history.accuracy) == 8
        assert history.best_accuracy > 0.6
        assert history.accuracy[-1] > history.accuracy[0]

    def test_deterministic(self):
        """Same seed, same subset, identical result -- to the last decimal."""
        kwargs = {"contributors": [0, 2], "n_clients": 4, "rounds": 3, "seed": 5}
        a = run_fedavg(**kwargs)
        b = run_fedavg(**kwargs)
        assert a.accuracy == b.accuracy
        assert a.loss == b.loss

    def test_different_subsets_give_different_models(self):
        a = run_fedavg(contributors=[0, 1], n_clients=6, rounds=3, alpha=0.1, seed=0)
        b = run_fedavg(contributors=[3, 4], n_clients=6, rounds=3, alpha=0.1, seed=0)
        assert a.accuracy != b.accuracy

    def test_population_size_does_not_change_a_subset_result(self):
        """Selecting {0,1} must give the same model whether the population is
        6 or 20 -- otherwise subset comparisons are confounded."""
        a = run_fedavg(contributors=[0, 1], n_clients=6, rounds=3, seed=1)
        b = run_fedavg(contributors=[0, 1], n_clients=20, rounds=3, seed=1)
        assert a.accuracy == b.accuracy

    def test_single_contributor_is_allowed(self):
        history = run_fedavg(contributors=[0], n_clients=4, rounds=2, seed=0)
        assert len(history.accuracy) == 2

    def test_contributors_are_deduplicated_and_sorted(self):
        history = run_fedavg(contributors=[2, 0, 2], n_clients=4, rounds=1, seed=0)
        assert history.contributors == [0, 2]

    def test_empty_contributor_set_is_rejected(self):
        with pytest.raises(ValueError):
            run_fedavg(contributors=[], n_clients=4)

    def test_there_is_a_utility_signal_to_trade_against(self):
        """**The property the whole evaluation rests on.**

        Federating more contributors must measurably beat federating one. This
        is the (approximate, empirical) monotone utility axis that anti-monotone
        capability is traded against; without a gap here there is no Pareto
        frontier to study and no paper.

        A guard, not a demonstration: if someone retunes CLASS_SEPARATION and
        the task saturates, every contributor subset scores the same and the
        evaluation silently becomes vacuous.
        """
        one = run_fedavg(contributors=[0], n_clients=6, rounds=8, alpha=1.0, seed=0)
        all_of_them = run_fedavg(
            contributors=list(range(6)), n_clients=6, rounds=8, alpha=1.0, seed=0
        )
        assert all_of_them.final_accuracy - one.final_accuracy > 0.15

    def test_iid_data_removes_the_signal(self):
        """The control. Under IID every subset sees the same distribution, so
        federating adds nothing and the task saturates -- which is why the
        default alpha is moderate rather than large."""
        one = run_fedavg(contributors=[0], n_clients=6, rounds=8, alpha=100.0, seed=0)
        many = run_fedavg(
            contributors=list(range(6)), n_clients=6, rounds=8, alpha=100.0, seed=0
        )
        assert abs(many.final_accuracy - one.final_accuracy) < 0.05


class TestHistory:
    def test_reports_final_and_best(self):
        h = History(contributors=[0], accuracy=[0.1, 0.9, 0.5], loss=[3.0, 1.0, 2.0])
        assert h.final_accuracy == 0.5
        assert h.best_accuracy == 0.9

    def test_empty_history_is_safe(self):
        h = History(contributors=[])
        assert h.final_accuracy == 0.0
        assert h.best_accuracy == 0.0
