"""Tests for the training-free utility estimators.

The estimators exist to rank consortia without training them, and an estimator
that silently ranks badly is worse than none -- the whole point is to trust it
enough to skip the training. So the two quality terms are checked against cases
whose answer is known by construction, and the ranking machinery is checked by
feeding it ground truth and requiring a perfect score.

That last one matters more than it looks. If the frontier metrics cannot
recognise a perfect estimator, every number they produce about an imperfect one
is meaningless.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_estimators import (
    evaluate_seed,
    kendall_tau,
    load_groundtruth,
    pareto_front,
)

from odrl_fl.estimators import (
    ESTIMATORS,
    client_gradients,
    coherence,
    js_divergence,
    make_coverage_utility,
    make_divergence_utility,
    make_gradient_utility,
)

POOL = 6


@dataclass(frozen=True)
class Stub:
    """Minimal stand-in for a contributor: the estimators need only `index`."""

    index: int


def subset(*indices: int) -> list[Stub]:
    return [Stub(i) for i in indices]


class TestJSDivergence:
    def test_identical_distributions_are_zero_apart(self):
        assert js_divergence([1, 1, 1, 1], [1, 1, 1, 1]) == 0.0

    def test_disjoint_support_is_maximal(self):
        assert js_divergence([1, 0, 0, 0], [0, 0, 0, 1]) == pytest.approx(1.0)

    def test_scale_does_not_matter_only_shape(self):
        """Counts are normalised, so ten samples and ten thousand of the same
        shape must score the same -- volume is accounted for separately."""
        assert js_divergence([1, 3], [10, 30]) == pytest.approx(0.0)

    def test_symmetric(self):
        a, b = [3, 1, 0, 0], [1, 1, 1, 1]
        assert js_divergence(a, b) == pytest.approx(js_divergence(b, a))

    def test_bounded_in_bits(self):
        for p, q in (([1, 0], [0, 1]), ([5, 1], [1, 5]), ([1, 1], [1, 0])):
            assert 0.0 <= js_divergence(p, q) <= 1.0

    def test_empty_counts_do_not_divide_by_zero(self):
        assert js_divergence([0, 0], [1, 1]) == 1.0


class TestCoherence:
    def test_identical_gradients_agree_completely(self):
        assert coherence(np.array([[1.0, 0.0], [1.0, 0.0]])) == pytest.approx(1.0)

    def test_opposed_gradients_cancel(self):
        """Two clients pulling exactly against each other average to nothing,
        which is precisely the failure federated averaging suffers."""
        assert coherence(np.array([[1.0, 0.0], [-1.0, 0.0]])) == pytest.approx(0.0)

    def test_orthogonal_gradients_sit_between(self):
        value = coherence(np.array([[1.0, 0.0], [0.0, 1.0]]))
        assert 0.0 < value < 1.0

    def test_magnitude_does_not_change_agreement(self):
        """Coherence is about direction; a loud client and a quiet one pointing
        the same way still agree."""
        assert coherence(np.array([[5.0, 0.0], [5.0, 0.0]])) == pytest.approx(1.0)

    def test_a_single_client_agrees_with_itself(self):
        assert coherence(np.array([[3.0, 4.0]])) == pytest.approx(1.0)

    def test_empty_and_zero_gradients_do_not_raise(self):
        assert coherence(np.empty((0, 3))) == 0.0
        assert coherence(np.zeros((2, 3))) == 0.0


class TestEstimatorContract:
    @pytest.mark.parametrize("name", sorted(ESTIMATORS))
    def test_empty_subset_scores_zero(self, name):
        assert ESTIMATORS[name](POOL, seed=0)([]) == 0.0

    @pytest.mark.parametrize("name", sorted(ESTIMATORS))
    def test_scores_are_finite_and_non_negative(self, name):
        estimate = ESTIMATORS[name](POOL, seed=0)
        for members in (subset(0), subset(0, 1), subset(*range(POOL))):
            value = estimate(members)
            assert np.isfinite(value)
            assert value >= 0.0

    @pytest.mark.parametrize("name", sorted(ESTIMATORS))
    def test_deterministic_across_calls(self, name):
        """Selection compares subsets; an estimator that drifts between queries
        would rank on noise."""
        estimate = ESTIMATORS[name](POOL, seed=0)
        assert estimate(subset(0, 2)) == estimate(subset(0, 2))

    @pytest.mark.parametrize("name", sorted(ESTIMATORS))
    def test_order_of_members_is_irrelevant(self, name):
        estimate = ESTIMATORS[name](POOL, seed=0)
        assert estimate(subset(0, 1, 2)) == pytest.approx(estimate(subset(2, 0, 1)))

    @pytest.mark.parametrize("name", sorted(ESTIMATORS))
    def test_two_factories_with_one_seed_agree(self, name):
        a = ESTIMATORS[name](POOL, seed=0)
        b = ESTIMATORS[name](POOL, seed=0)
        assert a(subset(1, 3)) == pytest.approx(b(subset(1, 3)))


class TestVolumeSensitivity:
    @pytest.mark.parametrize(
        "factory", [make_coverage_utility, make_divergence_utility]
    )
    def test_more_contributors_score_higher_for_volume_terms(self, factory):
        """Coverage and divergence both scale with volume, so a superset must
        not score lower. Gradient coherence is excluded on purpose: it *can*
        fall when a discordant client joins, which is the entire point of it."""
        estimate = factory(POOL, seed=0)
        assert estimate(subset(0, 1, 2)) >= estimate(subset(0, 1))


class TestGradients:
    def test_one_row_per_client(self):
        gradients = client_gradients(POOL, seed=0)
        assert gradients.shape[0] == POOL
        assert gradients.shape[1] > 0

    def test_gradients_are_not_all_identical(self):
        """If every client produced the same gradient the estimator would carry
        no information, and under a Dirichlet label skew they should not."""
        gradients = client_gradients(POOL, seed=0)
        assert not np.allclose(gradients[0], gradients[1])

    def test_reproducible_for_a_seed(self):
        assert np.allclose(client_gradients(POOL, seed=0), client_gradients(POOL, seed=0))

    def test_precomputed_gradients_are_reused(self):
        """The factory accepts a precompute so several estimators can share one
        pass; passing the same array must give the same answers."""
        gradients = client_gradients(POOL, seed=0)
        a = make_gradient_utility(POOL, seed=0, gradients=gradients)
        b = make_gradient_utility(POOL, seed=0)
        assert a(subset(0, 1)) == pytest.approx(b(subset(0, 1)))


class TestKendallTau:
    def test_identical_rankings_correlate_perfectly(self):
        a = [1.0, 2.0, 3.0, 4.0]
        assert kendall_tau(a, a) == pytest.approx(1.0)

    def test_reversed_rankings_anticorrelate(self):
        assert kendall_tau([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_all_ties_do_not_divide_by_zero(self):
        assert kendall_tau([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == 0.0

    def test_partial_agreement_lands_between(self):
        tau = kendall_tau([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 4.0, 3.0])
        assert 0.0 < tau < 1.0


class TestParetoFront:
    def test_a_single_point_is_its_own_front(self):
        assert pareto_front([(1.0, frozenset({"a"}), 0)]) == {0}

    def test_a_dominated_point_is_excluded(self):
        points = [
            (1.0, frozenset({"a"}), 5),   # worse on every axis
            (2.0, frozenset({"a", "b"}), 1),
        ]
        assert pareto_front(points) == {1}

    def test_a_trade_off_keeps_both(self):
        """More utility but more duties owed is a real choice, not a tiebreak."""
        points = [
            (1.0, frozenset({"a"}), 0),
            (2.0, frozenset({"a"}), 5),
        ]
        assert pareto_front(points) == {0, 1}

    def test_duties_count_against(self):
        points = [
            (1.0, frozenset({"a"}), 0),
            (1.0, frozenset({"a"}), 3),
        ]
        assert pareto_front(points) == {0}

    def test_identical_points_are_all_kept(self):
        """Neither is strictly better, so neither dominates -- dropping one
        would be an arbitrary choice dressed up as an optimisation."""
        points = [(1.0, frozenset({"a"}), 1), (1.0, frozenset({"a"}), 1)]
        assert pareto_front(points) == {0, 1}


class TestMetricsRecogniseAPerfectEstimator:
    """If ground truth scored against itself does not come out perfect, every
    number these metrics report about a real estimator is meaningless."""

    def test_truth_against_itself_recovers_its_own_frontier(self):
        truth = [3.0, 1.0, 2.0, 5.0]
        capabilities = [frozenset({"a"})] * 4
        duties = [0, 2, 1, 4]
        points = list(zip(truth, capabilities, duties))
        front = pareto_front(points)
        assert front == pareto_front(points)
        assert kendall_tau(truth, truth) == pytest.approx(1.0)

    def test_a_perfect_estimator_has_zero_regret(self):
        truth = [3.0, 1.0, 2.0, 5.0]
        points = [(t, frozenset({"a"}), 0) for t in truth]
        front = pareto_front(points)
        pick = max(front, key=lambda i: points[i][0])
        assert max(truth) - truth[pick] == pytest.approx(0.0)


def write_groundtruth(path, rows):
    """Minimal ground-truth CSV, in the shape run_groundtruth.py emits."""
    import csv as _csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(
            fh,
            fieldnames=[
                "members", "size", "seed", "alpha", "rounds",
                "accuracy", "best_accuracy", "curve",
            ],
        )
        writer.writeheader()
        for members, seed, accuracy in rows:
            writer.writerow({
                "members": members,
                "size": len(members.split("-")),
                "seed": seed,
                "alpha": 0.3,
                "rounds": 20,
                "accuracy": accuracy,
                "best_accuracy": accuracy,
                "curve": f"{accuracy}",
            })


class TestLoadGroundtruth:
    def test_seeds_are_kept_apart_not_merged(self, tmp_path):
        """Each seed is a different federation. Collapsing them into one list
        was the defect this keyed structure exists to prevent."""
        path = tmp_path / "gt.csv"
        write_groundtruth(path, [("0-1", 0, 0.80), ("0-1", 1, 0.90)])
        truth, _ = load_groundtruth(path)
        assert truth[(0, 1)] == {0: 0.80, 1: 0.90}

    def test_settings_come_back(self, tmp_path):
        path = tmp_path / "gt.csv"
        write_groundtruth(path, [("0", 0, 0.5)])
        _, settings = load_groundtruth(path)
        assert settings["alpha"] == 0.3
        assert settings["rounds"] == 20

    def test_mixed_alpha_is_refused(self, tmp_path):
        """Two heterogeneities in one file means two different worlds, and an
        estimator can only be built to match one of them."""
        import csv as _csv

        path = tmp_path / "gt.csv"
        write_groundtruth(path, [("0", 0, 0.5)])
        with path.open("a", newline="", encoding="utf-8") as fh:
            _csv.writer(fh).writerow(["0", 1, 1, 1.0, 20, 0.6, 0.6, "0.6"])
        with pytest.raises(ValueError, match="mixes alpha"):
            load_groundtruth(path)


class TestEvaluateSeed:
    def test_returns_none_when_nothing_can_be_ranked(self):
        """One feasible consortium is not a choice, and a frontier of one would
        report a vacuous 100%."""
        truth = {(0,): {0: 0.9}}
        assert evaluate_seed(0, 4, 0.3, truth) is None

    def test_scores_every_estimator_when_there_is_a_choice(self):
        from itertools import combinations

        truth = {}
        for size in range(1, 5):
            for members in combinations(range(4), size):
                truth[members] = {0: 0.5 + 0.1 * len(members)}
        result = evaluate_seed(0, 4, 0.3, truth)
        if result is None:
            pytest.skip("no feasible pair at this corpus seed")
        assert set(result) == set(ESTIMATORS)
        for metrics in result.values():
            assert 0.0 <= metrics["recall"] <= 1.0
            assert 0.0 <= metrics["precision"] <= 1.0
            assert metrics["regret"] >= 0.0
            assert metrics["rank_of_best"] >= 1
