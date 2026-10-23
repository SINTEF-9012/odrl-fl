"""Two-stage selection: the shortlist, its cost, and its regret.

Hand-built curves throughout, so a failure here is a defect in the selection
logic and never a change in how the task happens to train.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from odrl_fl.shortlist import regret, rounds_spent, shortlist, sweep


class TestShortlist:
    def test_takes_the_highest_scoring(self):
        scores = {(0,): 0.1, (1,): 0.9, (2,): 0.5}
        assert shortlist(scores, 2) == [(1,), (2,)]

    def test_best_comes_first(self):
        assert shortlist({(0,): 0.2, (1,): 0.8}, 2) == [(1,), (0,)]

    def test_a_shortlist_longer_than_the_field_is_the_field(self):
        assert len(shortlist({(0,): 0.1, (1,): 0.2}, 10)) == 2

    def test_ties_break_on_members_not_insertion_order(self):
        """A campaign split across array tasks must agree with itself."""
        forward = shortlist({(0,): 0.5, (1,): 0.5, (2,): 0.5}, 2)
        backward = shortlist({(2,): 0.5, (1,): 0.5, (0,): 0.5}, 2)
        assert forward == backward == [(0,), (1,)]

    def test_an_empty_shortlist_is_rejected(self):
        with pytest.raises(ValueError):
            shortlist({(0,): 1.0}, 0)


class TestCost:
    def test_every_candidate_is_explored_and_only_the_shortlist_finished(self):
        # 100 candidates x 8 rounds, then 3 of them for the remaining 12.
        assert rounds_spent(100, 8, 3, 20) == 100 * 8 + 3 * 12

    def test_exploring_the_whole_budget_is_the_exhaustive_cost(self):
        assert rounds_spent(50, 20, 3, 20) == 50 * 20

    def test_exploring_past_the_budget_is_rejected(self):
        with pytest.raises(ValueError):
            rounds_spent(10, 21, 1, 20)

    def test_two_stages_are_cheaper_than_one(self):
        """The claim the paper makes, as an inequality rather than a number."""
        assert rounds_spent(500, 8, 3, 20) < rounds_spent(500, 20, 3, 20)


class TestRegret:
    def test_zero_when_the_best_is_shortlisted(self):
        truth = {(0,): 0.7, (1,): 0.9}
        assert regret([(1,)], truth) == pytest.approx(0.0)

    def test_the_shortlist_is_credited_with_its_best_member(self):
        """A deployment that finishes k candidates keeps the best of them."""
        truth = {(0,): 0.7, (1,): 0.9, (2,): 0.5}
        assert regret([(2,), (1,)], truth) == pytest.approx(0.0)

    def test_measures_the_gap_to_the_optimum(self):
        truth = {(0,): 0.7, (1,): 0.9}
        assert regret([(0,)], truth) == pytest.approx(0.2)

    def test_an_unknown_candidate_is_an_error_not_a_zero(self):
        with pytest.raises(KeyError):
            regret([(9,)], {(0,): 0.5})


class TestSweep:
    #: Three candidates. (2,) leads early and loses; (0,) is slow and wins.
    #: That ordering is the whole point: a selector that cannot be fooled by
    #: an early leader is not being tested by this fixture.
    CURVES: ClassVar[dict[tuple[int, ...], list[float]]] = {
        (0,): [0.10, 0.30, 0.60, 0.90],
        (1,): [0.20, 0.40, 0.55, 0.70],
        (2,): [0.50, 0.50, 0.50, 0.50],
    }
    FINAL: ClassVar[dict[tuple[int, ...], float]] = {
        m: c[-1] for m, c in CURVES.items()
    }

    def test_an_early_leader_that_loses_is_the_case_this_measures(self):
        """Round 1 picks the wrong consortium; the full budget does not."""
        early = next(p for p in sweep(self.CURVES, self.FINAL, [1], [1]))
        late = next(p for p in sweep(self.CURVES, self.FINAL, [4], [1]))
        assert early.regret == pytest.approx(0.40)
        assert late.regret == pytest.approx(0.0)

    def test_a_wider_shortlist_recovers_it(self):
        wide = next(p for p in sweep(self.CURVES, self.FINAL, [1], [3]))
        assert wide.regret == pytest.approx(0.0)

    def test_cost_is_reported_as_a_fraction_of_exhaustive(self):
        p = next(p for p in sweep(self.CURVES, self.FINAL, [2], [1]))
        assert p.rounds == 3 * 2 + 1 * 2
        assert p.fraction == pytest.approx(8 / 12)

    def test_exploring_the_whole_budget_costs_the_whole_budget(self):
        p = next(p for p in sweep(self.CURVES, self.FINAL, [4], [1]))
        assert p.fraction == pytest.approx(1.0)

    def test_selecting_and_scoring_on_one_run_is_flagged_as_a_bound(self):
        """The winner's-curse caveat is recorded, not left to the reader."""
        assert all(
            not p.holdout for p in sweep(self.CURVES, self.FINAL, [1, 2], [1])
        )

    def test_separate_replicates_are_flagged_as_a_measurement(self):
        other = {m: v + 0.01 for m, v in self.FINAL.items()}
        assert all(p.holdout for p in sweep(self.CURVES, other, [1, 2], [1]))

    def test_candidates_outside_the_ground_truth_are_dropped(self):
        """The policy filter leaves a subset; scoring must follow it."""
        feasible = {(0,): self.FINAL[(0,)], (1,): self.FINAL[(1,)]}
        p = next(p for p in sweep(self.CURVES, feasible, [1], [1]))
        assert p.rounds == rounds_spent(2, 1, 1, 4)

    def test_shortlists_wider_than_the_field_are_skipped(self):
        points = sweep(self.CURVES, self.FINAL, [1], [1, 3, 5])
        assert [p.k for p in points] == [1, 3]

    def test_curves_of_different_lengths_are_rejected(self):
        bad = {**self.CURVES, (3,): [0.1, 0.2]}
        with pytest.raises(ValueError, match="differ in length"):
            sweep(bad, {**self.FINAL, (3,): 0.2}, [1], [1])

    def test_no_shared_candidates_is_an_error(self):
        with pytest.raises(ValueError, match="share no candidates"):
            sweep(self.CURVES, {(9,): 0.5}, [1], [1])
