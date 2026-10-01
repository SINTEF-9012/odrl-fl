"""Tests for the real TCGA-BRCA cohort.

Skipped when the data is absent rather than downloading during a test run: the
cohort belongs to the GDC, and a test suite should not reach across the network
without being asked. `python -c "from odrl_fl.tcga import fetch; fetch()"`
retrieves it.

The tests that matter are the survival ones. Cox partial likelihood and the
concordance index are easy to get subtly wrong in ways that still produce
plausible-looking numbers, so both are checked against cases whose answer is
known by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from odrl_fl.odrl import Action, capability, compose
from odrl_fl.tcga import (
    DATA_DIR,
    N_FEATURES,
    CoxModel,
    _site_seed,
    c_index,
    cox_loss,
    evaluate_cox,
    gdc_open_access_policy,
    load_sites,
    pooled,
    split,
    standardise,
    train_cox,
)

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "brca.csv").exists(),
    reason="TCGA-BRCA not fetched; run odrl_fl.tcga.fetch()",
)


@pytest.fixture(scope="module")
def sites():
    return standardise(load_sites(min_patients=20))


class TestCohort:
    def test_the_cohort_loads_with_the_expected_shape(self, sites):
        assert len(sites) >= 10
        for site in sites:
            assert site.x.shape[1] == N_FEATURES
            assert site.y.shape[1] == 2
            assert len(site.x) == len(site.y)

    def test_institutions_are_real_and_named(self, sites):
        names = {s.name for s in sites}
        assert "University of Pittsburgh" in names
        assert "Walter Reed" in names

    def test_sites_differ_in_size_by_an_order_of_magnitude(self, sites):
        """The volume imbalance is the data's own, not a sampling choice."""
        counts = [s.n_patients for s in sites]
        assert max(counts) / min(counts) > 5

    def test_events_are_rare(self, sites):
        """13.8% -- which is why the metric is a C-index, not accuracy."""
        _x, y = pooled(sites)
        rate = y[:, 0].mean()
        assert 0.05 < rate < 0.25

    def test_a_split_keeps_events_on_both_sides(self, sites):
        """Stratified, because an unstratified small site can yield a test fold
        with no events at all, making the C-index undefined."""
        for site in sites:
            if site.n_events < 2:
                continue
            (_, ytr), (_, yte) = split(site, seed=0)
            assert ytr[:, 0].sum() >= 1
            assert yte[:, 0].sum() >= 1

    def test_standardising_does_not_change_the_outcomes(self, sites):
        raw = load_sites(min_patients=20)
        for before, after in zip(raw, sites):
            assert np.array_equal(before.y, after.y)


class TestConcordance:
    def test_perfect_ranking_scores_one(self):
        y = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=np.float32)
        assert c_index(np.array([3.0, 2.0, 1.0]), y) == 1.0

    def test_reversed_ranking_scores_zero(self):
        y = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=np.float32)
        assert c_index(np.array([1.0, 2.0, 3.0]), y) == 0.0

    def test_ties_score_a_half(self):
        y = np.array([[1.0, 1.0], [1.0, 2.0]], dtype=np.float32)
        assert c_index(np.array([1.0, 1.0]), y) == 0.5

    def test_censored_patients_still_contribute_as_comparators(self):
        """A censored patient who outlived an event is a valid comparison."""
        y = np.array([[1.0, 1.0], [0.0, 5.0]], dtype=np.float32)
        assert c_index(np.array([2.0, 1.0]), y) == 1.0

    def test_no_comparable_pairs_yields_a_half(self):
        """Everyone censored: nothing to rank, so no information."""
        y = np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float32)
        assert c_index(np.array([1.0, 2.0]), y) == 0.5


class TestCoxLoss:
    def test_the_loss_rewards_ranking_the_failure_highest(self):
        y = torch.tensor([[1.0, 1.0], [0.0, 5.0]])
        good = cox_loss(torch.tensor([2.0, 0.0]), y)
        bad = cox_loss(torch.tensor([0.0, 2.0]), y)
        assert good < bad

    def test_a_cohort_with_no_events_contributes_nothing(self):
        """Not a crash and not a NaN -- some small sites have zero events."""
        y = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        loss = cox_loss(torch.tensor([1.0, 2.0]), y)
        assert torch.isfinite(loss)
        assert float(loss) == 0.0

    def test_the_loss_is_finite_on_real_data(self, sites):
        x, y = pooled(sites)
        loss = cox_loss(CoxModel()(torch.from_numpy(x)), torch.from_numpy(y))
        assert torch.isfinite(loss)


class TestTraining:
    def test_training_improves_concordance_on_pooled_data(self, sites):
        """The signal check. If this fails there is no utility axis at all."""
        torch.manual_seed(0)
        trains, tests = zip(*(split(s, seed=0) for s in sites))
        xtr = np.concatenate([t[0] for t in trains])
        ytr = np.concatenate([t[1] for t in trains])
        xte = np.concatenate([t[0] for t in tests])
        yte = np.concatenate([t[1] for t in tests])

        model = CoxModel()
        before = evaluate_cox(model, xte, yte)
        train_cox(model, xtr, ytr, epochs=200)
        after = evaluate_cox(model, xte, yte)

        assert after > before
        assert after > 0.6

    def test_training_is_deterministic_under_a_seed(self, sites):
        """Model init draws from the global generator, so a comparison between
        two consortia is only meaningful if this holds."""
        (xtr, ytr), (xte, yte) = split(sites[0], seed=0)
        scores = []
        for _ in range(2):
            torch.manual_seed(7)
            model = CoxModel()
            train_cox(model, xtr, ytr, epochs=20)
            scores.append(evaluate_cox(model, xte, yte))
        assert scores[0] == scores[1]


class TestRealTerms:
    def test_the_open_access_policy_grants_everything_and_forbids_one_thing(self):
        policy = gdc_open_access_policy()
        assert capability(policy) == {
            Action.USE,
            Action.AGGREGATE,
            Action.DERIVE,
            Action.DISTRIBUTE,
        }
        assert {p.action for p in policy.prohibitions} == {Action.IDENTIFY}

    def test_composing_the_real_terms_loses_nothing(self):
        """The honest negative result: open-access terms do not erode.

        Erosion needs heterogeneous conditions, and this tier has none. That is
        why the phenomenon lives in controlled access, and why it cannot be
        demonstrated on data anyone can download.
        """
        policies = [
            gdc_open_access_policy(uid=f"urn:policy:site-{i}") for i in range(16)
        ]
        assert compose(policies).capability == capability(policies[0])

    def test_the_re_identification_ban_has_no_standard_vocabulary_term(self):
        """`identify` is a profile extension: neither ODRL core nor DUO defines
        it, though it is the only enforceable term on this data."""
        assert Action.IDENTIFY.value == "identify"


class TestSplitIsReproducible:
    """A split derived from the builtin `hash` is reproducible within a process
    and never between processes, because Python randomises string hashing per
    interpreter. That shipped for three increments: a test failed intermittently,
    and every committed TCGA figure came from a fold nobody could reconstruct.
    """

    def test_the_site_seed_is_a_fixed_number_forever(self):
        """Hardcoded on purpose. If someone reaches for `hash()` again this
        fails under almost every PYTHONHASHSEED, which is the whole point."""
        assert _site_seed("Mayo Clinic") == 1675962478

    def test_different_sites_get_different_seeds(self):
        """Stability must not collapse into every site sharing one fold."""
        assert _site_seed("Walter Reed") != _site_seed("Roswell Park")

    def test_the_seed_does_not_depend_on_process_state(self):
        """The builtin hash varies per interpreter; this must not."""
        assert _site_seed("Walter Reed") == 3045256978

    def test_splitting_twice_gives_the_same_fold(self, sites):
        first = split(sites[0], seed=0)
        second = split(sites[0], seed=0)
        for a, b in zip(first, second):
            assert np.array_equal(a[0], b[0])
            assert np.array_equal(a[1], b[1])

    def test_a_different_seed_gives_a_different_fold(self, sites):
        """Determinism must not have been bought by ignoring the seed."""
        (xtr_a, _), _ = split(sites[0], seed=0)
        (xtr_b, _), _ = split(sites[0], seed=1)
        assert not np.array_equal(xtr_a, xtr_b)

    def test_the_fold_itself_is_pinned(self, sites):
        """Golden value over the whole derivation, not just its seed: catches a
        change to the stratification or the rounding as well as to the hash."""
        (xtr, ytr), (_, yte) = split(sites[0], seed=0)
        assert len(ytr) == 112
        assert len(yte) == 38
        assert xtr[0][0] == pytest.approx(-1.398966, abs=1e-5)
