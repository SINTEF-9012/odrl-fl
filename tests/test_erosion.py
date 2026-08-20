"""Tests for the erosion study.

The measurements decide whether the project continues, so the metrics
themselves need pinning: a `purpose_retention` that silently returned 1.0, or a
bootstrap interval that did not bracket its mean, would produce a confident
verdict from nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from odrl_fl.corpus import generate_corpus
from odrl_fl.erosion import (
    bootstrap_ci,
    breadth,
    evaluate_criteria,
    measure,
    purpose_retention,
    purpose_universe,
    run_study,
    sample_subsets,
    summarise,
)
from odrl_fl.odrl import compose
from odrl_fl.odrl.duo import translate


def offer(codes, uid, **params):
    policy, _ = translate(codes, uid=uid, params=params or None)
    return policy


GRU = ["DUO:0000042"]
HMB = ["DUO:0000006"]
DS = ["DUO:0000007"]
CC = ["DUO:0000043"]
NRES = ["DUO:0000004"]


class TestBreadth:
    def test_universe_contains_only_leaves(self):
        leaves = purpose_universe()
        assert "research" not in leaves, "an interior node would double-count"
        assert "clinical-research" in leaves
        assert len(leaves) > 5

    def test_broader_terms_cover_more(self):
        assert (
            breadth(frozenset({"any-purpose"}))
            > breadth(frozenset({"research"}))
            > breadth(frozenset({"biomedical-research"}))
        )

    def test_a_leaf_covers_itself(self):
        assert breadth(frozenset({"ds-cancer"})) == 1

    def test_unknown_terms_contribute_nothing(self):
        assert breadth(frozenset({"not-a-purpose"})) == 0


class TestPurposeRetention:
    def test_unrestricted_purpose_retains_everything(self):
        assert purpose_retention(compose([offer(NRES, "urn:n")]).agreement) == 1.0

    def test_narrower_permissions_retain_less(self):
        broad = purpose_retention(compose([offer(GRU, "urn:g")]).agreement)
        medium = purpose_retention(compose([offer(HMB, "urn:h")]).agreement)
        narrow = purpose_retention(
            compose([offer(DS, "urn:d", disease="ds-cancer")]).agreement
        )
        assert 1.0 > broad > medium > narrow > 0.0

    def test_no_surviving_permission_retains_nothing(self):
        """GRU and CC do not overlap, so nothing survives."""
        result = compose([offer(GRU, "urn:g"), offer(CC, "urn:c")])
        assert not result.capability
        assert purpose_retention(result.agreement) == 0.0

    def test_composition_never_widens_retention(self):
        """The continuous counterpart of anti-monotonicity."""
        alone = purpose_retention(compose([offer(GRU, "urn:g")]).agreement)
        with_narrower = purpose_retention(
            compose([offer(GRU, "urn:g"), offer(HMB, "urn:h")]).agreement
        )
        assert with_narrower <= alone


class TestConflictAttribution:
    def test_purpose_clash_is_attributed_to_purpose(self):
        result = compose([offer(GRU, "urn:g"), offer(CC, "urn:c")])
        assert set(result.conflicts.values()) == {"purpose"}

    def test_region_clash_is_attributed_to_spatial(self):
        eu = offer(GRU + ["DUO:0000022"], "urn:eu", region="EU")
        us = offer(GRU + ["DUO:0000022"], "urn:us", region="US")
        assert set(compose([eu, us]).conflicts.values()) == {"spatial"}

    def test_different_diseases_conflict(self):
        """DS is disease-parameterised: consent for cancer research does not
        cover cardiovascular research."""
        a = offer(DS, "urn:a", disease="ds-cancer")
        b = offer(DS, "urn:b", disease="ds-neurological")
        result = compose([a, b])
        assert not result.capability
        assert set(result.conflicts.values()) == {"purpose"}

    def test_compatible_policies_record_no_conflict(self):
        assert not compose([offer(GRU, "urn:a"), offer(GRU, "urn:b")]).conflicts


class TestBootstrap:
    def test_interval_brackets_the_mean(self):
        rng = np.random.default_rng(0)
        values = list(rng.normal(5.0, 1.0, size=200))
        low, high = bootstrap_ci(values, rng=rng)
        assert low < float(np.mean(values)) < high

    def test_interval_is_tight_for_constant_data(self):
        low, high = bootstrap_ci([3.0] * 50)
        assert low == high == 3.0

    def test_handles_degenerate_input(self):
        assert bootstrap_ci([7.0]) == (7.0, 7.0)
        low, high = bootstrap_ci([])
        assert np.isnan(low) and np.isnan(high)

    def test_wider_data_gives_a_wider_interval(self):
        rng = np.random.default_rng(1)
        narrow = bootstrap_ci(list(rng.normal(0, 0.1, 100)), rng=rng)
        wide = bootstrap_ci(list(rng.normal(0, 5.0, 100)), rng=rng)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


class TestMeasurement:
    def test_identical_contributors_lose_nothing(self):
        corpus = generate_corpus(6, heterogeneity=0.0, seed=0)
        result = measure(corpus, heterogeneity=0.0, seed=0)
        assert result.retains_any
        assert result.capability_count == 4
        assert not result.loss_reasons

    def test_duties_accumulate_as_capability_falls(self):
        """Permissions intersect while duties union, so a growing federation
        loses capability and gains obligation at the same time."""
        corpus = generate_corpus(30, heterogeneity=1.0, seed=0)
        small = measure(corpus[:2], heterogeneity=1.0, seed=0)
        large = measure(corpus[:20], heterogeneity=1.0, seed=0)
        assert large.duty_count >= small.duty_count
        assert large.capability_count <= small.capability_count

    def test_sample_subsets_respects_size_and_bounds(self):
        corpus = generate_corpus(10, seed=0)
        rng = np.random.default_rng(0)
        subsets = sample_subsets(corpus, 4, 5, rng)
        assert len(subsets) == 5
        for subset in subsets:
            assert len(subset) == 4
            assert len({c.index for c in subset}) == 4

    def test_sample_subsets_declines_oversized_requests(self):
        corpus = generate_corpus(3, seed=0)
        assert sample_subsets(corpus, 5, 4, np.random.default_rng(0)) == []


class TestStudy:
    @pytest.fixture(scope="class")
    @classmethod
    def summaries(cls):
        return run_study(
            pool_size=12,
            sizes=(1, 5, 10),
            heterogeneities=(0.0, 1.0),
            corpus_seeds=(0, 1),
            draws=12,
        )

    def test_covers_every_cell(self, summaries):
        assert {(s.size, s.heterogeneity) for s in summaries} == {
            (size, h) for h in (0.0, 1.0) for size in (1, 5, 10)
        }

    def test_homogeneous_corpora_do_not_erode(self, summaries):
        flat = [s for s in summaries if s.heterogeneity == 0.0]
        assert len({s.capability_mean for s in flat}) == 1

    def test_diverse_corpora_erode_with_size(self, summaries):
        diverse = sorted(
            (s for s in summaries if s.heterogeneity == 1.0), key=lambda s: s.size
        )
        assert diverse[0].capability_mean > diverse[-1].capability_mean

    def test_rows_are_serialisable(self, summaries):
        row = summaries[0].as_row()
        assert row["size"] and "capability_mean" in row
        assert all(isinstance(v, (int, float, str)) for v in row.values())


class TestCriteria:
    @pytest.fixture(scope="class")
    @classmethod
    def criteria(cls):
        return evaluate_criteria(
            run_study(
                pool_size=20,
                sizes=(1, 10),
                heterogeneities=(0.0, 1.0),
                corpus_seeds=(0, 1),
                draws=20,
            )
        )

    def test_all_four_are_evaluated(self, criteria):
        assert {c.name for c in criteria} == {
            "non-trivial",
            "non-degenerate",
            "attributable",
            "diversity-driven",
        }

    def test_blocking_criteria_are_the_ones_that_stop_the_project(self, criteria):
        """Fixed in advance, so a threshold cannot be renegotiated later."""
        blocking = {c.name for c in criteria if c.blocking}
        assert blocking == {"non-trivial", "non-degenerate"}

    def test_erosion_is_non_trivial_and_non_degenerate(self, criteria):
        """The two that decide whether there is a phenomenon at all."""
        for criterion in criteria:
            if criterion.blocking:
                assert criterion.passed, f"{criterion.name}: {criterion.detail}"

    def test_every_criterion_explains_itself(self, criteria):
        assert all(c.detail for c in criteria)


class TestSummarise:
    def test_reports_counts_and_intervals(self):
        corpus = generate_corpus(8, heterogeneity=1.0, seed=0)
        rng = np.random.default_rng(0)
        measurements = [
            measure(subset, heterogeneity=1.0, seed=0)
            for subset in sample_subsets(corpus, 3, 10, rng)
        ]
        summary = summarise(measurements, size=3, heterogeneity=1.0, rng=rng)
        assert summary.n == 10
        assert summary.capability_ci[0] <= summary.capability_mean <= summary.capability_ci[1]
        assert 0.0 <= summary.retains_any <= 1.0
