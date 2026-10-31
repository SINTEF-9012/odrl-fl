"""Tests for the measured EGA corpus and the cohesion estimator.

Two kinds of test here, deliberately separated. The cohesion algebra is exact
and gets checked against cases whose answer is known by construction -- it is
the part that turns an observed rate into a number the README will quote, so it
has to be right for reasons better than "the output looked plausible". The
corpus tests need the cached fetch and are skipped when it is absent, following
`test_tcga.py`: a test suite should not reach across the network unasked.
"""

from __future__ import annotations

import numpy as np
import pytest

from odrl_fl.ega import (
    CACHE,
    Contributor,
    agreement_between_dacs,
    agreement_within_dac,
    by_dac,
    cohesion_bracket,
    empirical_coincidence,
    load,
    provenance,
    solve_cohesion,
)


def forward(c: float, p: float) -> float:
    """The clustered model's agreement rate, stated independently of the
    implementation so the inversion is checked against the derivation rather
    than against itself."""
    return p + c**2 * (1.0 - p)


def contributor(dac: str, *codes: str, n: int | None = None) -> Contributor:
    return Contributor(
        policy_id=f"P{dac}{'-'.join(codes)}",
        dac_id=dac,
        institution=f"Institution {dac}",
        codes=tuple(sorted(codes)),
        n_samples=n,
    )


class TestSolveCohesion:
    @pytest.mark.parametrize("c", [0.0, 0.1, 0.3, 0.5, 0.6, 0.9, 1.0])
    @pytest.mark.parametrize("p", [0.01, 0.05, 0.2, 0.5])
    def test_round_trips_against_the_forward_model(self, c, p):
        assert solve_cohesion(forward(c, p), p) == pytest.approx(c, abs=1e-9)

    def test_agreement_at_chance_means_no_cohesion(self):
        assert solve_cohesion(0.2, 0.2) == 0.0

    def test_perfect_agreement_means_total_cohesion(self):
        assert solve_cohesion(1.0, 0.2) == pytest.approx(1.0)

    def test_below_chance_is_floored_at_zero_not_imaginary(self):
        """The clustered model can only add homophily, never subtract it, so
        there is no cohesion that explains sub-chance agreement."""
        assert solve_cohesion(0.05, 0.4) == 0.0

    def test_above_one_is_clamped(self):
        assert solve_cohesion(1.5, 0.2) == 1.0

    def test_degenerate_inputs_do_not_raise(self):
        assert np.isnan(solve_cohesion(float("nan"), 0.2))
        assert np.isnan(solve_cohesion(0.5, 1.0))

    def test_cohesion_rises_with_agreement(self):
        p = 0.1
        values = [solve_cohesion(a, p) for a in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert values == sorted(values)


class TestAgreement:
    def test_a_lone_policy_contributes_no_pair(self):
        """One policy says nothing about internal consistency; counting it as
        either perfect or zero agreement would be an invention."""
        assert agreement_within_dac([contributor("A", "DUO:0000042")]) == []

    def test_identical_policies_agree_completely(self):
        corpus = [contributor("A", "DUO:0000042") for _ in range(4)]
        assert agreement_within_dac(corpus) == [1.0]

    def test_distinct_policies_do_not_agree_at_all(self):
        corpus = [contributor("A", f"DUO:000000{i}") for i in range(4)]
        assert agreement_within_dac(corpus) == [0.0]

    def test_partial_agreement_counts_pairs_not_policies(self):
        # Three policies, two alike: one agreeing pair out of three.
        corpus = [
            contributor("A", "DUO:0000042"),
            contributor("A", "DUO:0000042"),
            contributor("A", "DUO:0000007"),
        ]
        assert agreement_within_dac(corpus) == [pytest.approx(1 / 3)]

    def test_one_value_per_institution_not_per_pair(self):
        """A prolific DAC must not dominate: 20 policies would be 190 of the
        pairs and would turn a typical institution into the loudest one."""
        corpus = [contributor("BIG", "DUO:0000042") for _ in range(20)]
        corpus += [contributor("SMALL", f"DUO:000000{i}") for i in range(2)]
        assert sorted(agreement_within_dac(corpus)) == [0.0, 1.0]

    def test_between_dacs_ignores_same_dac_pairs(self):
        corpus = [
            contributor("A", "DUO:0000042"),
            contributor("A", "DUO:0000042"),
            contributor("B", "DUO:0000007"),
        ]
        # Only the A-B pairs count, and neither agrees.
        assert agreement_between_dacs(corpus) == 0.0

    def test_between_dacs_finds_cross_institution_agreement(self):
        corpus = [contributor("A", "DUO:0000042"), contributor("B", "DUO:0000042")]
        assert agreement_between_dacs(corpus) == 1.0

    def test_contributors_without_a_dac_are_excluded(self):
        corpus = [contributor("", "DUO:0000042"), contributor("A", "DUO:0000042")]
        assert list(by_dac(corpus)) == ["A"]


class TestEmpiricalCoincidence:
    def test_a_single_repeated_combination_always_coincides(self):
        corpus = [contributor(str(i), "DUO:0000042") for i in range(5)]
        assert empirical_coincidence(corpus) == pytest.approx(1.0)

    def test_all_distinct_gives_one_over_n(self):
        corpus = [contributor(str(i), f"DUO:000000{i}") for i in range(4)]
        assert empirical_coincidence(corpus) == pytest.approx(0.25)

    def test_falls_back_to_the_committed_table(self):
        """Without a corpus the value must stay comparable with the synthetic
        models, which are calibrated against that same table."""
        p = empirical_coincidence()
        assert 0.0 < p < 1.0


class TestBracket:
    def test_bracket_is_ordered_and_bounded(self):
        corpus = [contributor("A", "DUO:0000042") for _ in range(3)]
        corpus += [contributor(f"S{i}", f"DUO:000000{i}") for i in range(6)]
        bracket = cohesion_bracket(corpus, resamples=200)
        assert bracket["agreement_between"] <= bracket["agreement_within"]
        assert 0.0 <= bracket["cohesion_lower"] <= 1.0
        assert 0.0 <= bracket["cohesion_upper"] <= 1.0
        assert bracket["cohesion_lower"] <= bracket["cohesion_upper"]

    def test_interval_brackets_the_estimate(self):
        corpus = []
        for i in range(8):
            # Half the institutions internally consistent, half not.
            code = "DUO:0000042" if i % 2 else f"DUO:000001{i}"
            corpus += [contributor(f"D{i}", code), contributor(f"D{i}", code)]
            corpus += [contributor(f"D{i}", f"DUO:000009{i}")]
        b = cohesion_bracket(corpus, resamples=500)
        assert b["agreement_within_lo"] <= b["agreement_within"]
        assert b["agreement_within"] <= b["agreement_within_hi"]

    def test_no_multi_policy_dacs_yields_nan_not_a_number(self):
        """With nothing to measure, the upper bound must be absent rather than
        silently defaulting to something quotable."""
        corpus = [contributor(str(i), "DUO:0000042") for i in range(4)]
        b = cohesion_bracket(corpus, resamples=100)
        assert b["n_dacs_multi"] == 0
        assert np.isnan(b["agreement_within"])


@pytest.mark.skipif(
    not CACHE.exists(), reason="EGA corpus not fetched; run python -m odrl_fl.ega --fetch"
)
class TestCachedCorpus:
    def test_every_contributor_has_a_dac_and_codes(self):
        for c in load():
            assert c.dac_id
            assert c.codes
            assert all(code.startswith("DUO:") for code in c.codes)

    def test_volume_is_absent_or_positive_never_zero(self):
        """A missing volume must stay explicit: 12% of annotated policies carry
        no sample count, and calling those empty would both understate every
        volume-weighted quantity and invent a fact about the institution."""
        for c in load():
            assert c.n_samples is None or c.n_samples > 0

    def test_codes_are_sorted_so_sets_compare_by_equality(self):
        """Agreement is measured by tuple equality, which is only meaningful if
        the order is canonical."""
        for c in load():
            assert list(c.codes) == sorted(c.codes)

    def test_provenance_records_what_was_fetched(self):
        p = provenance()
        assert p["policies_annotated"] > 0
        assert 0.0 < p["adoption_rate"] <= 1.0
        assert p["fetched"]

    def test_the_real_bracket_is_ordered(self):
        b = cohesion_bracket(load(), resamples=500)
        assert b["agreement_between"] <= b["agreement_within"]
        assert b["cohesion_lower"] <= b["cohesion_upper"]
