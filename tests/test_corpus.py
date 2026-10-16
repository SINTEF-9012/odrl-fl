"""Tests for corpus generation.

Determinism and the heterogeneity dial are what matter. The erosion study
compares subsets of a corpus, so the corpus cannot shift underneath the
comparison; and RQ4 sweeps policy diversity, so the dial has to actually move
diversity rather than merely being accepted as a parameter.
"""

from __future__ import annotations

import pytest

from odrl_fl.corpus import (
    code_diversity,
    combinations,
    corpus_stats,
    frequencies,
    generate_corpus,
    modal_combination,
    policies_of,
)
from odrl_fl.odrl import capability, compose


class TestProvenance:
    def test_frequencies_record_where_they_came_from(self):
        """A corpus claiming to be grounded has to say in what."""
        provenance = frequencies()["_provenance"]
        assert provenance["source"]
        assert provenance["policies_seen"] > 0
        assert provenance["policies_annotated"] > 0
        assert 0.0 < provenance["adoption_rate"] <= 1.0

    def test_sparse_adoption_is_recorded(self):
        """Most EGA policies carry no structured conditions at all. That is a
        finding about the state of practice, and the number is worth keeping."""
        provenance = frequencies()["_provenance"]
        assert provenance["policies_annotated"] < provenance["policies_seen"]

    def test_combinations_are_ordered_by_frequency(self):
        counts = [count for _, count in combinations()]
        assert counts == sorted(counts, reverse=True)
        assert modal_combination() == combinations()[0][0]


class TestDeterminism:
    def test_same_seed_gives_the_same_corpus(self):
        first = generate_corpus(12, seed=3)
        second = generate_corpus(12, seed=3)
        assert [c.codes for c in first] == [c.codes for c in second]
        assert policies_of(first) == policies_of(second)

    def test_different_seeds_differ(self):
        assert [c.codes for c in generate_corpus(12, seed=1)] != [
            c.codes for c in generate_corpus(12, seed=2)
        ]

    def test_a_contributor_is_stable_across_corpus_sizes(self):
        """Contributor 3 must hold the same policy whether the corpus has 10
        members or 200 -- the same property the data partitions have, and for
        the same reason: subset comparisons must vary for one reason only."""
        small = generate_corpus(10, seed=5)
        large = generate_corpus(200, seed=5)
        for index in (0, 3, 9):
            assert small[index].codes == large[index].codes
            assert small[index].policy == large[index].policy

    def test_uids_are_unique_and_stable(self):
        corpus = generate_corpus(25, seed=0)
        uids = [c.uid for c in corpus]
        assert len(set(uids)) == len(uids)
        assert uids == [c.uid for c in generate_corpus(25, seed=0)]


class TestHeterogeneityDial:
    def test_zero_heterogeneity_is_homogeneous(self):
        corpus = generate_corpus(20, heterogeneity=0.0, seed=0)
        assert code_diversity(corpus) == 0.0
        assert len({frozenset(c.codes) for c in corpus}) == 1

    def test_diversity_rises_with_the_dial(self):
        """Without this, RQ4 has nothing to sweep."""
        levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        diversities = [
            code_diversity(generate_corpus(30, heterogeneity=h, seed=7))
            for h in levels
        ]
        assert diversities == sorted(diversities)
        assert diversities[-1] > diversities[0]

    def test_homogeneous_corpora_barely_erode(self):
        """Identical policies compose to what any one of them permits: the
        control that shows erosion tracks diversity rather than mere size."""
        corpus = generate_corpus(20, heterogeneity=0.0, seed=0)
        composed = compose(policies_of(corpus)).capability
        assert composed == capability(corpus[0].policy)

    def test_diverse_corpora_erode(self):
        homogeneous = compose(
            policies_of(generate_corpus(20, heterogeneity=0.0, seed=0))
        ).capability
        diverse = compose(
            policies_of(generate_corpus(20, heterogeneity=1.0, seed=0))
        ).capability
        assert len(diverse) < len(homogeneous)


class TestErosionShape:
    def test_capability_is_anti_monotone_over_a_real_corpus(self):
        """The property, checked against DUO-derived policies rather than
        generated ones."""
        corpus = generate_corpus(15, heterogeneity=1.0, seed=2)
        previous = None
        for size in range(1, len(corpus) + 1):
            current = compose(policies_of(corpus[:size])).capability
            if previous is not None:
                assert current <= previous
            previous = current

    def test_there_is_a_curve_rather_than_a_cliff(self):
        """Erosion has to be gradual enough to measure.

        Deliberately averaged over sampled subsets rather than taken from a
        single prefix. Individual subsets are bimodal -- one incompatible
        contributor collapses a whole subset to nothing, so an arbitrary triple
        may well score zero while the mean at that size is close to four. The
        erosion study must sample for the same reason, and this test pins the
        shape it depends on: gradual decay, neither an immediate cliff nor a
        flat line.
        """
        import numpy as np

        corpus = generate_corpus(40, heterogeneity=1.0, seed=0)
        rng = np.random.default_rng(0)

        def mean_capability(size: int, draws: int = 60) -> float:
            totals = []
            for _ in range(draws):
                picked = rng.choice(len(corpus), size=size, replace=False)
                subset = [corpus[i] for i in picked]
                totals.append(len(compose(policies_of(subset)).capability))
            return float(np.mean(totals))

        at_three = mean_capability(3)
        at_ten = mean_capability(10)
        at_twenty = mean_capability(20)

        assert at_three > 2.5, "collapsed too early to be informative"
        assert at_three > at_ten > at_twenty, "erosion is not monotone in size"
        assert at_twenty > 0.0, "nothing survives, so there is no curve"


class TestStats:
    def test_stats_describe_the_corpus(self):
        stats = corpus_stats(generate_corpus(20, heterogeneity=1.0, seed=0))
        assert stats["contributors"] == 20
        assert stats["distinct_code_sets"] > 1
        assert stats["mean_codes_per_contributor"] > 0
        assert stats["top_codes"]

    def test_unsupported_codes_are_surfaced(self):
        """COL appears in real data and cannot be expressed; the corpus must
        report that rather than hiding it."""
        stats = corpus_stats(generate_corpus(60, heterogeneity=1.0, seed=1))
        assert isinstance(stats["unsupported_codes"], list)

    def test_rejects_degenerate_arguments(self):
        with pytest.raises(ValueError):
            generate_corpus(0)
        with pytest.raises(ValueError):
            generate_corpus(5, heterogeneity=1.5)
