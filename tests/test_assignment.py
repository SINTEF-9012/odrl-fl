"""Tests for the code-assignment models.

These exist to defuse the erosion study's main threat to validity: EGA supplies
real code frequencies but says nothing about which institution holds which
policy, so the corpus has to assume something. Each model is a different
assumption, and the study is run under all of them.

The most important test here is `test_pairwise_compatibility_is_not_transitive`.
It underpins both the sensitivity result and the case for increment 5.
"""

from __future__ import annotations

import numpy as np
import pytest

from odrl_fl.assignment import (
    MODELS,
    build_sampler,
    model_names,
    purposes_compatible,
)
from odrl_fl.corpus import code_diversity, generate_corpus, policies_of
from odrl_fl.odrl import compose


class TestCompatibility:
    def test_subsumption_makes_purposes_compatible(self):
        assert purposes_compatible(
            frozenset({"research"}), frozenset({"biomedical-research"})
        )

    def test_disjoint_purposes_are_incompatible(self):
        assert not purposes_compatible(
            frozenset({"research"}), frozenset({"clinical-care"})
        )

    def test_an_unconstrained_side_is_compatible_with_anything(self):
        assert purposes_compatible(frozenset(), frozenset({"ds-cancer"}))

    def test_pairwise_compatibility_is_not_transitive(self):
        """The property that makes selection a search rather than a filter.

        Two contributors can each be compatible with a common anchor and
        incompatible with one another, so a consortium cannot be assembled by
        vetting candidates against a reference policy one at a time. It has to
        be evaluated as a set -- which is what increment 5 has to do, and why
        even a purpose-matched consortium still erodes.
        """
        research = frozenset({"research"})
        cancer = frozenset({"ds-cancer"})
        neuro = frozenset({"ds-neurological"})

        assert purposes_compatible(research, cancer)
        assert purposes_compatible(research, neuro)
        assert not purposes_compatible(cancer, neuro)


class TestModels:
    def test_every_model_is_registered_and_described(self):
        assert set(model_names()) == set(MODELS)
        for name, model in MODELS.items():
            assert model.name == name
            assert model.description

    @pytest.mark.parametrize("name", sorted(MODELS))
    def test_every_model_produces_a_usable_corpus(self, name):
        corpus = generate_corpus(12, assignment=name, seed=0)
        assert len(corpus) == 12
        assert all(c.codes for c in corpus)

    @pytest.mark.parametrize("name", sorted(MODELS))
    def test_every_model_is_deterministic(self, name):
        first = generate_corpus(10, assignment=name, seed=3)
        second = generate_corpus(10, assignment=name, seed=3)
        assert [c.codes for c in first] == [c.codes for c in second]

    def test_unknown_model_is_rejected(self):
        with pytest.raises(ValueError, match="unknown assignment model"):
            build_sampler("no-such-model", np.random.default_rng(0))

    def test_default_is_the_baseline(self):
        assert [c.codes for c in generate_corpus(8, seed=1)] == [
            c.codes for c in generate_corpus(8, assignment="empirical", seed=1)
        ]


class TestModelBehaviour:
    def test_cohesion_reduces_diversity(self):
        """The dial the objection turns on: more homophily, more alike."""
        diversities = [
            code_diversity(generate_corpus(30, assignment=name, seed=5))
            for name in ("clustered-0.3", "clustered-0.6", "clustered-0.9")
        ]
        assert diversities == sorted(diversities, reverse=True)

    def test_cohesion_reduces_erosion(self):
        """And therefore erodes less -- the honest limitation, measured.

        This is the finding that stops the headline number being quoted without
        its assumption attached.
        """

        def retained(name: str) -> int:
            corpus = generate_corpus(20, assignment=name, seed=5)
            return len(compose(policies_of(corpus[:10])).capability)

        assert retained("clustered-0.9") >= retained("clustered-0.3")

    def test_marginal_sampling_destroys_co_occurrence(self):
        """Real annotations bundle conditions; marginal draws do not, so the
        code sets it produces are almost all distinct."""
        corpus = generate_corpus(25, assignment="marginal", seed=0)
        assert code_diversity(corpus) > 0.95

    def test_purpose_matched_still_erodes(self):
        """The load-bearing check.

        A consortium already self-selected for compatible purposes must still
        lose capability, because compatibility is not transitive. If this ever
        started passing trivially, the argument for explicit selection would
        collapse.
        """
        corpus = generate_corpus(30, assignment="purpose-matched", seed=0)
        alone = len(compose(policies_of(corpus[:1])).capability)
        together = len(compose(policies_of(corpus[:15])).capability)
        assert together < alone

    def test_erosion_occurs_under_every_model(self):
        """Direction is robust even where magnitude is not."""
        for name in sorted(MODELS):
            corpus = generate_corpus(40, assignment=name, seed=0)
            alone = len(compose(policies_of(corpus[:1])).capability)
            large = len(compose(policies_of(corpus[:30])).capability)
            assert large <= alone, f"{name} widened capability, which is impossible"
