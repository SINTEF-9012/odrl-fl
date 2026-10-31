"""Tests for the DUO -> ODRL mapping.

The exhaustiveness tests come first and matter most. A DUO code that is
silently unmapped would drop a real condition from a contributor's offer, and
the composed capability would then exceed what was granted -- wrong results
that look entirely plausible. The same principle as the parser's loud
rejection, applied to the vocabulary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from odrl_fl.odrl import Action, LeftOperand, capability, compose
from odrl_fl.odrl.duo import (
    DEFAULT_ACTIONS,
    UnknownDUOCode,
    code_class,
    describe,
    duo_table,
    is_composable,
    translate,
)

VOCAB = Path(__file__).resolve().parent.parent / "vocab"

#: Handled by `translate`. A mapping using anything else would fall through.
KNOWN_KINDS = {
    "purpose",
    "purpose_param",
    "purpose_exclude",
    "constraint",
    "constraint_and_exclude",
    "spatial",
    "until",
    "moratorium",
    "prohibit",
    "duty",
    "pre_duty",
    "none",
    "unsupported",
}

#: Combinations observed on real EGA policy records.
REAL_COMBINATIONS = {
    "EGAP00001000001": ["DUO:0000019", "DUO:0000026", "DUO:0000028", "DUO:0000042"],
    "EGAP00001000009": ["DUO:0000042"],
    "EGAP00001000011": ["DUO:0000042", "DUO:0000027", "DUO:0000006", "DUO:0000018"],
}


def constraint_for(policy, action, operand):
    permission = policy.permission_for(action)
    return next(
        (c for c in permission.constraints if c.left_operand is operand), None
    )


class TestExhaustiveness:
    def test_every_mapping_uses_a_handled_kind(self):
        for code, entry in duo_table().items():
            assert entry["odrl"]["kind"] in KNOWN_KINDS, f"{code} has an unhandled kind"

    def test_every_observed_code_is_mapped(self):
        """Nothing seen in real annotations may be missing from the table."""
        observed = json.loads(
            (VOCAB / "duo_frequencies.json").read_text(encoding="utf-8")
        )["code_frequencies"]
        missing = sorted(set(observed) - set(duo_table()))
        assert not missing, f"observed in EGA data but unmapped: {missing}"

    def test_every_code_is_classified(self):
        valid = {
            "permission",
            "restriction",
            "externally-adjudicated",
            "duty",
            "out-of-fragment",
            "structural",
            "deprecated",
        }
        for code, entry in duo_table().items():
            assert entry["class"] in valid, f"{code} has class {entry['class']!r}"

    def test_every_code_declares_its_fidelity(self):
        """Where DUO and ODRL do not line up cleanly, say so."""
        for code, entry in duo_table().items():
            assert entry["fidelity"] in {"exact", "interpretive", "n/a"}, code

    def test_an_unknown_code_is_rejected(self):
        with pytest.raises(UnknownDUOCode):
            translate(["DUO:9999999"], uid="urn:policy:x")

    def test_unknown_code_can_be_tolerated_explicitly(self):
        _, unsupported = translate(
            ["DUO:0000042", "DUO:9999999"], uid="urn:policy:x", strict=False
        )
        assert "DUO:9999999" in unsupported


class TestClassification:
    def test_col_is_out_of_fragment(self):
        """The interesting exclusion: 'collaboration required' means the
        provider must be a participant, so it depends on the very contributor
        set being composed -- self-referential, exactly the class ruled out."""
        assert code_class("DUO:0000020") == "out-of-fragment"
        assert not is_composable("DUO:0000020")

        policy, unsupported = translate(
            ["DUO:0000042", "DUO:0000020"], uid="urn:policy:col"
        )
        assert unsupported == ("DUO:0000020",)
        # The rest of the offer still translates.
        assert Action.USE in capability(policy)

    def test_ontology_categories_contribute_nothing(self):
        assert code_class("DUO:0000001") == "structural"
        bare, _ = translate(["DUO:0000042"], uid="urn:policy:a")
        with_category, _ = translate(
            ["DUO:0000042", "DUO:0000001"], uid="urn:policy:a"
        )
        assert bare == with_category

    def test_deprecated_codes_map_through_their_replacements(self):
        """GRU-CC and RU are retired but still carry 132 code occurrences in
        the EGA sample, so they cannot simply be refused."""
        assert code_class("DUO:0000005") == "deprecated"
        gru_cc, unsupported = translate(["DUO:0000005"], uid="urn:policy:d")
        assert not unsupported
        purposes = constraint_for(gru_cc, Action.USE, LeftOperand.PURPOSE)
        assert purposes.right_operand == frozenset({"research", "clinical-care"})

    def test_describe_is_informative(self):
        text = describe("DUO:0000042")
        assert "GRU" in text and "permission" in text


class TestTranslation:
    @pytest.mark.parametrize("accession", sorted(REAL_COMBINATIONS))
    def test_real_ega_combinations_translate(self, accession):
        policy, unsupported = translate(
            REAL_COMBINATIONS[accession], uid=f"urn:policy:{accession}"
        )
        assert not unsupported
        assert set(policy.permitted_actions) == set(DEFAULT_ACTIONS)

    def test_real_combinations_compose(self):
        policies = [
            translate(codes, uid=f"urn:policy:{acc}")[0]
            for acc, codes in REAL_COMBINATIONS.items()
        ]
        result = compose(policies)
        assert result.capability == set(DEFAULT_ACTIONS)

    def test_gru_and_hmb_redundancy_is_reduced(self):
        """GRU and HMB co-occur in real EGA records. GRU subsumes HMB, so the
        composed purpose must collapse to the broader term rather than listing
        both."""
        policy, _ = translate(
            ["DUO:0000042", "DUO:0000006"], uid="urn:policy:redundant"
        )
        composed = compose([policy]).agreement
        purposes = constraint_for(composed, Action.USE, LeftOperand.PURPOSE)
        assert purposes.right_operand == frozenset({"research"})

    def test_orthogonal_predicates_conjoin_rather_than_collide(self):
        """US requires an approved user, NPU a not-for-profit organisation.

        Both constrain the receiving party and are jointly satisfiable. Sharing
        one operand would intersect their value sets to nothing and report an
        irreconcilable conflict, so each independent predicate has its own.
        """
        approved_user, _ = translate(
            ["DUO:0000042", "DUO:0000026"], uid="urn:policy:us"
        )
        not_for_profit, _ = translate(
            ["DUO:0000042", "DUO:0000045"], uid="urn:policy:npu"
        )
        result = compose([approved_user, not_for_profit])

        assert Action.USE in result.capability, "jointly satisfiable, not a conflict"
        permission = result.agreement.permission_for(Action.USE)
        operands = {c.left_operand for c in permission.constraints}
        assert LeftOperand.RECIPIENT in operands
        assert LeftOperand.ORGANISATION in operands

    def test_nmds_prohibits_methods_research_derivation(self):
        """NMDS is exactly the constrained-prohibition shape the algebra
        supports: one positive set-valued constraint."""
        policy, _ = translate(["DUO:0000042", "DUO:0000015"], uid="urn:policy:nmds")
        prohibition = next(iter(policy.prohibitions))
        assert prohibition.action is Action.DERIVE
        constraint = next(iter(prohibition.constraints))
        assert constraint.right_operand == frozenset({"methods-research"})

    def test_ncu_excludes_commercial_purposes(self):
        policy, _ = translate(["DUO:0000042", "DUO:0000046"], uid="urn:policy:ncu")
        excluded = [
            c
            for c in policy.permission_for(Action.USE).constraints
            if c.left_operand is LeftOperand.PURPOSE
            and c.operator.value == "isNoneOf"
        ]
        assert excluded and excluded[0].right_operand == frozenset({"commercial"})

    def test_parameterised_codes_use_supplied_values(self):
        policy, _ = translate(
            ["DUO:0000042", "DUO:0000022", "DUO:0000025"],
            uid="urn:policy:params",
            params={"region": "FR", "expiry": "2029-01-01"},
        )
        assert constraint_for(
            policy, Action.USE, LeftOperand.SPATIAL
        ).right_operand == frozenset({"FR"})
        assert (
            constraint_for(policy, Action.USE, LeftOperand.DATETIME).right_operand
            == "2029-01-01"
        )

    def test_moratorium_binds_distribution_not_use(self):
        """A publication moratorium delays release, so it constrains
        distribution rather than every action."""
        policy, _ = translate(
            ["DUO:0000042", "DUO:0000024"],
            uid="urn:policy:mor",
            params={"moratorium_until": "2027-01-01"},
        )
        assert constraint_for(policy, Action.DISTRIBUTE, LeftOperand.DATETIME)
        assert constraint_for(policy, Action.USE, LeftOperand.DATETIME) is None

    def test_irb_becomes_a_pre_duty(self):
        policy, _ = translate(["DUO:0000042", "DUO:0000021"], uid="urn:policy:irb")
        duties = policy.permission_for(Action.USE).duties
        assert {d.action for d in duties} == {Action.OBTAIN_CONSENT}

    def test_pub_becomes_an_obligation(self):
        policy, _ = translate(["DUO:0000042", "DUO:0000019"], uid="urn:policy:pub")
        assert {d.action for d in policy.obligations} == {Action.INFORM}

    def test_conflicting_purposes_across_contributors(self):
        """GRU permits research, CC permits clinical care; they do not overlap,
        so a federation spanning both has no permitted use.

        One of the mechanisms behind capability erosion in the corpus.
        """
        research, _ = translate(["DUO:0000042"], uid="urn:policy:gru")
        clinical, _ = translate(["DUO:0000043"], uid="urn:policy:cc")
        assert not compose([research, clinical]).capability

    def test_translation_is_order_independent(self):
        codes = ["DUO:0000042", "DUO:0000026", "DUO:0000019"]
        first, _ = translate(codes, uid="urn:policy:o")
        second, _ = translate(list(reversed(codes)), uid="urn:policy:o")
        assert first == second
