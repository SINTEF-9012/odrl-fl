"""Tests for policy-carrying model bundles.

Two properties carry the increment. **Verification is substantive**: it
recomposes the agreement from the recorded contributor declarations rather than
checking a digest against itself, so widening the terms requires forging every
contributor's declaration to justify the change. And **policy-carrying is
recursive**: a bundle's agreement is a Policy, so reusing the model composes its
terms with the newcomers' and capability can only narrow.

`TestTampering` is the set to read if the guarantee is ever questioned, and
`test_an_unsigned_manifest_can_be_rewritten_wholesale` states the limit rather
than leaving it implied.
"""

from __future__ import annotations

import json

import pytest
import torch

from odrl_fl.bundle import (
    MANIFEST_NAME,
    MODEL_NAME,
    ContributorRecord,
    Manifest,
    VerificationError,
    as_contributor_policy,
    build_manifest,
    export,
    load,
    policy_digest,
    verify,
)
from odrl_fl.corpus import generate_corpus
from odrl_fl.duties import Ledger, Obligation
from odrl_fl.odrl import (
    Action,
    Constraint,
    LeftOperand,
    Operator,
    Permission,
    Policy,
    PolicyType,
    capability,
    compose,
)
from odrl_fl.selection import requirement
from odrl_fl.task import Net

ACTIONS = (Action.USE, Action.AGGREGATE)


def offer(uid: str, *purposes: str) -> Policy:
    constraints = frozenset()
    if purposes:
        constraints = frozenset(
            {Constraint(LeftOperand.PURPOSE, Operator.IS_ANY_OF, frozenset(purposes))}
        )
    return Policy(
        uid=uid,
        type=PolicyType.OFFER,
        assigner=f"urn:party:{uid.rsplit(':', 1)[-1]}",
        permissions=frozenset(
            Permission(action=a, constraints=constraints) for a in ACTIONS
        ),
    )


def bundle_at(tmp_path, policies=None, ledger=None, with_model=True):
    policies = policies or [offer("urn:p:a", "research"), offer("urn:p:b", "research")]
    records = [
        ContributorRecord(node_id=i, policy=p, codes=("DUO:0000042",), n_train=100)
        for i, p in enumerate(policies)
    ]
    manifest = build_manifest(
        compose(policies).agreement,
        records,
        request=requirement(*ACTIONS),
        ledger=ledger,
        metrics={"accuracy": 0.9},
    )
    directory = tmp_path / "bundle"
    export(directory, manifest, Net().state_dict() if with_model else None)
    return directory


# ---------------------------------------------------------------------------


class TestExport:
    def test_a_bundle_holds_the_model_and_the_terms(self, tmp_path):
        directory = bundle_at(tmp_path)
        assert (directory / MODEL_NAME).exists()
        assert (directory / MANIFEST_NAME).exists()

    def test_the_manifest_records_what_the_holder_may_do(self, tmp_path):
        manifest = load(bundle_at(tmp_path))
        assert manifest.capability == {Action.USE, Action.AGGREGATE}

    def test_every_contributor_is_recorded_with_its_declaration(self, tmp_path):
        manifest = load(bundle_at(tmp_path))
        assert len(manifest.contributors) == 2
        assert all(c.policy.permissions for c in manifest.contributors)

    def test_the_model_digest_covers_the_bytes_on_disk(self, tmp_path):
        directory = bundle_at(tmp_path)
        manifest = load(directory)
        assert manifest.model_digest
        verify(directory)  # would raise if the digest were of something else

    def test_a_bundle_can_be_written_without_weights(self, tmp_path):
        """Useful for inspecting terms alone; verification must still work."""
        directory = bundle_at(tmp_path, with_model=False)
        assert not (directory / MODEL_NAME).exists()
        verify(directory)

    def test_the_manifest_is_valid_json_and_human_readable(self, tmp_path):
        raw = json.loads((bundle_at(tmp_path) / MANIFEST_NAME).read_text())
        assert raw["capability"] == ["aggregate", "use"]
        assert raw["agreement"]["@type"] == "Agreement"


class TestOutstandingDuties:
    def ledger(self):
        return Ledger(
            outstanding=(
                Obligation(action="inform", owed_by=("urn:party:a",), detail="publish"),
            ),
            gates_cleared={"obtainConsent": "IRB-2026-0142"},
            rejections=[(2, 7, "budget exceeded")],
        )

    def test_outstanding_duties_travel_with_the_model(self, tmp_path):
        """The point of the increment: only the holder can discharge these."""
        manifest = load(bundle_at(tmp_path, ledger=self.ledger()))
        assert [o.action for o in manifest.outstanding] == ["inform"]
        assert manifest.outstanding[0].owed_by == ("urn:party:a",)

    def test_outstanding_duties_are_never_recorded_as_discharged(self, tmp_path):
        manifest = load(bundle_at(tmp_path, ledger=self.ledger()))
        assert all(not o.discharged for o in manifest.outstanding)

    def test_cleared_gates_and_their_evidence_are_carried(self, tmp_path):
        manifest = load(bundle_at(tmp_path, ledger=self.ledger()))
        assert manifest.gates_cleared == {"obtainConsent": "IRB-2026-0142"}

    def test_dropped_updates_are_carried(self, tmp_path):
        """An auditor should see that a contributor failed its duty mid-run."""
        manifest = load(bundle_at(tmp_path, ledger=self.ledger()))
        assert manifest.rejections == ((2, 7, "budget exceeded"),)

    def test_the_description_tells_the_holder_what_is_owed(self, tmp_path):
        lines = load(bundle_at(tmp_path, ledger=self.ledger())).describe()
        assert any("YOU OWE" in line and "inform" in line for line in lines)


class TestVerification:
    def test_an_untouched_bundle_verifies(self, tmp_path):
        assert verify(bundle_at(tmp_path)) is not None

    def test_verification_recomposes_rather_than_trusting_the_record(self, tmp_path):
        """The substantive check, on a real DUO-derived corpus."""
        policies = [c.policy for c in generate_corpus(6, seed=0)[:3]]
        directory = bundle_at(tmp_path, policies=policies)
        manifest = verify(directory)
        assert manifest.agreement == compose(policies).agreement

    def test_a_manifest_with_no_contributors_is_refused(self, tmp_path):
        directory = bundle_at(tmp_path)
        raw = json.loads((directory / MANIFEST_NAME).read_text())
        raw["contributors"] = []
        (directory / MANIFEST_NAME).write_text(json.dumps(raw))
        with pytest.raises(VerificationError, match="no contributors"):
            verify(directory)

    def test_an_unreadable_manifest_version_is_refused(self, tmp_path):
        directory = bundle_at(tmp_path)
        raw = json.loads((directory / MANIFEST_NAME).read_text())
        raw["version"] = 99
        (directory / MANIFEST_NAME).write_text(json.dumps(raw))
        with pytest.raises(VerificationError, match="not readable"):
            verify(directory)


class TestTampering:
    def test_swapping_the_weights_is_detected(self, tmp_path):
        directory = bundle_at(tmp_path)
        torch.save(dict(Net().state_dict()), directory / MODEL_NAME)
        with pytest.raises(VerificationError, match="does not match the digest"):
            verify(directory)

    def test_deleting_the_weights_is_detected(self, tmp_path):
        directory = bundle_at(tmp_path)
        (directory / MODEL_NAME).unlink()
        with pytest.raises(VerificationError, match="no model is present"):
            verify(directory)

    def test_widening_the_agreement_is_detected(self, tmp_path):
        """The attack that matters: claim more rights than were granted."""
        directory = bundle_at(
            tmp_path, policies=[offer("urn:p:a", "ds-cancer"), offer("urn:p:b", "ds-cancer")]
        )
        raw = json.loads((directory / MANIFEST_NAME).read_text())
        for permission in raw["agreement"]["permission"]:
            for constraint in permission.get("constraint", []):
                if constraint["leftOperand"] == "purpose":
                    constraint["rightOperand"] = ["research"]
        raw.pop("agreement_digest", None)  # recomputed by a competent forger
        (directory / MANIFEST_NAME).write_text(json.dumps(raw))

        with pytest.raises(VerificationError, match="do not compose"):
            verify(directory)

    def test_altering_a_contributor_declaration_is_detected(self, tmp_path):
        directory = bundle_at(tmp_path)
        raw = json.loads((directory / MANIFEST_NAME).read_text())
        raw["contributors"][0]["policy"]["uid"] = "urn:p:someone-else"
        with pytest.raises(VerificationError, match="digest does not match"):
            Manifest.from_dict(raw)

    def test_a_stale_agreement_digest_is_detected(self, tmp_path):
        directory = bundle_at(tmp_path)
        raw = json.loads((directory / MANIFEST_NAME).read_text())
        raw["agreement_digest"] = "0" * 64
        (directory / MANIFEST_NAME).write_text(json.dumps(raw))
        with pytest.raises(VerificationError, match="has been altered"):
            verify(directory)

    def test_an_unsigned_manifest_can_be_rewritten_wholesale(self, tmp_path):
        """The stated limit, asserted so it is never mistaken for a guarantee.

        A forger who rewrites the contributor declarations *and* the agreement
        consistently produces a bundle that verifies. Detection covers accident
        and one-sided edits, not forgery; closing it needs signatures.
        """
        directory = bundle_at(tmp_path)
        forged = [offer("urn:p:a"), offer("urn:p:b")]  # unconstrained purposes
        manifest = build_manifest(
            compose(forged).agreement,
            [ContributorRecord(node_id=i, policy=p) for i, p in enumerate(forged)],
        )
        export(directory, manifest, Net().state_dict())

        verified = verify(directory)
        assert verified.capability == {Action.USE, Action.AGGREGATE}


class TestRecursion:
    def test_a_bundle_agreement_composes_into_a_further_federation(self, tmp_path):
        """What makes "policy-carrying" more than a label."""
        manifest = load(bundle_at(tmp_path))
        reused = as_contributor_policy(manifest)
        newcomer = offer("urn:p:c", "research")

        second = compose([reused, newcomer]).agreement
        assert capability(second) <= manifest.capability

    def test_reuse_can_only_narrow_capability(self, tmp_path):
        """A restricted model cannot be laundered into an unrestricted one."""
        manifest = load(
            bundle_at(tmp_path, policies=[offer("urn:p:a", "ds-cancer"),
                                          offer("urn:p:b", "ds-cancer")])
        )
        permissive = offer("urn:p:open")  # no purpose constraint at all
        second = compose([as_contributor_policy(manifest), permissive]).agreement

        assert capability(second) <= manifest.capability
        purposes = {
            frozenset(c.right_operand)
            for p in second.permissions
            for c in p.constraints
            if c.left_operand is LeftOperand.PURPOSE
        }
        assert purposes == {frozenset({"ds-cancer"})}

    def test_the_derived_agreement_names_the_bundle_it_came_from(self, tmp_path):
        manifest = load(bundle_at(tmp_path))
        second = compose(
            [as_contributor_policy(manifest), offer("urn:p:c", "research")]
        ).agreement
        assert manifest.agreement.uid in second.derived_from


class TestDigests:
    def test_a_policy_digest_is_stable(self, tmp_path):
        policy = offer("urn:p:a", "research")
        assert policy_digest(policy) == policy_digest(offer("urn:p:a", "research"))

    def test_different_terms_give_different_digests(self):
        assert policy_digest(offer("urn:p:a", "research")) != policy_digest(
            offer("urn:p:a", "ds-cancer")
        )
