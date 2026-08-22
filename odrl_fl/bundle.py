"""Policy-carrying models: shipping a model with the terms it was built under.

A trained federated model is a derivative of every contributor's data, and the
composed Agreement says what may lawfully be done with it. Separate those two
and the terms are lost the first time the file is copied -- which is the normal
state of affairs, and the reason a model's provenance is usually a paragraph in
a README rather than something anyone can check.

A bundle keeps them together and makes the pairing **checkable**:

    bundle/
      model.pt        the weights
      manifest.json   the agreement, who contributed, what is outstanding

`verify` recomputes the agreement from the contributor policies recorded in the
manifest and confirms it matches the agreement the manifest claims, then checks
the weights still hash to what was recorded. The bundle is self-contained: no
registry, no external state, nothing to consult. Editing the agreement to widen
a purpose fails verification, because the recorded contributor policies no
longer compose to it.

**Why this is the increment that matters for duties.** Increment 7 found that
of the duties real DUO annotations produce, almost none can be enforced during
training -- publication, returning findings, attribution and compensation are
all discharged after the model exists, by whoever holds it. A holder who does
not know what is owed cannot discharge any of them. Carrying the ledger with
the weights is the mechanism that makes those obligations actionable at all.

**Policy-carrying is recursive.** An Agreement is a Policy, so a bundle's
agreement can be fed straight back into `compose` as an input when the model is
reused in another federation. Capability erodes further, which is correct:
reusing a restricted model cannot widen what may be done with the result. That
recursion is what makes "policy-carrying" more than a label.

**What this is not.** The manifest is not signed. Anyone able to rewrite the
weights can rewrite the manifest to match, and verification would pass. It
detects accident and unilateral edits to one half of the pair, not a determined
forger; that needs signatures over the digest and a key distribution story,
which is future work rather than an oversight.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from odrl_fl.duties import Ledger, Obligation
from odrl_fl.odrl import (
    Action,
    Policy,
    capability,
    compose,
    parse_policy,
    policy_to_jsonld,
)

MANIFEST_NAME = "manifest.json"
MODEL_NAME = "model.pt"

#: Bumped when the manifest layout changes incompatibly. A reader that does not
#: recognise the version refuses rather than guessing at a layout it predates.
MANIFEST_VERSION = 1


class VerificationError(Exception):
    """A bundle does not attest to what it claims."""


def digest_of(payload: Any) -> str:
    """SHA-256 over a canonical JSON encoding.

    Canonical because a digest that depends on key order or whitespace is not a
    digest of the content. `policy_to_jsonld` already emits collections sorted,
    so a policy has one serialisation and therefore one digest.
    """
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def policy_digest(policy: Policy) -> str:
    return digest_of(policy_to_jsonld(policy))


@dataclass(frozen=True)
class ContributorRecord:
    """One party's contribution, as the manifest records it."""

    node_id: int
    policy: Policy
    codes: tuple[str, ...] = ()
    n_train: int = 0

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "policy": policy_to_jsonld(self.policy),
            "policy_digest": policy_digest(self.policy),
            "duo_codes": list(self.codes),
            "n_train": self.n_train,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> ContributorRecord:
        policy = parse_policy(payload["policy"])
        recorded = payload.get("policy_digest")
        if recorded and recorded != policy_digest(policy):
            raise VerificationError(
                f"node {payload.get('node_id')}: recorded policy digest does not "
                "match the recorded policy"
            )
        return cls(
            node_id=int(payload["node_id"]),
            policy=policy,
            codes=tuple(payload.get("duo_codes") or ()),
            n_train=int(payload.get("n_train") or 0),
        )


@dataclass(frozen=True)
class Manifest:
    """Everything needed to say what a model is, and to check the claim."""

    agreement: Policy
    contributors: tuple[ContributorRecord, ...]
    request: Policy | None = None
    excluded: Mapping[str, str] = field(default_factory=dict)
    outstanding: tuple[Obligation, ...] = ()
    gates_cleared: Mapping[str, str] = field(default_factory=dict)
    rejections: tuple[tuple[int, int, str], ...] = ()
    model_digest: str = ""
    metrics: Mapping[str, float] = field(default_factory=dict)
    created: str = ""
    version: int = MANIFEST_VERSION

    @property
    def capability(self) -> frozenset[Action]:
        """What the holder of this model may do with it."""
        return capability(self.agreement)

    @property
    def agreement_digest(self) -> str:
        return policy_digest(self.agreement)

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "created": self.created,
            "agreement": policy_to_jsonld(self.agreement),
            "agreement_digest": self.agreement_digest,
            "capability": sorted(a.value for a in self.capability),
            "contributors": [c.as_dict() for c in self.contributors],
            "request": policy_to_jsonld(self.request) if self.request else None,
            "excluded": dict(self.excluded),
            # Recorded duties are carried open. The holder of this model owes
            # them; nothing in a training run could have discharged them.
            "outstanding_duties": [
                {
                    "action": o.action,
                    "owed_by": list(o.owed_by),
                    "detail": o.detail,
                    "discharged": o.discharged,
                }
                for o in self.outstanding
            ],
            "gates_cleared": dict(self.gates_cleared),
            "rejected_updates": [
                {"round": r, "node_id": n, "reason": why}
                for r, n, why in self.rejections
            ],
            "model_digest": self.model_digest,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> Manifest:
        version = int(payload.get("version", 0))
        if version != MANIFEST_VERSION:
            raise VerificationError(
                f"manifest version {version} is not readable by this build "
                f"(expected {MANIFEST_VERSION})"
            )
        agreement = parse_policy(payload["agreement"])
        recorded = payload.get("agreement_digest")
        if recorded and recorded != policy_digest(agreement):
            raise VerificationError(
                "the recorded agreement digest does not match the recorded "
                "agreement -- the manifest has been altered"
            )
        return cls(
            agreement=agreement,
            contributors=tuple(
                ContributorRecord.from_dict(c) for c in payload["contributors"]
            ),
            request=(
                parse_policy(payload["request"]) if payload.get("request") else None
            ),
            excluded={str(k): str(v) for k, v in (payload.get("excluded") or {}).items()},
            outstanding=tuple(
                Obligation(
                    action=str(o["action"]),
                    owed_by=tuple(o.get("owed_by") or ()),
                    detail=str(o.get("detail") or ""),
                    discharged=bool(o.get("discharged")),
                )
                for o in payload.get("outstanding_duties") or ()
            ),
            gates_cleared=dict(payload.get("gates_cleared") or {}),
            rejections=tuple(
                (int(r["round"]), int(r["node_id"]), str(r["reason"]))
                for r in payload.get("rejected_updates") or ()
            ),
            model_digest=str(payload.get("model_digest") or ""),
            metrics=dict(payload.get("metrics") or {}),
            created=str(payload.get("created") or ""),
            version=version,
        )

    def describe(self) -> list[str]:
        """A human-readable account. What a holder needs to know, in order."""
        lines = [
            f"model    {self.model_digest[:16] or '(no digest)'}",
            f"terms    {self.agreement.uid}",
            f"built by {len(self.contributors)} contributors",
            "you may  " + (
                ", ".join(sorted(a.value for a in self.capability)) or "-- nothing --"
            ),
        ]
        if self.gates_cleared:
            for action, reference in sorted(self.gates_cleared.items()):
                lines.append(f"cleared  {action}, evidence {reference}")
        for obligation in self.outstanding:
            owed = (
                f"all {len(obligation.owed_by)} contributors"
                if len(obligation.owed_by) > 2
                else ", ".join(obligation.owed_by) or "the consortium"
            )
            lines.append(f"YOU OWE  {obligation.action} -- to {owed}")
        if self.rejections:
            lines.append(f"dropped  {len(self.rejections)} update(s) for duty failures")
        return lines


def build_manifest(
    agreement: Policy,
    contributors: Sequence[ContributorRecord],
    *,
    request: Policy | None = None,
    excluded: Mapping[int, str] | None = None,
    ledger: Ledger | None = None,
    metrics: Mapping[str, float] | None = None,
    model_digest: str = "",
    created: str | None = None,
) -> Manifest:
    ledger = ledger or Ledger()
    return Manifest(
        agreement=agreement,
        contributors=tuple(contributors),
        request=request,
        excluded={str(k): v for k, v in (excluded or {}).items()},
        outstanding=tuple(ledger.outstanding),
        gates_cleared=dict(ledger.gates_cleared),
        rejections=tuple(ledger.rejections),
        model_digest=model_digest,
        metrics=dict(metrics or {}),
        created=created or datetime.now(UTC).isoformat(timespec="seconds"),
    )


def _model_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(
    directory: str | Path,
    manifest: Manifest,
    state_dict: Mapping[str, Any] | None = None,
) -> Path:
    """Write a bundle. Returns its directory.

    The model digest is computed from the bytes actually written, not from the
    object in memory, so the manifest attests to the file a reader will load.
    """
    import torch

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    if state_dict is not None:
        model_path = out / MODEL_NAME
        torch.save(dict(state_dict), model_path)
        manifest = replace(manifest, model_digest=_model_digest(model_path))

    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def load(directory: str | Path) -> Manifest:
    """Read a bundle's manifest. Does not verify it -- call `verify` for that."""
    path = Path(directory) / MANIFEST_NAME
    return Manifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def verify(directory: str | Path, *, check_model: bool = True) -> Manifest:
    """Confirm a bundle attests to what it claims. Raises `VerificationError`.

    Three checks, and the first is the substantive one:

      1. the recorded contributor policies **recompose** to the recorded
         agreement -- so the terms cannot be widened without also rewriting
         every contributor's declaration to justify the change;
      2. the declared capability matches what the agreement actually grants;
      3. the weights on disk still hash to what the manifest recorded.

    Not a defence against a determined forger -- see the module docstring.
    """
    directory = Path(directory)
    manifest = load(directory)

    if not manifest.contributors:
        raise VerificationError("the manifest records no contributors")

    recomposed = compose([c.policy for c in manifest.contributors]).agreement
    if recomposed != manifest.agreement:
        raise VerificationError(
            "the recorded contributor policies do not compose to the recorded "
            f"agreement (they yield {recomposed.uid}, the manifest claims "
            f"{manifest.agreement.uid})"
        )

    declared = frozenset(manifest.capability)
    if declared != capability(recomposed):
        raise VerificationError(
            "the declared capability is not what the agreement grants"
        )

    if check_model and manifest.model_digest:
        model_path = directory / MODEL_NAME
        if not model_path.exists():
            raise VerificationError(
                "the manifest records a model digest but no model is present"
            )
        actual = _model_digest(model_path)
        if actual != manifest.model_digest:
            raise VerificationError(
                "the model file does not match the digest the manifest records"
            )

    return manifest


def as_contributor_policy(manifest: Manifest) -> Policy:
    """The bundle's agreement, ready to be composed into a further federation.

    The recursion that makes the term "policy-carrying" mean something. Reusing
    this model in another consortium composes its terms with the newcomers',
    and capability can only narrow -- a restricted model cannot be laundered
    into an unrestricted one by training something else on it.
    """
    return manifest.agreement
