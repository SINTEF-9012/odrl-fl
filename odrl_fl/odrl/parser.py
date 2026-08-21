"""Read and write ODRL policies as JSON-LD.

Handles the pragmatics of real ODRL documents rather than an idealised form:
terms appear bare (`use`) or prefixed (`odrl:use`) or as full IRIs; JSON-LD
allows a single object wherever a list is permitted; `target` and `assigner`
may be strings or objects with `@id`.

The parser is **strict about the subset**. A `count` constraint raises rather
than being skipped, because a policy that silently loses a restriction would
yield a composed capability nobody granted -- and it would look entirely
plausible in the results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from odrl_fl.odrl.model import (
    SELF_REFERENTIAL_OPERANDS,
    STATEFUL_OPERANDS,
    Action,
    Constraint,
    Duty,
    LeftOperand,
    MalformedPolicy,
    Operator,
    Permission,
    Policy,
    PolicyType,
    Prohibition,
    UnsupportedConstruct,
)


def _term(value: Any) -> str:
    """Strip JSON-LD prefixes and IRI paths down to the bare term.

    `odrl:use`, `http://www.w3.org/ns/odrl/2/use` and `use` all denote the same
    thing, and real policies use all three.
    """
    if isinstance(value, dict):
        value = value.get("@id") or value.get("id") or value.get("@value")
    if value is None:
        raise MalformedPolicy("expected a term, found nothing")
    text = str(value)
    for separator in ("#", "/", ":"):
        if separator in text:
            text = text.rsplit(separator, 1)[-1]
    return text


def _identity(value: Any) -> str:
    """Resolve a party or asset reference, leaving its namespace intact.

    Deliberately *not* `_term`. `assigner`, `assignee` and `target` name
    entities rather than vocabulary, and an identity has no canonical short
    form to compact towards: reducing `urn:party:hospital-a` to `hospital-a`
    discards what makes it unique and collides it with `urn:asset:hospital-a`.
    `uid` has always been read this way; these three were not.

    It matters on the wire. Increment 6 announces policies as JSON-LD between
    contributor and server, and increment 8 hashes that serialisation into a
    provenance record. Neither survives a parser that cannot round-trip.
    """
    if isinstance(value, dict):
        value = value.get("@id") or value.get("id") or value.get("@value")
    if value is None:
        raise MalformedPolicy("expected an identifier, found nothing")
    return str(value)


def _as_list(value: Any) -> list[Any]:
    """JSON-LD permits a bare object anywhere a list is allowed."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _enum(cls, value: Any, what: str):
    term = _term(value)
    try:
        return cls(term)
    except ValueError as exc:
        supported = ", ".join(sorted(m.value for m in cls))
        raise UnsupportedConstruct(
            f"{what} '{term}' is outside the supported subset (have: {supported})"
        ) from exc


def parse_constraint(doc: dict[str, Any]) -> Constraint:
    if not isinstance(doc, dict):
        raise MalformedPolicy(f"constraint must be an object, got {type(doc).__name__}")

    for container in ("and", "or", "xone", "andSequence"):
        if container in doc:
            raise UnsupportedConstruct(
                f"logical constraint '{container}' is not supported; constraints on a "
                "rule are conjunctive, and disjunction needs a normal form before "
                "composition"
            )

    if "leftOperand" not in doc:
        raise MalformedPolicy("constraint is missing 'leftOperand'")

    operand = _term(doc["leftOperand"])

    if operand in STATEFUL_OPERANDS:
        raise UnsupportedConstruct(
            f"'{operand}' is stateful and has no defined composition across parties: "
            "if A permits 20 uses and B permits 30, the derived model's budget is "
            "neither 20, 30, nor 50"
        )
    if operand in SELF_REFERENTIAL_OPERANDS:
        raise UnsupportedConstruct(
            f"'{operand}' depends on the contributor set being composed, which makes "
            "composition a fixpoint rather than a fold"
        )

    left = _enum(LeftOperand, operand, "left operand")
    operator = _enum(Operator, doc.get("operator", "eq"), "operator")

    if "rightOperand" not in doc and "rightOperandReference" not in doc:
        raise MalformedPolicy(f"constraint on '{operand}' is missing 'rightOperand'")
    raw = doc.get("rightOperand", doc.get("rightOperandReference"))

    if isinstance(raw, list):
        value: Any = frozenset(_term(v) for v in raw)
    elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
    elif isinstance(raw, dict) and "@value" in raw:
        inner = raw["@value"]
        value = float(inner) if isinstance(inner, (int, float)) else str(inner)
    else:
        value = _term(raw)

    return Constraint(left_operand=left, operator=operator, right_operand=value)


def _parse_constraints(doc: dict[str, Any]) -> frozenset[Constraint]:
    return frozenset(
        parse_constraint(c) for c in _as_list(doc.get("constraint") or doc.get("refinement"))
    )


def parse_duty(doc: dict[str, Any]) -> Duty:
    if "action" not in doc:
        raise MalformedPolicy("duty is missing 'action'")
    for unsupported in ("consequence", "remedy"):
        if unsupported in doc:
            raise UnsupportedConstruct(
                f"'{unsupported}' (a duty triggered by violation) is not supported; "
                "a violated duty voids the permission instead"
            )
    return Duty(
        action=_enum(Action, doc["action"], "action"),
        constraints=_parse_constraints(doc),
    )


def parse_permission(doc: dict[str, Any]) -> Permission:
    if "action" not in doc:
        raise MalformedPolicy("permission is missing 'action'")
    return Permission(
        action=_enum(Action, doc["action"], "action"),
        constraints=_parse_constraints(doc),
        duties=frozenset(parse_duty(d) for d in _as_list(doc.get("duty"))),
    )


def parse_prohibition(doc: dict[str, Any]) -> Prohibition:
    if "action" not in doc:
        raise MalformedPolicy("prohibition is missing 'action'")
    if "remedy" in doc:
        raise UnsupportedConstruct(
            "'remedy' (a duty triggered by violating a prohibition) is not supported"
        )
    return Prohibition(
        action=_enum(Action, doc["action"], "action"),
        constraints=_parse_constraints(doc),
    )


def parse_policy(doc: dict[str, Any] | str) -> Policy:
    """Parse an ODRL JSON-LD policy into the typed model."""
    if isinstance(doc, str):
        doc = json.loads(doc)
    if not isinstance(doc, dict):
        raise MalformedPolicy("a policy must be a JSON object")

    uid = doc.get("uid") or doc.get("@id")
    if not uid:
        raise MalformedPolicy("policy is missing 'uid'")

    policy_type = doc.get("@type") or doc.get("type") or "Offer"
    try:
        parsed_type = PolicyType(_term(policy_type))
    except ValueError as exc:
        raise UnsupportedConstruct(
            f"policy type '{_term(policy_type)}' is not supported "
            f"(have: {', '.join(t.value for t in PolicyType)})"
        ) from exc

    permissions = frozenset(parse_permission(p) for p in _as_list(doc.get("permission")))
    prohibitions = frozenset(
        parse_prohibition(p) for p in _as_list(doc.get("prohibition"))
    )
    obligations = frozenset(
        parse_duty(d) for d in _as_list(doc.get("obligation") or doc.get("duty"))
    )

    if not (permissions or prohibitions or obligations):
        raise MalformedPolicy(f"policy '{uid}' contains no rules")

    def optional(key: str) -> str | None:
        return _identity(doc[key]) if doc.get(key) is not None else None

    return Policy(
        uid=str(uid),
        type=parsed_type,
        assigner=optional("assigner"),
        assignee=optional("assignee"),
        target=optional("target"),
        permissions=permissions,
        prohibitions=prohibitions,
        obligations=obligations,
        # `inheritFrom` is ODRL's own term for a policy derived from others,
        # which is exactly what composition produces.
        derived_from=frozenset(str(v) for v in _as_list(doc.get("inheritFrom"))),
    )


# --- Serialisation ---------------------------------------------------------


def _constraint_to_jsonld(constraint: Constraint) -> dict[str, Any]:
    if isinstance(constraint.right_operand, frozenset):
        right: Any = sorted(constraint.right_operand)
    else:
        right = constraint.right_operand
    return {
        "leftOperand": constraint.left_operand.value,
        "operator": constraint.operator.value,
        "rightOperand": right,
    }


def _rule_to_jsonld(rule: Permission | Prohibition | Duty) -> dict[str, Any]:
    out: dict[str, Any] = {"action": rule.action.value}
    if rule.constraints:
        out["constraint"] = [
            _constraint_to_jsonld(c) for c in sorted(rule.constraints)
        ]
    duties = getattr(rule, "duties", frozenset())
    if duties:
        out["duty"] = [_rule_to_jsonld(d) for d in sorted(duties)]
    return out


def policy_to_jsonld(policy: Policy) -> dict[str, Any]:
    """Serialise back to JSON-LD.

    Collections are emitted in sorted order so the output is byte-stable --
    which matters for hashing a policy into a provenance record later.
    """
    doc: dict[str, Any] = {
        "@context": "http://www.w3.org/ns/odrl.jsonld",
        "@type": policy.type.value,
        "uid": policy.uid,
    }
    for key in ("assigner", "assignee", "target"):
        value = getattr(policy, key)
        if value:
            doc[key] = value
    if policy.derived_from:
        doc["inheritFrom"] = sorted(policy.derived_from)

    if policy.permissions:
        doc["permission"] = [_rule_to_jsonld(p) for p in sorted(policy.permissions)]
    if policy.prohibitions:
        doc["prohibition"] = [_rule_to_jsonld(p) for p in sorted(policy.prohibitions)]
    if policy.obligations:
        doc["obligation"] = [_rule_to_jsonld(d) for d in sorted(policy.obligations)]
    return doc


def load_policy(path: str | Path) -> Policy:
    return parse_policy(json.loads(Path(path).read_text(encoding="utf-8")))


def load_policies(directory: str | Path) -> list[Policy]:
    """Load every `*.json` policy in a directory, sorted by filename."""
    return [load_policy(p) for p in sorted(Path(directory).glob("*.json"))]
