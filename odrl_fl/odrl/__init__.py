"""ODRL policies: typed model, JSON-LD parser, and (from increment 2) the
composition algebra.

The subset supported here is deliberately narrow and is documented in
`model.py`. Constructs outside it raise `UnsupportedConstruct` rather than
being ignored, because a silently dropped constraint yields a composed
capability nobody granted.
"""

from odrl_fl.odrl.composition import (
    CompositionResult,
    LossReason,
    Unsatisfiable,
    capability,
    compose,
    conjoin,
    merge_duties,
    satisfies,
)
from odrl_fl.odrl.model import (
    SELF_REFERENTIAL_OPERANDS,
    SET_VALUED,
    STATEFUL_OPERANDS,
    Action,
    Constraint,
    Duty,
    LeftOperand,
    MalformedPolicy,
    ODRLError,
    Operator,
    Permission,
    Policy,
    PolicyType,
    Prohibition,
    UnsupportedConstruct,
)
from odrl_fl.odrl.parser import (
    load_policies,
    load_policy,
    parse_constraint,
    parse_policy,
    policy_to_jsonld,
)
from odrl_fl.odrl.taxonomy import FLAT, Taxonomy, default_taxonomy

__all__ = [
    "FLAT",
    "SELF_REFERENTIAL_OPERANDS",
    "SET_VALUED",
    "STATEFUL_OPERANDS",
    "Action",
    "CompositionResult",
    "Constraint",
    "Duty",
    "LeftOperand",
    "LossReason",
    "MalformedPolicy",
    "ODRLError",
    "Operator",
    "Permission",
    "Policy",
    "PolicyType",
    "Prohibition",
    "Taxonomy",
    "Unsatisfiable",
    "UnsupportedConstruct",
    "capability",
    "compose",
    "conjoin",
    "default_taxonomy",
    "load_policies",
    "load_policy",
    "merge_duties",
    "parse_constraint",
    "parse_policy",
    "policy_to_jsonld",
    "satisfies",
]
