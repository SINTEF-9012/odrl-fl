"""ODRL policies: typed model, JSON-LD parser, and (from increment 2) the
composition algebra.

The subset supported here is deliberately narrow and is documented in
`model.py`. Constructs outside it raise `UnsupportedConstruct` rather than
being ignored, because a silently dropped constraint yields a composed
capability nobody granted.
"""

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

__all__ = [
    "SELF_REFERENTIAL_OPERANDS",
    "SET_VALUED",
    "STATEFUL_OPERANDS",
    "Action",
    "Constraint",
    "Duty",
    "LeftOperand",
    "MalformedPolicy",
    "ODRLError",
    "Operator",
    "Permission",
    "Policy",
    "PolicyType",
    "Prohibition",
    "UnsupportedConstruct",
    "load_policies",
    "load_policy",
    "parse_constraint",
    "parse_policy",
    "policy_to_jsonld",
]
