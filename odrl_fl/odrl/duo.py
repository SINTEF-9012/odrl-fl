"""Translating GA4GH Data Use Ontology codes into ODRL offers.

DUO is a controlled vocabulary of data use conditions, already attached to real
datasets. It supplies the *terms*; ODRL supplies the *grammar* that lets those
terms be composed across contributors. This module is the bridge, and the
mapping table it reads (`vocab/duo.json`) is a deliverable in its own right.

**Not every DUO code fits.** Classifying them is part of the contribution:

- `permission`             sets the purpose the data may serve (GRU, HMB, DS)
- `restriction`            a constraint or prohibition the algebra folds
                           (NCU, GS, TS, NMDS)
- `externally-adjudicated` composes syntactically, but an authority decides its
                           truth out of band (IRB, US, PS, IS)
- `duty`                   an obligation rather than a restriction (PUB, RTN)
- `out-of-fragment`        cannot be expressed at all -- **COL**, which requires
                           the provider to be a participant and so depends on
                           the very contributor set being composed
- `structural`             an ontology category, not a usable condition
- `deprecated`             obsolete but still present in real annotations

That last class is not hypothetical. Among 404 annotated EGA policies, GRU-CC
(`DUO:0000005`) and RU (`DUO:0000014`) account for 132 code occurrences despite
having been retired from the ontology, so any tool meeting real data has to
decide what to do with them. Here they are mapped through the replacements the
ontology itself records.

**What a contributor grants by default.** DUO is a *limitation* vocabulary: it
says what may not be done, and is silent on the derived-model actions this work
is about. So a contributor is taken to grant `use`, `aggregate`, `derive` and
`distribute`, with codes restricting from there. That is a modelling assumption
rather than something DUO states, `DEFAULT_ACTIONS` makes it explicit, and
increment 4 should sensitivity-test it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from odrl_fl.odrl.model import (
    Action,
    Constraint,
    Duty,
    LeftOperand,
    Operator,
    Permission,
    Policy,
    PolicyType,
    Prohibition,
)

DUO_PATH = Path(__file__).resolve().parent.parent.parent / "vocab" / "duo.json"

#: Actions a contributor is taken to grant absent any restricting code.
DEFAULT_ACTIONS: tuple[Action, ...] = (
    Action.USE,
    Action.AGGREGATE,
    Action.DERIVE,
    Action.DISTRIBUTE,
)

#: Defaults for codes carrying a parameter the ontology does not fix.
DEFAULT_PARAMS: dict[str, str] = {
    "region": "EU",
    "expiry": "2028-12-31",
    "moratorium_until": "2027-01-01",
    "disease": "ds-cancer",
}


class UnknownDUOCode(KeyError):
    """A code absent from the mapping table.

    Raised rather than skipped, for the same reason the parser rejects
    unsupported constructs: a dropped condition grants capability nobody
    allowed, and it would look entirely plausible in the results.
    """


@lru_cache(maxsize=1)
def duo_table() -> dict[str, dict[str, Any]]:
    """The code -> mapping table, loaded once."""
    return json.loads(DUO_PATH.read_text(encoding="utf-8"))["codes"]


def code_class(code: str) -> str:
    table = duo_table()
    if code not in table:
        raise UnknownDUOCode(code)
    return table[code]["class"]


def is_composable(code: str) -> bool:
    """Does this code contribute anything the algebra can fold?"""
    return code_class(code) not in ("out-of-fragment", "structural")


def describe(code: str) -> str:
    entry = duo_table()[code]
    return (
        f"{code} ({entry['shorthand']}) {entry['label']} "
        f"[{entry['class']}/{entry['fidelity']}]"
    )


def translate(
    codes: Iterable[str],
    *,
    uid: str,
    assigner: str | None = None,
    target: str | None = None,
    params: Mapping[str, str] | None = None,
    actions: Iterable[Action] = DEFAULT_ACTIONS,
    strict: bool = True,
) -> tuple[Policy, tuple[str, ...]]:
    """Build an ODRL Offer from DUO codes.

    Returns the Offer together with the codes that could not be expressed, so a
    caller can report how much of a real annotation the fragment failed to
    capture rather than losing it silently.

    `params` supplies values the codes leave open -- a region for GS, an expiry
    for TS, a moratorium date for MOR.
    """
    table = duo_table()
    values = {**DEFAULT_PARAMS, **(params or {})}

    purposes: set[str] = set()
    purpose_excludes: set[str] = set()
    # Keyed by operand: codes restricting *different* dimensions must stay in
    # separate constraints so they conjoin. Merging them into one operand would
    # union their values into a disjunction and grant more than was offered.
    scoped: dict[str, set[str]] = {}
    shared: list[Constraint] = []
    prohibitions: list[Prohibition] = []
    obligations: list[Duty] = []
    pre_duties: list[Duty] = []
    moratoria: list[Constraint] = []
    unsupported: list[str] = []

    for code in sorted(set(codes)):
        entry = table.get(code)
        if entry is None:
            if strict:
                raise UnknownDUOCode(
                    f"{code} is not in the mapping table; add it to vocab/duo.json "
                    "rather than letting a condition be silently dropped"
                )
            unsupported.append(code)
            continue

        spec = entry["odrl"]
        kind = spec["kind"]

        if kind == "none":
            continue
        if kind == "unsupported":
            unsupported.append(code)
        elif kind == "purpose":
            purposes.update(spec["values"])
        elif kind == "purpose_param":
            # DS-[XX]: the permitted purpose is the specific disease consented
            # to, not disease-specific research in general.
            purposes.add(values[spec["param"]])
        elif kind == "purpose_exclude":
            purpose_excludes.update(spec["values"])
        elif kind == "constraint":
            scoped.setdefault(spec["operand"], set()).update(spec["values"])
        elif kind == "constraint_and_exclude":
            scoped.setdefault(spec["operand"], set()).update(spec["values"])
            purpose_excludes.update(spec["exclude"])
        elif kind == "spatial":
            shared.append(
                Constraint(
                    LeftOperand.SPATIAL,
                    Operator.IS_ANY_OF,
                    frozenset({values[spec["param"]]}),
                )
            )
        elif kind == "until":
            shared.append(
                Constraint(LeftOperand.DATETIME, Operator.LTEQ, values[spec["param"]])
            )
        elif kind == "moratorium":
            moratoria.append(
                Constraint(LeftOperand.DATETIME, Operator.GTEQ, values[spec["param"]])
            )
        elif kind == "prohibit":
            constraints = (
                frozenset(
                    {
                        Constraint(
                            LeftOperand.PURPOSE,
                            Operator.IS_ANY_OF,
                            frozenset(spec["purpose"]),
                        )
                    }
                )
                if spec.get("purpose")
                else frozenset()
            )
            prohibitions.append(
                Prohibition(action=Action(spec["action"]), constraints=constraints)
            )
        elif kind == "duty":
            obligations.append(Duty(action=Action(spec["action"])))
        elif kind == "pre_duty":
            pre_duties.append(Duty(action=Action(spec["action"])))
        else:  # pragma: no cover - guarded by the exhaustiveness test
            raise ValueError(f"unhandled mapping kind '{kind}' for {code}")

    base: list[Constraint] = list(shared)
    if purposes:
        base.append(
            Constraint(LeftOperand.PURPOSE, Operator.IS_ANY_OF, frozenset(purposes))
        )
    if purpose_excludes:
        base.append(
            Constraint(
                LeftOperand.PURPOSE, Operator.IS_NONE_OF, frozenset(purpose_excludes)
            )
        )
    for operand, values_for_operand in sorted(scoped.items()):
        base.append(
            Constraint(
                LeftOperand(operand), Operator.IS_ANY_OF, frozenset(values_for_operand)
            )
        )

    permissions = []
    for action in actions:
        constraints = list(base)
        # A publication moratorium delays release, so it binds distribution
        # rather than use.
        if action is Action.DISTRIBUTE:
            constraints.extend(moratoria)
        permissions.append(
            Permission(
                action=action,
                constraints=frozenset(constraints),
                duties=frozenset(pre_duties),
            )
        )

    policy = Policy(
        uid=uid,
        type=PolicyType.OFFER,
        assigner=assigner,
        target=target,
        permissions=frozenset(permissions),
        prohibitions=frozenset(prohibitions),
        obligations=frozenset(obligations),
    )
    return policy, tuple(unsupported)
