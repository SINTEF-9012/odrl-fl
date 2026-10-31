"""Subsumption hierarchies for set-valued constraint operands.

Composition intersects the value sets of set-valued constraints, and doing that
naively is wrong in a way that would quietly corrupt the headline result.
`spatial isAnyOf {EU}` and `spatial isAnyOf {DE}` are compatible -- Germany is
in the EU -- but plain set intersection gives the empty set and reports a
conflict. Every such false conflict inflates measured capability erosion, so
increment 4's curve would be partly an artifact of this module being absent.

A taxonomy is a partial order per operand: a term subsumes the terms below it.
The **meet** of two value sets is the set of terms covered by both, reduced to
its most general members:

    {EU} ⊓ {DE}      = {DE}        Germany is the common ground
    {EU} ⊓ {US}      = {}          nothing in common
    {EU, US} ⊓ {DE}  = {DE}
    {research} ⊓ {clinical-research} = {clinical-research}

Terms not mentioned in the taxonomy subsume only themselves, so an unknown
vocabulary degrades to exact matching rather than failing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

#: Vocabulary lives outside `policies/`, which holds only policy documents --
#: `load_policies` globs that directory.
DEFAULT_TAXONOMY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "vocab" / "taxonomy.json"
)


@dataclass(frozen=True)
class Taxonomy:
    """Partial orders over constraint values, one per left operand.

    `broader[operand][term]` holds the terms that `term` is directly broader
    than. Transitivity is resolved on lookup and cached.
    """

    broader: Mapping[str, Mapping[str, frozenset[str]]] = field(default_factory=dict)
    #: Transitive-closure memo. The erosion study composes thousands of policy
    #: sets, and every composition walks this order repeatedly.
    _memo: dict[tuple[str, str], frozenset[str]] = field(
        default_factory=dict, repr=False, compare=False
    )

    # -- order ------------------------------------------------------------

    def descendants(self, operand: str, term: str) -> frozenset[str]:
        """Every term at or below `term`, transitively."""
        key = (operand, term)
        cached = self._memo.get(key)
        if cached is not None:
            return cached

        edges = self.broader.get(operand, {})
        seen: set[str] = set()
        stack = [term]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(edges.get(current, ()))

        result = frozenset(seen)
        self._memo[key] = result
        return result

    def subsumes(self, operand: str, broad: str, narrow: str) -> bool:
        """Is `broad` the same as, or more general than, `narrow`?"""
        return narrow in self.descendants(operand, broad)

    def meet(
        self, operand: str, left: Iterable[str], right: Iterable[str]
    ) -> frozenset[str]:
        """Greatest lower bound of two value sets.

        A term survives if it is covered by *some* member of each side. The
        result is then reduced to maximal elements: if both `EU` and `DE`
        survive, `EU` already permits `DE`, so keeping both is redundant.
        """
        left_set, right_set = frozenset(left), frozenset(right)

        # Candidates are the terms named on either side. Nothing below both can
        # be expressed without inventing vocabulary, so this is complete.
        covered = {
            term
            for term in left_set | right_set
            if any(self.subsumes(operand, a, term) for a in left_set)
            and any(self.subsumes(operand, b, term) for b in right_set)
        }
        return frozenset(
            term
            for term in covered
            if not any(
                other != term and self.subsumes(operand, other, term)
                for other in covered
            )
        )

    def join(self, operand: str, left: Iterable[str], right: Iterable[str]) -> frozenset[str]:
        """Union, reduced to maximal elements.

        Used for negative (`isNoneOf`) constraints, where prohibiting `EU` and
        prohibiting `DE` prohibits exactly `EU`.
        """
        combined = frozenset(left) | frozenset(right)
        return frozenset(
            term
            for term in combined
            if not any(
                other != term and self.subsumes(operand, other, term)
                for other in combined
            )
        )

    def difference(
        self, operand: str, left: Iterable[str], right: Iterable[str]
    ) -> frozenset[str]:
        """Terms permitted by `left` and not excluded by `right`.

        Exclusion is inherited downwards: excluding `EU` excludes `DE`. A term
        broader than an excluded one survives, since only part of it is
        excluded -- conservative, and the alternative would need vocabulary
        that does not exist ("EU except DE").
        """
        excluded: set[str] = set()
        for term in right:
            excluded |= self.descendants(operand, term)
        return frozenset(term for term in left if term not in excluded)

    # -- construction -----------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Mapping[str, Iterable[str]]]) -> Taxonomy:
        return cls(
            broader={
                operand: {
                    term: frozenset(narrower) for term, narrower in terms.items()
                }
                for operand, terms in data.items()
                if not operand.startswith("_")
            }
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_TAXONOMY_PATH) -> Taxonomy:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def extended(
        self, additions: Mapping[str, Mapping[str, Iterable[str]]]
    ) -> Taxonomy:
        """A new taxonomy with extra edges merged in.

        The hook for increment 3, which layers the DUO hierarchy on top.
        """
        merged: dict[str, dict[str, frozenset[str]]] = {
            operand: dict(terms) for operand, terms in self.broader.items()
        }
        for operand, terms in additions.items():
            if operand.startswith("_"):
                continue
            target = merged.setdefault(operand, {})
            for term, narrower in terms.items():
                target[term] = target.get(term, frozenset()) | frozenset(narrower)
        return Taxonomy(broader=merged)


#: Empty taxonomy: every term subsumes only itself, so composition falls back
#: to exact set intersection. Useful for isolating the effect of subsumption.
FLAT = Taxonomy()


@lru_cache(maxsize=1)
def default_taxonomy() -> Taxonomy:
    """The shipped hierarchy, loaded once."""
    return Taxonomy.load()
