"""Two-stage member selection: filter by policy, then rank by a short run.

**Why there are two stages.** The policy filter of `odrl_fl.selection` is free
and sound -- anti-monotonicity means a candidate it removes cannot be rescued by
adding members -- but it answers only which consortia are *permitted*. Among the
permitted ones it has nothing to say, and increment 13 established that nothing
cheap does: all three training-free estimators in `odrl_fl.estimators` collapse
to "take the largest feasible set", and the novel one anti-correlates with
accuracy. So the second stage has to train. The question this module answers is
how little it can get away with.

**What a short run can and cannot do.** It can separate bad consortia from good
ones. It cannot reliably separate good from best: ranking the candidates by their
round-8 accuracy agrees with the final ranking at Kendall tau 0.535, and the
truly best candidate sits around rank 20 of 128. What makes the shortlist work
anyway is that by that point the top of the field has flattened, so being wrong
about the order costs almost nothing in final accuracy. The claim to make is
near-optimality cheaply, never recovery of the optimum.

Which round that happens at is a property of the task, not a constant. Report
`sweep` as a curve and let a deployment pick its own point on it.

**The winner's curse, and how `evaluate_on` deals with it.** An argmax over a
thousand noisy candidates is biased upward, because the winner is partly
whichever candidate drew the kindest initialisation. Measuring regret against
that winner therefore flatters the selector. Passing separate replicates to
`select_on` and `evaluate_on` removes the bias: the shortlist is chosen from one
set of training runs and scored on another, exactly as a held-out split works.
With one replicate the two coincide and the result is an upper bound on error,
which `sweep` records in its `holdout` field rather than leaving to be inferred.

Everything here is a pure function over plain numbers. No training happens in
this module, which is what lets the whole cost-fidelity surface be computed from
a curve file in under a second.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: A consortium, identified by its sorted members.
Members = tuple[int, ...]

#: Per-round accuracy of one training run, round 1 first.
Curve = Sequence[float]


def shortlist(scores: Mapping[Members, float], k: int) -> list[Members]:
    """The `k` highest-scoring candidates, best first.

    Ties break on the members themselves rather than on dictionary order, so
    that a shortlist is a function of its input and not of how the mapping was
    built. Without that, a campaign split across array tasks could report
    different shortlists for identical data.
    """
    if k < 1:
        raise ValueError("a shortlist holds at least one candidate")
    return sorted(scores, key=lambda m: (-scores[m], m))[:k]


def rounds_spent(
    n_candidates: int, explore_rounds: int, k: int, total_rounds: int
) -> int:
    """Training rounds a two-stage run costs.

    Every candidate is explored, then only the shortlist is finished. The first
    term dominates, which is why the policy filter matters: it is what makes
    `n_candidates` small enough for the second stage to be affordable at all.
    """
    if not 1 <= explore_rounds <= total_rounds:
        raise ValueError("explore_rounds must lie within the training budget")
    return n_candidates * explore_rounds + k * (total_rounds - explore_rounds)


def regret(picked: Sequence[Members], truth: Mapping[Members, float]) -> float:
    """How much final accuracy the shortlist gave up against the best candidate.

    The shortlist is credited with its *best* member, because a deployment that
    finishes `k` candidates keeps the best of them. Zero means the optimum was
    inside the shortlist, never that it was ranked first.
    """
    if not picked:
        raise ValueError("nothing was picked")
    missing = [m for m in picked if m not in truth]
    if missing:
        raise KeyError(f"no ground truth for {missing[0]}")
    return max(truth.values()) - max(truth[m] for m in picked)


@dataclass(frozen=True)
class Point:
    """One point on the cost-fidelity curve."""

    explore_rounds: int
    k: int
    regret: float
    rounds: int
    #: Rounds as a fraction of training every candidate to completion.
    fraction: float
    #: Whether selection and scoring used different replicates. False means
    #: the figure is an upper bound on the error, not a measurement of it.
    holdout: bool


def sweep(
    select_on: Mapping[Members, Curve],
    evaluate_on: Mapping[Members, float],
    explore_rounds: Sequence[int],
    shortlist_sizes: Sequence[int] = (1, 3, 5),
) -> list[Point]:
    """The cost-fidelity curve over `explore_rounds` x `shortlist_sizes`.

    `select_on` holds the curves the shortlist is chosen from. `evaluate_on`
    holds the final accuracy it is scored against, which should come from
    different replicates -- see the module docstring on the winner's curse.
    """
    candidates = set(select_on) & set(evaluate_on)
    if not candidates:
        raise ValueError("select_on and evaluate_on share no candidates")

    lengths = {len(select_on[m]) for m in candidates}
    if len(lengths) != 1:
        raise ValueError(f"curves differ in length: {sorted(lengths)}")
    total_rounds = lengths.pop()

    truth = {m: evaluate_on[m] for m in candidates}
    holdout = select_on is not evaluate_on and not all(
        select_on[m][-1] == evaluate_on[m] for m in candidates
    )

    out = []
    for r in explore_rounds:
        scores = {m: select_on[m][r - 1] for m in candidates}
        for k in shortlist_sizes:
            if k > len(candidates):
                continue
            spent = rounds_spent(len(candidates), r, k, total_rounds)
            out.append(
                Point(
                    explore_rounds=r,
                    k=k,
                    regret=regret(shortlist(scores, k), truth),
                    rounds=spent,
                    fraction=spent / (len(candidates) * total_rounds),
                    holdout=holdout,
                )
            )
    return out
