"""How DUO codes get assigned to institutions -- and whether it matters.

The main threat to validity in the erosion result. EGA gives us real code
*frequencies* and real *co-occurrence*, but says nothing about which kind of
institution holds which policy. The corpus therefore has to assume something,
and the baseline assumes contributors draw independently from the empirical
distribution of code sets.

That assumption is contestable in a direction that cuts against the result.
Consortia are not random samples of the world's institutions: a cancer network
is mostly cancer-focused sites, whose consent conditions would be far more
compatible than chance. If erosion vanishes under realistic clustering, the
headline is an artifact of the sampling model.

So five assignment models are implemented and the curve is measured under each:

    empirical       frequency-weighted whole combinations, drawn independently
                    (the baseline)
    marginal        each code drawn independently from its marginal frequency,
                    destroying the observed co-occurrence structure
    uniform         uniform over distinct observed combinations, removing the
                    frequency weighting so rare conditions count equally
    clustered       homophily: contributors adopt a shared archetype with
                    probability `cohesion`, else draw empirically
    purpose-matched the strongest form of the objection -- a consortium that
                    has already self-selected for compatible research purposes

`purpose-matched` deserves its name. If erosion largely disappears there, the
finding is not weakened but sharpened: it says consortium formation *already*
performs implicit policy selection, and that the explicit version is what this
project automates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from odrl_fl.odrl import Action, LeftOperand, Operator, default_taxonomy
from odrl_fl.odrl.duo import translate

#: A per-contributor draw. Bound to corpus-level state where a model needs it.
Sampler = Callable[[np.random.Generator], tuple[str, ...]]


@lru_cache(maxsize=1)
def _combination_purposes() -> tuple[tuple[tuple[str, ...], frozenset[str]], ...]:
    """Each observed combination with the purposes it permits.

    An empty purpose set means the combination carries no permission code and
    so constrains nothing -- compatible with everything.
    """
    from odrl_fl.corpus import combinations

    out = []
    for codes, _count in combinations():
        policy, _ = translate(codes, uid="urn:probe", actions=(Action.USE,))
        permission = policy.permission_for(Action.USE)
        purposes: frozenset[str] = frozenset()
        if permission is not None:
            positive = [
                c
                for c in permission.constraints
                if c.left_operand is LeftOperand.PURPOSE
                and c.operator is Operator.IS_ANY_OF
            ]
            if positive:
                purposes = frozenset().union(*(c.right_operand for c in positive))
        out.append((tuple(codes), purposes))
    return tuple(out)


def purposes_compatible(left: frozenset[str], right: frozenset[str]) -> bool:
    """Could two contributors ever agree on a purpose?

    An unconstrained side is compatible with anything; otherwise the taxonomy
    meet must be non-empty, which is exactly the test composition applies.
    """
    if not left or not right:
        return True
    return bool(default_taxonomy().meet("purpose", left, right))


# --- The models -------------------------------------------------------------


def _weighted_combinations() -> tuple[list[tuple[str, ...]], np.ndarray]:
    from odrl_fl.corpus import combinations

    observed = combinations()
    weights = np.array([count for _, count in observed], dtype=float)
    return [codes for codes, _ in observed], weights / weights.sum()


def _empirical_sampler(_: np.random.Generator) -> Sampler:
    codes, weights = _weighted_combinations()

    def draw(rng: np.random.Generator) -> tuple[str, ...]:
        return codes[int(rng.choice(len(codes), p=weights))]

    return draw


def _uniform_sampler(_: np.random.Generator) -> Sampler:
    codes, _weights = _weighted_combinations()

    def draw(rng: np.random.Generator) -> tuple[str, ...]:
        return codes[int(rng.integers(len(codes)))]

    return draw


def _marginal_sampler(_: np.random.Generator) -> Sampler:
    """Draw codes independently, discarding co-occurrence.

    Real annotations bundle conditions together -- NPUNCU travels with HMB, and
    so on. Sampling marginally breaks those bundles, which tests how much of
    the result depends on the joint structure rather than on the code mix.
    """
    from odrl_fl.corpus import frequencies

    counts = frequencies()["code_frequencies"]
    annotated = frequencies()["_provenance"]["policies_annotated"]
    codes = sorted(counts)
    # Per-code probability of appearing on any one policy.
    rates = np.array([min(1.0, counts[c] / annotated) for c in codes], dtype=float)

    def draw(rng: np.random.Generator) -> tuple[str, ...]:
        picked = tuple(c for c, r in zip(codes, rates) if rng.random() < r)
        # A policy with no conditions at all is not observed in practice; fall
        # back to the most common single code.
        return picked or (codes[int(np.argmax(rates))],)

    return draw


def _clustered_sampler(corpus_rng: np.random.Generator, cohesion: float = 0.6) -> Sampler:
    """Homophily around one archetype drawn per corpus.

    The realistic objection: a consortium forms around a shared theme, so its
    members' conditions resemble each other far more than chance. Unlike the
    `heterogeneity` dial, which clusters on the globally *modal* code set, the
    archetype here is drawn per corpus -- a cancer network clusters on cancer,
    not on whatever is most common overall.
    """
    codes, weights = _weighted_combinations()
    archetype = codes[int(corpus_rng.choice(len(codes), p=weights))]

    def draw(rng: np.random.Generator) -> tuple[str, ...]:
        if rng.random() < cohesion:
            return archetype
        return codes[int(rng.choice(len(codes), p=weights))]

    return draw


def _purpose_matched_sampler(corpus_rng: np.random.Generator) -> Sampler:
    """A consortium that already self-selected for compatible purposes.

    The strongest form of the objection, and the most informative: contributors
    are drawn only from combinations whose purposes could ever agree with the
    anchor's. If erosion largely survives even here, the sampling model is not
    doing the work.
    """
    entries = _combination_purposes()
    codes, weights = _weighted_combinations()
    anchor_index = int(corpus_rng.choice(len(codes), p=weights))
    anchor_purposes = entries[anchor_index][1]

    eligible = [
        combo for combo, purposes in entries
        if purposes_compatible(anchor_purposes, purposes)
    ]
    lookup = {combo: weight for combo, weight in zip(codes, weights)}
    eligible_weights = np.array([lookup.get(c, 0.0) for c in eligible], dtype=float)
    if eligible_weights.sum() <= 0:
        eligible_weights = np.ones(len(eligible), dtype=float)
    eligible_weights /= eligible_weights.sum()

    def draw(rng: np.random.Generator) -> tuple[str, ...]:
        return eligible[int(rng.choice(len(eligible), p=eligible_weights))]

    return draw


def _measured_sampler(_: np.random.Generator) -> Sampler:
    """Draw real code sets from the real institutions that published them.

    The other six models each answer "what if institutions were paired with
    conditions like *this*?" This one does not ask. It draws whole code sets
    from EGA's own DAC population, so whatever correlation structure real data
    access committees have is the structure the corpus gets -- and there is no
    `cohesion` parameter to choose, because nothing is being assumed.

    Institutions are drawn uniformly and a policy is then drawn from within the
    chosen one. Sampling policies directly instead would weight each institution
    by how many policies it happens to publish, and one EGA DAC publishes twenty
    -- which would quietly make the corpus mostly that one institution.
    """
    from odrl_fl.ega import CACHE, by_dac, load

    if not CACHE.exists():
        raise FileNotFoundError(
            f"the 'measured' assignment model needs the EGA corpus at {CACHE}; "
            "run `python -m odrl_fl.ega --fetch`"
        )

    groups = [
        [c.codes for c in members] for members in by_dac(load()).values()
    ]

    def draw(rng: np.random.Generator) -> tuple[str, ...]:
        institution = groups[int(rng.integers(len(groups)))]
        return institution[int(rng.integers(len(institution)))]

    return draw


@dataclass(frozen=True)
class Model:
    name: str
    build: Callable[[np.random.Generator], Sampler]
    description: str


def _clustered_at(cohesion: float) -> Callable[[np.random.Generator], Sampler]:
    def build(corpus_rng: np.random.Generator) -> Sampler:
        return _clustered_sampler(corpus_rng, cohesion=cohesion)

    return build


MODELS: dict[str, Model] = {
    "empirical": Model(
        "empirical",
        _empirical_sampler,
        "frequency-weighted combinations, drawn independently (baseline)",
    ),
    "marginal": Model(
        "marginal",
        _marginal_sampler,
        "codes drawn independently, destroying observed co-occurrence",
    ),
    "uniform": Model(
        "uniform",
        _uniform_sampler,
        "uniform over distinct combinations, ignoring how common each is",
    ),
    "clustered-0.3": Model(
        "clustered-0.3",
        _clustered_at(0.3),
        "mild homophily around a per-corpus archetype",
    ),
    "clustered-0.6": Model(
        "clustered-0.6",
        _clustered_at(0.6),
        "strong homophily -- the realistic form of the objection",
    ),
    "clustered-0.9": Model(
        "clustered-0.9",
        _clustered_at(0.9),
        "near-identical consortium, an upper bound on cohesion",
    ),
    "purpose-matched": Model(
        "purpose-matched",
        _purpose_matched_sampler,
        "a consortium already self-selected for compatible purposes",
    ),
    "measured": Model(
        "measured",
        _measured_sampler,
        "real code sets drawn from the real EGA institutions that published them",
    ),
}


#: The models that *assume* an assignment, in the order the sensitivity study
#: reports them. `measured` is deliberately not here: it observes the
#: assignment rather than assuming one, so averaging it in with the assumptions
#: would confuse "how much does this depend on what we assumed" with "what is
#: actually the case". Models registered later by `register_clustered` are
#: likewise excluded -- they are derived from a measurement.
ASSUMED = (
    "empirical",
    "marginal",
    "uniform",
    "clustered-0.3",
    "clustered-0.6",
    "clustered-0.9",
    "purpose-matched",
)


def register_clustered(cohesion: float) -> str:
    """Register a clustered model at an arbitrary cohesion, returning its name.

    The three fixed rungs (0.3, 0.6, 0.9) were chosen before anything had been
    measured. The bracket that `odrl_fl.ega` recovers from real DAC structure
    lands wherever it lands, so comparing the measured value against the
    assumed ones needs a model at exactly that cohesion rather than at the
    nearest convenient rung.
    """
    if not 0.0 <= cohesion <= 1.0:
        raise ValueError(f"cohesion must be in [0, 1], got {cohesion}")
    name = f"clustered-{cohesion:.3f}"
    if name not in MODELS:
        MODELS[name] = Model(
            name,
            _clustered_at(cohesion),
            f"homophily at a measured cohesion of {cohesion:.3f}",
        )
    return name


def build_sampler(model: str, corpus_rng: np.random.Generator) -> Sampler:
    if model not in MODELS:
        raise ValueError(
            f"unknown assignment model '{model}'; have {', '.join(sorted(MODELS))}"
        )
    return MODELS[model].build(corpus_rng)


def model_names() -> Sequence[str]:
    return tuple(MODELS)
