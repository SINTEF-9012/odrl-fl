"""The measured corpus: who actually holds which conditions.

`assignment.py` exists because the erosion corpus had to assume something about
which institution holds which data use conditions, and the README is blunt about
what that costs -- erosion at ten contributors ranges from 5% to 76% across the
seven assumed models, and under `clustered-0.6` the pre-registered go/no-go
criterion would not have been met. The direction of the result is robust. The
magnitude has been a modelling choice wearing the clothes of a measurement.

It does not have to be. EGA's `/policies` endpoint returns `dac_accession_id`
alongside `duos` -- the Data Access Committee that issues each policy -- and
`experiments/fetch_duo_frequencies.py` has been discarding that field. Keep it
and the grouping structure is observable: which conditions travel together
because the same institution issued them.

## What is measured, and what is still assumed

A DAC is *not* a consortium, and treating one as the other would only swap one
assumption for another. A DAC is a single institution issuing several policies,
which is the most homophilous grouping obtainable -- so within-DAC agreement is
an **upper bound** on how alike a real consortium's members are. Two unrelated
DACs are the least alike, a **lower bound**. Real consortia sit somewhere
between, and this module reports the bracket rather than a point.

## Recovering cohesion from an agreement rate

`_clustered_sampler` defines cohesion exactly: with probability `c` a
contributor adopts a shared archetype code set, otherwise it draws from the
empirical distribution. For two contributors in one corpus, writing `p` for the
chance two independent empirical draws coincide and `w` for the archetype's own
weight, the probability their code sets are identical is

    agreement = c² + (1-c)²·p + 2c(1-c)·w

The archetype is itself drawn from the same weighted distribution, so
`E[w] = sum(w_i · w_i) = p`. Substituting collapses the quadratic:

    agreement = c² + p(1-c)² + 2p·c(1-c)
              = c² + p(1-c)[(1-c) + 2c]
              = c² + p(1-c)(1+c)
              = p + c²(1-p)

which inverts to `c = sqrt((agreement - p) / (1 - p))`. Below `p` the data is
saying contributors agree *less* than chance, which the clustered model cannot
express; cohesion is reported as zero there rather than as an imaginary number.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations as pairs_of
from pathlib import Path

import numpy as np

BASE = "https://metadata.ega-archive.org"
CACHE = Path(__file__).resolve().parent.parent / "vocab" / "ega_corpus.json"


@dataclass(frozen=True)
class Contributor:
    """One real policy, with the institution that issued it.

    `n_samples` is `None` rather than 0 when EGA does not publish a volume.
    88% of annotated policies carry one; treating the silent 12% as empty
    institutions would understate every volume-weighted quantity downstream,
    and silently invent a fact about an institution that published none.
    """

    policy_id: str
    dac_id: str
    institution: str
    codes: tuple[str, ...]
    n_samples: int | None = None


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def _page(endpoint: str, limit: int, offset: int, timeout: float = 40.0) -> list[dict]:
    url = f"{BASE}/{endpoint}?limit={limit}&offset={offset}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _paginate(
    endpoint: str,
    *,
    max_records: int,
    page_size: int = 200,
    pause: float = 0.15,
    label: str = "",
) -> list[dict]:
    """Walk an endpoint, stopping on a short page, an error, or the cap.

    Errors stop the walk rather than crashing: a partial corpus that says how
    partial it is beats losing an eight-minute fetch to one timeout.
    """
    out: list[dict] = []
    offset = 0
    while len(out) < max_records:
        try:
            page = _page(endpoint, page_size, offset)
        except Exception as exc:  # noqa: BLE001 - report and stop, do not crash
            print(f"  {label or endpoint}: stopped at {offset}: "
                  f"{type(exc).__name__}: {exc}")
            break
        if not page:
            break
        out.extend(page)
        offset += len(page)
        print(f"  {label or endpoint}: {len(out)}", end="\r", flush=True)
        if len(page) < page_size:
            break
        time.sleep(pause)
    print(f"  {label or endpoint}: {len(out)} records")
    return out


def _dac_titles(accessions: Sequence[str], pause: float = 0.05) -> dict[str, str]:
    """Resolve DAC accessions to institution names, one request each.

    Slower per record than the listing endpoint but bounded and reliable: the
    listing walk times out past roughly offset 5000, and there are only a few
    hundred committees that issue an annotated policy.
    """
    titles: dict[str, str] = {}
    for i, accession in enumerate(accessions, 1):
        try:
            with urllib.request.urlopen(
                f"{BASE}/dacs/{accession}", timeout=30
            ) as response:
                titles[accession] = (json.load(response).get("title") or "").strip()
        except Exception:  # noqa: BLE001 - a missing name is not worth failing over
            titles[accession] = ""
        if i % 25 == 0:
            print(f"  dac titles: {i}/{len(accessions)}", flush=True)
        time.sleep(pause)
    resolved = sum(1 for t in titles.values() if t)
    print(f"  dac titles: {resolved}/{len(accessions)} resolved")
    return titles


def fetch(
    *,
    max_policies: int = 6000,
    max_datasets: int = 25000,
    out: Path | str = CACHE,
) -> dict:
    """Pull policies, DACs and datasets, and write the cached corpus.

    Run once; the result is committed and every downstream caller reads it via
    `load()`. The experiments must not depend on a live API or they stop being
    reproducible -- the same rule `fetch_duo_frequencies.py` already follows.
    """
    print(f"Fetching from {BASE} ...")
    policies = _paginate("policies", max_records=max_policies, label="policies")
    datasets = _paginate("datasets", max_records=max_datasets, label="datasets")

    # DAC titles are fetched one at a time rather than by walking /dacs. The
    # listing endpoint degrades past roughly offset 5000 and starts timing out,
    # and only the DACs that actually issue an annotated policy are wanted --
    # a few hundred targeted lookups instead of an unbounded walk that stalls.
    wanted = sorted(
        {p["dac_accession_id"] for p in policies if p.get("duos") and p.get("dac_accession_id")}
    )
    titles = _dac_titles(wanted)

    # Volume is per dataset; a policy can govern several, so they sum.
    volume: dict[str, int] = {}
    for d in datasets:
        policy_id = d.get("policy_accession_id")
        n = d.get("num_samples")
        if policy_id and isinstance(n, int) and n > 0:
            volume[policy_id] = volume.get(policy_id, 0) + n

    contributors = []
    annotated = 0
    for p in policies:
        codes = p.get("duos")
        if not codes:
            continue
        annotated += 1
        accession = p.get("accession_id") or ""
        dac_id = p.get("dac_accession_id") or ""
        contributors.append(
            {
                "policy_id": accession,
                "dac_id": dac_id,
                "institution": titles.get(dac_id, ""),
                "codes": sorted(codes),
                "n_samples": volume.get(accession),
            }
        )

    with_volume = sum(1 for c in contributors if c["n_samples"] is not None)
    payload = {
        "_provenance": {
            "source": "EGA metadata API",
            "url": BASE,
            "fetched": datetime.now(tz=UTC).date().isoformat(),
            "policies_seen": len(policies),
            "policies_annotated": annotated,
            "adoption_rate": round(annotated / max(len(policies), 1), 4),
            "dacs_resolved": sum(1 for t in titles.values() if t),
            "datasets_seen": len(datasets),
            "contributors_with_volume": with_volume,
            "note": (
                "One record per EGA policy carrying structured DUO conditions, "
                "keyed to the Data Access Committee that issues it. The DAC is "
                "the unit of institutional identity here: it is the closest "
                "observable proxy for 'one institution's terms', and is used to "
                "bound consortium cohesion from above, never to stand in for a "
                "consortium itself."
            ),
        },
        "contributors": contributors,
    }

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")
    print(f"  {annotated} annotated policies "
          f"({annotated / max(len(policies), 1):.1%} adoption)")
    print(f"  {len({c['dac_id'] for c in contributors})} distinct DACs")
    print(f"  {with_volume} carry a sample volume")
    return payload


def load(path: Path | str = CACHE) -> list[Contributor]:
    """Read the cached corpus. Raises if it is absent -- callers should skip."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Contributor(
            policy_id=c["policy_id"],
            dac_id=c["dac_id"],
            institution=c.get("institution", ""),
            codes=tuple(c["codes"]),
            n_samples=c.get("n_samples"),
        )
        for c in payload["contributors"]
    ]


def provenance(path: Path | str = CACHE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["_provenance"]


# --------------------------------------------------------------------------
# Measuring agreement
# --------------------------------------------------------------------------


def by_dac(corpus: Sequence[Contributor]) -> dict[str, list[Contributor]]:
    groups: dict[str, list[Contributor]] = {}
    for c in corpus:
        if c.dac_id:
            groups.setdefault(c.dac_id, []).append(c)
    return groups


def _agreement(members: Sequence[Contributor]) -> float | None:
    """Fraction of pairs within one group holding an identical code set.

    `None` for a group of one: a single policy generates no pairs and so
    carries no information about internal consistency. Counting it as either
    perfect or zero agreement would be an invention.
    """
    if len(members) < 2:
        return None
    total = same = 0
    for a, b in pairs_of(members, 2):
        total += 1
        same += a.codes == b.codes
    return same / total


def agreement_within_dac(corpus: Sequence[Contributor]) -> list[float]:
    """Per-DAC internal agreement, one value per institution with >= 2 policies.

    Returned per DAC rather than pooled over pairs on purpose. One EGA DAC holds
    20 policies and would contribute 190 of the pairs, dominating a pooled rate
    and turning "how consistent is a typical institution" into "how consistent
    is the most prolific one".
    """
    scores = [_agreement(members) for members in by_dac(corpus).values()]
    return [s for s in scores if s is not None]


def agreement_between_dacs(corpus: Sequence[Contributor]) -> float:
    """Agreement across every pair of policies issued by *different* DACs.

    The floor: how often two unrelated institutions coincide by chance.
    """
    members = [c for c in corpus if c.dac_id]
    total = same = 0
    for a, b in pairs_of(members, 2):
        if a.dac_id == b.dac_id:
            continue
        total += 1
        same += a.codes == b.codes
    return same / total if total else float("nan")


def empirical_coincidence(corpus: Sequence[Contributor] | None = None) -> float:
    """`p`: the chance two independent draws from the corpus coincide.

    Computed as sum of squared frequencies. When no corpus is given this falls
    back to the committed combination table, so the value stays comparable with
    the synthetic assignment models that are calibrated against it.
    """
    if corpus is None:
        from odrl_fl.assignment import _weighted_combinations

        _, weights = _weighted_combinations()
        return float(np.sum(weights**2))

    counts: dict[tuple[str, ...], int] = {}
    for c in corpus:
        counts[c.codes] = counts.get(c.codes, 0) + 1
    total = sum(counts.values())
    if total < 2:
        return float("nan")
    weights = np.array(list(counts.values()), dtype=float) / total
    return float(np.sum(weights**2))


def solve_cohesion(agreement: float, p_emp: float) -> float:
    """Invert `agreement = p + c²(1-p)` for the clustered model's `c`.

    Clamped to [0, 1]. An agreement below `p` means contributors coincide less
    often than independent draws would, which the clustered model has no way to
    express -- it can only add homophily, never subtract it. Reporting zero
    there is the honest floor; a negative or imaginary cohesion would not be.
    """
    if not np.isfinite(agreement) or not np.isfinite(p_emp) or p_emp >= 1.0:
        return float("nan")
    excess = (agreement - p_emp) / (1.0 - p_emp)
    if excess <= 0.0:
        return 0.0
    return float(min(1.0, np.sqrt(excess)))


def cohesion_bracket(
    corpus: Sequence[Contributor],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """The measured bracket, with a bootstrap interval on the upper bound.

    Resampling is over *DACs*, not over pairs, because the DACs are what was
    sampled -- 26 institutions happen to publish more than one machine-readable
    policy, and the uncertainty that matters is which 26.
    """
    from odrl_fl.erosion import bootstrap_ci

    p_emp = empirical_coincidence(corpus)
    within = agreement_within_dac(corpus)
    between = agreement_between_dacs(corpus)

    within_mean = float(np.mean(within)) if within else float("nan")
    lo, hi = (
        bootstrap_ci(within, resamples=resamples, rng=np.random.default_rng(seed))
        if within
        else (float("nan"), float("nan"))
    )

    return {
        "p_emp": p_emp,
        "n_dacs_multi": len(within),
        "agreement_within": within_mean,
        "agreement_within_lo": lo,
        "agreement_within_hi": hi,
        "agreement_between": between,
        "cohesion_upper": solve_cohesion(within_mean, p_emp),
        "cohesion_upper_lo": solve_cohesion(lo, p_emp),
        "cohesion_upper_hi": solve_cohesion(hi, p_emp),
        "cohesion_lower": solve_cohesion(between, p_emp),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="EGA measured corpus")
    parser.add_argument("--fetch", action="store_true", help="Refresh the cache")
    parser.add_argument("--max-policies", type=int, default=20000)
    args = parser.parse_args(argv)

    if args.fetch:
        fetch(max_policies=args.max_policies)
        return 0

    if not CACHE.exists():
        print(f"No cached corpus at {CACHE}. Run with --fetch.")
        return 1

    corpus = load()
    bracket = cohesion_bracket(corpus)
    print(f"{len(corpus)} annotated policies, "
          f"{len(by_dac(corpus))} DACs, "
          f"{bracket['n_dacs_multi']} with 2+ policies")
    print(f"  chance coincidence p       {bracket['p_emp']:.4f}")
    print(f"  agreement between DACs     {bracket['agreement_between']:.4f}")
    print(f"  agreement within a DAC     {bracket['agreement_within']:.4f} "
          f"[{bracket['agreement_within_lo']:.4f}, "
          f"{bracket['agreement_within_hi']:.4f}]")
    print(f"  cohesion bracket           {bracket['cohesion_lower']:.3f} .. "
          f"{bracket['cohesion_upper']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
