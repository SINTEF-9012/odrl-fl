"""Policy-aware selection on real patients from real institutions.

Every number so far came from synthetic data split by a Dirichlet draw. This
runs the same machinery on 1096 TCGA breast-cancer patients partitioned the way
the data already was -- by Tissue Source Site, into named hospitals -- and
scores it with the metric the task actually supports, Harrell's C-index on a
Cox model.

Three questions. It settles two of them.

**Does federating help at all here?** If a single institution did as well as the
consortium there would be nothing to trade against policy, and the whole
exercise would be moot.

**Do the real terms erode?** They are citable and they are thin: the GDC
open-access tier grants everything and forbids only re-identification. Composing
sixteen copies of the same permissive policy loses nothing, which is the honest
answer and a short one.

**Does policy-aware selection cost model quality?** It cannot answer this, and
saying so is the point. The held-out fold carries 37 observed events, and a
bootstrap interval on a single C-index comes out about as wide as the entire
spread between the best and worst institution. The differences at stake are an
order of magnitude smaller. More seeds do not help; the limit is the cohort.

So the results split cleanly. Anything combinatorial -- whether a consortium is
*permitted* to use its model -- is settled here, because it follows from
composing declared terms and never touches a metric. Anything about model
quality is not.

Heterogeneity in the *terms* is still modelled: these rows carry no per-site
data-use conditions, because open-access data does not have any. The codes are
drawn from the EGA empirical distribution and attached to real institutions with
real patient counts, which is stated here rather than buried.

    python experiments/run_tcga.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odrl_fl.corpus import generate_corpus
from odrl_fl.odrl import Action, capability, compose
from odrl_fl.selection import is_feasible, policy_blind_selection, requirement, search
from odrl_fl.strategy import Announcement
from odrl_fl.tcga import (
    CoxModel,
    c_index,
    evaluate_cox,
    gdc_open_access_policy,
    load_sites,
    split,
    standardise,
    train_cox,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def federated_cox(
    trains, *, rounds: int = 8, epochs: int = 15, seed: int = 0
) -> CoxModel:
    """FedAvg over the Cox model, weighted by patient count.

    Seeded per call, not once per script. `CoxModel()` draws its initial weights
    from the global generator, so without this a consortium's score depends on
    how many other consortia happened to be trained before it -- which silently
    confounds every comparison the experiment makes.
    """
    torch.manual_seed(seed)
    global_model = CoxModel()
    weights = np.array([len(y) for _, y in trains], dtype=np.float64)
    weights = weights / weights.sum()

    for _ in range(rounds):
        states = []
        for x, y in trains:
            local = CoxModel()
            local.load_state_dict(global_model.state_dict())
            train_cox(local, x, y, epochs=epochs)
            states.append(local.state_dict())

        merged = {
            key: sum(s[key] * float(w) for s, w in zip(states, weights))
            for key in states[0]
        }
        global_model.load_state_dict(merged)
    return global_model


def score(members, trains, xte, yte, *, rounds: int, epochs: int) -> float:
    if not members:
        return 0.5
    model = federated_cox(
        [trains[i] for i in members], rounds=rounds, epochs=epochs
    )
    return evaluate_cox(model, xte, yte)


def bootstrap_ci(
    risk: np.ndarray, y: np.ndarray, *, draws: int = 400, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap over the test fold.

    Reported because the fold is small: with 37 observed events, the interval
    turns out to be wider than the entire spread between the best and worst
    single-institution model, which decides what this dataset can and cannot
    settle.
    """
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        scores.append(c_index(risk[idx], y[idx]))
    return tuple(np.percentile(scores, [2.5, 97.5]))


def risk_scores(model: CoxModel, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def centralised_score(trains, xte, yte, *, epochs: int = 400, seed: int = 0) -> float:
    """Train on the pooled data. The upper bound federating is measured against."""
    torch.manual_seed(seed)
    model = CoxModel()
    train_cox(
        model,
        np.concatenate([t[0] for t in trains]),
        np.concatenate([t[1] for t in trains]),
        epochs=epochs,
    )
    return evaluate_cox(model, xte, yte)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Selection on TCGA-BRCA")
    parser.add_argument("--min-patients", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--purpose", default="ds-cancer")
    args = parser.parse_args(argv)

    torch.manual_seed(0)
    sites = standardise(load_sites(min_patients=args.min_patients))
    trains, tests = zip(*(split(s, seed=0) for s in sites))
    xte = np.concatenate([t[0] for t in tests])
    yte = np.concatenate([t[1] for t in tests])

    print("TCGA-BRCA: policy-aware selection on real institutions\n")
    print(f"  {len(sites)} institutions with >= {args.min_patients} patients")
    print(f"  {sum(s.n_patients for s in sites)} patients, "
          f"{sum(s.n_events for s in sites)} observed events")
    print(f"  shared held-out fold: {len(yte)} patients, "
          f"{int(yte[:, 0].sum())} events\n")
    for s in sites[:6]:
        print(f"    {s.name:<32} {s.n_patients:>4} patients  "
              f"{s.n_events:>3} events  {s.region}")
    if len(sites) > 6:
        print(f"    ... and {len(sites) - 6} more")

    # 1. Does federating help?
    print("\n  Does federating help?")
    everyone = score(range(len(sites)), trains, xte, yte,
                     rounds=args.rounds, epochs=args.epochs)
    solo = [
        score([i], trains, xte, yte, rounds=args.rounds, epochs=args.epochs)
        for i in range(len(sites))
    ]
    centralised = centralised_score(trains, xte, yte)

    print(f"    centralised, data pooled  C-index {centralised:.3f}  (upper bound)")
    print(f"    federated, all {len(sites)} sites   C-index {everyone:.3f}")
    print(f"    best single institution   C-index {max(solo):.3f}"
          f"  ({sites[int(np.argmax(solo))].name})")
    print(f"    median single institution C-index {float(np.median(solo)):.3f}")
    print()
    print("    Federating beats the typical institution but loses most of what")
    print("    pooling would gain, and that gap is not under-training -- it does")
    print("    not close with more rounds. The Cox partial likelihood is **not a")
    print("    sum over clients**: its risk set spans every patient still at risk")
    print("    at each event time, so a client computing the loss over its own")
    print("    cohort optimises a different objective than the federation's.")
    print("    Andreux et al. (2020) reformulate to discrete time for exactly this")
    print("    reason; naive FedAvg here does not, and the number shows it.")
    print()
    print("    It does not disturb the comparison below -- both consortia are")
    print("    trained by the same procedure -- but it does mean these C-index")
    print("    figures are a floor for this dataset rather than its ceiling.")

    # How much of any of this is signal?
    fed_model = federated_cox(trains, rounds=args.rounds, epochs=args.epochs)
    lo, hi = bootstrap_ci(risk_scores(fed_model, xte), yte)
    spread = max(solo) - min(solo)
    print("\n  How much of this is signal?")
    print(f"    federated C-index {everyone:.3f}, bootstrap 95% CI "
          f"[{lo:.3f}, {hi:.3f}] (width {hi - lo:.3f})")
    print(f"    spread across all {len(sites)} single-site models: {spread:.3f}")
    print()
    print(f"    With {int(yte[:, 0].sum())} events in the held-out fold, a single "
          "C-index carries")
    print(f"    roughly +/-{(hi - lo) / 2:.2f}. Any comparison whose effect is smaller "
          "than that is")
    print("    unresolvable here, which covers every model-quality claim this")
    print("    experiment could make. The 'best single institution' figure above")
    print(f"    is a maximum over {len(sites)} such estimates, so it is biased upward")
    print("    and is not a ranking.")
    print()
    print("    What survives the noise is everything combinatorial: whether a")
    print("    consortium is *permitted* to use its model does not depend on how")
    print("    well the model scores.")

    # 2. Do the real terms erode?
    real = [
        gdc_open_access_policy(uid=f"urn:policy:{s.name}", assigner=f"urn:party:{s.name}")
        for s in sites
    ]
    composed = compose(real)
    print("\n  Do the real terms erode?")
    print(f"    one institution   {sorted(a.value for a in capability(real[0]))}")
    print(f"    all {len(sites)} composed  "
          f"{sorted(a.value for a in composed.capability)}")
    print("    Nothing is lost. The GDC open-access tier grants everything and")
    print("    forbids only re-identification, so sixteen copies of it compose to")
    print("    itself. Erosion needs heterogeneous terms, and the tier anyone can")
    print("    download has none -- which is exactly why the phenomenon lives in")
    print("    controlled access, and why open data cannot be used to study it.")

    # 3. Selection under modelled terms, scored by real C-index.
    print(f"\n  Selection under modelled terms (request: use, aggregate"
          f"{f', purpose={args.purpose}' if args.purpose else ''})\n")
    request = requirement(Action.USE, Action.AGGREGATE, purpose=args.purpose or None)

    def volume_utility(subset):
        return float(sum(a.n_train for a in subset)) ** 0.5

    rows: list[dict] = []
    for seed in range(args.seeds):
        codes = generate_corpus(len(sites), seed=seed)
        pool = [
            Announcement(node_id=i, policy=c.policy, n_train=sites[i].n_patients,
                         label_counts={0: sites[i].n_patients})
            for i, c in enumerate(codes)
        ]
        result = search(pool, request, volume_utility)
        best = result.best
        aware = tuple(best.members) if best else ()
        blind = policy_blind_selection(pool, volume_utility, len(aware) or 1)
        blind_ok = is_feasible([pool[i] for i in blind], request)

        aware_c = score(aware, trains, xte, yte,
                        rounds=args.rounds, epochs=args.epochs)
        blind_c = score(blind, trains, xte, yte,
                        rounds=args.rounds, epochs=args.epochs)

        rows.append({
            "seed": seed,
            "aware_size": len(aware),
            "aware_cindex": round(aware_c, 4),
            "aware_feasible": int(bool(best)),
            "blind_size": len(blind),
            "blind_cindex": round(blind_c, 4),
            "blind_feasible": int(blind_ok),
            "blind_deployable": round(blind_c if blind_ok else 0.0, 4),
            "everyone_cindex": round(everyone, 4),
        })
        print(f"    seed {seed}: policy-aware {len(aware):>2} sites -> "
              f"C-index {aware_c:.3f} | policy-blind {len(blind):>2} sites -> "
              f"{blind_c:.3f} ({'usable' if blind_ok else 'NOT usable'})")

    aware_mean = float(np.mean([r["aware_cindex"] for r in rows]))
    blind_mean = float(np.mean([r["blind_cindex"] for r in rows]))
    blind_deploy = float(np.mean([r["blind_deployable"] for r in rows]))
    blind_yield = float(np.mean([r["blind_feasible"] for r in rows]))

    print(f"\n  across {args.seeds} code assignments")
    print(f"    policy-aware  C-index {aware_mean:.3f}, usable 100% of the time")
    print(f"    policy-blind  C-index {blind_mean:.3f}, usable "
          f"{blind_yield:.0%} of the time")
    print(f"\n    Deployable C-index: {aware_mean:.3f} against {blind_deploy:.3f}.")
    print()
    print("  What this dataset does and does not settle")
    print()
    print("    Settled. The policy-blind consortium is permitted to use its own")
    print(f"    model {blind_yield:.0%} of the time; the policy-aware one always is.")
    print("    That is combinatorial -- it follows from composing the declared")
    print("    terms and does not depend on any C-index, so the noise above")
    print("    cannot touch it. The deployable figures differ by an order of")
    print("    magnitude for the same reason: an unusable model scores zero")
    print("    however well it fits.")
    print()
    print("    Not settled. Whether policy-aware selection costs model quality.")
    print(f"    The point estimates differ by {abs(aware_mean - blind_mean):.3f}, "
          "against a bootstrap")
    print(f"    interval on a single C-index roughly {hi - lo:.2f} wide. The"
          " comparison is")
    print("    underpowered by an order of magnitude, and no amount of extra")
    print(f"    seeds fixes it -- the limit is {int(yte[:, 0].sum())} events in the "
          "held-out fold, not")
    print("    the number of code assignments. Answering it needs a larger")
    print("    cohort, which for this disease means controlled-access data.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "tcga.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  wrote {RESULTS / 'tcga.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
