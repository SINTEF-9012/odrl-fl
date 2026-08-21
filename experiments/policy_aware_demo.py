"""Selection to trained model, end to end.

The first point in the project where a model is trained on a consortium chosen
by policy. Everything before this measured the decision; this runs it and then
trains.

It simulates the declaration round rather than standing up a federation, so it
needs no SuperLink and no Ray: each contributor's announcement is built exactly
as `client_app.declare` would build it, and handed to exactly the `admit` the
server calls. Only the transport is missing.

The comparison is against the policy-blind consortium of the same size -- the
standard practice of picking whoever has the most useful data. It often trains
the better model, and is usually not permitted to use it.

    python experiments/policy_aware_demo.py
    python experiments/policy_aware_demo.py --purpose research --nodes 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odrl_fl.corpus import generate_corpus
from odrl_fl.fedavg import run_fedavg
from odrl_fl.odrl import Action
from odrl_fl.selection import is_feasible, policy_blind_selection, requirement
from odrl_fl.strategy import (
    Announcement,
    NoFeasibleConsortium,
    admit,
    declaration,
    declared_utility,
    parse_announcement,
)
from odrl_fl.task import N_CLASSES, load_partition

#: Spread of contributor data volumes. Must match what `run_fedavg` trains
#: on, or nodes would declare volumes they do not hold.
SIZE_VARIATION = 1.2


def announce(n_nodes: int, alpha: float, seed: int) -> list[Announcement]:
    """Build what each node would send, and read it back as the server would.

    Deliberately round-tripped through the wire format rather than constructed
    directly: if serialisation ever loses something, this demo should break
    alongside the real path instead of quietly diverging from it.
    """
    corpus = generate_corpus(n_nodes, seed=seed)
    announcements = []
    for contributor in corpus:
        partition = load_partition(
            contributor.index,
            n_nodes,
            alpha=alpha,
            seed=seed,
            size_variation=SIZE_VARIATION,
        )
        record = declaration(
            contributor.policy,
            n_train=partition.n_train,
            label_counts=partition.label_counts,
            codes=contributor.codes,
        )
        announcements.append(parse_announcement(contributor.index, record))
    return announcements


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Policy-aware federation, end to end")
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require", nargs="+", default=["use", "aggregate"])
    parser.add_argument(
        "--purpose",
        default="ds-cancer",
        help="Purpose the federation needs the data for. Empty means actions "
             "only, which nearly every contributor grants.",
    )
    args = parser.parse_args(argv)

    request = requirement(
        *(Action(a) for a in args.require), purpose=args.purpose or None
    )
    utility = declared_utility(N_CLASSES)

    print("Policy-aware federation")
    print(f"  {args.nodes} nodes declaring, seed {args.seed}")
    print(f"  requirement: {', '.join(args.require)}"
          + (f" for purpose={args.purpose}" if args.purpose else ""))

    announcements = announce(args.nodes, args.alpha, args.seed)
    print(f"\n  declaration round: {len(announcements)} nodes answered")
    for a in announcements:
        print(f"    node {a.node_id:>2}  {a.n_train:>4} samples  "
              f"{len(a.classes)}/{N_CLASSES} classes  {', '.join(a.codes) or '-'}")

    try:
        admission = admit(announcements, request, utility)
    except NoFeasibleConsortium as exc:
        print(f"\n  ABORTED: {exc}")
        print("\n  No fallback is attempted. Training on everyone would produce")
        print("  precisely the model nobody is permitted to use.")
        return 1

    print(f"\n  admitted {len(admission.admitted)} of {len(announcements)}: "
          f"{list(admission.admitted)}")
    for node_id, reason in sorted(admission.excluded.items()):
        print(f"    excluded node {node_id} -- {reason}")
    print(f"  the federation may: "
          f"{', '.join(sorted(a.value for a in admission.capability))}")
    if admission.duties:
        print(f"  and owes: {', '.join(admission.duties)}")

    # The policy-blind comparison: same budget, chosen purely for data.
    blind = policy_blind_selection(announcements, utility, len(admission.admitted))
    blind_nodes = [announcements[i].node_id for i in blind]
    blind_ok = is_feasible([announcements[i] for i in blind], request)

    print(f"\n  a policy-blind consortium of the same size would pick {blind_nodes}")
    print(f"  is it permitted to use its own model? {'yes' if blind_ok else 'NO'}")

    print(f"\n  training {args.rounds} rounds on the admitted consortium ...")
    history = run_fedavg(
        contributors=admission.admitted,
        n_clients=args.nodes,
        rounds=args.rounds,
        alpha=args.alpha,
        seed=args.seed,
        size_variation=SIZE_VARIATION,
    )
    print(f"  accuracy on the shared held-out test set: "
          f"{history.final_accuracy:.3f}")

    blind_history = run_fedavg(
        contributors=blind_nodes,
        n_clients=args.nodes,
        rounds=args.rounds,
        alpha=args.alpha,
        seed=args.seed,
        size_variation=SIZE_VARIATION,
    )
    print(f"\n  the policy-blind model reaches {blind_history.final_accuracy:.3f}, "
          f"and is{'' if blind_ok else ' not'} deployable")

    if not blind_ok:
        print(f"\n  Deployable accuracy: {history.final_accuracy:.3f} against 0.000.")
        print("  Whether the policy-blind model scores higher varies by seed and is")
        print("  beside the point -- an undeployable model has no accuracy worth")
        print("  quoting. What selection buys is the right to use the result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
