"""Train a federated model over a chosen set of contributors.

The experiment entry point. Later increments call `run_fedavg` directly while
sweeping subsets; this wrapper is for running one configuration by hand.

    python experiments/run_fedavg.py --clients 6 --rounds 5
    python experiments/run_fedavg.py --clients 6 --contributors 0 2 4
    python experiments/run_fedavg.py --clients 6 --alpha 0.05     # strong skew
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odrl_fl.fedavg import run_fedavg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run federated averaging")
    parser.add_argument("--clients", type=int, default=6, help="Population size")
    parser.add_argument(
        "--contributors", type=int, nargs="+", default=None,
        help="Client ids to federate (default: all of them)",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Dirichlet concentration: small => strong label skew, large => IID",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    contributors = (
        args.contributors if args.contributors is not None else list(range(args.clients))
    )

    print(f"Population {args.clients}, contributors {contributors}, alpha {args.alpha}")
    history = run_fedavg(
        contributors=contributors,
        n_clients=args.clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        alpha=args.alpha,
        seed=args.seed,
    )

    for i, (loss, acc) in enumerate(zip(history.loss, history.accuracy), start=1):
        print(f"  round {i:>2}  loss {loss:8.3f}  accuracy {acc:.4f}")
    print(f"final accuracy {history.final_accuracy:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
