"""Flower ServerApp: policy-aware FedAvg.

Membership is decided before the first round. The server asks every connected
node to declare its data-use terms, composes the declarations, and searches for
a consortium whose combined terms grant what the federation actually needs.
Only that consortium trains.

It then acts on what the agreement *demands*, in the only three ways a system
can. Preconditions **gate**: the run refuses to start without evidence that
somebody discharged them. Runtime duties are **dispatched** into the training
config and their claimed discharge verified before an update is aggregated.
Everything else -- publish results, return findings, attribute, compensate --
can only be **recorded**, because no training run can perform it. See
`odrl_fl.duties` for why that last category is the largest one in practice.

The run ends by exporting a **policy-carrying model**: the weights, the composed
agreement, every contributor's declaration, and the ledger of what remains
outstanding, in one verifiable bundle. That closes the loop opened by the duty
classification -- the obligations a training run cannot discharge are exactly
the ones whoever holds the model must, and they cannot act on what they were
never told.

Still to come: re-evaluating membership when a member withdraws consent mid-run
(increment 9).

Set `require-purpose` to see selection bite. A request naming only actions is
nearly always satisfiable, so almost nobody gets excluded and the run resembles
plain FedAvg. That is the honest behaviour rather than a bug: when the
requirement is weak, selection has little to do.
"""

from __future__ import annotations

from logging import INFO

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp

from odrl_fl.bundle import ContributorRecord, build_manifest, export
from odrl_fl.duties import Ledger, check_gates
from odrl_fl.odrl import Action
from odrl_fl.selection import requirement
from odrl_fl.strategy import (
    Announcement,
    PolicyAwareFedAvg,
    admit,
    collect_announcements,
    declared_utility,
)
from odrl_fl.task import N_CLASSES, Net, evaluate, global_test_loader

app = ServerApp()


def make_central_evaluate(seed: int):
    """Score the global model centrally on the shared held-out test set.

    Deliberately the *same* metric `odrl_fl.fedavg` reports, so the Flower path
    and the fast experiment path are directly comparable. Federated evaluation
    on clients' own splits is not comparable across contributor subsets --
    each subset would be scored on a different distribution.
    """
    testloader = global_test_loader(seed=seed)

    def central_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        net = Net()
        net.load_state_dict(arrays.to_torch_state_dict())
        loss, accuracy = evaluate(net, testloader)
        return MetricRecord({"central_loss": loss, "central_accuracy": accuracy})

    return central_evaluate


def parse_evidence(raw: str) -> dict[str, str]:
    """Read `duty-evidence` -- "obtainConsent=IRB-2026-0142,anonymize=DPIA-7".

    Thin on purpose: it records that a human asserted a reference, and nothing
    here checks the reference is real. The gate's value is that the assertion
    has to be made explicitly before the run proceeds, not that this code
    validates ethics paperwork.
    """
    evidence: dict[str, str] = {}
    for item in raw.split(","):
        if "=" in item:
            action, _, reference = item.partition("=")
            evidence[action.strip()] = reference.strip()
    return evidence


def report(announcements: list[Announcement], admission) -> None:
    """Log the decision in full. An unexplained exclusion is not auditable."""
    log(INFO, "policy round: %s nodes declared", len(announcements))
    log(
        INFO,
        "admitted %s of %s declared: %s",
        len(admission.admitted),
        len(announcements),
        list(admission.admitted),
    )
    for node_id, reason in sorted(admission.excluded.items()):
        log(INFO, "  excluded node %s -- %s", node_id, reason)
    log(
        INFO,
        "the federation may: %s",
        ", ".join(sorted(a.value for a in admission.capability)) or "nothing",
    )
    plan = admission.plan
    if plan.gates:
        log(INFO, "before starting, must evidence: %s",
            ", ".join(sorted(d.action.value for d in plan.gates)))
    if plan.directives:
        log(INFO, "dispatched to clients each round: %s",
            ", ".join(f"{d.action}"
                      + (f" (eps<={d.epsilon})" if d.epsilon else "")
                      for d in plan.directives))
    for obligation in plan.outstanding:
        log(INFO, "OUTSTANDING %s -- owed by %s, not dischargeable by training",
            obligation.action, ", ".join(obligation.owed_by) or "the consortium")


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = int(context.run_config.get("num-server-rounds", 5))
    local_epochs = int(context.run_config.get("local-epochs", 1))
    fraction_train = float(context.run_config.get("fraction-train", 1.0))

    # What this federation needs to be able to do. Naming a purpose is what
    # makes the requirement bite -- see the module docstring.
    actions = [
        Action(a.strip())
        for a in str(
            context.run_config.get("require-actions", "use,aggregate")
        ).split(",")
        if a.strip()
    ]
    purpose = str(context.run_config.get("require-purpose") or "") or None
    request = requirement(*actions, purpose=purpose)

    # Decide membership before a single round happens. `admit` raises
    # NoFeasibleConsortium if nothing composes -- deliberately fatal, because
    # falling back to training on everyone produces exactly the unusable model
    # this work is about.
    announcements, unreachable = collect_announcements(grid)
    admission = admit(
        announcements,
        request,
        declared_utility(N_CLASSES),
        rejected=unreachable,
    )
    report(announcements, admission)

    # The gate. A duty that must be discharged before the run cannot be
    # discharged *by* the run, so the only faithful thing a system can do is
    # refuse to start without evidence that somebody discharged it.
    evidence = parse_evidence(str(context.run_config.get("duty-evidence") or ""))
    cleared = check_gates(admission.plan.gates, evidence)
    for action, reference in sorted(cleared.items()):
        log(INFO, "precondition %s cleared, evidence %s", action, reference)

    ledger = Ledger(
        outstanding=admission.plan.outstanding, gates_cleared=cleared
    )

    strategy = PolicyAwareFedAvg(
        admitted=admission.admitted,
        directives=admission.plan.directives,
        ledger=ledger,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_train,
    )

    seed = int(context.run_config.get("seed", 0))
    torch.manual_seed(seed)

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(Net().state_dict()),
        num_rounds=num_rounds,
        train_config=ConfigRecord({"local-epochs": local_epochs}),
        evaluate_config=ConfigRecord({}),
        evaluate_fn=make_central_evaluate(seed),
    )

    # Ship the model with the terms it was built under. Separated, the terms
    # are lost the first time the file is copied -- and the duties that survive
    # a training run are precisely the ones only the *holder* can discharge.
    by_node = {a.node_id: a for a in announcements}
    records = [
        ContributorRecord(
            node_id=node_id,
            policy=by_node[node_id].policy,
            codes=by_node[node_id].codes,
            n_train=by_node[node_id].n_train,
        )
        for node_id in admission.admitted
    ]
    manifest = build_manifest(
        admission.agreement,
        records,
        request=request,
        excluded=admission.excluded,
        ledger=ledger,
        metrics={
            k: float(v)
            for k, v in (result.evaluate_metrics_serverapp.get(num_rounds) or {}).items()
        },
    )
    out = export(
        context.run_config.get("bundle-dir", "artifacts/model-bundle"),
        manifest,
        result.arrays.to_torch_state_dict(),
    )

    log(INFO, "policy-carrying model written to %s", out)
    for line in manifest.describe():
        log(INFO, "  %s", line)
