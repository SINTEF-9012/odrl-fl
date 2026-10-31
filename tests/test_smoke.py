"""End-to-end smoke test.

Exercises the fast experiment path, which is what every later increment builds
on. The Flower path (`server_app` / `client_app`) is verified separately and
manually, because it needs a SuperLink and SuperNode processes -- see the
README. It is not automated here: Flower's simulation backend requires Ray, and
Ray ships no Windows wheels for Python 3.13.

What is automated here is that the Flower apps at least *import* and expose the
handlers Flower expects, which catches the most common way that path breaks
without needing to start a federation.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


def test_a_small_federation_learns():
    from odrl_fl.fedavg import run_fedavg

    history = run_fedavg(
        contributors=[0, 1], n_clients=4, rounds=5, alpha=1.0, seed=0
    )
    assert len(history.accuracy) == 5
    assert history.final_accuracy > history.accuracy[0]


def test_flower_apps_are_importable_and_wired():
    """Guards the Flower integration without starting a federation."""
    from flwr.clientapp import ClientApp
    from flwr.serverapp import ServerApp

    from odrl_fl.client_app import app as client_app
    from odrl_fl.server_app import app as server_app

    assert isinstance(server_app, ServerApp)
    assert isinstance(client_app, ClientApp)


def test_central_evaluate_scores_a_model():
    """The metric the Flower path reports must match the fast path's
    definition: global model, shared held-out test set."""
    from flwr.app import ArrayRecord

    from odrl_fl.server_app import make_central_evaluate
    from odrl_fl.task import Net

    metrics = make_central_evaluate(seed=0)(1, ArrayRecord(Net().state_dict()))
    assert 0.0 <= metrics["central_accuracy"] <= 1.0
    assert metrics["central_loss"] >= 0.0


def test_report_cost_logs_the_pruning_counters(caplog):
    """The instrumentation that brought down five federations in job 3077.

    `report_cost` read `result.candidates_evaluated`, but the counters live on
    `result.stats`. The AttributeError propagated out of the logging and killed
    the simulation it was measuring, at every population, after each one had
    already done its real work. This exercises it against a real `Admission`
    so that the attribute names are checked rather than assumed.
    """
    import logging

    from odrl_fl.corpus import generate_corpus
    from odrl_fl.selection import requirement
    from odrl_fl.server_app import report_cost
    from odrl_fl.strategy import Announcement, admit, declared_utility
    from odrl_fl.task import N_CLASSES

    corpus = generate_corpus(6, seed=0)
    announcements = [
        Announcement(
            node_id=1000 + c.index,
            policy=c.policy,
            n_train=100,
            label_counts={0: 50, 1: 50},
        )
        for c in corpus
    ]
    admission = admit(
        announcements,
        requirement("use", "aggregate"),
        declared_utility(N_CLASSES),
    )

    with caplog.at_level(logging.INFO):
        report_cost(announcements, admission, 0.5, 0.25)

    lines = [r.getMessage() for r in caplog.records]
    assert any("COST nodes=6" in m for m in lines)
    assert any("declaration_bytes_mean=" in m for m in lines)
    # The line that was never reached, because the attribute was wrong.
    assert any("COST evaluated=" in m for m in lines), lines
