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
