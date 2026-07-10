"""ReinforcePolicy + trainer tests. Offline mechanics; network training run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dvrp_core.models.core import NodeLocation, Request, State, Vehicle

from dvrp_rl.policies import ReinforcePolicy


def _state() -> State:
    req = Request(
        id=0, origin=NodeLocation(0), destination=NodeLocation(1), passengers=1,
        received_time=100.0, earliest_pickup=100.0, latest_dropoff=460.0,
    )
    veh = Vehicle(id=0, depot_id=0, location=NodeLocation(0), capacity=4)
    return State(
        current_time=100.0, vehicles=[veh], accepted_trips={}, passengers={},
        new_request=req, transit_routes=[], bus_states=[],
    )


def test_greedy_threshold():
    s = _state()
    assert ReinforcePolicy(4.0, bias=10.0).accept(s) is True    # p≈1
    assert ReinforcePolicy(4.0, bias=-10.0).accept(s) is False  # p≈0


def test_prob_in_unit_interval():
    _x, p = ReinforcePolicy(4.0, bias=0.3).prob_accept(_state())
    assert 0.0 <= p <= 1.0


def test_sampling_records_trajectory():
    p = ReinforcePolicy(4.0, bias=0.0, rng=np.random.default_rng(0))
    for _ in range(6):
        p.accept(_state())
    assert len(p.trajectory) == 6
    x, a, prob = p.trajectory[0]
    assert x.shape == (4,) and a in (0, 1) and 0.0 <= prob <= 1.0


def test_save_load_roundtrip(tmp_path: Path):
    from dvrp_rl.train import load_policy, save_policy

    orig = ReinforcePolicy(4.0, weights=np.array([0.1, -0.2, 0.3, -0.4]), bias=1.5)
    path = tmp_path / "m.json"
    save_policy(orig, path)
    loaded = load_policy(path)
    assert np.allclose(loaded.w, orig.w)
    assert loaded.b == orig.b
    assert loaded.detour_tolerance == orig.detour_tolerance
    assert loaded.accept(_state()) == orig.accept(_state())


CONFIG = Path(__file__).resolve().parent.parent / "configs" / "binghampton.yaml"


@pytest.mark.network
def test_training_runs_and_lands_in_the_accept_region():
    """A short training run must produce a valid, non-degenerate greedy policy:
    it converges toward accept-all (near-optimal here), so its eval service rate
    is high — not collapsed to reject-all."""
    from dvrp_rl.evaluate import evaluate
    from dvrp_rl.train import train_reinforce

    policy, history = train_reinforce(
        CONFIG, train_seeds=[0, 1], n_episodes=20, n_steps=120, seed=0,
    )
    assert len(history["returns"]) == 20
    rate = evaluate(CONFIG, lambda _s: policy, [100], n_steps=120)["mean_service_rate"]
    assert 0.7 <= rate <= 1.0, "learned policy should sit in the accept-heavy region, not collapse"
