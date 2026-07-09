"""RolloutPolicy tests.

The decision rule (pick the higher-served branch) is tested offline with a
scripted fake env — no network. A separate network test checks it produces
valid actions and matches/beats AcceptAll on a real env (plan M3 exit gate).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dvrp_core.models.core import NodeLocation, Request, State, Vehicle

from dvrp_rl.policies import RolloutPolicy


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


class _Metrics:
    def __init__(self):
        self.served_requests = 0


class _FakeEnv:
    """Ends a rollout with ``served_requests = accept_val`` if the first
    stepped action is a Trip (accept), else ``reject_val``. Lets us assert
    RolloutPolicy.accept() picks the higher-served branch, offline."""

    def __init__(self, accept_val: int, reject_val: int, horizon_done: int = 5):
        self._geography = object()  # shared via deepcopy memo; unused here
        self._accept_val = accept_val
        self._reject_val = reject_val
        self._horizon_done = horizon_done
        self._first_seen = False
        self._steps = 0
        self.metrics = _Metrics()
        self._state = _state()

    def step(self, action):
        if not self._first_seen:
            self._first_seen = True
            self.metrics.served_requests = self._accept_val if action is not None else self._reject_val
        self._steps += 1
        return self._state, 0.0, self._steps >= self._horizon_done, {}


def _bound(accept_val: int, reject_val: int) -> RolloutPolicy:
    p = RolloutPolicy(horizon=5)
    p.bind_env(_FakeEnv(accept_val, reject_val))
    return p


def test_accepts_when_accepting_serves_more():
    assert _bound(accept_val=10, reject_val=5).accept(_state()) is True


def test_rejects_when_rejecting_serves_more():
    assert _bound(accept_val=5, reject_val=10).accept(_state()) is False


def test_ties_favor_accept():
    assert _bound(accept_val=6, reject_val=6).accept(_state()) is True


def test_accept_without_bind_raises():
    with pytest.raises(RuntimeError):
        RolloutPolicy().accept(_state())


# --- real-env gate ---------------------------------------------------------

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "binghampton.yaml"


class _CountingRollout(RolloutPolicy):
    """RolloutPolicy that counts how many requests it rejects."""

    def __init__(self, horizon: int = 30):
        super().__init__(horizon=horizon)
        self.rejects = 0

    def create_trips(self, state):
        action = super().create_trips(state)
        if action is None:
            self.rejects += 1
        return action


@pytest.mark.network
def test_rollout_beats_accept_all_and_actually_rejects():
    """Plan M3 exit gate. Two conditions so a *degenerate always-accept* rollout
    (which would merely tie AcceptAll) fails:
      (a) rollout's mean service rate is STRICTLY greater than AcceptAll's;
      (b) the rollout genuinely rejects >= 1 request over an episode.
    """
    from dvrp_rl.evaluate import evaluate, run_episode
    from dvrp_rl.policies import AcceptAll

    seeds = [100, 101, 102]
    accept = evaluate(CONFIG, lambda _s: AcceptAll(), seeds, n_steps=300)
    rollout = evaluate(CONFIG, lambda _s: RolloutPolicy(horizon=30), seeds, n_steps=300)
    assert rollout["mean_service_rate"] > accept["mean_service_rate"]

    counter = _CountingRollout(horizon=30)
    run_episode(CONFIG, counter, seed=100, n_steps=300)
    assert counter.rejects >= 1, "rollout degenerated to always-accept"
