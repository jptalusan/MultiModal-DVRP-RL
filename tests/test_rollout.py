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
    # oracle=True: the comparison logic is identical to sampled mode, and it
    # skips the demand-reseed the fake env has no model for.
    p = RolloutPolicy(horizon=5, oracle=True)
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
        RolloutPolicy(oracle=True).accept(_state())


def test_sampled_mode_refuses_deterministic_demand():
    """Sampled mode must fail loudly (not silently become an oracle) when the
    demand can't be resampled — here the fake env has no demand model."""
    with pytest.raises(RuntimeError, match="stochastic"):
        RolloutPolicy(horizon=5).bind_env(_FakeEnv(1, 1))


# --- real-env gate ---------------------------------------------------------

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "binghampton.yaml"


class _CountingRollout(RolloutPolicy):
    """RolloutPolicy that counts how many requests it rejects."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rejects = 0

    def create_trips(self, state):
        action = super().create_trips(state)
        if action is None:
            self.rejects += 1
        return action


@pytest.mark.network
def test_oracle_rollout_beats_accept_all_and_actually_rejects():
    """M3 exit gate for the ORACLE planner (perfect foresight). Two conditions
    so a degenerate always-accept rollout (which would merely tie AcceptAll)
    fails: (a) STRICTLY higher mean service rate; (b) it rejects >= 1 request.
    (The sampled rollout is not expected to beat AcceptAll — see RESULTS.md.)
    """
    from dvrp_rl.evaluate import evaluate, run_episode
    from dvrp_rl.policies import AcceptAll

    seeds = [100, 101, 102]
    accept = evaluate(CONFIG, lambda _s: AcceptAll(), seeds, n_steps=300)
    oracle = evaluate(CONFIG, lambda _s: RolloutPolicy(horizon=30, oracle=True), seeds, n_steps=300)
    assert oracle["mean_service_rate"] > accept["mean_service_rate"]

    counter = _CountingRollout(horizon=30, oracle=True)
    run_episode(CONFIG, counter, seed=100, n_steps=300)
    assert counter.rejects >= 1, "oracle rollout degenerated to always-accept"


@pytest.mark.network
def test_sampled_rollout_is_non_oracle_and_reproducible():
    """The default (sampled) rollout must (a) be reproducible for a fixed
    sample_seed, and (b) actually depend on the sampled future — different
    sample_seeds give different results — i.e. it is NOT using the true future.
    """
    from dvrp_rl.evaluate import run_episode

    def rate(sample_seed):
        return run_episode(
            CONFIG, RolloutPolicy(horizon=15, n_samples=3, sample_seed=sample_seed),
            seed=100, n_steps=80,
        )["service_rate"]

    assert rate(7) == rate(7), "sampled rollout must be reproducible for a fixed sample_seed"
    assert rate(7) != rate(999), "different sample_seeds should differ — proves it's not the true future"
