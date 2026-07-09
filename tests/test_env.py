"""Env-wrapper tests: build + step, and that an injected policy is used.

Marked ``network`` — builds a real MOSAIC env (first-run OSM fetch)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dvrp_rl.env import make_env_from_config

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "binghampton.yaml"


@pytest.mark.network
def test_build_from_config_steps_and_reports_metrics():
    env, policy = make_env_from_config(CONFIG, seed=7)
    state = env.reset()
    assert len(state.vehicles) == 3  # fleet size comes from the config depot

    for _ in range(50):
        action = policy.create_trips(state)
        state, _r, done, _i = env.step(action)
        if done:
            break
    env.drain()
    assert env.metrics.total_requests > 0
    env.close()


@pytest.mark.network
def test_injected_policy_instance_is_the_one_driven():
    """A Policy instance passed in must be returned as-is AND actually drive
    the env. Reject-everything -> zero served proves our decision was used."""
    from dvrp_core.models.core import State
    from dvrp_core.policy import Policy

    class RejectAllSpy(Policy):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def create_trips(self, state: State):
            self.calls += 1
            return None  # reject

    spy = RejectAllSpy()
    env, policy = make_env_from_config(CONFIG, policy=spy, seed=7)
    assert policy is spy, "make_env must use our instance as-is"

    state = env.reset()
    for _ in range(50):
        state, _r, done, _i = env.step(policy.create_trips(state))
        if done:
            break
    env.drain()

    assert spy.calls > 0, "our policy was never consulted"
    assert env.metrics.total_requests > 0
    assert env.metrics.served_requests == 0, "reject-all should serve nobody"
    env.close()
