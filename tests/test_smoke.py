"""End-to-end smoke test — the M0 exit criterion.

Builds a real MOSAIC env, runs a short uniform-demand episode with the
stock ``on_demand_only`` policy, drains, and asserts metrics exist and
are self-consistent.

Marked ``network``: the first run fetches the OSM graph via osmnx, so it
is skipped in CI by default (no network / would be flaky). Run locally:

    pytest -m network
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dvrp_rl.scenario import build_spec, load_config

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "binghampton.yaml"


@pytest.mark.network
def test_short_episode_produces_metrics():
    from dvrp_core.env.library import make_env

    spec = build_spec(load_config(CONFIG))
    env, policy = make_env(spec, policy="on_demand_only", demand="uniform", seed=42)

    state = env.reset()
    assert len(state.vehicles) == 3, "fleet size should come from the config depot"

    for _ in range(150):
        action = policy.create_trips(state)
        state, _reward, done, _info = env.step(action)
        if done:
            break
    env.drain()

    m = env.metrics
    assert m.total_requests > 0, "uniform demand should have generated requests"
    assert m.served_requests + m.rejected_requests <= m.total_requests
    assert 0.0 <= m.service_rate <= 1.0
    env.close()
