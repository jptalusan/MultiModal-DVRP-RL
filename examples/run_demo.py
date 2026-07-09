"""M0 demo: prove the MOSAIC dependency runs end to end.

Builds a scenario from a config, constructs a MOSAIC env with the stock
``on_demand_only`` policy, runs one short uniform-demand episode, drains,
and prints the resulting metrics. This is the one command the README
points at.

    python examples/run_demo.py                      # Binghampton (small)
    python examples/run_demo.py configs/nashville.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

from dvrp_core.env.library import make_env

from dvrp_rl.scenario import build_spec, load_config

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "binghampton.yaml"
N_STEPS = 200


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    print(f"config: {config_path}")

    spec = build_spec(load_config(config_path))
    print("building env (first run fetches + caches the OSM graph under cache/) ...")

    env, policy = make_env(spec, solver="greedy", policy="on_demand_only", demand="uniform", seed=42)
    state = env.reset()
    print(f"env ready · vehicles: {len(state.vehicles)} · t0: {state.current_time:.1f}s")

    for _ in range(N_STEPS):
        action = policy.create_trips(state)
        state, _reward, done, _info = env.step(action)
        if done:
            break
    env.drain()

    m = env.metrics
    print(
        f"episode done · requests: {m.total_requests} "
        f"(served {m.served_requests}, rejected {m.rejected_requests}) "
        f"· service_rate: {m.service_rate:.1%}"
    )
    env.close()


if __name__ == "__main__":
    main()
