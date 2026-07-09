"""Evaluate policies over seeds and report service rate.

Establishes the baseline number (plan M2): run each policy on the config's
eval seeds and report mean/std service_rate, so M3 can show a clear lift.

    python -m dvrp_rl.evaluate                      # Binghampton
    python -m dvrp_rl.evaluate configs/nashville.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

from dvrp_core.policy import Policy

from dvrp_rl.env import make_env_from_config
from dvrp_rl.policies import AcceptAll, RandomPolicy, RolloutPolicy
from dvrp_rl.scenario import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "binghampton.yaml"
N_STEPS = 300


def run_episode(config: str | Path | dict, policy: Policy, *, seed: int, n_steps: int = N_STEPS) -> dict[str, Any]:
    """Run one episode with ``policy`` on the demand stream fixed by ``seed``."""
    env, policy = make_env_from_config(config, policy=policy, seed=seed)
    if hasattr(policy, "bind_env"):
        policy.bind_env(env)  # planning policies (rollout) clone this env per decision
    state = env.reset()
    for _ in range(n_steps):
        state, _reward, done, _info = env.step(policy.create_trips(state))
        if done:
            break
    env.drain()
    m = env.metrics
    result = {
        "seed": seed,
        "service_rate": m.service_rate,
        "total_requests": m.total_requests,
        "served_requests": m.served_requests,
    }
    env.close()
    return result


def evaluate(
    config: str | Path | dict,
    make_policy: Callable[[int], Policy],
    seeds: list[int],
    *,
    n_steps: int = N_STEPS,
) -> dict[str, Any]:
    """Evaluate a policy across seeds.

    ``make_policy(seed)`` builds a fresh policy per episode (so stateful
    policies — Random, later RL — are seeded reproducibly per seed).
    """
    episodes = [run_episode(config, make_policy(s), seed=s, n_steps=n_steps) for s in seeds]
    rates = [e["service_rate"] for e in episodes]
    return {
        "mean_service_rate": mean(rates),
        # sample std (n-1); undefined for a single seed
        "std_service_rate": stdev(rates) if len(rates) > 1 else 0.0,
        "episodes": episodes,
    }


def main() -> None:
    config = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    seeds = load_config(config).get("seeds", {}).get("eval", [0, 1, 2])
    print(f"config: {config}\nseeds: {seeds}")
    print("(the sampled rollout does K clones/decision — expect a few minutes)\n")

    contenders: dict[str, Callable[[int], Policy]] = {
        "AcceptAll": lambda _s: AcceptAll(),
        "Random(0.5)": lambda s: RandomPolicy(accept_prob=0.5, seed=s),
        # oracle = perfect future foresight (upper bound); sampled = honest,
        # plans against sampled futures (deployable-style). See RESULTS.md.
        "Rollout(oracle)": lambda _s: RolloutPolicy(horizon=30, oracle=True),
        "Rollout(sampled,K=5)": lambda s: RolloutPolicy(horizon=30, n_samples=5, sample_seed=s),
    }
    for name, make_policy in contenders.items():
        res = evaluate(config, make_policy, seeds)
        print(f"{name:<22} service_rate: {res['mean_service_rate']:.1%} ± {res['std_service_rate']:.1%}")


if __name__ == "__main__":
    main()
