"""Train the REINFORCE accept/reject policy (policy gradient).

Per episode: run the policy in sampling mode on a training seed, take the
episode ``service_rate`` as the return, subtract an EMA baseline, and ascend the
score-function gradient ``sum_t (a_t - p_t) * x_t``. Weights are saved to JSON so
the demo / evaluator can load a trained policy without retraining.

    python -m dvrp_rl.train                       # train on Binghampton, eval vs AcceptAll
    python -m dvrp_rl.train configs/nashville.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from dvrp_rl.env import make_env_from_config
from dvrp_rl.policies.reinforce import ReinforcePolicy
from dvrp_rl.scenario import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "binghampton.yaml"
DEFAULT_MODEL = Path(__file__).resolve().parent.parent.parent / "models" / "reinforce_binghampton.json"


def train_reinforce(
    config: str | Path | dict,
    *,
    train_seeds: list[int],
    n_episodes: int = 400,
    n_steps: int = 300,
    lr: float = 5.0,
    warm_start_bias: float = 3.0,
    baseline_decay: float = 0.9,
    entropy_coef: float = 0.0,
    seed: int = 0,
) -> tuple[ReinforcePolicy, dict[str, Any]]:
    """Train a ReinforcePolicy; return (greedy policy, history).

    Warm-started near accept-all (``bias`` high → p≈1) so training starts from a
    strong prior and learns *which* requests to reject. Episodes cycle through
    ``train_seeds`` (kept disjoint from eval seeds to avoid leakage).

    REINFORCE: advantage = return − EMA baseline; the gradient is averaged over
    the trajectory (so ``lr`` doesn't depend on episode length). Warm-starting
    near accept-all keeps training stable; ``entropy_coef`` (default off) adds an
    entropy bonus toward p=0.5 to force more exploration if you want it — but on
    this problem the advantage signal is weak and raising it tends to push the
    policy into over-rejecting, so it's off by default.
    """
    cfg = load_config(config) if isinstance(config, (str, Path)) else config
    detour = float(cfg["detour_tolerance"])
    rng = np.random.default_rng(seed)
    policy = ReinforcePolicy(detour, bias=warm_start_bias, rng=rng)

    baseline: float | None = None
    returns: list[float] = []
    for ep in range(n_episodes):
        ep_seed = train_seeds[ep % len(train_seeds)]
        env, _ = make_env_from_config(config, seed=ep_seed)  # we drive with `policy`
        policy.trajectory = []
        state = env.reset()
        for _ in range(n_steps):
            state, _r, done, _i = env.step(policy.create_trips(state))
            if done:
                break
        env.drain()
        ret = env.metrics.service_rate
        env.close()

        baseline = ret if baseline is None else baseline_decay * baseline + (1 - baseline_decay) * ret
        advantage = ret - baseline
        n = max(len(policy.trajectory), 1)
        grad_w = np.zeros_like(policy.w)
        grad_b = 0.0
        for x, a, p in policy.trajectory:
            score = (a - p) * advantage                       # policy-gradient term
            entropy = entropy_coef * p * (1 - p) * np.log((1 - p + 1e-9) / (p + 1e-9))  # dH/dz
            g = score + entropy
            grad_w += g * x
            grad_b += g
        policy.w += lr * grad_w / n
        policy.b += lr * grad_b / n
        returns.append(ret)

    policy._rng = None  # switch to greedy (deployable) for eval/use
    history = {"returns": returns, "final_w": policy.w.tolist(), "final_b": policy.b}
    return policy, history


def save_policy(policy: ReinforcePolicy, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "weights": policy.w.tolist(),
        "bias": policy.b,
        "detour_tolerance": policy.detour_tolerance,
    }, indent=2))


def load_policy(path: str | Path) -> ReinforcePolicy:
    d = json.loads(Path(path).read_text())
    return ReinforcePolicy(d["detour_tolerance"], weights=np.array(d["weights"]), bias=d["bias"])


def main() -> None:
    from dvrp_rl.evaluate import evaluate
    from dvrp_rl.policies import AcceptAll

    config = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    cfg = load_config(config)
    train_seeds = cfg.get("seeds", {}).get("train", [0, 1, 2, 3])
    eval_seeds = cfg.get("seeds", {}).get("eval", [100, 101, 102])
    print(f"config: {config}\ntrain seeds: {train_seeds}\neval seeds: {eval_seeds}\ntraining ...")

    policy, history = train_reinforce(config, train_seeds=train_seeds)
    save_policy(policy, DEFAULT_MODEL)
    r0, rN = np.mean(history["returns"][:20]), np.mean(history["returns"][-20:])
    print(f"train return: {r0:.1%} (first 20) -> {rN:.1%} (last 20); weights={np.round(policy.w, 2)} b={policy.b:.2f}")

    acc = evaluate(config, lambda _s: AcceptAll(), eval_seeds)
    learned = evaluate(config, lambda _s: load_policy(DEFAULT_MODEL), eval_seeds)
    print("\nEVAL (held-out seeds):")
    print(f"  AcceptAll  service_rate: {acc['mean_service_rate']:.1%} ± {acc['std_service_rate']:.1%}")
    print(f"  REINFORCE  service_rate: {learned['mean_service_rate']:.1%} ± {learned['std_service_rate']:.1%}")


if __name__ == "__main__":
    main()
