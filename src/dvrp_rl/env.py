"""Build a MOSAIC env from one of our configs.

Thin wrapper over ``dvrp_core.env.library.make_env`` so callers (the demo,
the evaluator, the trainer) share one entry point and don't each repeat
``load_config`` -> ``build_spec`` -> ``make_env`` or import ``dvrp_core``
directly. This is the only module in the package that imports MOSAIC's
builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dvrp_core.env.library import make_env

from dvrp_rl.scenario import build_spec, load_config


def make_env_from_config(
    config: str | Path | dict[str, Any],
    *,
    solver: str = "greedy",
    policy: str | Any = "on_demand_only",
    demand: str = "uniform",
    seed: int | None = None,
    cache_dir: str = "cache",
):
    """Build ``(env, policy)`` from a config path or a loaded config dict.

    ``policy`` is passed through to ``make_env``: a registered name, or a
    ``Policy`` **instance** (e.g. one of ours) to inject and drive
    directly. All other kwargs mirror ``make_env``.
    """
    cfg = load_config(config) if isinstance(config, (str, Path)) else config
    spec = build_spec(cfg)
    return make_env(spec, solver=solver, policy=policy, demand=demand, seed=seed, cache_dir=cache_dir)
