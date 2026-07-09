# MultiModal-DVRP-RL

Learn a dispatch **policy** (RL or MCTS) against the
[MOSAIC](https://github.com/smarttransit-ai/MOSAIC) DVRP simulator. MOSAIC is
the *environment* (installed as a pinned pip dependency); this repo owns only
the learning code.

> **Prime directive:** clone this repo, follow the README, and run a learning
> agent against MOSAIC — on a small area with uniform demand — in one command,
> without touching the MOSAIC source or its backend.

See [`plan.md`](plan.md) for goals, milestones, and design decisions, and
[`needs.md`](needs.md) for gaps in MOSAIC we work around (we don't modify it).
**Status: M2 (baselines + evaluation).** Dependency proven, config→env
wrapper, feature layer, and `AcceptAll`/`Random` baselines with an evaluation
harness are in place. On the Binghampton box, `AcceptAll` serves **~90%** and
`Random(0.5)` **~49%** — the number M3's learned policy must beat.

## Access to MOSAIC

MOSAIC is a private repo. The dependency is pinned to the tag
`v0.1.1-rc.1` and installed over SSH, so you need an SSH key with read access
to `smarttransit-ai/MOSAIC` (`ssh -T git@github.com` should greet you).

## Install

Requires **Python ≥ 3.12** (MOSAIC's floor). Using [uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.12
uv venv --python 3.12
uv pip install -e ".[dev]"
```

or with plain pip in a 3.12 virtualenv:

```bash
pip install -e ".[dev]"
```

This pulls MOSAIC (`dvrp-gym`) from the pinned git tag. To move to a newer
MOSAIC rc, bump the tag in `pyproject.toml` and reinstall
(`pip install --force-reinstall` for a same-version re-pull).

## Run (one command)

```bash
python examples/run_demo.py                    # small Binghampton box
python examples/run_demo.py configs/nashville.yaml
```

The **first run** fetches the OSM road graph for the polygon via `osmnx` and
caches it under `cache/` (needs network once; subsequent runs reuse the
pickle). It runs one short uniform-demand episode with MOSAIC's stock
`on_demand_only` policy and prints the service rate.

Compare the baselines across the config's eval seeds:

```bash
python -m dvrp_rl.evaluate                     # AcceptAll vs Random(0.5)
```

## Test

```bash
pytest -m "not network"    # fast, offline: spec building + depot-in-polygon
pytest -m network          # full end-to-end episode (first run fetches OSM)
```

## Layout

```
configs/          scenario configs (polygon, depot, fleet, demand, seeds)
src/dvrp_rl/
  scenario.py     config YAML -> MOSAIC make_env spec dict
  env.py          config -> (env, policy); the one place that imports MOSAIC's builder
  features.py     State -> fixed feature vector (geography-free)
  policies/       AcceptRejectPolicy base + AcceptAll / Random baselines
  evaluate.py     run N seeds per policy, report service_rate (python -m dvrp_rl.evaluate)
examples/
  run_demo.py     the one-command entry point
tests/            offline scenario/feature/policy tests + network-marked env/eval/smoke tests
```

The Gymnasium adapter (for SB3/RLlib) is intentionally deferred — see
`needs.md`. Our v0 policies drive MOSAIC's env directly.
