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
**Status: M3 (a planning policy beats the baseline).** Dependency proven,
config→env wrapper, feature layer, `AcceptAll`/`Random` baselines, and a
`Rollout` planner (one-step lookahead via MOSAIC's deepcopy). On the Binghampton
box `Rollout` **91.6%** > `AcceptAll` **88.1%** > `Random(0.5)` **48.8%**, winning
10/10 seeds (+3.5pp; +6.8pp under congestion). Note `Rollout` is an **oracle
upper bound** — it plans with perfect foresight of future demand (see the caveat
in [`RESULTS.md`](RESULTS.md)); a deployable learner is the next step.

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

Compare all policies across the config's eval seeds:

```bash
python -m dvrp_rl.evaluate                     # AcceptAll vs Random(0.5) vs Rollout(H=30)
python -m dvrp_rl.evaluate configs/nashville.yaml
```

## Reproducing the results

The numbers in [`RESULTS.md`](RESULTS.md) come straight from the command above.
On `configs/binghampton.yaml` (eval seeds `[100…109]`, 300 steps/episode, MOSAIC
`v0.1.1-rc.1`, `greedy` solver) you should see:

```
AcceptAll      service_rate: 88.1% ± 2.3%
Random(0.5)    service_rate: 48.8% ± 2.8%
Rollout(H=30)  service_rate: 91.6% ± 1.5%
```

`Rollout` wins 10/10 seeds vs `AcceptAll` (paired, +3.5pp mean). Run
`python -m dvrp_rl.evaluate configs/binghampton_congested.yaml` for the 3×-demand
comparison (+6.8pp).

**Determinism:** results are reproducible for a fixed `(config, seed)` — the
demand stream is seeded, so re-running yields identical service rates (within a
cached graph; see the determinism note in `plan.md`). `Rollout` is ~50× slower
than `AcceptAll` (it clones + simulates the env twice per request); expect
~11 s/episode on the Binghampton box.

`Rollout` is a MOSAIC-`Policy` subclass injected via `make_env(policy=…)`, so it
reuses MOSAIC's scenario/geography/demand/solver plumbing unchanged — it plans
by deep-copying the live env, never by modifying the simulator. **It is an oracle
planner** (the deepcopy carries the seeded demand RNG, so it sees future demand);
treat its rate as an upper bound, not a deployable result.

## Test

```bash
pytest -m "not network"    # fast, offline: scenario/features/policies + rollout decision rule
pytest -m network          # real-env: episode, injection, eval harness, Rollout ≥ AcceptAll gate
```

## Layout

```
configs/          scenario configs (polygon, depot, fleet, demand, seeds)
src/dvrp_rl/
  scenario.py     config YAML -> MOSAIC make_env spec dict
  env.py          config -> (env, policy); the one place that imports MOSAIC's builder
  features.py     State -> fixed feature vector (geography-free)
  policies/       AcceptRejectPolicy base + AcceptAll / Random baselines + Rollout planner
  evaluate.py     run N seeds per policy, report service_rate (python -m dvrp_rl.evaluate)
examples/
  run_demo.py     the one-command entry point
tests/            offline scenario/feature/policy tests + network-marked env/eval/smoke tests
```

The Gymnasium adapter (for SB3/RLlib) is intentionally deferred — see
`needs.md`. Our v0 policies drive MOSAIC's env directly.
