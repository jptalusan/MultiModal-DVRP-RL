# MultiModal-DVRP-RL

Learn a dispatch **policy** (rollout / RL) against the
[MOSAIC](https://github.com/smarttransit-ai/MOSAIC) DVRP simulator. MOSAIC is
the *environment* (installed as a pinned pip dependency); this repo owns only
the learning code.

> **Prime directive:** clone this repo, follow the README, and run a learning
> agent against MOSAIC — on a small area with uniform demand — in one command,
> without touching the MOSAIC source or its backend.

Policies so far: `AcceptAll`/`Random` baselines and a `Rollout` planner (one-step
lookahead via MOSAIC's deepcopy). Benchmark numbers are in [`RESULTS.md`](RESULTS.md).

## Access to MOSAIC (private repo)

MOSAIC is private. The dependency is pinned to the tag `v0.1.1-rc.1`. Being a
member of the `smarttransit-ai` org grants **access**, but `git` still needs a
**credential** to authenticate the clone. Two paths — pick whichever you already
use with GitHub:

**A) SSH (default — the dependency URL is `git+ssh://…`).** You need an SSH key
registered to your (org-member) GitHub account. Verify:

```bash
ssh -T git@github.com     # should greet you by username
```

If that works, you're done — nothing else to configure.

**B) HTTPS + token** (if you don't use SSH). Keep the pinned URL but tell git to
rewrite the SSH form to HTTPS, then rely on your token / credential helper:

```bash
git config --global url."https://github.com/".insteadOf "ssh://git@github.com/"
gh auth login        # or store a PAT with `repo` scope via your credential helper
```

Either way, if install fails with a 403 / "repository not found", it's an
auth/access problem — confirm your account can open
`https://github.com/smarttransit-ai/MOSAIC` in a browser.

## Install

Requires **Python ≥ 3.12** (MOSAIC's floor). With [uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.12
uv venv --python 3.12
uv pip install -e ".[dev]"          # add ,notebook for the demo notebook: ".[dev,notebook]"
```

or with plain pip in a 3.12 virtualenv:

```bash
pip install -e ".[dev]"
```

This pulls MOSAIC (`dvrp-gym`) from the pinned git tag (the base install — no
backend/DB). To move to a newer MOSAIC rc, bump the tag in `pyproject.toml` and
reinstall; for a *same-version* re-pull use `pip install --force-reinstall`.

## Run the demo (one command)

```bash
python examples/run_demo.py                    # small Binghampton box
python examples/run_demo.py configs/nashville.yaml
```

`run_demo.py` builds a MOSAIC env from the config, runs one short uniform-demand
episode with MOSAIC's stock `on_demand_only` policy, and prints the service rate,
e.g.:

```
env ready · vehicles: 3 · t0: 66.7s
episode done · requests: 200 (served 169, rejected 31) · service_rate: 84.5%
```

The **first run** fetches the OSM road graph for the polygon via `osmnx` and
caches it under `cache/` — this needs network once (a minute or two); every run
after reuses the pickle and is fast. `cache/` is git-ignored.

For a guided, sectioned walkthrough (env → baselines → rollout), open the
notebook: [`notebooks/demo.ipynb`](notebooks/demo.ipynb) (install the `notebook`
extra first, then `jupyter lab`).

## Compare policies

```bash
python -m dvrp_rl.evaluate                     # AcceptAll vs Random vs Rollout (oracle + sampled)
python -m dvrp_rl.evaluate configs/nashville.yaml
python -m dvrp_rl.evaluate configs/binghampton_congested.yaml   # 3× demand
```

Train the learned (REINFORCE) policy and compare it to the baseline on held-out
seeds:

```bash
python -m dvrp_rl.train                        # trains, saves to models/, evals vs AcceptAll
```

## Policies

All policies are MOSAIC `Policy` subclasses; for `on_demand_only` each decides
**accept** (return a `Trip`) or **reject** (`None`) per incoming request.

| Policy | What it does |
|---|---|
| `AcceptAll` | Accept every request. The baseline to beat (= MOSAIC's stock policy). |
| `RandomPolicy(accept_prob, seed)` | Accept each request with a fixed probability. |
| `RolloutPolicy(horizon, oracle, n_samples, sample_seed)` | One-step lookahead: simulate accept vs reject `horizon` steps ahead on env clones, take the branch that serves more. |
| `ReinforcePolicy(detour_tolerance, weights, bias)` | Learned linear-logistic policy over the feature vector — decides from the current state only (no foresight, no cloning). Train with `python -m dvrp_rl.train`. |

`RolloutPolicy` has two modes:

```python
from dvrp_rl.policies import RolloutPolicy

# ORACLE — the clone keeps the true demand RNG, so it plans with perfect
# foresight of future requests. An upper bound, NOT deployable.
RolloutPolicy(horizon=30, oracle=True)

# SAMPLED (default) — reseeds each clone's demand RNG, so it plans against
# sampled, plausible futures (never the real one), averaged over n_samples with
# common random numbers. Reproducible for a fixed sample_seed, but not an oracle.
RolloutPolicy(horizon=30, n_samples=5, sample_seed=0)
```

Sampled mode **requires stochastic demand** (e.g. `uniform`): on deterministic
`file`-replay demand it raises, rather than silently replaying the true future.
`RolloutPolicy` needs the env to plan against — the evaluation harness calls
`policy.bind_env(env)` for you (see `evaluate.run_episode`).

## Reproducing the results

The numbers in [`RESULTS.md`](RESULTS.md) come straight from `python -m
dvrp_rl.evaluate`. On `configs/binghampton.yaml` (eval seeds `[100…104]`, 300
steps/episode, MOSAIC `v0.1.1-rc.1`, `greedy` solver) you should see roughly:

```
AcceptAll               service_rate: 88.6% ± 2.3%
Random(0.5)             service_rate: 49.5% ± 0.8%
Rollout(oracle)         service_rate: 91.7% ± 2.0%   # perfect-foresight upper bound
Rollout(sampled,K=5)    service_rate: 81.7% ± 4.0%   # honest, no foresight
```

**Determinism:** reproducible for a fixed `(config, seed)` — the demand stream is
seeded, so re-running gives identical service rates (within a cached graph). The
sampled rollout is *also* reproducible (its future-sampling is seeded) yet is
**not** an oracle. The rollout is ~10–50× slower than `AcceptAll` (K clones +
simulations per request), so the full command takes a few minutes.

## Test

```bash
pytest -m "not network"    # fast, offline: scenario/features/policies + rollout decision rule
pytest -m network          # real-env: episode, injection, eval harness, rollout gates
```

CI runs the offline set. Network tests need a MOSAIC credential + a one-time OSM
fetch, so they're excluded from CI by default (see `.github/workflows/ci.yml`).

## Layout

```
configs/          scenario configs (polygon, depot, fleet, demand, seeds)
src/dvrp_rl/
  scenario.py     config YAML -> MOSAIC make_env spec dict
  env.py          config -> (env, policy); the one place that imports MOSAIC's builder
  features.py     State -> fixed feature vector (geography-free)
  policies/       AcceptRejectPolicy base + AcceptAll / Random + Rollout planner
  evaluate.py     run N seeds per policy, report service_rate (python -m dvrp_rl.evaluate)
examples/
  run_demo.py     the one-command entry point
notebooks/
  demo.ipynb      guided walkthrough: env -> baselines -> rollout
tests/            offline scenario/feature/policy tests + network-marked env/eval/smoke tests
```

The Gymnasium adapter (for SB3/RLlib) is intentionally deferred — see
`needs.md`. Our v0 policies drive MOSAIC's env directly.

## Roadmap

Done: install + env wrapper, feature layer, `AcceptAll`/`Random` baselines, a
`Rollout` planner (oracle beats `AcceptAll`; honest sampled variant trails and
improves with more samples), and a deployable **REINFORCE** policy — which
*matches* accept-all but doesn't beat it (accept-all is near-optimal here; the
beneficial-reject signal needs foresight or richer features). **Next lever:
geometric features** (nearest-vehicle slack / insertion cost) to give a learned
policy a discriminative signal. Design notes and milestones are in
[`plan.md`](plan.md); MOSAIC gaps we work around are in [`needs.md`](needs.md).
