# MultiModal-DVRP-RL — plan

A standalone repo that installs the **MOSAIC** simulator as a pip dependency and
learns a dispatch **policy** (RL or MCTS) against it. The MOSAIC engine is the
*environment*; this repo owns only the learning code.

> **Prime directive:** anyone can `git clone` this repo, follow the README, and
> run a learning agent against the MOSAIC engine — on a small area with uniform
> demand — in one command, without touching the MOSAIC source or its backend.

## Goal / non-goals

**Goal (v0):** train + evaluate a simple learned policy for the **`on_demand_only`**
setting and show it beats a naive baseline on service rate, reproducibly.

**Non-goals (v0):** multimodal/transit, real MOVE-OD demand, GTFS, the MOSAIC
API/DB/frontend, large cities, distributed training, SOTA algorithms. Keep it
small and legible; leave hooks to grow.

## Dependency on MOSAIC (the environment)

- Install the **pre-release rc** as a git dependency (base install — **no**
  backend/uvicorn/DB; those are the `[api]` extra and are not needed):
  ```toml
  # pyproject.toml
  dependencies = [
    "dvrp-gym @ git+ssh://git@github.com/smarttransit-ai/MOSAIC.git@v0.1.1-rc.1",
    "numpy>=1.26",
  ]
  # optional: dvrp-gym[ortools] only if we want the OR-Tools solver baseline
  ```
- **Pin the exact rc tag** (`v0.1.1-rc.1`), never a moving branch. Bump the pin
  when MOSAIC cuts a newer rc / the real `v0.1.1`. (Same-version reinstalls need
  `pip install --force-reinstall …@<ref>`.)
- Public entry points we rely on:
  - `from dvrp_core.env.library import make_env` — build a ready-to-step env.
  - `from dvrp_core.policy import Policy` — base class for our learned policy.
  - `make_env(spec, policy=<Policy instance>)` — **injection hook**: pass our
    policy and reuse MOSAIC's scenario/geography/demand/solver plumbing.
  - `env.reset()` → `State`; `env.step(action)` → `(state, reward, done, info)`
    (legacy 4-tuple); `env.drain()`, `env.metrics`.
- First run downloads the OSM graph for our polygon via `osmnx` and caches it
  under `cache/` — so keep the area **small** (fast all-pairs precompute).

## The environment / scenario (v0)

- **Demand:** `uniform` (seed-reproducible; good for train/eval splits by seed).
- **Area (small, fast):** default to a compact box; target **Binghampton,
  Memphis** (a ~2 km neighborhood). *TODO: finalize the polygon* — approximate
  bbox ≈ lat 35.14–35.16, lon −89.99…−89.95; verify against the real neighborhood
  before committing. Fallback: the Nashville box from MOSAIC's quickstart (known-good).
- **Fleet:** 1 depot inside the polygon, a few vehicles (e.g. 3 × cap 4). Depot
  location + fleet size are config, not hardcoded.
- All of the above lives in a single `spec` dict + a config file — one place to edit.

## The learning problem (on_demand_only)

Per step, MOSAIC hands the policy `state.new_request` (a `Request`) plus the fleet
(`state.vehicles`, `state.accepted_trips`). For `on_demand_only`, `create_trips(state)`
returns **a `Trip` (accept — the greedy solver then inserts it) or `None` (reject)**.
So the learned decision is essentially **accept / reject each incoming request** to
maximize service quality; insertion is delegated to the solver.

- **Observation / features** (`features.py`): request OD + direct drive time,
  current time, fleet utilization (idle vs busy), nearest-vehicle slack, accepted
  load, etc. Keep a small, documented feature vector.
- **Action:** binary (accept / reject) for v0. (Later: which-vehicle / insertion hints.)
- **Reward — a real design task, flag early:** the env's raw per-step reward is
  `1.0` on accept / `0.0` on reject, which naively rewards "accept everything."
  v0 options to evaluate: (a) end-of-episode `service_rate` as the objective
  (episodic REINFORCE/return), (b) a shaped per-step signal via the event bus
  (`event_bus=` on `DVRPEnv`) penalizing infeasible/late service (VMT, wait). Pick
  one, document it, and sanity-check that "accept-all" is *not* trivially optimal.

## Approaches (implement one simple one first; keep both structured)

1. **Baselines (always build these first — they define "beat the baseline"):**
   - `AcceptAll` and `Random` policies (MOSAIC `Policy` subclasses). Establishes
     the reference `service_rate` the learner must beat.
2. **MCTS / rollout policy (recommended v0 — no training infra):** for each
   incoming request, use `copy.deepcopy(env)` to roll out "accept" vs "reject"
   with a base policy a few steps ahead, pick the higher-value branch. Directly
   exercises MOSAIC's deepcopy-for-planning capability; no gradients, easy to reason about.
3. **Simple RL policy:** a small featurized value/Q function (tabular or a
   1–2-layer MLP) trained with episodic REINFORCE or DQN on the accept/reject
   decision. Optional Gymnasium adapter (see below) to use SB3/RLlib later.

Ship **one** end to end (2 or 3) that beats the baseline; structure the code so the
other slots in.

## Gymnasium adapter (only if/when we use SB3/RLlib)

MOSAIC's `DVRPEnv` is not a `gymnasium.Env`: `step` is a 4-tuple `(state, reward,
done, info)`, `reset` returns a bare `State`, and there are no spaces. If we adopt
SB3/RLlib, this repo owns a thin `GymnasiumAdapter`: `reset()→(obs, info)`,
`step()→(obs, reward, terminated, truncated, info)`, define `observation_space`
(the feature vector) + `action_space` (Discrete(2)), and host `State`→features /
action-decode. Keep `gymnasium`/SB3 as **this repo's** deps, never MOSAIC's.

## Proposed repo structure

```
MultiModal-DVRP-RL/
  pyproject.toml            # deps incl. dvrp-gym @ …@v0.1.1-rc.1
  README.md                 # clone → install → run (the prime directive)
  plan.md                   # this
  configs/
    binghampton.yaml         # polygon, depot, fleet, demand, seeds
  src/dvrp_rl/
    __init__.py
    scenario.py             # build the spec dict from config (uniform, small area)
    env.py                  # make_env wrapper; optional GymnasiumAdapter
    features.py             # State → feature vector
    policies/
      __init__.py
      baseline.py           # AcceptAll, Random (Policy subclasses)
      mcts.py               # rollout/MCTS accept-reject policy
      rl.py                 # simple RL policy + trainer
    train.py                # training / rollout driver
    evaluate.py             # run N seeds, report service_rate + metrics; compare policies
  examples/
    run_demo.py             # the one-command entry the README points to
  tests/
    test_scenario.py        # spec builds; depot inside polygon
    test_env.py             # env builds + steps; injected policy is used
    test_policies.py        # each policy returns a valid Action (Trip|None)
    test_smoke.py           # short end-to-end episode + drain + metrics
  .github/workflows/ci.yml  # lint + typecheck + pytest (smoke uses a tiny cached area)
```

## Milestones (each gated by the quality workflow below)

- **M0 — Bootstrap + prove the dependency.** Repo skeleton, `pyproject` pinning
  `v0.1.1-rc.1`, README stub, CI. Smoke test: install MOSAIC, `make_env(spec)`,
  run one short uniform episode with the default policy, assert metrics exist.
  *Exit:* `pip install` from the git tag works; one command runs an episode.
- **M1 — Env + features layer.** `scenario.py`, `env.py` (+ optional Gym adapter),
  `features.py`. Tests for spec/depot/feature shapes.
- **M2 — Baselines + evaluation harness.** `AcceptAll`/`Random`, `evaluate.py`
  reporting `service_rate` across seeds. Establishes the baseline number.
- **M3 — One learned policy beats the baseline.** MCTS-rollout (recommended) or
  simple RL, wired via `make_env(policy=…)`. Show a clear service-rate lift vs M2.
- **M3+ — Deployable learned policy (REINFORCE).** A small policy-gradient
  learner (logistic / 1–2-layer MLP) over `features.py` (geography-free) that
  decides accept/reject from the *current* state only — no env cloning, no
  foresight. Trained on the true demand distribution to amortize what the
  sampled rollout does expensively per-decision. *Exit:* a deployable (no-oracle)
  policy that beats `AcceptAll` on eval seeds, wired via `make_env(policy=…)`.
- **M4 — README walkthrough + reproducibility + demo notebook.** Anyone clones →
  installs → `python examples/run_demo.py` → sees a policy evaluate and beat the
  baseline. Partly done (reproducibility section + configs that reproduce
  RESULTS.md). Remaining steps (do BEFORE M3+):
  1. **README — MOSAIC install via pip+git.** Document *both* auth paths, since
     org membership grants access but git still needs a credential: (a) SSH — the
     `git+ssh://` URL in `pyproject`, requires an SSH key on your (org-member)
     account (`ssh -T git@github.com`); (b) HTTPS+token — for org members who
     don't use SSH. Include the exact `uv`/`pip` commands, `allow-direct-references`,
     same-version `--force-reinstall`, and a private-repo 403/permission troubleshooting note.
  2. **README — usage.** `run_demo` description (OSM first-run/cache, expected
     output); policy usage incl. `oracle=True` vs sampled (`n_samples`, `horizon`,
     `sample_seed`), what each means, and the deterministic-demand guard.
  3. **`notebooks/demo.ipynb`** — markdown + commented cells, sectioned:
     setup/build-env, one demo episode, baselines (AcceptAll/Random via
     `evaluate`), fast MCTS demo (oracle + sampled at tiny horizon/steps — for
     demonstration, not benchmarking), pointer to RESULTS.md + REINFORCE next.
     Add a `[notebook]` extra (jupyter/nbconvert); verify via `nbconvert --execute`.
  4. **Packaging.** Fresh-venv clean-install test (prime directive), OSM-cache
     note for fresh clones/CI, README layout/status → M4, plan.md progress update.
  *Verify:* clean `uv venv` + install + `run_demo` from scratch; notebook executes.
- **M5 (optional) — Second approach / SB3 via the Gym adapter.**

## Progress log

- **M0 ✅** Repo skeleton, `pyproject` pinning `v0.1.1-rc.1`, README, smoke test.
  Proven: install works, `run_demo.py` runs one episode.
- **M1 ✅** `scenario.py`, `env.py` (config→env), `features.py` (4-feature,
  geography-free vector). Gym adapter deferred (see needs.md), not built.
- **M2 ✅** `AcceptAll`/`Random` baselines + `evaluate.py`. Baseline established.
- **M3 ✅ (with an honest caveat)** `RolloutPolicy` — one-step lookahead via
  MOSAIC deepcopy. Verify-fleet (2 testers + 2 reviewers) confirmed the code
  correct. Two modes:
  - *oracle* (perfect future foresight): beats `AcceptAll` +3.1pp (Binghampton),
    +7.2pp congested — the planning ceiling, **not deployable**.
  - *sampled* (default; reseeds demand RNG → plans against sampled, not-real
    futures): the honest, deployable-style planner. Currently *trails* AcceptAll
    (−6.9pp at K=5) because it sacrifices real requests for hypothetical ones;
    the gap shrinks with more samples (−2.7pp at K=20). Motivates M3+.
  See RESULTS.md for numbers. Repo public: github.com/jptalusan/MultiModal-DVRP-RL.
- **M4 ✅** Packaging: README overhaul (MOSAIC pip+git install with *both* SSH and
  HTTPS-token auth paths, `run_demo`/`evaluate` usage, `oracle` vs sampled docs);
  `notebooks/demo.ipynb` (env → episode → baselines → fast rollout, commented) with
  a `[notebook]` extra, executed clean via `nbconvert`; fresh-venv clean install +
  `run_demo` verified from scratch (prime directive holds).
- **M3+ ✅ (honest negative result)** `ReinforcePolicy` (deployable linear-logistic
  policy over the geography-free features, decides from the current state only —
  no foresight/cloning) + `train.py` (REINFORCE, EMA baseline, train/eval seed
  split). Robustly **converges to accept-all and ties the baseline (88.6%); does
  not beat it** — across warm-start/lr/entropy/episode sweeps it either matches
  accept-all or over-rejects. Accept-all is near-optimal here; the beneficial-
  reject signal needs foresight (oracle) or richer *geometric* features. The
  deployable-learner infrastructure is in place and correct.
- **M3++ (proposed next) — geometric features.** Give the learner nearest-vehicle
  slack / insertion detour cost (via the `env._geography` hook in needs.md) so it
  can discriminate "this request blocks others" without foresight. The concrete
  lever to actually beat accept-all.
- **M5 (optional)** — SB3/RLlib via the Gym adapter. Pending.

## Development workflow — the full repertoire (required per milestone)

Mirror MOSAIC's own process. For every milestone:

1. **Plan.** Use the `plan` skill / EnterPlanMode before non-trivial code — state
   assumptions, tradeoffs, and a verify step per item. Update this `plan.md` as
   scope firms up.
2. **Implement in small PRs**, surgical changes, matching structure.
3. **Test.** Unit (scenario/env/features/policy validity) + integration (short
   episode) + a CI smoke test. A change isn't done until tests prove the goal
   (e.g. M3: an automated test asserts learner service_rate ≥ baseline).
4. **Review.** `/code-review` on each diff (correctness + simplification); `/simplify`
   for cleanup-only passes.
5. **Verify-fleet.** Before merging M3+ (the learning milestones), run
   `/verify-fleet` — independent testers (try to break: reward hacking, degenerate
   accept-all, seed leakage, non-determinism) + reviewers (correctness/design), then
   synthesize. High-assurance gate for anything claiming "the policy learns."
6. **Report.** Close each milestone with the succinct `report` format (what changed,
   test result, review verdict, follow-ups, agent count).

## Risks / open questions to resolve early

- **Reward design** — the #1 modeling risk (accept-all must not be trivially
  optimal). Decide the objective in M2/M3.
- **Determinism** — MOSAIC has a flagged seed-reproducibility anomaly for uniform
  demand across fresh runs (possible graph-build nondeterminism). Confirm our
  train/eval seeds are reproducible *within a cached graph*; pin the cache.
- **Binghampton polygon** — finalize the exact boundary; keep it small for fast
  all-pairs.
- **First-run OSM fetch** — needs network once; document it and commit/ship a
  cache strategy so CI + fresh clones aren't slow or network-flaky.
- **rc pin churn** — track MOSAIC rc/release bumps; re-pin deliberately.
- **Action space growth** — v0 is accept/reject; document the path to
  which-vehicle / insertion decisions without rewriting the interface.
