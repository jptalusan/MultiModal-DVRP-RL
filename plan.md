# MultiModal-DVRP-RL — plan

A standalone repo that installs the **MOSAIC** simulator as a pip dependency and
learns a dispatch **policy** against it. MOSAIC is the *environment*; this repo
owns only the learning code. We never modify MOSAIC (gaps → `needs.md`).

> **Prime directive:** anyone can `git clone` this repo, follow the README, and
> run a policy against the MOSAIC engine — on a small area with uniform demand —
> in one command, without touching the MOSAIC source or its backend.

**Status:** M0–M4 + M3+ done (see Progress log). On-demand accept/reject is
built, benchmarked, and packaged. Next candidate: **M3++** (geometric features, to
actually beat accept-all). Fixed-line transit is deferred and tracked in
[issue #1](https://github.com/jptalusan/MultiModal-DVRP-RL/issues/1).

## Goal / non-goals

**Goal (v0):** evaluate policies for the **`on_demand_only`** setting and show,
reproducibly, how each compares to a naive baseline on service rate.

**Non-goals (v0):** transit/fixed-line (deferred → issue #1), real MOVE-OD demand,
the MOSAIC API/DB/frontend, large cities, distributed training, SOTA algorithms.
Keep it small and legible.

## Dependency on MOSAIC (verified facts)

- Pinned git dependency, base install (no `[api]` backend/DB extra):
  `dvrp-gym @ git+ssh://git@github.com/smarttransit-ai/MOSAIC.git@v0.1.1`.
  Hatchling needs `[tool.hatch.metadata] allow-direct-references = true`.
- **Pin the exact release tag**, never a moving branch. Same-version re-pull needs
  `pip install --force-reinstall`.
- Entry points we rely on:
  - `make_env(spec, *, solver, policy, demand, seed, cache_dir) -> (env, policy)`
    from `dvrp_core.env.library` — `policy` may be a **`Policy` instance** (our
    injection hook) or a registered name.
  - `Policy.create_trips(state) -> Action`; `env.reset()`, `env.step(action)`
    → 4-tuple, `env.drain()`, `env.close()`, `env.metrics.service_rate`.
- `served_requests` increments **at dispatch** (not dropoff), so a rollout needs
  no `drain()`. Requests are generated **lazily inside `step`**.
- First run fetches the OSM graph via `osmnx` into `cache/` (git-ignored);
  travel times are computed lazily, so area size affects fetch, not an upfront
  all-pairs cost. Keep areas small anyway.

## Scenarios (configs/)

| Config | Area | Demand | Purpose |
|---|---|---|---|
| `binghampton.yaml` | ~2 km Memphis box | `uniform`, rate 0.02 | default; the headline numbers |
| `binghampton_congested.yaml` | same | `uniform`, rate 0.06 (3×) | saturated fleet — where accept-all is weak |
| `nashville.yaml` | MOSAIC quickstart box | `uniform` | known-good fallback |

Each carries a depot + fleet and disjoint `seeds.train` / `seeds.eval`.
*Open:* the Binghampton polygon is still the plan's approximate bbox, not a
verified neighborhood boundary.

## The learning problem (on_demand_only)

Per step MOSAIC hands the policy `state.new_request` plus the fleet. The action
is **a `Trip` (accept — the greedy solver then inserts it) or `None` (reject)**;
insertion and vehicle choice are the *solver's* job, not the policy's. An
accepted trip can still be rejected by the solver if no feasible insertion
exists — which is why accept-all is ~88.6%, not 100%.

- **Features** (`features.py`): 4 geography-free values — recovered direct trip
  time (from the request's time window) + fleet busy-fraction / occupancy /
  mean-manifest. Locations in `State` are opaque graph-node indices with no
  lat/lon and no geography handle (see `needs.md`), so geometric features are
  currently out of reach for a policy.
- **Reward (resolved):** end-of-episode `service_rate` as the return. Verified
  that accept-all is **not** trivially optimal (88.6% uncongested; 42.9% under
  3× demand), so there is real headroom to reject well.

## Milestones

**Done:** M0 bootstrap · M1 env + features · M2 baselines + eval harness ·
M3 rollout planner · M3+ REINFORCE · M4 packaging. Details in the Progress log.

**M3++ (candidate) — geometric features.** Give the learner nearest-vehicle
slack / insertion detour cost via the `env._geography` hook (`needs.md`) so it
can tell "this request blocks others" *without* foresight. Bounded upside: a
deployable policy can at best approach the K→∞ sampled-rollout limit, which is
≥ accept-all but strictly below the oracle — so expect a marginal win
uncongested and a plausible real win under congestion. *Exit:* a deployable
policy that beats accept-all on held-out seeds, at least in the congested config.

**M5 (optional) — SB3/RLlib via a Gymnasium adapter.** Deferred: `DVRPEnv` is
not a `gymnasium.Env` (bare-`State` reset, 4-tuple step, no spaces). If we adopt
SB3 we own a thin adapter in *this* repo. Not built — no consumer yet.

## M6 — Fixed-line transit (deferred → issue #1)

Running MOSAIC's stock `fixed_line_only` policy as its own scenario (buses +
walking legs, no on-demand vehicles) is **deferred**. The full investigation —
what MOSAIC provides for free, the ~40 lines we'd add, the three gotchas (none
needing a MOSAIC change), honest expectations, phases, and open decisions — lives
in [issue #1](https://github.com/jptalusan/MultiModal-DVRP-RL/issues/1).

Key points to remember: it needs a **GTFS feed** and a polygon that actually
contains bus routes; the sim clock starts at **midnight**, so `start_time` must be
set to a daytime value or everything is rejected; and results are **not comparable**
to the on-demand baseline (uniform random demand vs fixed routes). Our accept/reject
learning layer does **not** transfer — transit actions and features differ.

## Progress log

- **M0 ✅** Repo skeleton, `pyproject` pinning `v0.1.1-rc.1`, README, smoke test.
  Install works; `run_demo.py` runs one episode.
- **M1 ✅** `scenario.py`, `env.py` (config→env), `features.py` (4-feature,
  geography-free vector). Gym adapter deferred (see `needs.md`), not built.
- **M2 ✅** `AcceptAll`/`Random` baselines + `evaluate.py`. Baseline established.
- **M3 ✅ (with an honest caveat)** `RolloutPolicy` — one-step lookahead via
  MOSAIC deepcopy. Verify-fleet (2 testers + 2 reviewers) confirmed the code
  correct. Two modes:
  - *oracle* (perfect future foresight): beats `AcceptAll` +3.1pp (Binghampton),
    +7.2pp congested — the planning ceiling, **not deployable**.
  - *sampled* (default; reseeds the demand RNG → plans against sampled, not-real
    futures): the honest, deployable-style planner. *Trails* AcceptAll (−6.9pp at
    K=5); the gap shrinks with more samples (−2.7pp at K=20). A second fleet
    (tester + reviewer) confirmed it is genuinely non-oracle and reproducible,
    and flagged the deterministic-demand trap now guarded in `bind_env`.
- **M4 ✅** Packaging: README overhaul (MOSAIC pip+git install with *both* SSH and
  HTTPS-token auth paths, usage docs, `oracle` vs sampled); `notebooks/demo.ipynb`
  (env → episode → baselines → fast rollout) with a `[notebook]` extra, executed
  clean via `nbconvert`; fresh-venv clean install + `run_demo` verified from
  scratch (prime directive holds).
- **M3+ ✅ (honest negative result)** `ReinforcePolicy` (deployable
  linear-logistic policy over the geography-free features — current state only,
  no foresight/cloning) + `train.py` (REINFORCE, EMA baseline, train/eval seed
  split). Robustly **converges to accept-all and ties the baseline (88.6%); does
  not beat it** — across warm-start/lr/entropy/episode sweeps it either matches
  accept-all or over-rejects. Accept-all is near-optimal here; the
  beneficial-reject signal needs foresight or richer *geometric* features. The
  deployable-learner infrastructure is in place and correct.
- Repo public: github.com/jptalusan/MultiModal-DVRP-RL.
- **Next:** M3++ (geometric features) and/or M6 (fixed-line). M5 optional.

## Development workflow (per milestone)

1. **Plan** before non-trivial code — assumptions, tradeoffs, a verify step per
   item. Keep this file current.
2. **Implement** in small, surgical changes matching existing structure.
3. **Test.** Offline unit tests + a network integration test. A change isn't done
   until a test proves the goal.
4. **Review.** `/code-review` per diff; `/simplify` for cleanup-only passes.
5. **Verify-fleet** for anything claiming "the policy learns/wins" — independent
   testers (reward hacking, degenerate accept-all, seed leakage, non-determinism)
   + reviewers, then synthesize.
6. **Report** in the succinct `report` format; append benchmark rows to
   `RESULTS.md`.

## Risks / open questions

- **Binghampton polygon** — still the approximate bbox; finalize against the real
  neighborhood (keep it small).
- **CI can't install MOSAIC** — private repo needs a deploy key / token, so
  network-marked tests are excluded from CI. Deferred deliberately.
- **First-run OSM fetch** — needs network once; `cache/` is git-ignored, so fresh
  clones and CI pay for it. Consider a shipped cache if CI ever runs them.
- **rc pin churn** — track MOSAIC rc/release bumps; re-pin deliberately.
- **Beating accept-all without foresight** — open. Bounded by the K→∞ rollout
  limit; geometric features (M3++) are the concrete lever, congestion the regime
  where it should matter.
- **Action space growth** — accept/reject today; which-vehicle / insertion hints
  (and transit actions, M6) would need interface work.
