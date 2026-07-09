# Results

Service-rate results per policy, appended as we add them. "Service rate" is
MOSAIC's `env.metrics.service_rate` = served requests / total requests
(policy-rejected requests count as unserved).

**Reproduce:** `python -m dvrp_rl.evaluate [config.yaml]` (uses the config's
`eval` seeds). Numbers below are from the committed configs, unchanged.

> **Oracle caveat for `Rollout`.** The rollout plans by `deepcopy`-ing the env,
> which copies the *seeded demand RNG* — so it decides each request with
> **perfect foresight of all future demand**. This makes its accept-vs-reject
> A/B fair (both branches replay the identical future), but it is an
> **upper-bound / oracle planner, not a deployable dispatcher** (a real system
> can't see the future). Read `Rollout`'s rate as a ceiling, not a shippable
> number. It is also not a *learned* policy — it's a one-step-lookahead
> (Bertsekas) rollout of the `AcceptAll` base, so beating/tying `AcceptAll` is
> expected by construction; the question is the size of the lift.

## Scenario: Binghampton (`configs/binghampton.yaml`)

- Area: ~2 km Memphis box (approximate — see plan.md TODO). Depot: 1, fleet 3 × cap 4.
- Demand: `uniform`, `request_rate=0.02`. Eval seeds: `[100…109]` (10). 300 steps/episode.
- Environment: MOSAIC `dvrp-gym @ v0.1.1-rc.1`, solver `greedy`.

| Policy | Service rate (mean ± sample std) | Milestone | Notes |
|---|---|---|---|
| `AcceptAll` | 88.1% ± 2.3% | M2 | Accept every request; equals MOSAIC's stock `on_demand_only`. The baseline to beat. |
| `Random(0.5)` | 48.8% ± 2.8% | M2 | Accept each request w.p. 0.5 (seeded per episode). |
| `Rollout(H=30)` | **91.6% ± 1.5%** | M3 | 1-step lookahead: sim accept vs reject 30 steps ahead (AcceptAll base), pick more-served branch. Oracle planner (see caveat). ~11 s/episode (~50× AcceptAll). |
| RL (REINFORCE) | _pending_ | M3+ | To follow — a *deployable* (no-foresight) learner. |

**Rollout vs AcceptAll is a paired comparison** (identical per-seed demand):
`Rollout` wins **10/10 seeds**, mean paired delta **+3.5pp** (per-seed deltas
range +0.7…+6.0pp). 10/10 same-direction wins is a sign-test p≈0.001, so the
lift is real, if modest — at low congestion `AcceptAll` is already near-optimal,
and the rollout only helps on the rare request whose acceptance blocks others.

### Congestion check (`configs/binghampton_congested.yaml`, `request_rate=0.06`, 3× demand)

Accept-all is **not** trivially optimal — when the fleet saturates the rollout's
edge grows. Same 10 seeds, 300 steps:

| Policy | Service rate | Notes |
|---|---|---|
| `AcceptAll` | 42.0% ± 2.9% | greedy acceptance blocks the fleet |
| `Rollout(H=30)` | **48.8% ± 2.5%** | **+6.8pp** mean paired delta, wins **10/10 seeds**, by rejecting requests that would saturate vehicles |
