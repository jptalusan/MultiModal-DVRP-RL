# Results

Service-rate results per policy, appended as we add them. "Service rate" is
MOSAIC's `env.metrics.service_rate` = served requests / total requests
(policy-rejected requests count as unserved).

**Reproduce:** `python -m dvrp_rl.evaluate [config.yaml]`

## Scenario: Binghampton (`configs/binghampton.yaml`)

- Area: ~2 km Memphis box (approximate — see plan.md TODO). Depot: 1, fleet 3 × cap 4.
- Demand: `uniform`, `request_rate=0.02`. Eval seeds: `[100, 101, 102]`. Horizon: 300 steps/episode.
- Environment: MOSAIC `dvrp-gym @ v0.1.1-rc.1`, solver `greedy`.

| Policy | Service rate (mean ± std) | Milestone | Notes |
|---|---|---|---|
| `AcceptAll` | 90.2% ± 0.8% | M2 | Accept every request; equals MOSAIC's stock `on_demand_only`. The baseline to beat. |
| `Random(0.5)` | 49.4% ± 0.8% | M2 | Accept each request w.p. 0.5 (seeded per episode). |
| `Rollout(H=30)` | **92.9% ± 0.8%** | M3 | 1-step lookahead: sim accept vs reject 30 steps ahead (AcceptAll base), pick more-served branch. **Beats AcceptAll on every seed (+2.7pp).** ~12.5 s/episode (~38× AcceptAll). |
| RL (REINFORCE) | _pending_ | M3+ | To follow if rollout's per-decision cost is prohibitive. |

> The gap between `AcceptAll` (90%) and `Random` (49%) shows accepting is
> usually right here — a learned policy wins only by *selectively* rejecting
> requests that would block serving others. `Rollout` does exactly that.

### Congestion check (`request_rate=0.06`, 3× demand)

Accept-all is **not** trivially optimal — when the fleet saturates, the
rollout's edge grows:

| Policy | Service rate | Notes |
|---|---|---|
| `AcceptAll` | 44.1% ± 2.2% | greedy acceptance blocks the fleet |
| `Rollout(H=30)` | **50.9% ± 2.9%** | **+6.8pp** by rejecting requests that would saturate vehicles |
