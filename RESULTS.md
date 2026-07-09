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
| `AcceptAll` | **90.2% ± 0.8%** | M2 | Accept every request; equals MOSAIC's stock `on_demand_only`. The baseline to beat. |
| `Random(0.5)` | 49.4% ± 0.8% | M2 | Accept each request w.p. 0.5 (seeded per episode). |
| MCTS-rollout | _pending_ | M3 | — |
| RL (REINFORCE) | _pending_ | M3 | — |

> The gap between `AcceptAll` (90%) and `Random` (49%) shows accepting is
> usually right here — a learned policy wins only by *selectively* rejecting
> requests that would block serving others.
