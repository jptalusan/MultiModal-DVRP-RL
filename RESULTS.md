# Results

Service-rate results per policy, appended as we add them. "Service rate" is
MOSAIC's `env.metrics.service_rate` = served requests / total requests
(policy-rejected requests count as unserved).

**Reproduce:** `python -m dvrp_rl.evaluate [config.yaml]` (uses the config's
`eval` seeds; the sampled rollout takes a few minutes).

## The two rollout modes

`Rollout` is a one-step-lookahead (Bertsekas) planner over the `AcceptAll` base
— **not** a learned policy. It has two modes:

- **oracle** (`oracle=True`) — the clone keeps the env's seeded demand RNG, so it
  plans against the *true* future: **perfect foresight**. An upper bound, not
  deployable. Use it to measure the value of planning if demand were known.
- **sampled** (default) — the clone's demand RNG is *reseeded*, so it plans
  against *sampled, plausible* futures from the same distribution — never the
  real one. Averaged over `K` futures with common random numbers. Reproducible
  for a fixed `sample_seed` (seeded), but not an oracle. This is the honest,
  deployable-style planner.

## Scenario: Binghampton (`configs/binghampton.yaml`)

- Area: ~2 km Memphis box (approximate — see plan.md). Depot: 1, fleet 3 × cap 4.
- Demand: `uniform`, `request_rate=0.02`. Eval seeds `[100…104]` (5). 300 steps/episode.
- Environment: MOSAIC `dvrp-gym @ v0.1.1-rc.1`, solver `greedy`.

| Policy | Service rate (mean ± sample std) | vs AcceptAll | Notes |
|---|---|---|---|
| `AcceptAll` | 88.6% ± 2.3% | — | Baseline; equals MOSAIC's stock `on_demand_only`. |
| `Random(0.5)` | 49.5% ± 0.8% | −39pp | Accept each request w.p. 0.5. |
| `Rollout` **oracle** (H=30) | **91.7% ± 2.0%** | **+3.1pp, wins 5/5** | Perfect-foresight upper bound. |
| `Rollout` **sampled** (H=30, K=5) | 81.7% ± 4.0% | −6.9pp, wins 0/5 | Honest planner; see below. |
| RL (REINFORCE) | _pending_ | — | Deployable learner — next (plan.md). |

**Reading this:** planning *with* foresight beats the baseline (+3.1pp) — that's
the ceiling. The honest **sampled** planner currently does *worse* than
AcceptAll: with only a few sampled futures it sacrifices a real, servable request
to keep capacity free for *hypothetical* future requests that don't materialize.
This is mostly variance, and shrinks as `K` grows:

| Sampled rollout | Δ vs AcceptAll (3 seeds, 150 steps) |
|---|---|
| K=5 | −6.2pp |
| K=20 | −2.7pp |

Closing the gap needs many (expensive) samples or variance reduction — which is
exactly the case for a **learned** policy (REINFORCE) that amortizes the planning
into a fast, foresight-free decision. That's the next milestone.

### Congestion check (`configs/binghampton_congested.yaml`, `request_rate=0.06`, 3×)

Accept-all is **not** trivially optimal — with a saturated fleet the oracle's
ceiling widens (same 5 seeds, 300 steps):

| Policy | Service rate | vs AcceptAll |
|---|---|---|
| `AcceptAll` | 42.9% ± 2.6% | — |
| `Rollout` **oracle** (H=30) | **50.1% ± 2.8%** | **+7.2pp, wins 5/5** |
