# needs.md — gaps in the MOSAIC package we consume

We treat MOSAIC (`dvrp-gym`, pinned at `v0.1.1-rc.1`) as a read-only
dependency: we do **not** modify it. This file records places where the
public API doesn't give us something we'd want, and the workaround we
adopted. It's a wishlist to raise upstream / revisit on the next rc — not
a bug tracker.

## 1. No geometry / geography access from inside a policy

**What we hit (M1).** A policy only receives `State` (via
`Policy.create_trips(state)`). In `State`, every location is a
`NodeLocation(node: int)` — an opaque index into MOSAIC's internal graph
node array — or a `PathLocation`. There is **no lat/lon** on these, and
`State` carries **no geography / travel-time handle**. So a policy cannot,
on its own, compute:
- arbitrary point-to-point drive/walk times or distances,
- nearest-vehicle-to-request distance,
- any geometric feature beyond what's already encoded in the state.

**Workaround we used.** Trip length is still recoverable *without* geometry:
uniform demand sets `latest_dropoff = received_time + detour_tolerance *
max(direct_time, 60)` and `earliest_pickup = received_time`, so
`direct_time ≈ (latest_dropoff − earliest_pickup) / detour_tolerance`.
Our `features.py` uses that. But this only recovers the *new request's*
own length — not fleet geometry (e.g. how far the nearest idle vehicle is).

**What would remove the gap.** Either (a) expose the geography object on
`State` (or pass it to `create_trips`), or (b) add read-only helpers like
`state.drive_time(loc_a, loc_b)`. Until then, geometric fleet features
(nearest-vehicle slack, detour cost of insertion) are out of reach for a
policy and we stick to time-window + fleet-aggregate features.

## 1b. No public "clone for planning" hook (rollout policy)

`RolloutPolicy` plans by `copy.deepcopy`-ing the live env. To do that well it
reaches into private attributes, because MOSAIC exposes no planning API:
- `env._geography` — pinned in the deepcopy `memo` so the (large, read-only)
  road graph + travel-time matrix are *shared*, not copied.
- `env._event_log` — detached before the copy (it grows per step; copying it
  makes per-decision cost O(n) in episode length).
- `env._demand_model._rng` — reseeded on the clone to sample a *different*
  future (non-oracle planning). Without this, the deepcopy carries the seeded
  RNG and the rollout replays the true future (perfect foresight).

**What would remove the gap.** A public `env.clone_for_planning(*, resample_demand=…)`
that shares immutable geography, drops the event log, and optionally reseeds
demand — would let us stop touching private state.

## 2. `DVRPEnv` is not a `gymnasium.Env`

**What it is.** `env.reset()` returns a bare `State` (not `(obs, info)`)
and `env.step(action)` returns a 4-tuple `(state, reward, done, info)`
(not the 5-tuple `(obs, reward, terminated, truncated, info)`), and there
are no `observation_space` / `action_space`. Standard RL libraries
(Stable-Baselines3, RLlib) require the Gymnasium interface.

**Workaround / decision.** Our v0 approaches (MCTS-rollout, hand-rolled
REINFORCE) drive the env directly, so we do **not** need this yet. If we
adopt SB3/RLlib later (plan M5), we write a `GymnasiumAdapter` in *this*
repo (features → `observation_space`, accept/reject → `Discrete(2)`).
Not a MOSAIC change — just noting why the adapter is deferred.

## 3. Transit-scenario papercuts (nice-to-have, not blockers)

Found while planning the fixed-line milestone (plan M6). **Neither blocks
us** — both have free workarounds on our side — but both are easy footguns
for anyone standing up a transit-only run:

- **`start_time` defaults to midnight.** `make_env` uses
  `spec.get("start_time", 0.0)`, but GTFS is time-of-day, so with a transit
  policy and the default start time *every* request is rejected (no service
  at 00:00) with no hint as to why. *Workaround:* pass a daytime
  `start_time` (e.g. `28800` = 08:00). *Upstream nicety:* warn (or raise)
  when transit routes are loaded and `start_time` falls outside the feed's
  service hours.
- **A depot is required even for transit-only runs.** `_validate` demands
  `depots` with ≥1 entry regardless of the policy, though a
  `fixed_line_only` run uses no on-demand vehicles. *Workaround:* pass a
  dummy depot with `num_vehicles: 0` — verified this builds and steps fine.
  *Upstream nicety:* relax the depot requirement (or make it
  policy-dependent) for transit-only policies.
