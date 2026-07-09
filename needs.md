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
