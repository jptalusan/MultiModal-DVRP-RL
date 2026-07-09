"""Rollout / one-step-lookahead accept-reject planner — no training.

This is *not* a learned policy: it's a Bertsekas-style rollout that improves
the ``AcceptAll`` base policy by one step of lookahead. For each incoming
request it simulates *accepting* vs *rejecting* it, lets ``AcceptAll`` run the
env forward a fixed ``horizon``, and takes whichever branch serves more. By
construction such a rollout is (near-)guaranteed to do no worse than its base,
so beating/tying ``AcceptAll`` is expected, not surprising — the interesting
question is by how much.

ORACLE CAVEAT — this policy is an upper bound, not a deployable dispatcher.
Both branches roll out on ``deepcopy`` clones of the live env, and the copy
includes the *seeded demand RNG*. So each rollout replays the exact future
request stream: the planner decides with **perfect foresight of all future
demand**. That makes the A/B fair (both branches see the identical future),
but it inflates the advantage relative to any real dispatcher, which cannot
see the future. Read its service rate as an oracle/upper-bound number.

Clones share the immutable geography (road graph + travel-time matrix) via a
``deepcopy`` ``memo`` — copying it would dominate cost and blow up on larger
areas. Sharing is safe *only* because ``NetworkGeography`` is read-only during
a step (verified: no mutating methods); do not share it if geography ever gains
mutable state. Reaching ``env._geography`` / ``env._event_log`` uses private
attributes (see needs.md): MOSAIC exposes no public "clone for planning" hook.

``horizon`` note: at low demand a short horizon can't see any cost of
accepting, so ``horizon`` of 0-1 degenerates to plain ``AcceptAll`` (100%
accept) at 2x the compute. Use a horizon long enough to cover a trip's blocking
window (default 30 ≈ 1500 s of sim at request_rate 0.02).
"""

from __future__ import annotations

import copy

from dvrp_core.models.core import State

from dvrp_rl.policies.base import AcceptRejectPolicy
from dvrp_rl.policies.baseline import AcceptAll


class RolloutPolicy(AcceptRejectPolicy):
    """Accept iff a horizon rollout serves at least as many when accepting."""

    def __init__(self, horizon: int = 30):
        super().__init__()
        self.horizon = horizon
        self._env = None

    def bind_env(self, env) -> None:
        """Give the policy the env to plan against (cloned once per decision)."""
        self._env = env

    def _clone(self):
        env = self._env
        geo = env._geography
        # Detach the ever-growing event log before copying: it's irrelevant to
        # the rollout's served-count and copying it would make per-decision cost
        # grow with episode length (O(n^2) over an episode). Restored in finally.
        log = getattr(env, "_event_log", None)
        if log is not None:
            env._event_log = []
        try:
            return copy.deepcopy(env, memo={id(geo): geo})
        finally:
            if log is not None:
                env._event_log = log

    def _served_if(self, state: State, *, accept: bool) -> int:
        """Served-request count after taking ``accept`` now, then AcceptAll for ``horizon`` steps."""
        clone = self._clone()
        base = AcceptAll()
        # The clone already holds trips (ids assigned by us on real accepts).
        # Start the rollout base's id counter past ours so its new trips can't
        # collide with — and corrupt — existing trips in the clone.
        base._next_trip_id = self._next_trip_id
        # AcceptAll.create_trips builds the Trip for the current request; None = reject.
        first_action = base.create_trips(state) if accept else None
        s, _r, done, _i = clone.step(first_action)
        steps = 0
        while not done and steps < self.horizon:
            s, _r, done, _i = clone.step(base.create_trips(s))
            steps += 1
        return clone.metrics.served_requests

    def accept(self, state: State) -> bool:
        if self._env is None:
            raise RuntimeError("RolloutPolicy.bind_env(env) must be called before use")
        # Tie -> accept: serving the current request is free upside when it
        # doesn't reduce what the fleet serves later.
        return self._served_if(state, accept=True) >= self._served_if(state, accept=False)
