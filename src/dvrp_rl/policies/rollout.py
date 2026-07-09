"""Rollout ("MCTS"-style) accept/reject policy — no training required.

For each incoming request we do a one-step lookahead: simulate *accepting*
vs *rejecting* it, then let a base policy (AcceptAll) run the env forward a
fixed horizon, and take whichever branch ends with more served requests.
Both branches roll out on independent ``deepcopy`` clones of the live env,
so they see the *same* future demand (the demand RNG is copied with the
env) — a fair A/B on this one decision.

Clones share the immutable geography (road graph + travel-time matrix) via
a ``deepcopy`` ``memo``; copying it would dominate cost and blow up on
larger areas. This reaches ``env._geography`` (a private attribute — see
needs.md): MOSAIC exposes no public "clone for planning" hook, and reading
it here is the documented workaround.
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
        geo = self._env._geography
        return copy.deepcopy(self._env, memo={id(geo): geo})

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
