"""Rollout / one-step-lookahead accept-reject planner — no training.

This is *not* a learned policy: it's a Bertsekas-style rollout that improves
the ``AcceptAll`` base policy by one step of lookahead. For each incoming
request it simulates *accepting* vs *rejecting* it, lets ``AcceptAll`` run the
env forward a fixed ``horizon``, and takes whichever branch serves more. By
construction such a rollout is (near-)guaranteed to do no worse than its base,
so beating/tying ``AcceptAll`` is expected — the interesting question is how much.

Two modes (``oracle`` flag):

* ``oracle=False`` (default, honest Monte-Carlo rollout): each clone's demand
  RNG is **reseeded**, so the rollout plans against *sampled, plausible*
  futures drawn from the same uniform-demand distribution — **not** the real
  future. We average over ``n_samples`` futures and evaluate accept vs reject
  on the *same* sampled futures (common random numbers → a fair paired A/B).
  The sampling is itself seeded (``sample_seed``), so runs are reproducible;
  "deterministic" only in that sense — the planner never sees the true future.
  This is what a real dispatcher could do: plan under demand uncertainty.

* ``oracle=True``: skip reseeding, so the clone keeps the live env's seeded RNG
  and replays the *exact* future — perfect foresight. Not deployable; use it
  only to measure the upper-bound / foresight advantage.

Clones share the immutable geography via a ``deepcopy`` ``memo`` — copying it
would dominate cost and blow up on larger areas. Sharing is safe *only* because
``NetworkGeography`` is read-only during a step (verified: no mutating methods).
Reaching ``env._geography`` / ``env._event_log`` / ``_demand_model._rng`` uses
private attributes (see needs.md): MOSAIC exposes no public planning hook.

``horizon`` note: at low demand a short horizon can't see any cost of accepting,
so ``horizon`` of 0-1 degenerates to plain ``AcceptAll`` (100% accept). Use a
horizon long enough to cover a trip's blocking window (default 30 ≈ 1500 s of
sim at request_rate 0.02).
"""

from __future__ import annotations

import copy
import random

from dvrp_core.models.core import State

from dvrp_rl.policies.base import AcceptRejectPolicy
from dvrp_rl.policies.baseline import AcceptAll


class RolloutPolicy(AcceptRejectPolicy):
    """Accept iff a horizon rollout serves at least as many when accepting."""

    def __init__(self, horizon: int = 30, n_samples: int = 5, oracle: bool = False, sample_seed: int = 0):
        super().__init__()
        self.horizon = horizon
        self.oracle = oracle
        # Oracle uses the one true future (one rollout per branch); sampled mode
        # averages over n_samples drawn futures.
        self.n_samples = 1 if oracle else max(1, n_samples)
        self._sample_rng = random.Random(sample_seed)
        self._env = None

    def bind_env(self, env) -> None:
        """Give the policy the env to plan against (cloned once per decision).

        In sampled (non-oracle) mode this refuses *deterministic* demand: if
        reseeding the demand RNG can't change the future (no ``_rng``, or a
        file/replay model that ignores it), the rollout would silently replay
        the TRUE future — a hidden oracle. We raise rather than mislabel it.
        """
        self._env = env
        if not self.oracle:
            dm = getattr(env, "_demand_model", None)
            name = type(dm).__name__
            if dm is None or not hasattr(dm, "_rng") or "File" in name or "Replay" in name:
                raise RuntimeError(
                    f"sampled rollout (oracle=False) needs stochastic, RNG-driven demand; "
                    f"demand model {name!r} looks deterministic — reseeding would be a no-op "
                    f"and the rollout would silently replay the true future. Use oracle=True, "
                    f"or a stochastic demand (e.g. uniform)."
                )

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

    def _served_if(self, state: State, *, accept: bool, future_seed: int | None) -> int:
        """Served count after taking ``accept`` now, then AcceptAll for ``horizon`` steps.

        ``future_seed`` (sampled mode) reseeds the clone's demand RNG so the
        rollout sees a sampled future rather than the real one; ``None`` (oracle
        mode) keeps the copied RNG, i.e. the true future.
        """
        clone = self._clone()
        if future_seed is not None:
            dm = getattr(clone, "_demand_model", None)
            if dm is not None and hasattr(dm, "_rng"):
                dm._rng = random.Random(future_seed)  # sample a plausible future, not the real one
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

    def _value(self, state: State, *, accept: bool, seeds: list[int | None]) -> float:
        return sum(self._served_if(state, accept=accept, future_seed=s) for s in seeds) / len(seeds)

    def accept(self, state: State) -> bool:
        if self._env is None:
            raise RuntimeError("RolloutPolicy.bind_env(env) must be called before use")
        # Common random numbers: accept and reject are scored on the SAME sampled
        # futures, so the comparison isolates the effect of the action. In oracle
        # mode the single "seed" is None (the real future).
        seeds: list[int | None]
        seeds = [None] if self.oracle else [self._sample_rng.randrange(2**31) for _ in range(self.n_samples)]
        # Tie -> accept: serving the current request is free upside when it
        # doesn't reduce what the fleet serves later.
        return self._value(state, accept=True, seeds=seeds) >= self._value(state, accept=False, seeds=seeds)
