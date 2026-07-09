"""Baseline accept/reject policies — the reference a learner must beat."""

from __future__ import annotations

import random

from dvrp_core.models.core import State

from dvrp_rl.policies.base import AcceptRejectPolicy


class AcceptAll(AcceptRejectPolicy):
    """Accept every request. Equivalent to MOSAIC's stock ``on_demand_only``;
    defines the service rate a smarter policy must beat by rejecting well."""

    def accept(self, state: State) -> bool:
        return True


class RandomPolicy(AcceptRejectPolicy):
    """Accept each request with a fixed probability. Seeded for reproducibility."""

    def __init__(self, accept_prob: float = 0.5, seed: int = 0):
        super().__init__()
        self.accept_prob = accept_prob
        self._rng = random.Random(seed)

    def accept(self, state: State) -> bool:
        return self._rng.random() < self.accept_prob
