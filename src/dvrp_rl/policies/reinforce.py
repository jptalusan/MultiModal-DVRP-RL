"""A deployable learned accept/reject policy trained with REINFORCE.

Unlike ``RolloutPolicy``, this policy decides from the *current state's feature
vector alone* — no env cloning, no future foresight. It's a linear-logistic
policy over ``features.extract_features``:

    p_accept = sigmoid(w · x + b)

At evaluation it's greedy (accept iff ``p_accept >= 0.5``); during training a
sampling RNG is set, so it samples ``accept ~ Bernoulli(p)`` and records the
per-step ``(features, action, p)`` trajectory for the policy-gradient update
(see ``dvrp_rl.train``). A linear model is deliberate: 4 features and a
near-linear "reject long trips when the fleet is busy" boundary don't need an
MLP (and it keeps us numpy-only). An MLP is a later option if this underfits.
"""

from __future__ import annotations

import numpy as np

from dvrp_core.models.core import State

from dvrp_rl.features import N_FEATURES, extract_features
from dvrp_rl.policies.base import AcceptRejectPolicy


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))  # clip to avoid exp overflow


class ReinforcePolicy(AcceptRejectPolicy):
    """Linear-logistic accept/reject policy. Greedy unless a sampling RNG is set."""

    def __init__(
        self,
        detour_tolerance: float,
        *,
        weights: np.ndarray | None = None,
        bias: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        super().__init__()
        self.detour_tolerance = float(detour_tolerance)
        self.w = np.zeros(N_FEATURES, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
        self.b = float(bias)
        # rng set -> training mode (sample + record); None -> greedy eval.
        self._rng = rng
        self.trajectory: list[tuple[np.ndarray, int, float]] = []

    def prob_accept(self, state: State) -> tuple[np.ndarray, float]:
        """Return ``(features, p_accept)`` for the current request."""
        x = extract_features(state, detour_tolerance=self.detour_tolerance)
        p = float(_sigmoid(float(self.w @ x + self.b)))
        return x, p

    def accept(self, state: State) -> bool:
        x, p = self.prob_accept(state)
        if self._rng is None:
            return bool(p >= 0.5)  # greedy (deployable) decision
        a = self._rng.random() < p  # stochastic (training) decision
        self.trajectory.append((x, int(a), p))
        return bool(a)
