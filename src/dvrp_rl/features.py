"""Turn a MOSAIC ``State`` into a small, fixed feature vector.

A policy only ever receives ``State`` (see ``Policy.create_trips``). This
module is the single place that maps that state to numbers a learner /
rollout can use. The vector is deliberately small and every entry is
*live* in v0 (varies within a scenario) — no constant or collinear
padding.

What we include, and why each is distinct:

* ``req_direct_time``   — the new request's direct drive time. Not on the
  state directly (locations are opaque graph-node indices with no lat/lon;
  see needs.md), but recoverable from the time window: uniform demand sets
  ``latest_dropoff = received_time + detour_tolerance * max(direct, 60)``
  and ``earliest_pickup = received_time``, so
  ``direct ≈ (latest_dropoff − earliest_pickup) / detour_tolerance``.
* ``fleet_busy_frac``   — fraction of vehicles with a non-empty manifest
  (breadth: how many vehicles are engaged).
* ``fleet_occupancy``   — onboard trips / total capacity (how full the
  fleet is *now*; onboard/manifest are pruned as stops complete).
* ``fleet_mean_manifest`` — mean pending stops per vehicle (depth of
  committed future workload).

Deliberately omitted (considered, not forgotten): request ``passengers``
(constant = 1 in v0 uniform demand); request *slack* ``latest_dropoff −
current_time`` (collinear with ``req_direct_time`` because
``current_time == earliest_pickup`` at decision time); raw ``current_time``
(demand is stationary, so it carries little beyond what fleet load already
encodes). Add them when a scenario makes them informative.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # keep this module free of a runtime MOSAIC import
    from dvrp_core.models.core import State

FEATURE_NAMES: tuple[str, ...] = (
    "req_direct_time",
    "fleet_busy_frac",
    "fleet_occupancy",
    "fleet_mean_manifest",
)
N_FEATURES: int = len(FEATURE_NAMES)

# Rough O(1) scalers so features land in a comparable range for a small
# network. Not learned; just keep magnitudes sane.
_TIME_SCALE = 600.0       # seconds (~10 min typical trip)
_MANIFEST_SCALE = 4.0     # stops


def extract_features(state: State, *, detour_tolerance: float) -> np.ndarray:
    """Map ``state`` (with a pending ``new_request``) to a feature vector.

    Parameters
    ----------
    state
        The state handed to ``create_trips``; ``state.new_request`` must
        be set (a request is being decided).
    detour_tolerance
        The scenario's detour tolerance — needed to recover the request's
        direct drive time from its time window. It's a config constant.

    Returns
    -------
    np.ndarray
        ``float32`` array of shape ``(N_FEATURES,)``, ordered as
        ``FEATURE_NAMES``.
    """
    req = state.new_request
    req_direct_time = (req.latest_dropoff - req.earliest_pickup) / detour_tolerance / _TIME_SCALE

    vehicles = state.vehicles
    n = max(len(vehicles), 1)  # guard: a real fleet always has vehicles
    busy = sum(1 for v in vehicles if v.manifest)
    onboard = sum(len(v.onboard_trips) for v in vehicles)
    total_capacity = max(sum(v.capacity for v in vehicles), 1)
    mean_manifest = sum(len(v.manifest) for v in vehicles) / n

    return np.array(
        [
            req_direct_time,
            busy / n,
            onboard / total_capacity,
            mean_manifest / _MANIFEST_SCALE,
        ],
        dtype=np.float32,
    )
