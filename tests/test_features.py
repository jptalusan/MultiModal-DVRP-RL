"""Feature-extraction tests. Offline — builds synthetic States, no network."""

from __future__ import annotations

import numpy as np
import pytest

from dvrp_rl.features import FEATURE_NAMES, N_FEATURES, extract_features

# MOSAIC data models are plain dataclasses — safe to import/build offline.
from dvrp_core.models.core import ManifestStop, NodeLocation, Request, State, Vehicle

DETOUR = 4.0


def _request(direct_time: float, *, received: float = 100.0) -> Request:
    """A request whose time window encodes a given direct drive time."""
    return Request(
        id=0,
        origin=NodeLocation(0),
        destination=NodeLocation(1),
        passengers=1,
        received_time=received,
        earliest_pickup=received,
        latest_dropoff=received + DETOUR * direct_time,
    )


def _vehicle(vid: int, *, manifest_len: int = 0, onboard: int = 0, capacity: int = 4) -> Vehicle:
    manifest = [ManifestStop(location=NodeLocation(0), trip_id=i, stop_type=0) for i in range(manifest_len)]
    return Vehicle(
        id=vid,
        depot_id=0,
        location=NodeLocation(0),
        capacity=capacity,
        manifest=manifest,
        onboard_trips=list(range(onboard)),
    )


def _state(request: Request, vehicles: list[Vehicle], *, current_time: float = 100.0) -> State:
    return State(
        current_time=current_time,
        vehicles=vehicles,
        accepted_trips={},
        passengers={},
        new_request=request,
        transit_routes=[],
        bus_states=[],
    )


def test_shape_dtype_and_names_agree():
    state = _state(_request(300.0), [_vehicle(0)])
    feats = extract_features(state, detour_tolerance=DETOUR)
    assert feats.shape == (N_FEATURES,)
    assert feats.dtype == np.float32
    assert len(FEATURE_NAMES) == N_FEATURES
    assert np.all(np.isfinite(feats))


def test_direct_time_is_recovered_from_the_window():
    # window width = DETOUR * 600 -> recovered direct = 600s -> scaled by /600 -> 1.0
    feats = extract_features(_state(_request(600.0), [_vehicle(0)]), detour_tolerance=DETOUR)
    idx = FEATURE_NAMES.index("req_direct_time")
    assert feats[idx] == pytest.approx(1.0)


def test_idle_fleet_is_all_zeros_for_fleet_features():
    feats = extract_features(_state(_request(300.0), [_vehicle(0), _vehicle(1)]), detour_tolerance=DETOUR)
    for name in ("fleet_busy_frac", "fleet_occupancy", "fleet_mean_manifest"):
        assert feats[FEATURE_NAMES.index(name)] == pytest.approx(0.0)


def test_fleet_features_track_load():
    # 4 vehicles cap 4 (total cap 16); 2 have manifests; 4 trips onboard total.
    vehicles = [
        _vehicle(0, manifest_len=2, onboard=2),
        _vehicle(1, manifest_len=2, onboard=2),
        _vehicle(2),
        _vehicle(3),
    ]
    feats = extract_features(_state(_request(300.0), vehicles), detour_tolerance=DETOUR)
    assert feats[FEATURE_NAMES.index("fleet_busy_frac")] == pytest.approx(0.5)      # 2/4 busy
    assert feats[FEATURE_NAMES.index("fleet_occupancy")] == pytest.approx(4 / 16)   # onboard/capacity
    assert feats[FEATURE_NAMES.index("fleet_mean_manifest")] == pytest.approx((4 / 4) / 4.0)  # mean 1 stop /scale


def test_longer_trip_gives_larger_direct_time_feature():
    idx = FEATURE_NAMES.index("req_direct_time")
    short = extract_features(_state(_request(120.0), [_vehicle(0)]), detour_tolerance=DETOUR)[idx]
    long = extract_features(_state(_request(900.0), [_vehicle(0)]), detour_tolerance=DETOUR)[idx]
    assert long > short
