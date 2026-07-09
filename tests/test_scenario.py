"""Scenario/spec-building tests. No network — safe for CI."""

from __future__ import annotations

from pathlib import Path

import pytest

from dvrp_rl.scenario import bbox_to_polygon, build_spec, load_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def test_bbox_to_polygon_is_a_closed_geojson_ring():
    poly = bbox_to_polygon(south=35.14, west=-89.99, north=35.16, east=-89.95)
    assert poly["type"] == "Polygon"
    ring = poly["coordinates"][0]
    assert ring[0] == ring[-1], "GeoJSON ring must close"
    assert len(ring) == 5
    # coordinates are [lon, lat]
    assert all(-90 < lon < -86 and 35 < lat < 37 for lon, lat in ring)


@pytest.mark.parametrize("config_path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_spec_builds_with_required_keys(config_path):
    spec = build_spec(load_config(config_path))
    for key in ("polygon", "depots", "request_rate", "detour_tolerance"):
        assert key in spec
    assert spec["polygon"]["type"] == "Polygon"
    assert isinstance(spec["depots"], list) and spec["depots"]


@pytest.mark.parametrize("config_path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_depot_is_inside_the_polygon(config_path):
    """A depot outside the polygon silently snaps to the nearest node — guard it."""
    from shapely.geometry import Point, shape

    spec = build_spec(load_config(config_path))
    poly = shape(spec["polygon"])
    for depot in spec["depots"]:
        assert poly.contains(Point(depot["lon"], depot["lat"])), (
            f"depot {depot['name']!r} is outside the service polygon"
        )
