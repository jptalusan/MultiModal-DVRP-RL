"""Turn a YAML config into a MOSAIC scenario ``spec`` dict.

MOSAIC's ``make_env`` (see ``dvrp_core.env.library``) takes a plain
JSON-style dict — the dict *is* the config. This module is the single
place that builds that dict from our config file, so the polygon, depot,
fleet, demand and seeds live in ``configs/*.yaml``, never hardcoded.

Required spec keys MOSAIC validates: ``polygon`` (GeoJSON Polygon),
``depots``, ``request_rate``, ``detour_tolerance``. Everything else is a
tuning knob with a default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def bbox_to_polygon(*, south: float, west: float, north: float, east: float) -> dict[str, Any]:
    """Build a GeoJSON Polygon (a closed rectangle) from a bbox.

    GeoJSON rings are ``[lon, lat]`` and must close (first point repeated).
    """
    return {
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
        ]],
    }


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a scenario config YAML into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_spec(config: dict[str, Any]) -> dict[str, Any]:
    """Build a MOSAIC ``make_env`` spec dict from a loaded config.

    The config's ``area`` provides either a ready ``polygon`` (GeoJSON)
    or a ``bbox`` (``south/west/north/east``) we convert. Depots, demand
    and the tuning knobs pass through.
    """
    area = config["area"]
    if "polygon" in area:
        polygon = area["polygon"]
    elif "bbox" in area:
        polygon = bbox_to_polygon(**area["bbox"])
    else:
        raise ValueError("config['area'] must provide either 'polygon' or 'bbox'")

    spec: dict[str, Any] = {
        "polygon": polygon,
        "depots": config["depots"],
        "request_rate": config["request_rate"],
        "detour_tolerance": config["detour_tolerance"],
        "max_walk_time": config.get("max_walk_time", 600.0),
        "max_wait_time": config.get("max_wait_time", 1800.0),
        "random_seed": config.get("random_seed", 42),
    }
    return spec
