"""multimodal-dvrp-rl: learn a dispatch policy against the MOSAIC simulator."""

from dvrp_rl.env import make_env_from_config
from dvrp_rl.features import FEATURE_NAMES, N_FEATURES, extract_features
from dvrp_rl.policies import AcceptAll, AcceptRejectPolicy, RandomPolicy
from dvrp_rl.scenario import bbox_to_polygon, build_spec, load_config

# NB: dvrp_rl.evaluate is a runnable driver (python -m dvrp_rl.evaluate) — kept
# out of this eager import so `-m` doesn't re-execute an already-imported module.
__all__ = [
    "FEATURE_NAMES",
    "N_FEATURES",
    "AcceptAll",
    "AcceptRejectPolicy",
    "RandomPolicy",
    "bbox_to_polygon",
    "build_spec",
    "extract_features",
    "load_config",
    "make_env_from_config",
]
