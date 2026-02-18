# v2/__init__.py

"""
V2 — Activation Intervention
Direct residual stream intervention with trajectory recovery measurement.
"""

from .intervention import run_intervention_experiment
from .trajectory import Trajectory, TrajectoryComparison
from .cache import SeedCache, build_seed_cache
from .diagnostics_bridge import DiagnosticsManager
from .backend import load_model_with_backend
from .sae_adapter import load_sae_decoder_vector, make_sae_intervention
