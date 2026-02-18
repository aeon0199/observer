# v1/__init__.py

"""
V1 — Branching Diagnostic
Non-invasive inference-time stability measurement using KV-cache branching.
"""

from .runner import run_branching_observer_loop
from .metrics import compute_cot_metrics, COTMetrics
from .cache import build_seed_cache, SeedCache