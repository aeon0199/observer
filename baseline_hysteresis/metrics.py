# v1/metrics.py

"""
Stability Metrics
-----------------
Computes drift, recovery, hysteresis from telemetry frames.
"""

import numpy as np
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class COTMetrics:
    """Computed stability metrics from a run."""
    drift: float
    hysteresis: float
    recovery: float
    
    drift_hidden: float
    drift_entropy: float
    drift_logit: float
    drift_svd: float
    
    hysteresis_hidden: float
    hysteresis_entropy: float
    hysteresis_logit: float
    hysteresis_svd: float
    
    regime: str
    baseline_stable: bool
    cache_verified: bool


def entropy_distance(h1: float, h2: float) -> float:
    """Absolute difference in entropy."""
    return abs(h1 - h2)


def norm_distance(n1: float, n2: float) -> float:
    """Relative difference in norms."""
    if n1 < 1e-10 and n2 < 1e-10:
        return 0.0
    return abs(n1 - n2) / (max(n1, n2) + 1e-10)


def svd_distance(svd1: Dict, svd2: Dict) -> float:
    """Distance between SVD spectral signatures."""
    sv1 = svd1.get("singular_values", [])
    sv2 = svd2.get("singular_values", [])
    
    if not sv1 or not sv2:
        return 0.0
    
    sv1 = np.array(sv1)
    sv2 = np.array(sv2)
    
    min_len = min(len(sv1), len(sv2))
    sv1 = sv1[:min_len]
    sv2 = sv2[:min_len]
    
    norm = max(np.linalg.norm(sv1), np.linalg.norm(sv2))
    if norm < 1e-10:
        return 0.0
    
    return np.linalg.norm(sv1 - sv2) / norm


def compute_composite_distance(stats1: Dict, stats2: Dict) -> Dict[str, float]:
    """Compute all component distances."""
    d_hidden = norm_distance(
        stats1.get("hidden_norm", 0),
        stats2.get("hidden_norm", 0)
    )
    
    d_entropy = entropy_distance(
        stats1.get("entropy", 0),
        stats2.get("entropy", 0)
    )
    
    d_logit = norm_distance(
        stats1.get("logit_norm", 0),
        stats2.get("logit_norm", 0)
    )
    
    d_svd = svd_distance(
        stats1.get("svd", {}),
        stats2.get("svd", {})
    )
    
    composite = (
        1.0 * d_hidden +
        1.0 * d_entropy +
        0.5 * d_logit +
        1.0 * d_svd
    )
    
    return {
        "hidden": d_hidden,
        "entropy": d_entropy,
        "logit": d_logit,
        "svd": d_svd,
        "composite": composite
    }


def classify_regime(recovery: float) -> str:
    """Classify stability regime."""
    if recovery > 0.8:
        return "elastic"
    elif recovery > 0.4:
        return "partial"
    elif recovery >= 0:
        return "plastic"
    else:
        return "runaway"


def compute_cot_metrics(
    before_stats: Dict[str, Any],
    after_stats: Dict[str, Any],
    revert_stats: Dict[str, Any],
    cache_verified: bool = True
) -> COTMetrics:
    """Compute full metrics from three-stage telemetry."""
    drift_components = compute_composite_distance(before_stats, after_stats)
    hysteresis_components = compute_composite_distance(before_stats, revert_stats)
    
    D = drift_components["composite"]
    H = hysteresis_components["composite"]
    
    eps = 1e-8
    R = 1.0 - (H / (D + eps))
    
    regime = classify_regime(R)
    baseline_stable = H < 0.01
    
    return COTMetrics(
        drift=D,
        hysteresis=H,
        recovery=R,
        
        drift_hidden=drift_components["hidden"],
        drift_entropy=drift_components["entropy"],
        drift_logit=drift_components["logit"],
        drift_svd=drift_components["svd"],
        
        hysteresis_hidden=hysteresis_components["hidden"],
        hysteresis_entropy=hysteresis_components["entropy"],
        hysteresis_logit=hysteresis_components["logit"],
        hysteresis_svd=hysteresis_components["svd"],
        
        regime=regime,
        baseline_stable=baseline_stable,
        cache_verified=cache_verified
    )


def format_metrics_summary(metrics: COTMetrics) -> str:
    """Format metrics for display."""
    lines = [
        "",
        "═" * 60,
        "  STABILITY METRICS",
        "═" * 60,
        "",
        f"  Drift (D):      {metrics.drift:.6f}",
        f"  Hysteresis (H): {metrics.hysteresis:.6f}",
        f"  Recovery (R):   {metrics.recovery:.4f}",
        f"  Regime:         {metrics.regime.upper()}",
        "",
        "  Component Breakdown:",
        f"    Hidden norm:  drift={metrics.drift_hidden:.4f}  hyst={metrics.hysteresis_hidden:.4f}",
        f"    Entropy:      drift={metrics.drift_entropy:.4f}  hyst={metrics.hysteresis_entropy:.4f}",
        f"    Logit norm:   drift={metrics.drift_logit:.4f}  hyst={metrics.hysteresis_logit:.4f}",
        f"    SVD spectrum: drift={metrics.drift_svd:.4f}  hyst={metrics.hysteresis_svd:.4f}",
        "",
        f"  Cache verified: {metrics.cache_verified}",
        "═" * 60,
    ]
    return "\n".join(lines)