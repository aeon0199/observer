from __future__ import annotations

from typing import Dict, Optional

import torch

from runtime_lab.config.schemas import DiagnosticsConfig
from .layer_probe import LayerProbe, LayerProbeConfig
from .predictor import DivergencePredictor
from .spectral import SpectralTrajectoryProbe, SpectralTrajectoryConfig
from .windowed_svd import WindowedSVDProbe, WindowedSVDProbeConfig


def summarize_diagnostics_health(records: list[Dict]) -> Dict:
    total = 0
    degraded = 0
    invalid = 0
    issue_counts: Dict[str, int] = {}

    for diagnostics in records:
        if not diagnostics:
            continue
        total += 1
        health = diagnostics.get("health", {}) or {}
        if health.get("degraded"):
            degraded += 1
        if health.get("valid") is False:
            invalid += 1
        for item in health.get("issues", []) or []:
            issue = str(item.get("issue", "unknown"))
            issue_counts[issue] = int(issue_counts.get(issue, 0) + 1)

    return {
        "steps_observed": int(total),
        "degraded_steps": int(degraded),
        "invalid_steps": int(invalid),
        "ok": bool(invalid == 0),
        "issue_counts": issue_counts,
    }


class DiagnosticsManager:
    def __init__(self, config: Optional[DiagnosticsConfig] = None):
        self.config = config or DiagnosticsConfig()
        self.enabled = bool(self.config.enabled)
        self.probe_layers = [int(i) for i in self.config.probe_layers]
        self.init_error: Optional[str] = None

        self.predictor = None
        self.layer_probe = None
        self.svd_probe = None
        self.spectral_probe = None

        if not self.enabled:
            return

        try:
            self.predictor = DivergencePredictor(
                window_size=self.config.predictor_window,
                proj_dim=self.config.predictor_proj_dim,
                ridge=self.config.predictor_ridge,
            )
            self.svd_probe = WindowedSVDProbe(
                WindowedSVDProbeConfig(
                    window_size=self.config.svd_window,
                    top_k=self.config.svd_top_k,
                )
            )
            self.spectral_probe = SpectralTrajectoryProbe(
                SpectralTrajectoryConfig(
                    window_size=self.config.spectral_window,
                    n_bands=self.config.spectral_bands,
                )
            )
            if self.probe_layers:
                self.layer_probe = LayerProbe(
                    LayerProbeConfig(
                        layers=self.probe_layers,
                        window_size=self.config.layer_window,
                    )
                )
        except Exception as e:
            self.enabled = False
            self.init_error = str(e)

    def reset(self):
        if not self.enabled:
            return
        if self.predictor is not None:
            self.predictor.reset()
        if self.layer_probe is not None:
            self.layer_probe.reset()
        if self.svd_probe is not None:
            self.svd_probe.reset()
        if self.spectral_probe is not None:
            self.spectral_probe.reset()

    def step(
        self,
        hidden: Optional[torch.Tensor],
        layer_states: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict:
        if (not self.enabled) or hidden is None:
            return {}

        out: Dict = {}
        health: Dict = {"enabled": True, "valid": True, "degraded": False, "issues": []}

        try:
            out["divergence"] = float(self.predictor.step(hidden))
            predictor_health = getattr(self.predictor, "last_health", None) or {}
            health["valid"] = bool(predictor_health.get("valid", True))
            health["degraded"] = bool(predictor_health.get("degraded", False))
            health["issues"].extend(list(predictor_health.get("issues", []) or []))
        except Exception as e:
            out["divergence_error"] = str(e)
            health["valid"] = False
            health["degraded"] = True
            health["issues"].append({"issue": "predictor_exception", "message": str(e)})

        try:
            if self.spectral_probe is not None:
                out["spectral"] = self.spectral_probe.step(hidden)
            else:
                out["spectral"] = {"disabled": True}
        except Exception as e:
            out["spectral_error"] = str(e)
            health["degraded"] = True
            health["issues"].append({"issue": "spectral_exception", "message": str(e)})

        try:
            out["svd"] = self.svd_probe.step(hidden)
        except Exception as e:
            out["svd_error"] = str(e)
            health["degraded"] = True
            health["issues"].append({"issue": "svd_exception", "message": str(e)})

        if self.layer_probe is not None:
            try:
                out["layer_stiffness"] = self.layer_probe.step(layer_states or {})
            except Exception as e:
                out["layer_stiffness_error"] = str(e)
                health["degraded"] = True
                health["issues"].append({"issue": "layer_probe_exception", "message": str(e)})

        out["health"] = health

        return out
