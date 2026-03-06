from __future__ import annotations

from typing import Dict, Optional

import torch

from runtime_lab.config.schemas import DiagnosticsConfig
from .layer_probe import LayerProbe, LayerProbeConfig
from .predictor import DivergencePredictor
from .spectral import spectral_energy_metrics
from .windowed_svd import WindowedSVDProbe, WindowedSVDProbeConfig


class DiagnosticsManager:
    def __init__(self, config: Optional[DiagnosticsConfig] = None):
        self.config = config or DiagnosticsConfig()
        self.enabled = bool(self.config.enabled)
        self.probe_layers = [int(i) for i in self.config.probe_layers]
        self.init_error: Optional[str] = None

        self.predictor = None
        self.layer_probe = None
        self.svd_probe = None

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

    def step(
        self,
        hidden: Optional[torch.Tensor],
        layer_states: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict:
        if (not self.enabled) or hidden is None:
            return {}

        out: Dict = {}

        try:
            out["divergence"] = float(self.predictor.step(hidden))
        except Exception as e:
            out["divergence_error"] = str(e)

        try:
            out["spectral"] = spectral_energy_metrics(hidden)
        except Exception as e:
            out["spectral_error"] = str(e)

        try:
            out["svd"] = self.svd_probe.step(hidden)
        except Exception as e:
            out["svd_error"] = str(e)

        if self.layer_probe is not None:
            try:
                out["layer_stiffness"] = self.layer_probe.step(layer_states or {})
            except Exception as e:
                out["layer_stiffness_error"] = str(e)

        return out
