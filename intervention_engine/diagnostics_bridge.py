"""Bridge V1.5 diagnostics into V2 without package-path assumptions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch


def _load_module_from_path(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"[V2] Missing V1.5 module: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"[V2] Failed to load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_v15_modules() -> Dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    v15 = root / "v1.5 t"

    return {
        "predictor": _load_module_from_path("v2_bridge_v15_predictor", v15 / "predictor.py"),
        "spectral": _load_module_from_path("v2_bridge_v15_spectral", v15 / "spectral.py"),
        "layer_probe": _load_module_from_path("v2_bridge_v15_layer_probe", v15 / "layer_probe.py"),
        "utils": _load_module_from_path("v2_bridge_v15_utils", v15 / "utils.py"),
    }


class DiagnosticsManager:
    """Optional diagnostics pack (divergence + spectral + windowed SVD + layer stiffness)."""

    def __init__(self, enabled: bool = True, probe_layers: Optional[List[int]] = None):
        self.enabled = bool(enabled)
        self.probe_layers = [int(i) for i in (probe_layers or [])]
        self.init_error: Optional[str] = None

        self.predictor = None
        self.layer_probe = None
        self.svd_probe = None
        self._spectral_fn = None

        if not self.enabled:
            return

        try:
            mods = _load_v15_modules()
        except Exception as e:
            # Graceful fallback: keep runtime paths alive even when optional
            # diagnostics modules are not present in the repo.
            self.enabled = False
            self.init_error = str(e)
            return

        self.predictor = mods["predictor"].DivergencePredictor(window_size=8, ridge=1e-2)
        self._spectral_fn = mods["spectral"].spectral_energy_metrics

        WindowedSVDProbe = mods["utils"].WindowedSVDProbe
        WindowedSVDProbeConfig = mods["utils"].WindowedSVDProbeConfig
        self.svd_probe = WindowedSVDProbe(WindowedSVDProbeConfig(window_size=8))

        if self.probe_layers:
            LayerProbe = mods["layer_probe"].LayerProbe
            LayerProbeConfig = mods["layer_probe"].LayerProbeConfig
            self.layer_probe = LayerProbe(
                LayerProbeConfig(layers=self.probe_layers, window_size=5)
            )

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
            out["spectral"] = self._spectral_fn(hidden)
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
