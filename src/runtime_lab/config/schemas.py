from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CommonRunConfig:
    prompt: str
    model_key: Optional[str] = None
    max_new_tokens: int = 64
    backend: str = "hf"
    nnsight_remote: bool = False
    nnsight_device: Optional[str] = None
    seed: Optional[int] = 42


@dataclass
class DiagnosticsConfig:
    enabled: bool = True
    probe_layers: List[int] = field(default_factory=list)
    predictor_window: int = 8
    predictor_proj_dim: int = 64
    predictor_ridge: float = 1e-2
    svd_window: int = 8
    svd_top_k: int = 8
    layer_window: int = 5


@dataclass
class StressConfig(CommonRunConfig):
    intervention_layer: int = -1
    intervention_type: str = "additive"
    intervention_magnitude: float = 2.0
    intervention_start: int = 5
    intervention_duration: int = 10
    with_diagnostics: bool = True


@dataclass
class HysteresisConfig(CommonRunConfig):
    original_question_label: str = "ORIGINAL_QUESTION"


@dataclass
class ControlConfig(CommonRunConfig):
    measure_layer: int = -1
    act_layer: int = -1
    intervention_type: str = "scaling"
    shadow: bool = False
    ma_window: int = 3
    threshold_warn: float = 0.55
    threshold_crit: float = 0.85
    scale_warn: float = 0.90
    scale_crit: float = 0.75
    hold_warn: int = 3
    hold_crit: int = 6
