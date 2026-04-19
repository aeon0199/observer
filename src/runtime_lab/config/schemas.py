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
    # Sampling controls. temperature<=0 reproduces legacy greedy decoding.
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0


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
    # Token-time FFT window for SpectralTrajectoryProbe (real sequence-axis FFT).
    spectral_window: int = 16
    spectral_bands: int = 6


@dataclass
class StressConfig(CommonRunConfig):
    intervention_layer: int = -1
    intervention_type: str = "additive"
    intervention_magnitude: float = 2.0
    # When True, `intervention_magnitude` is interpreted as a fraction of the
    # current hidden-state L2 norm (i.e. magnitude=0.1 means the injected
    # delta has 10% the magnitude of the hidden state). This makes the knob
    # meaningful across models / layers whose activations live at very
    # different scales. When False, magnitude is absolute (legacy behavior).
    intervention_magnitude_relative: bool = True
    intervention_start: int = 5
    intervention_duration: int = 10
    with_diagnostics: bool = True


@dataclass
class HysteresisConfig(CommonRunConfig):
    original_question_label: str = "ORIGINAL_QUESTION"
    # Perturbation class. "prompt" (default, legacy) injects a synthetic
    # <REFLECTION> block into the context — this measures *prompt
    # contamination persistence*, not internal dynamics. "noise" instead
    # injects a seeded additive perturbation onto hidden states for a
    # configurable window during PERTURB, then removes it for REASK —
    # this is the actual "internal hysteresis" question.
    perturbation_mode: str = "prompt"
    noise_layer: int = -1
    noise_magnitude: float = 0.15  # relative to hidden norm
    noise_start: int = 3
    noise_duration: int = 8
    noise_seed: int = 1234


@dataclass
class ControlConfig(CommonRunConfig):
    measure_layer: int = -1
    act_layer: int = -1
    intervention_type: str = "scaling"
    additive_warn_magnitude: float = 0.3
    additive_crit_magnitude: float = 0.6
    additive_seed: int = 42
    additive_direction: str = "opposing"
    additive_reference: str = "ema"
    ema_alpha: float = 0.9
    ema_warmup_tokens: int = 3
    anchor_tokens: int = 3
    shadow: bool = False
    ma_window: int = 3
    threshold_warn: float = 0.55
    threshold_crit: float = 0.85
    scale_warn: float = 0.90
    scale_crit: float = 0.75
    hold_warn: int = 3
    hold_crit: int = 6
