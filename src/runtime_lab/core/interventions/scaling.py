from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import Intervention


class ScalingIntervention(Intervention):
    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)
        self.name = "scaling"

    def apply(self, hidden_state: torch.Tensor) -> torch.Tensor:
        modified = hidden_state.clone()
        modified[:, -1, :] = modified[:, -1, :] * self.scale
        return modified


@dataclass
class ScaleState:
    value: float = 1.0


class DynamicScalingIntervention(Intervention):
    def __init__(self, scale_state: ScaleState):
        self.scale_state = scale_state
        self.name = "dynamic_scaling"

    def apply(self, hidden_state: torch.Tensor) -> torch.Tensor:
        s = float(self.scale_state.value)
        if abs(s - 1.0) <= 1e-6:
            return hidden_state
        modified = hidden_state.clone()
        modified[:, -1, :] = modified[:, -1, :] * s
        return modified
