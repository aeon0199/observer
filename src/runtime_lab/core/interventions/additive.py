from __future__ import annotations

import torch

from .base import Intervention


class AdditiveIntervention(Intervention):
    def __init__(self, magnitude: float = 1.0, seed: int = 42):
        self.magnitude = float(magnitude)
        self.seed = int(seed)
        self._vector = None
        self.name = "additive"

    def apply(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self._vector is None:
            dim = hidden_state.shape[-1]
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed)
            vec = torch.randn(dim, generator=g, device="cpu", dtype=torch.float32)
            vec = vec / (vec.norm() + 1e-12) * self.magnitude
            self._vector = vec

        vec = self._vector.to(device=hidden_state.device, dtype=hidden_state.dtype)
        modified = hidden_state.clone()
        modified[:, -1, :] = modified[:, -1, :] + vec
        return modified
