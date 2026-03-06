from __future__ import annotations

import torch

from .base import Intervention


class ProjectionIntervention(Intervention):
    def __init__(self, subspace_dim: int = 10, seed: int = 42):
        self.subspace_dim = int(subspace_dim)
        self.seed = int(seed)
        self._projection_matrix = None
        self.name = "projection"

    def apply(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self._projection_matrix is None:
            dim = hidden_state.shape[-1]
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed)
            random_matrix = torch.randn(dim, self.subspace_dim, generator=g, device="cpu", dtype=torch.float32)
            q, _ = torch.linalg.qr(random_matrix)
            self._projection_matrix = torch.eye(dim, dtype=torch.float32) - q @ q.T

        proj = self._projection_matrix.to(device=hidden_state.device, dtype=hidden_state.dtype)
        modified = hidden_state.clone()
        modified[:, -1, :] = modified[:, -1, :] @ proj
        return modified
