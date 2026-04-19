from __future__ import annotations

import torch

from .base import Intervention


class AdditiveIntervention(Intervention):
    """Adds a fixed random direction to the last-token hidden state.

    `magnitude` means different things depending on `relative`:

      - relative=True  (default): the delta's L2 norm is set to
        `magnitude * ||h||` where h is the current last-token hidden state.
        So magnitude=0.1 injects a perturbation that's 10% as big as the
        current activation. This is the sensible unit for research — it
        scales with whatever activation norms the model happens to have.

      - relative=False: the delta's L2 norm is `magnitude` in absolute units.
        Legacy behavior; keep only when replicating older runs.
    """

    def __init__(
        self,
        magnitude: float = 0.1,
        seed: int = 42,
        relative: bool = True,
    ):
        self.magnitude = float(magnitude)
        self.seed = int(seed)
        self.relative = bool(relative)
        self._direction = None  # unit vector, set lazily
        self.name = "additive"

    def apply(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self._direction is None:
            dim = hidden_state.shape[-1]
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed)
            vec = torch.randn(dim, generator=g, device="cpu", dtype=torch.float32)
            vec = vec / (vec.norm() + 1e-12)
            self._direction = vec

        direction = self._direction.to(device=hidden_state.device, dtype=hidden_state.dtype)

        modified = hidden_state.clone()
        last = modified[:, -1, :]

        if self.relative:
            # Set delta norm to magnitude * ||h_last||, per batch element.
            h_norm = last.norm(dim=-1, keepdim=True)  # [B, 1]
            delta = direction.unsqueeze(0) * (self.magnitude * h_norm)
        else:
            delta = direction * self.magnitude

        modified[:, -1, :] = last + delta
        return modified
