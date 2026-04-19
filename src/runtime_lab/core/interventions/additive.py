from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import Intervention


@dataclass
class MagnitudeState:
    """Mutable magnitude holder for controller-driven additive intervention.

    The control loop sets `value` based on the controller's status; the
    DynamicAdditiveIntervention reads it at each apply() call. value=0
    means "no injection this step" — the intervention becomes a no-op.
    """
    value: float = 0.0


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


class DynamicAdditiveIntervention(Intervention):
    """Controller-driven additive intervention.

    Reads its magnitude from a shared `MagnitudeState` object on each apply().
    Direction is a fixed seeded unit vector (same as AdditiveIntervention).
    When magnitude==0, apply() returns the input unchanged — zero cost.

    Sign handling: the intervention uses the direction as-is. The control
    loop can set `magnitude_state.value` to a negative number to flip the
    sign (pushing in the opposite direction) without needing to re-seed.
    """

    def __init__(
        self,
        magnitude_state: MagnitudeState,
        seed: int = 42,
        relative: bool = True,
    ):
        self.magnitude_state = magnitude_state
        self.seed = int(seed)
        self.relative = bool(relative)
        self._direction = None
        self.name = "dynamic_additive"

    def apply(self, hidden_state: torch.Tensor) -> torch.Tensor:
        m = float(self.magnitude_state.value)
        if abs(m) <= 1e-8:
            return hidden_state

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
            h_norm = last.norm(dim=-1, keepdim=True)
            delta = direction.unsqueeze(0) * (m * h_norm)
        else:
            delta = direction * m

        modified[:, -1, :] = last + delta
        return modified
