from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


_EPS = 1e-8


@dataclass
class WindowedSVDProbeConfig:
    window_size: int = 8
    top_k: int = 8
    center: bool = True


class WindowedSVDProbe:
    def __init__(self, config: Optional[WindowedSVDProbeConfig] = None):
        self.config = config or WindowedSVDProbeConfig()
        if self.config.window_size < 2:
            raise ValueError("window_size must be >= 2")
        self._buf: List[torch.Tensor] = []

    def reset(self) -> None:
        self._buf.clear()

    def step(self, hidden: torch.Tensor) -> Dict[str, Any]:
        if not isinstance(hidden, torch.Tensor):
            return {"window_len": len(self._buf), "singular_values": [], "effective_rank": 0.0}

        h = hidden.detach().float().cpu().view(-1)
        self._buf.append(h)
        if len(self._buf) > int(self.config.window_size):
            self._buf.pop(0)

        t = len(self._buf)
        if t < 2:
            return {"window_len": t, "singular_values": [], "effective_rank": 0.0}

        x = torch.stack(self._buf, dim=0)
        if self.config.center:
            x = x - x.mean(dim=0, keepdim=True)

        g = x @ x.T
        eig = torch.linalg.eigvalsh(g).clamp(min=0.0)
        eig = torch.flip(eig, dims=[0])

        sv = torch.sqrt(eig + _EPS)
        sv_k = sv[: int(self.config.top_k)]

        s2 = eig
        total = float(s2.sum().item()) + _EPS
        p = s2 / total

        eff_rank = float(torch.exp(-(p * (p + _EPS).log()).sum()).item())
        top1_frac = float((s2[0] / total).item()) if s2.numel() else 0.0

        return {
            "window_len": t,
            "singular_values": [float(v) for v in sv_k.tolist()],
            "effective_rank": eff_rank,
            "top1_energy_frac": top1_frac,
        }
