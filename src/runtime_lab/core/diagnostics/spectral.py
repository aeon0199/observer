from __future__ import annotations

import math
from typing import Any, Dict

import torch


_EPS = 1e-8


def spectral_energy_metrics(hidden_state: torch.Tensor, n_bands: int = 6) -> Dict[str, Any]:
    if not isinstance(hidden_state, torch.Tensor):
        raise TypeError(f"hidden_state must be torch.Tensor, got {type(hidden_state)}")

    def _compute(x: torch.Tensor) -> Dict[str, Any]:
        x = x.detach().float().view(-1)
        d = int(x.numel())
        x = x - x.mean()

        spec = torch.fft.rfft(x, dim=0)
        power = (spec.real ** 2 + spec.imag ** 2).float()

        power_no_dc = power[1:] if power.numel() > 1 else power
        total_power = float(power_no_dc.sum().item())

        if total_power <= 0.0:
            return {
                "total_power": 0.0,
                "spectral_entropy": 0.0,
                "spectral_flatness": 0.0,
                "centroid": 0.0,
                "rolloff_85": 0.0,
                "high_frac": 0.0,
                "low_frac": 0.0,
                "band_fracs": [0.0 for _ in range(max(1, int(n_bands)))],
                "freq_bins": int(power_no_dc.numel()),
                "signal_dim": d,
            }

        p = power_no_dc / (power_no_dc.sum() + _EPS)

        ent = -(p * (p + _EPS).log()).sum()
        ent_norm = float((ent / math.log(float(p.numel()) + _EPS)).item())

        geom = torch.exp(torch.mean(torch.log(power_no_dc + _EPS)))
        arith = torch.mean(power_no_dc)
        flat = float((geom / (arith + _EPS)).item())

        freqs = torch.linspace(0.0, 1.0, steps=int(p.numel()), device=p.device, dtype=p.dtype)
        centroid = float((freqs * p).sum().item())

        roll_idx = int((torch.cumsum(p, dim=0) >= 0.85).nonzero(as_tuple=False)[0].item()) if p.numel() > 0 else 0
        rolloff = float(roll_idx / max(1, (p.numel() - 1)))

        k = max(1, int(0.2 * p.numel()))
        low_frac = float(p[:k].sum().item())
        high_frac = float(p[-k:].sum().item())

        nb = max(1, int(n_bands))
        band_fracs = []
        for b in range(nb):
            a = int((b * p.numel()) / nb)
            z = int(((b + 1) * p.numel()) / nb)
            band_fracs.append(float(p[a:z].sum().item()))

        return {
            "total_power": total_power,
            "spectral_entropy": ent_norm,
            "spectral_flatness": flat,
            "centroid": centroid,
            "rolloff_85": rolloff,
            "high_frac": high_frac,
            "low_frac": low_frac,
            "band_fracs": band_fracs,
            "freq_bins": int(p.numel()),
            "signal_dim": d,
        }

    try:
        return _compute(hidden_state)
    except Exception as e:
        try:
            out = _compute(hidden_state.detach().float().cpu())
            out["fft_fallback"] = "cpu"
            out["fft_error"] = str(e)
            return out
        except Exception as e2:
            return {
                "error": f"FFT failed on device and cpu fallback: {e}; fallback: {e2}",
                "signal_dim": int(hidden_state.numel()),
            }
