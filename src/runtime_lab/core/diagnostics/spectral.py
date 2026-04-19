"""Trajectory spectral probe.

Previous version FFT'd the flattened hidden-state vector, treating the (arbitrary)
neuron ordering as a frequency axis. That measured nothing physically meaningful —
permuting neurons would change every output, yet the underlying signal would be
identical. The results were effectively noise about an implementation detail.

This module now operates on the **token-time axis**: as hidden states stream in,
they are accumulated into a sliding window of shape [T, D]. An FFT is taken
along dim=0 (time), so each of D neurons contributes a per-dim time series; the
per-frequency power is averaged across dims to produce a scalar spectrum over
the trajectory. This captures real structure — e.g. whether activations
oscillate or drift at low vs. high cadence across generation steps.

A backwards-compat `spectral_energy_metrics(hidden_state)` shim is kept for any
external caller, but returns the zero/warmup state; the real signal now requires
the stateful SpectralTrajectoryProbe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


_EPS = 1e-8


def _empty_metrics(n_bands: int, signal_dim: int, window_len: int) -> Dict[str, Any]:
    return {
        "total_power": 0.0,
        "spectral_entropy": 0.0,
        "spectral_flatness": 0.0,
        "centroid": 0.0,
        "rolloff_85": 0.0,
        "high_frac": 0.0,
        "low_frac": 0.0,
        "band_fracs": [0.0 for _ in range(max(1, int(n_bands)))],
        "freq_bins": 0,
        "window_len": int(window_len),
        "signal_dim": int(signal_dim),
        "axis": "token-time",
    }


def _derive_features_from_power(power: torch.Tensor, n_bands: int) -> Dict[str, float]:
    """power is 1D, DC already removed, positive. Returns the scalar features."""
    total = float(power.sum().item())
    if total <= 0.0 or power.numel() == 0:
        return {
            "total_power": 0.0,
            "spectral_entropy": 0.0,
            "spectral_flatness": 0.0,
            "centroid": 0.0,
            "rolloff_85": 0.0,
            "high_frac": 0.0,
            "low_frac": 0.0,
            "band_fracs": [0.0] * max(1, n_bands),
        }

    p = power / (power.sum() + _EPS)

    ent = -(p * (p + _EPS).log()).sum()
    ent_norm = float((ent / math.log(float(p.numel()) + _EPS)).item())

    geom = torch.exp(torch.mean(torch.log(power + _EPS)))
    arith = torch.mean(power) + _EPS
    flat = float((geom / arith).item())

    freqs = torch.linspace(0.0, 1.0, steps=int(p.numel()), device=p.device, dtype=p.dtype)
    centroid = float((freqs * p).sum().item())

    cum = torch.cumsum(p, dim=0)
    nz = (cum >= 0.85).nonzero(as_tuple=False)
    roll_idx = int(nz[0].item()) if nz.numel() > 0 else int(p.numel() - 1)
    rolloff = float(roll_idx / max(1, (p.numel() - 1)))

    k = max(1, int(0.2 * p.numel()))
    low_frac = float(p[:k].sum().item())
    high_frac = float(p[-k:].sum().item())

    nb = max(1, int(n_bands))
    band_fracs: List[float] = []
    for b in range(nb):
        a = int((b * p.numel()) / nb)
        z = int(((b + 1) * p.numel()) / nb)
        band_fracs.append(float(p[a:z].sum().item()))

    return {
        "total_power": total,
        "spectral_entropy": ent_norm,
        "spectral_flatness": flat,
        "centroid": centroid,
        "rolloff_85": rolloff,
        "high_frac": high_frac,
        "low_frac": low_frac,
        "band_fracs": band_fracs,
    }


@dataclass
class SpectralTrajectoryConfig:
    window_size: int = 16
    n_bands: int = 6
    center: bool = True
    min_tokens: int = 4


class SpectralTrajectoryProbe:
    """Streaming FFT over the last `window_size` hidden-state vectors, along the
    time axis. Output axes-of-meaning:
      - low frequency power  ->  slow drift in activations over tokens
      - high frequency power ->  step-to-step oscillation / chattery dynamics
      - centroid / rolloff   ->  where the energy lives on the slow-fast spectrum
      - spectral_entropy     ->  flat (broadband) vs. peaked trajectory spectrum
    """

    def __init__(self, config: Optional[SpectralTrajectoryConfig] = None):
        self.config = config or SpectralTrajectoryConfig()
        self._buf: List[torch.Tensor] = []
        self._signal_dim: int = 0

    def reset(self) -> None:
        self._buf = []
        self._signal_dim = 0

    def step(self, hidden_state: torch.Tensor) -> Dict[str, Any]:
        if not isinstance(hidden_state, torch.Tensor):
            raise TypeError(f"hidden_state must be torch.Tensor, got {type(hidden_state)}")

        x = hidden_state.detach().float().cpu().reshape(-1)
        self._signal_dim = int(x.numel())
        self._buf.append(x)
        if len(self._buf) > int(self.config.window_size):
            self._buf.pop(0)

        T = len(self._buf)
        if T < max(2, int(self.config.min_tokens)):
            return _empty_metrics(self.config.n_bands, self._signal_dim, T)

        # Stack into [T, D] and center per-dim across time.
        try:
            X = torch.stack(self._buf, dim=0)
            if self.config.center:
                X = X - X.mean(dim=0, keepdim=True)

            # FFT along token axis: [T//2+1, D] complex
            fft = torch.fft.rfft(X, dim=0)
            power = (fft.real ** 2 + fft.imag ** 2)  # [F, D]

            # Drop DC, aggregate across dims.
            if power.shape[0] <= 1:
                return _empty_metrics(self.config.n_bands, self._signal_dim, T)
            power_nodc = power[1:]
            power_mean = power_nodc.mean(dim=1)  # [F-1]

        except Exception as e:
            out = _empty_metrics(self.config.n_bands, self._signal_dim, T)
            out["fft_error"] = str(e)
            return out

        features = _derive_features_from_power(power_mean, self.config.n_bands)

        # Self-test: shuffling the token order along the time axis SHOULD
        # change the spectrum (since FFT over time is order-sensitive).
        # If this ratio is ~1.0, something's wrong (we'd be neuron-axis again).
        # Only run when window is mature enough to make the check meaningful.
        invariance_ratio = None
        if T >= 8:
            try:
                perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(0))
                X_shuf = X[perm]
                fft_shuf = torch.fft.rfft(X_shuf, dim=0)
                power_shuf = (fft_shuf.real ** 2 + fft_shuf.imag ** 2)[1:].mean(dim=1)
                num = (power_mean - power_shuf).abs().sum().item()
                den = (power_mean.abs().sum().item() + 1e-8)
                invariance_ratio = float(num / den)  # 0=identical, high=very different
            except Exception:
                invariance_ratio = None

        return {
            **features,
            "freq_bins": int(power_mean.numel()),
            "window_len": int(T),
            "signal_dim": int(self._signal_dim),
            "axis": "token-time",
            "permutation_change": invariance_ratio,  # None until window>=8; >0 confirms sequence-axis behavior
        }


# Backwards-compat shim ------------------------------------------------------
#
# `spectral_energy_metrics` used to take a single hidden state and return
# (bogus) spectral features over the neuron axis. Callers that still import it
# will get a well-formed warmup payload rather than a meaningless result; they
# should migrate to SpectralTrajectoryProbe.

def spectral_energy_metrics(hidden_state: torch.Tensor, n_bands: int = 6) -> Dict[str, Any]:
    if not isinstance(hidden_state, torch.Tensor):
        raise TypeError(f"hidden_state must be torch.Tensor, got {type(hidden_state)}")
    d = int(hidden_state.detach().reshape(-1).numel())
    out = _empty_metrics(n_bands, d, 1)
    out["deprecated"] = (
        "spectral_energy_metrics is stateless; use SpectralTrajectoryProbe for "
        "meaningful token-time spectral features."
    )
    return out
