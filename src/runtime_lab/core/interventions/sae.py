from __future__ import annotations

from typing import Optional

import torch

from .base import Intervention


def _unwrap_sae_loaded(obj):
    if isinstance(obj, tuple) and len(obj) >= 1:
        return obj[0]
    return obj


def load_sae_decoder_vector(
    repo_id: str,
    feature_idx: int,
    layer: Optional[int] = None,
    sae_id: Optional[str] = None,
    normalize: bool = True,
    device: Optional[str] = None,
) -> torch.Tensor:
    try:
        from sae_lens import SAE
    except Exception as e:
        raise RuntimeError("SAE requested but `sae-lens` is not installed") from e

    if sae_id is None:
        if layer is None:
            raise ValueError("Provide either sae_id or layer for SAE loading")
        sae_id = f"layer_{int(layer)}"

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loaded = SAE.from_pretrained(release=repo_id, sae_id=sae_id, device=target_device)
    sae = _unwrap_sae_loaded(loaded)

    vec = sae.W_dec[int(feature_idx)].detach().float()
    if normalize:
        vec = vec / (vec.norm() + 1e-12)
    return vec.cpu()


class SAEIntervention(Intervention):
    def __init__(self, decoder_vector: torch.Tensor, strength: float = 5.0, name: str = "sae"):
        self.decoder_vector = decoder_vector.detach().float().cpu().view(-1)
        self.strength = float(strength)
        self.name = str(name)

    def apply(self, hidden_state: torch.Tensor) -> torch.Tensor:
        vec = self.decoder_vector.to(device=hidden_state.device, dtype=hidden_state.dtype)
        modified = hidden_state.clone()
        modified[:, -1, :] = modified[:, -1, :] + (self.strength * vec)
        return modified
