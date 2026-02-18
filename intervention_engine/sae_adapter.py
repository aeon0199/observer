# v2/sae_adapter.py

"""Optional SAE steering helpers for V2 interventions."""

from __future__ import annotations

from typing import Optional

import torch


def _unwrap_sae_loaded(obj):
    # sae-lens APIs have varied across versions; normalize to SAE object.
    if isinstance(obj, tuple):
        if len(obj) >= 1:
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
        raise RuntimeError(
            "[V2] SAE steering requested but `sae-lens` is not installed."
        ) from e

    if sae_id is None:
        if layer is None:
            raise ValueError("[V2] Provide either sae_id or layer for SAE loading.")
        sae_id = f"layer_{int(layer)}"

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loaded = SAE.from_pretrained(
        release=repo_id,
        sae_id=sae_id,
        device=target_device,
    )
    sae = _unwrap_sae_loaded(loaded)

    vec = sae.W_dec[int(feature_idx)].detach().float()
    if normalize:
        vec = vec / (vec.norm() + 1e-12)
    return vec.cpu()


def make_sae_intervention(
    decoder_vector: torch.Tensor,
    strength: float = 5.0,
    name: str = "sae",
):
    base_vec = decoder_vector.detach().float().cpu().view(-1)
    scale = float(strength)

    def intervene(hidden_state: torch.Tensor) -> torch.Tensor:
        vec = base_vec.to(device=hidden_state.device, dtype=hidden_state.dtype)
        modified = hidden_state.clone()
        modified[:, -1, :] = modified[:, -1, :] + (scale * vec)
        return modified

    intervene.__name__ = str(name)
    return intervene
