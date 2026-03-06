from __future__ import annotations

from typing import Any

from .additive import AdditiveIntervention
from .projection import ProjectionIntervention
from .sae import SAEIntervention, load_sae_decoder_vector
from .scaling import ScalingIntervention


def build_intervention(kind: str, **kwargs: Any):
    kind = str(kind).lower().strip()

    if kind == "scaling":
        return ScalingIntervention(scale=float(kwargs.get("scale", kwargs.get("magnitude", 1.0))))

    if kind == "additive":
        return AdditiveIntervention(
            magnitude=float(kwargs.get("magnitude", 1.0)),
            seed=int(kwargs.get("seed", 42)),
        )

    if kind == "projection":
        return ProjectionIntervention(
            subspace_dim=int(kwargs.get("subspace_dim", kwargs.get("magnitude", 10))),
            seed=int(kwargs.get("seed", 42)),
        )

    if kind == "sae":
        decoder_vector = kwargs.get("decoder_vector")
        if decoder_vector is None:
            decoder_vector = load_sae_decoder_vector(
                repo_id=kwargs["repo_id"],
                feature_idx=int(kwargs["feature_idx"]),
                layer=kwargs.get("layer"),
                sae_id=kwargs.get("sae_id"),
                normalize=bool(kwargs.get("normalize", True)),
                device=kwargs.get("device"),
            )
        return SAEIntervention(
            decoder_vector=decoder_vector,
            strength=float(kwargs.get("strength", 5.0)),
            name=str(kwargs.get("name", "sae")),
        )

    raise ValueError(f"Unknown intervention kind: {kind}")
