# v2/backend.py

"""Backend adapters for V2 runners.

Supports:
- hf: regular Hugging Face model loading (default)
- nnsight: optional wrapper for remote/large-model workflows
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

try:
    from .model_loader import load_model, load_model_registry, resolve_config
except ImportError:  # pragma: no cover
    from model_loader import load_model, load_model_registry, resolve_config


@dataclass
class BackendLoadResult:
    tokenizer: Any
    model: Any
    device: torch.device
    config: Any
    backend: str
    backend_meta: Dict[str, Any]


def _as_torch_device(value: Optional[str], fallback: torch.device) -> torch.device:
    if not value:
        return fallback
    return torch.device(str(value))


def _pick_hf_model_from_nnsight(wrapper: Any) -> Any:
    """Best-effort extraction of the HF-compatible model from a LanguageModel wrapper."""
    for attr in ("model", "_model", "base_model"):
        if hasattr(wrapper, attr):
            candidate = getattr(wrapper, attr)
            if candidate is not None:
                return candidate
    return wrapper


def load_model_with_backend(
    model_key: Optional[str] = None,
    registry_path: str = "models.json",
    backend: str = "hf",
    nnsight_remote: bool = False,
    nnsight_device: Optional[str] = None,
) -> BackendLoadResult:
    backend = str(backend or "hf").lower().strip()

    if backend == "hf":
        tokenizer, model, device, config = load_model(model_key=model_key, registry_path=registry_path)
        return BackendLoadResult(
            tokenizer=tokenizer,
            model=model,
            device=device,
            config=config,
            backend="hf",
            backend_meta={},
        )

    if backend != "nnsight":
        raise ValueError(f"[V2] Unsupported backend: {backend}")

    try:
        from nnsight import LanguageModel
    except Exception as e:
        raise RuntimeError(
            "[V2] backend='nnsight' requested but `nnsight` is not installed. "
            "Install optional deps and retry."
        ) from e

    registry = load_model_registry(registry_path)
    if model_key is None:
        model_key = registry.get("default_model", list(registry["models"].keys())[0])
    config = resolve_config(model_key, registry)

    target_device = nnsight_device or ("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = LanguageModel(
        config.hf_id,
        device_map=target_device,
        dispatch=True,
        remote=bool(nnsight_remote),
    )

    tokenizer = getattr(wrapper, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("[V2] Failed to access tokenizer from nnsight LanguageModel.")

    model = _pick_hf_model_from_nnsight(wrapper)
    device = _as_torch_device(target_device, torch.device("cpu"))

    return BackendLoadResult(
        tokenizer=tokenizer,
        model=model,
        device=device,
        config=config,
        backend="nnsight",
        backend_meta={
            "remote": bool(nnsight_remote),
            "device_map": target_device,
            "wrapper": wrapper,
        },
    )
