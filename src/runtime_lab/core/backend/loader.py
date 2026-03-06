from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelConfig:
    key: str
    hf_id: str
    device: Optional[str] = None
    torch_dtype: Optional[str] = None


@dataclass
class BackendLoadResult:
    tokenizer: Any
    model: Any
    device: torch.device
    config: ModelConfig
    backend: str
    backend_meta: Dict[str, Any]


def load_model_registry(path: str = "models.json") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model registry not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_config(model_key: Optional[str], registry: Dict[str, Any]) -> ModelConfig:
    models = registry.get("models", {})
    if not models:
        raise ValueError("Registry contains no models")

    if model_key is None:
        model_key = registry.get("default_model") or list(models.keys())[0]

    if model_key not in models:
        raise KeyError(f"Unknown model key: {model_key}")

    raw = models[model_key]
    return ModelConfig(
        key=str(model_key),
        hf_id=str(raw["hf_id"]),
        device=raw.get("device"),
        torch_dtype=raw.get("torch_dtype"),
    )


def _as_torch_dtype(name: Optional[str]):
    if not name:
        return None
    name = str(name).lower().strip()
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return table.get(name)


def _as_torch_device(value: Optional[str], fallback: str = "cpu") -> torch.device:
    return torch.device(str(value or fallback))


def _pick_hf_model_from_nnsight(wrapper: Any) -> Any:
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

    registry = load_model_registry(registry_path)
    config = resolve_config(model_key, registry)

    if backend == "hf":
        preferred_device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = _as_torch_device(preferred_device)
        dtype = _as_torch_dtype(config.torch_dtype)

        tokenizer = AutoTokenizer.from_pretrained(config.hf_id)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        model = AutoModelForCausalLM.from_pretrained(config.hf_id, **model_kwargs)
        model.to(device)
        model.eval()

        return BackendLoadResult(
            tokenizer=tokenizer,
            model=model,
            device=device,
            config=config,
            backend="hf",
            backend_meta={},
        )

    if backend != "nnsight":
        raise ValueError(f"Unsupported backend: {backend}")

    try:
        from nnsight import LanguageModel
    except Exception as e:
        raise RuntimeError("backend='nnsight' requested but `nnsight` is not installed") from e

    target_device = nnsight_device or config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = LanguageModel(
        config.hf_id,
        device_map=target_device,
        dispatch=True,
        remote=bool(nnsight_remote),
    )
    tokenizer = getattr(wrapper, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Failed to access tokenizer from nnsight LanguageModel")

    model = _pick_hf_model_from_nnsight(wrapper)
    device = _as_torch_device(target_device)

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
