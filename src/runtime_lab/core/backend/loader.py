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
    dtype: Optional[str] = None


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
        torch_dtype=raw.get("torch_dtype") or raw.get("dtype"),
        dtype=raw.get("dtype") or raw.get("torch_dtype"),
    )


def _as_torch_dtype(name: Optional[str]):
    if not name:
        return None
    name = str(name).lower().strip()
    if name in {"auto", "default"}:
        return None
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


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def _best_available_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def _device_available(device_name: str) -> bool:
    normalized = str(device_name or "").lower().strip()
    if normalized.startswith("cuda"):
        return bool(torch.cuda.is_available())
    if normalized.startswith("mps"):
        return _mps_available()
    if normalized.startswith("cpu"):
        return True
    return False


def _dtype_name(dtype: Optional[torch.dtype]) -> Optional[str]:
    if dtype is None:
        return None
    table = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float64: "float64",
    }
    return table.get(dtype, str(dtype))


def _default_dtype_for_device(device_name: str) -> torch.dtype:
    normalized = str(device_name or "cpu").lower().strip()
    if normalized.startswith("cuda"):
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _resolve_runtime_policy(
    config: ModelConfig,
    device_override: Optional[str] = None,
) -> Dict[str, Any]:
    requested_device = str(device_override or config.device or "auto").lower().strip()
    requested_dtype_name = str(config.dtype or config.torch_dtype or "auto").lower().strip()
    notes: list[str] = []

    if requested_device in {"", "auto", "default"}:
        resolved_device_name = _best_available_device()
        notes.append(f"auto-selected device '{resolved_device_name}'")
    elif _device_available(requested_device):
        resolved_device_name = requested_device
    else:
        resolved_device_name = _best_available_device()
        notes.append(
            f"requested device '{requested_device}' unavailable; fell back to '{resolved_device_name}'"
        )

    requested_dtype = _as_torch_dtype(requested_dtype_name)
    resolved_dtype = requested_dtype or _default_dtype_for_device(resolved_device_name)

    if resolved_device_name.startswith("mps") and resolved_dtype in {torch.float16, torch.bfloat16}:
        notes.append(
            f"overriding dtype '{_dtype_name(resolved_dtype)}' to 'float32' for MPS numerical stability"
        )
        resolved_dtype = torch.float32

    if resolved_device_name.startswith("cpu") and resolved_dtype in {torch.float16, torch.bfloat16}:
        notes.append(
            f"overriding dtype '{_dtype_name(resolved_dtype)}' to 'float32' for CPU compatibility"
        )
        resolved_dtype = torch.float32

    return {
        "requested_device": requested_device,
        "resolved_device": resolved_device_name,
        "requested_dtype": requested_dtype_name,
        "resolved_dtype": _dtype_name(resolved_dtype),
        "policy_notes": notes,
        "device": _as_torch_device(resolved_device_name),
        "dtype": resolved_dtype,
    }


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
        policy = _resolve_runtime_policy(config)
        device = policy["device"]
        dtype = policy["dtype"]

        tokenizer = AutoTokenizer.from_pretrained(config.hf_id)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {}
        if dtype is not None:
            model_kwargs["dtype"] = dtype

        model = AutoModelForCausalLM.from_pretrained(config.hf_id, **model_kwargs)
        model.to(device)
        model.eval()

        return BackendLoadResult(
            tokenizer=tokenizer,
            model=model,
            device=device,
            config=config,
            backend="hf",
            backend_meta={
                "requested_device": policy["requested_device"],
                "resolved_device": policy["resolved_device"],
                "requested_dtype": policy["requested_dtype"],
                "resolved_dtype": policy["resolved_dtype"],
                "policy_notes": policy["policy_notes"],
                "device_map": str(device),
            },
        )

    if backend != "nnsight":
        raise ValueError(f"Unsupported backend: {backend}")

    try:
        from nnsight import LanguageModel
    except Exception as e:
        raise RuntimeError("backend='nnsight' requested but `nnsight` is not installed") from e

    policy = _resolve_runtime_policy(config, device_override=nnsight_device or config.device)
    target_device = policy["resolved_device"]
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
            "requested_device": policy["requested_device"],
            "resolved_device": policy["resolved_device"],
            "requested_dtype": policy["requested_dtype"],
            "resolved_dtype": policy["resolved_dtype"],
            "policy_notes": policy["policy_notes"],
            "wrapper": wrapper,
        },
    )
