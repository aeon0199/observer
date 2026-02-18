# v1/model_loader.py

"""
Model Loader
------------
Loads HuggingFace models and tokenizers.
"""

import os
import json
import torch
from dataclasses import dataclass
from typing import Dict, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class LoadedModelConfig:
    key: str
    label: str
    hf_id: str
    max_new_tokens: int
    temperature: float
    top_p: float
    dtype: torch.dtype
    seed: int = 42


def select_device() -> torch.device:
    """Pick the best available backend."""
    if torch.cuda.is_available():
        print("[V1] Device: CUDA GPU")
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        print("[V1] Device: Apple MPS")
        return torch.device("mps")

    print("[V1] Device: CPU")
    return torch.device("cpu")


def load_model_registry(path: str = "models.json") -> Dict[str, Any]:
    resolved = path
    if not os.path.isabs(resolved):
        resolved = os.path.join(os.path.dirname(__file__), resolved)

    if not os.path.exists(resolved):
        raise FileNotFoundError(f"[V1] Model registry not found: {resolved}")

    with open(resolved, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_config(model_key: str, registry: Dict[str, Any]) -> LoadedModelConfig:
    if model_key not in registry["models"]:
        raise KeyError(f"[V1] Unknown model key: {model_key}")

    entry = registry["models"][model_key]

    if "70b" in model_key or "120b" in model_key:
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    return LoadedModelConfig(
        key=model_key,
        label=entry.get("label", model_key),
        hf_id=entry["hf_id"],
        max_new_tokens=entry.get("max_new_tokens", 128),
        temperature=entry.get("temperature", 0.0),
        top_p=entry.get("top_p", 1.0),
        dtype=dtype,
    )


def load_model(model_key: str = None, registry_path: str = "models.json") \
        -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device, LoadedModelConfig]:
    """Load a model from models.json."""

    registry = load_model_registry(registry_path)
    
    if model_key is None:
        model_key = registry.get("default_model", list(registry["models"].keys())[0])

    config = resolve_config(model_key, registry)
    device = select_device()

    # Device-aware dtype for better portability across backends.
    if device.type == "cpu":
        config.dtype = torch.float32
    elif device.type == "mps":
        config.dtype = torch.float16

    print(f"[V1] Loading model: {config.label}")
    print(f"[V1] HF ID / Path: {config.hf_id}")
    print(f"[V1] Precision: {config.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_id,
        trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.hf_id,
        torch_dtype=config.dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True
    )

    if device.type != "cuda":
        model.to(device)

    model.eval()

    print("[V1] Model loaded successfully.")

    return tokenizer, model, device, config
