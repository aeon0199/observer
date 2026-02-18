# v1/utils.py

"""
Utilities
---------
Tensor operations, metrics, JSON helpers.
"""

import json
import torch
import os
import numpy as np
from typing import Any, Dict


def to_numpy(x: Any) -> np.ndarray:
    """Convert tensor-like to CPU NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.array(x, dtype=np.float32)


def tensor_norm(t: Any) -> float:
    """L2 norm of a tensor or array."""
    arr = to_numpy(t)
    return float(np.linalg.norm(arr))


def tensor_entropy(logits: Any) -> float:
    """Compute entropy over logits."""
    arr = to_numpy(logits).astype(np.float64)
    arr -= np.max(arr)
    exp = np.exp(arr)
    probs = exp / (np.sum(exp) + 1e-12)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return float(entropy)


def topk_values(logits: Any, k: int = 5) -> Dict[str, Any]:
    """Extract top-k logits for the last token."""
    if isinstance(logits, torch.Tensor):
        x = logits.detach().cpu().float()
    else:
        x = torch.as_tensor(logits, dtype=torch.float32).detach().cpu()

    if x.ndim == 3:
        x = x[0, -1, :]
    elif x.ndim == 2:
        x = x[-1, :]
    elif x.ndim == 0:
        x = x.reshape(1)
    else:
        x = x.reshape(-1)

    if x.numel() == 0:
        return {"indices": [], "values": []}

    k = int(min(max(1, k), x.numel()))
    values, indices = torch.topk(x, k)
    return {
        "indices": indices.tolist(),
        "values": [float(v) for v in values]
    }


def svd_compress(t: Any, k: int = 5) -> Dict[str, Any]:
    """Compute compact SVD signature."""
    try:
        arr = to_numpy(t)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        U, S, Vt = np.linalg.svd(arr.astype(np.float32), full_matrices=False)
        k = min(k, len(S))
        return {"singular_values": S[:k].tolist()}

    except Exception as exc:
        return {"error": str(exc)}


def basic_stats(hidden: Any, logits: Any) -> Dict[str, Any]:
    """Return unified metric block."""
    return {
        "hidden_norm": tensor_norm(hidden),
        "logit_norm": tensor_norm(logits),
        "entropy": tensor_entropy(logits),
        "topk": topk_values(logits, k=5),
    }


def json_safe(x: Any) -> Any:
    """Recursively convert to JSON-serializable forms."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (np.ndarray, np.generic)):
        return x.tolist()
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    if isinstance(x, list):
        return [json_safe(i) for i in x]
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    return str(x)


def save_json(path: str, data: Any) -> None:
    """Atomic JSON writer."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2)
    os.replace(tmp, path)
