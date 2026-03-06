from __future__ import annotations

import copy
import hashlib
from typing import Any

import torch


def clone_past_key_values(past_key_values: Any) -> Any:
    if past_key_values is None:
        return None

    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        new_cache = None
        try:
            new_cache = past_key_values.__class__()
        except Exception:
            new_cache = None

        if new_cache is not None:
            try:
                new_cache.key_cache = [t.clone() for t in list(key_cache)]
                new_cache.value_cache = [t.clone() for t in list(value_cache)]
                for attr in ("seen_tokens", "_seen_tokens"):
                    if hasattr(past_key_values, attr):
                        setattr(new_cache, attr, getattr(past_key_values, attr))
                return new_cache
            except Exception:
                pass

    try:
        return copy.deepcopy(past_key_values)
    except Exception:
        pass

    def _clone(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.clone()
        if isinstance(x, tuple):
            return tuple(_clone(i) for i in x)
        if isinstance(x, list):
            return [_clone(i) for i in x]
        if isinstance(x, dict):
            return {k: _clone(v) for k, v in x.items()}
        return x

    return _clone(past_key_values)


def _extract_first_key_tensor(past_key_values: Any):
    try:
        return past_key_values[0][0]
    except Exception:
        pass
    try:
        return past_key_values.key_cache[0]
    except Exception:
        pass
    return None


def compute_cache_fingerprint(past_key_values: Any) -> str:
    first_key = _extract_first_key_tensor(past_key_values)
    if first_key is None or not isinstance(first_key, torch.Tensor):
        return "unavailable"

    x = first_key.detach().float().cpu().reshape(-1)
    if x.numel() == 0:
        return "empty"

    sample_n = min(256, x.numel())
    idx = torch.linspace(0, x.numel() - 1, steps=sample_n).long()
    sample = x[idx].numpy().tobytes()

    h = hashlib.sha256()
    h.update(str(tuple(first_key.shape)).encode("utf-8"))
    h.update(str(first_key.dtype).encode("utf-8"))
    h.update(sample)
    return h.hexdigest()[:16]
