from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch


def json_safe(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (np.ndarray, np.generic)):
        return x.tolist()
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    if isinstance(x, list):
        return [json_safe(i) for i in x]
    if isinstance(x, tuple):
        return [json_safe(i) for i in x]
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if hasattr(x, "__dict__"):
        return json_safe(vars(x))
    return str(x)


def save_json(path: str, data: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
