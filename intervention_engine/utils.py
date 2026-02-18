# v2/utils.py

"""
Utilities
---------
JSON helpers and common functions.
"""

import json
import os
import torch
import numpy as np
from typing import Any


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