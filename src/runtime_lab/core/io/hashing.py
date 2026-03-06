from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_config(config: Any) -> str:
    config_str = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]
