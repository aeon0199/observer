# v1/frames.py

"""
Telemetry Frames
----------------
Internal telemetry snapshot for each stage of a run.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

try:
    from .utils import json_safe
except ImportError:  # pragma: no cover
    from utils import json_safe


@dataclass
class COTFrame:
    """
    Internal telemetry capsule for one stage of the observer loop.
    """
    stage: str
    token_text: str
    timestamp: float

    hidden_norm: float
    entropy: float
    logit_norm: float

    topk: Any
    svd: Any

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "timestamp": self.timestamp,
            "token_text": self.token_text,
            "hidden_norm": self.hidden_norm,
            "entropy": self.entropy,
            "logit_norm": self.logit_norm,
            "topk": json_safe(self.topk),
            "svd": json_safe(self.svd),
            "extra": json_safe(self.extra),
        }
