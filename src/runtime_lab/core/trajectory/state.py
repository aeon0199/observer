from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


@dataclass
class TokenState:
    token_idx: int
    token_id: int
    token_text: str
    hidden_norm: float
    logit_norm: float
    entropy: float
    top1_prob: float
    top1_token: int
    intervention_active: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    states: List[TokenState] = field(default_factory=list)
    _hidden_vecs: List[torch.Tensor] = field(default_factory=list, repr=False)
    _logits: List[torch.Tensor] = field(default_factory=list, repr=False)
    prompt: str = ""
    model_id: str = ""
    intervention_layer_requested: int = -1
    intervention_layer_resolved: int = -1
    intervention_start: int = -1
    intervention_end: int = -1
    intervention_type: str = "none"

    def add_state(self, state: TokenState) -> None:
        self.states.append(state)

    def __len__(self) -> int:
        return len(self.states)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "token_idx": np.array([s.token_idx for s in self.states], dtype=np.int64),
            "hidden_norm": np.array([s.hidden_norm for s in self.states], dtype=np.float64),
            "logit_norm": np.array([s.logit_norm for s in self.states], dtype=np.float64),
            "entropy": np.array([s.entropy for s in self.states], dtype=np.float64),
            "top1_prob": np.array([s.top1_prob for s in self.states], dtype=np.float64),
            "intervention_active": np.array([s.intervention_active for s in self.states], dtype=bool),
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "model_id": self.model_id,
            "intervention": {
                "requested_layer": self.intervention_layer_requested,
                "resolved_layer": self.intervention_layer_resolved,
                "start_token": self.intervention_start,
                "end_token": self.intervention_end,
                "type": self.intervention_type,
            },
            "states": [
                {
                    "token_idx": s.token_idx,
                    "token_id": s.token_id,
                    "token_text": s.token_text,
                    "hidden_norm": s.hidden_norm,
                    "logit_norm": s.logit_norm,
                    "entropy": s.entropy,
                    "top1_prob": s.top1_prob,
                    "top1_token": s.top1_token,
                    "intervention_active": s.intervention_active,
                    "diagnostics": s.diagnostics,
                }
                for s in self.states
            ],
        }


def compute_entropy(logits: torch.Tensor) -> float:
    logits = logits.float()
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return float(entropy.mean().item())


def compute_top1(logits: torch.Tensor) -> Tuple[float, int]:
    probs = torch.softmax(logits.float(), dim=-1)
    top1_prob, top1_idx = probs.max(dim=-1)
    return float(top1_prob.item()), int(top1_idx.item())
