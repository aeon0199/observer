# v2/trajectory.py

"""
Trajectory Logging
------------------
Token-by-token trajectory capture for intervention experiments.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class TokenState:
    """State captured at a single token position."""
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
    """Complete trajectory through a generation run."""

    states: List[TokenState] = field(default_factory=list)

    # Internal (not serialized): used for research-grade comparisons.
    _hidden_vecs: List[torch.Tensor] = field(default_factory=list, repr=False)
    _logits: List[torch.Tensor] = field(default_factory=list, repr=False)

    prompt: str = ""
    model_id: str = ""
    intervention_layer: int = -1
    intervention_start: int = -1
    intervention_end: int = -1
    intervention_type: str = ""
    
    def add_state(self, state: TokenState):
        self.states.append(state)
    
    def __len__(self):
        return len(self.states)
    
    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert trajectory to numpy arrays."""
        return {
            "token_idx": np.array([s.token_idx for s in self.states]),
            "hidden_norm": np.array([s.hidden_norm for s in self.states]),
            "logit_norm": np.array([s.logit_norm for s in self.states]),
            "entropy": np.array([s.entropy for s in self.states]),
            "top1_prob": np.array([s.top1_prob for s in self.states]),
            "intervention_active": np.array([s.intervention_active for s in self.states]),
        }
    
    def to_json(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "prompt": self.prompt,
            "model_id": self.model_id,
            "intervention": {
                "layer": self.intervention_layer,
                "start_token": self.intervention_start,
                "end_token": self.intervention_end,
                "type": self.intervention_type
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
            ]
        }


_JS_EPS = 1e-12


def _softmax_np(x: np.ndarray) -> np.ndarray:
    """Stable softmax for 1D arrays."""
    x = x.astype(np.float64, copy=False)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + _JS_EPS)


def js_divergence_from_logits_bits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """Jensen–Shannon divergence in *bits* between next-token distributions."""
    a = logits_a.detach().float().cpu().numpy().reshape(-1)
    b = logits_b.detach().float().cpu().numpy().reshape(-1)

    p = _softmax_np(a)
    q = _softmax_np(b)
    m = 0.5 * (p + q)

    kl_pm = float(np.sum(p * (np.log2(p + _JS_EPS) - np.log2(m + _JS_EPS))))
    kl_qm = float(np.sum(q * (np.log2(q + _JS_EPS) - np.log2(m + _JS_EPS))))

    return 0.5 * (kl_pm + kl_qm)


@dataclass
class TrajectoryComparison:
    """Comparison between baseline and intervention trajectories."""

    baseline: Trajectory
    intervention: Trajectory

    # Primary summary metrics (computed from the best available distance).
    primary_metric: str = "hidden_cosine"
    deviation_during: float = 0.0
    recovery_after: float = 0.0
    convergence_rate: float = 0.0
    final_distance: float = 0.0
    recovery_ratio: float = 0.0

    # Additional diagnostics.
    token_match_rate: float = 0.0
    first_token_divergence: Optional[int] = None

    # Rich per-token metrics (JSON-friendly scalars).
    per_token: Dict[str, List[float]] = field(default_factory=dict)

    # Rich summaries for each metric family.
    summary: Dict[str, Any] = field(default_factory=dict)

    def compute_metrics(self):
        """Compute comparison metrics.

        Research-grade notes:
        - hidden_norm alone is not sufficient (same norm can hide large vector changes)
        - we prefer cosine / relative L2 on the captured hidden vector
        - we also compute JS divergence between logits distributions when available
        """

        min_len = min(len(self.baseline.states), len(self.intervention.states))
        if min_len == 0:
            return

        # Token-level match stats.
        base_ids = [s.token_id for s in self.baseline.states[:min_len]]
        intv_ids = [s.token_id for s in self.intervention.states[:min_len]]
        matches = [int(a == b) for a, b in zip(base_ids, intv_ids)]
        self.token_match_rate = float(sum(matches) / max(1, len(matches)))
        self.first_token_divergence = next((i for i, m in enumerate(matches) if m == 0), None)

        # Compute per-token distances (best-effort depending on captured internals).
        hidden_cos = None
        hidden_l2_rel = None
        js_bits = None

        if len(self.baseline._hidden_vecs) >= min_len and len(self.intervention._hidden_vecs) >= min_len:
            eps = 1e-12
            hidden_cos = []
            hidden_l2_rel = []
            for i in range(min_len):
                b_raw = self.baseline._hidden_vecs[i]
                v_raw = self.intervention._hidden_vecs[i]

                # If hooks failed and we inserted placeholders, treat vectors as unavailable.
                if not isinstance(b_raw, torch.Tensor) or not isinstance(v_raw, torch.Tensor):
                    hidden_cos = None
                    hidden_l2_rel = None
                    break
                if b_raw.numel() <= 1 or v_raw.numel() <= 1:
                    hidden_cos = None
                    hidden_l2_rel = None
                    break

                b = b_raw.float().view(-1)
                v = v_raw.float().view(-1)
                bn = float(torch.linalg.vector_norm(b).item())
                vn = float(torch.linalg.vector_norm(v).item())
                if bn < eps or vn < eps:
                    hidden_cos = None
                    hidden_l2_rel = None
                    break

                dot = float(torch.dot(b, v).item())
                cos = dot / (bn * vn + eps)
                # Numerical safety: keep cosine in [-1, 1] so 1-cos is non-negative.
                cos = float(max(-1.0, min(1.0, cos)))
                hidden_cos.append(float(max(0.0, 1.0 - cos)))

                l2 = float(torch.linalg.vector_norm(b - v).item())
                hidden_l2_rel.append(float(l2 / (bn + eps)))

        if len(self.baseline._logits) >= min_len and len(self.intervention._logits) >= min_len:
            js_bits = [
                float(js_divergence_from_logits_bits(self.baseline._logits[i], self.intervention._logits[i]))
                for i in range(min_len)
            ]

        # Fallback: normalized absolute hidden_norm differences.
        base_arr = self.baseline.to_arrays()
        intv_arr = self.intervention.to_arrays()
        hn_dist = np.abs(base_arr["hidden_norm"][:min_len] - intv_arr["hidden_norm"][:min_len])
        hn_mean = float(np.mean(base_arr["hidden_norm"][:min_len]))
        if hn_mean > 0:
            hn_dist = hn_dist / hn_mean
        hn_dist = [float(x) for x in hn_dist]

        # Choose primary metric.
        if hidden_cos is not None:
            self.primary_metric = "hidden_cosine"
            primary = hidden_cos
        elif js_bits is not None:
            self.primary_metric = "logit_js_bits"
            primary = js_bits
        else:
            self.primary_metric = "hidden_norm_rel"
            primary = hn_dist

        # Store per-token series.
        self.per_token = {"primary": primary, "hidden_norm_rel": hn_dist}
        if hidden_cos is not None:
            self.per_token["hidden_cosine"] = hidden_cos
        if hidden_l2_rel is not None:
            self.per_token["hidden_l2_rel"] = hidden_l2_rel
        if js_bits is not None:
            self.per_token["logit_js_bits"] = js_bits

        start = int(self.intervention.intervention_start)
        end = int(self.intervention.intervention_end)

        def _summarize(series: List[float]) -> Dict[str, float]:
            arr = np.array(series, dtype=np.float64)
            out: Dict[str, float] = {
                "mean_all": float(np.mean(arr)),
                "final": float(arr[-1]) if len(arr) else 0.0,
            }

            if 0 <= start < len(arr) and 0 < end <= len(arr) and start < end:
                during = arr[start:end]
                out["mean_during"] = float(np.mean(during)) if len(during) else 0.0
            else:
                out["mean_during"] = 0.0

            if 0 <= end < len(arr):
                post = arr[end:]
                out["mean_post"] = float(np.mean(post)) if len(post) else 0.0
            else:
                out["mean_post"] = 0.0

            return out

        # Summaries for each metric.
        self.summary = {}
        self.summary["primary"] = {"name": self.primary_metric, **_summarize(primary)}
        if hidden_cos is not None:
            self.summary["hidden_cosine"] = _summarize(hidden_cos)
        if hidden_l2_rel is not None:
            self.summary["hidden_l2_rel"] = _summarize(hidden_l2_rel)
        self.summary["hidden_norm_rel"] = _summarize(hn_dist)
        if js_bits is not None:
            self.summary["logit_js_bits"] = _summarize(js_bits)

        # Compute legacy-style "deviation during" / "recovery" etc on primary.
        primary_arr = np.array(primary, dtype=np.float64)
        if 0 <= start < len(primary_arr) and 0 < end <= len(primary_arr) and start < end:
            during = primary_arr[start:end]
            self.deviation_during = float(np.mean(during)) if len(during) else 0.0
        else:
            self.deviation_during = 0.0

        if 0 <= end < len(primary_arr):
            post = primary_arr[end:]
            self.final_distance = float(post[-1]) if len(post) else float(primary_arr[-1])

            if self.deviation_during > 0:
                self.recovery_after = float(self.deviation_during - self.final_distance)
                self.recovery_ratio = float(self.recovery_after / self.deviation_during)

            if len(post) > 1:
                x = np.arange(len(post))
                slope = float(np.polyfit(x, post, 1)[0])
                self.convergence_rate = float(-slope)

    def to_json(self) -> Dict[str, Any]:
        """Serialize to JSON."""
        return {
            "baseline": self.baseline.to_json(),
            "intervention": self.intervention.to_json(),
            "metrics": {
                "primary_metric": self.primary_metric,
                "deviation_during": self.deviation_during,
                "recovery_after": self.recovery_after,
                "convergence_rate": self.convergence_rate,
                "final_distance": self.final_distance,
                "recovery_ratio": self.recovery_ratio,
                "token_match_rate": self.token_match_rate,
                "first_token_divergence": self.first_token_divergence,
                "summary": self.summary,
                "per_token": self.per_token,
            },
        }


def compute_entropy(logits: torch.Tensor) -> float:
    """Compute entropy from logits."""
    logits = logits.float()
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return float(entropy.mean())


def compute_top1(logits: torch.Tensor) -> Tuple[float, int]:
    """Get top-1 probability and token id."""
    probs = torch.softmax(logits.float(), dim=-1)
    top1_prob, top1_idx = probs.max(dim=-1)
    return float(top1_prob), int(top1_idx)
