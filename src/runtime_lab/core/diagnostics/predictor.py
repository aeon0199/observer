from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


_EPS = 1e-8
_MAX_ABS = 1e4


def _to_1d_np(x: torch.Tensor) -> np.ndarray:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    return x.detach().float().cpu().view(-1).numpy().astype(np.float32, copy=False)


def _rademacher_projection(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.integers(0, 2, size=(in_dim, out_dim), dtype=np.int8)
    m = (m * 2 - 1).astype(np.float32)
    m /= np.sqrt(float(out_dim))
    return m


def _sanitize_array(x: np.ndarray) -> tuple[np.ndarray, Dict[str, float | bool]]:
    arr = np.asarray(x, dtype=np.float32)
    had_nonfinite = bool(not np.isfinite(arr).all())
    max_abs_before = float(np.max(np.abs(arr))) if arr.size else 0.0
    clipped = bool(max_abs_before > _MAX_ABS)

    if had_nonfinite:
        arr = np.nan_to_num(arr, nan=0.0, posinf=_MAX_ABS, neginf=-_MAX_ABS)
    if clipped:
        arr = np.clip(arr, -_MAX_ABS, _MAX_ABS)

    return arr.astype(np.float32, copy=False), {
        "had_nonfinite": had_nonfinite,
        "clipped": clipped,
        "max_abs_before": max_abs_before,
    }


@dataclass
class DivergenceDetails:
    n: int
    proj_dim: int
    l2: float
    cosine: float
    combined: float


class StateWindow:
    def __init__(self, maxlen: int):
        if maxlen < 2:
            raise ValueError("StateWindow maxlen must be >= 2")
        self.maxlen = int(maxlen)
        self._buf: List[np.ndarray] = []

    def reset(self) -> None:
        self._buf.clear()

    def add(self, v: np.ndarray) -> None:
        if v.ndim != 1:
            v = v.reshape(-1)
        self._buf.append(v.astype(np.float32, copy=False))
        if len(self._buf) > self.maxlen:
            self._buf.pop(0)

    def __len__(self) -> int:
        return len(self._buf)

    def matrix(self) -> np.ndarray:
        if not self._buf:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(self._buf, axis=0)


def _fit_var1_ridge(states: np.ndarray, ridge: float) -> np.ndarray:
    x = np.asarray(states[:-1, :], dtype=np.float64)
    y = np.asarray(states[1:, :], dtype=np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        xtx = x.T @ x
    k = xtx.shape[0]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        rhs = x.T @ y
    system = xtx + float(ridge) * np.eye(k, dtype=np.float64)
    try:
        solved = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        solved = np.linalg.lstsq(system, rhs, rcond=None)[0]
    return solved.astype(np.float32, copy=False)


def _divergence(pred: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(np.float64, copy=False)
    actual = actual.astype(np.float64, copy=False)

    diff = pred - actual
    pred_n = float(np.linalg.norm(pred))
    act_n = float(np.linalg.norm(actual))

    denom_l2 = 0.5 * (pred_n + act_n) + _EPS
    denom_l2 = float(max(denom_l2, 1e-3))
    l2 = float(np.linalg.norm(diff) / denom_l2)

    denom = (pred_n * act_n) + _EPS
    cosine = float(1.0 - float(np.dot(pred, actual) / denom))
    cosine = float(np.clip(cosine, 0.0, 2.0))

    combined = float(0.7 * l2 + 0.3 * cosine)
    return {"l2": l2, "cosine": cosine, "combined": combined}


class DivergencePredictor:
    def __init__(
        self,
        window_size: int = 8,
        proj_dim: int = 64,
        proj_seed: int = 0,
        ridge: float = 1e-2,
    ):
        if window_size < 3:
            raise ValueError("window_size must be >= 3")
        if proj_dim < 4:
            raise ValueError("proj_dim must be >= 4")

        self.window_size = int(window_size)
        self.proj_dim = int(proj_dim)
        self.proj_seed = int(proj_seed)
        self.ridge = float(ridge)

        self._window = StateWindow(self.window_size)
        self._proj: Optional[np.ndarray] = None
        self.last_details: Optional[DivergenceDetails] = None
        self.last_health: Dict[str, object] = {"valid": True, "degraded": False, "issues": []}

    def reset(self) -> None:
        self._window.reset()
        self.last_details = None
        self.last_health = {"valid": True, "degraded": False, "issues": []}

    def _reset_health(self) -> None:
        self.last_health = {"valid": True, "degraded": False, "issues": []}

    def _mark_issue(self, issue: str, **details: object) -> None:
        self.last_health["degraded"] = True
        issues = self.last_health.setdefault("issues", [])
        payload: Dict[str, object] = {"issue": issue}
        payload.update(details)
        issues.append(payload)

    def _project(self, hidden: torch.Tensor) -> np.ndarray:
        h = _to_1d_np(hidden)
        h, h_meta = _sanitize_array(h)
        if h_meta["had_nonfinite"] or h_meta["clipped"]:
            self._mark_issue("hidden_state_sanitized", **h_meta)
        if self._proj is None or self._proj.shape[0] != h.shape[0]:
            self._proj = _rademacher_projection(h.shape[0], self.proj_dim, self.proj_seed)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            z = h @ self._proj
        z, z_meta = _sanitize_array(z)
        if z_meta["had_nonfinite"] or z_meta["clipped"]:
            self._mark_issue("projection_sanitized", **z_meta)
        return z.astype(np.float32, copy=False)

    def step(self, hidden: torch.Tensor) -> float:
        self._reset_health()
        z = self._project(hidden)
        self._window.add(z)

        if len(self._window) < 3:
            self.last_details = DivergenceDetails(
                n=len(self._window),
                proj_dim=self.proj_dim,
                l2=0.0,
                cosine=0.0,
                combined=0.0,
            )
            return 0.0

        states = self._window.matrix()
        states, states_meta = _sanitize_array(states)
        if states_meta["had_nonfinite"] or states_meta["clipped"]:
            self._mark_issue("window_sanitized", **states_meta)
        train_states = states[:-1, :]
        a = _fit_var1_ridge(train_states, ridge=self.ridge)
        a, a_meta = _sanitize_array(a)
        if a_meta["had_nonfinite"] or a_meta["clipped"]:
            self._mark_issue("var_weights_sanitized", **a_meta)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            pred = states[-2, :] @ a
        pred, pred_meta = _sanitize_array(pred)
        if pred_meta["had_nonfinite"] or pred_meta["clipped"]:
            self._mark_issue("prediction_sanitized", **pred_meta)
        actual = states[-1, :]

        d = _divergence(pred, actual)
        if not np.isfinite(d["combined"]):
            self.last_health["valid"] = False
            self._mark_issue("divergence_nonfinite", combined=d["combined"])
            d = {"l2": 0.0, "cosine": 0.0, "combined": 0.0}
        self.last_details = DivergenceDetails(
            n=int(states.shape[0]),
            proj_dim=self.proj_dim,
            l2=float(d["l2"]),
            cosine=float(d["cosine"]),
            combined=float(d["combined"]),
        )
        return float(d["combined"])
