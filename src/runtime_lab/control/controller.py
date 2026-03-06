from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ControllerState:
    raw_score: float = 0.0
    avg_score: float = 0.0
    status: str = "STABLE"
    cooldown_counter: int = 0
    prev_effective_rank: Optional[float] = None


class StabilityController:
    def __init__(
        self,
        ma_window: int = 3,
        threshold_warn: float = 0.55,
        threshold_crit: float = 0.85,
        hold_warn: int = 3,
        hold_crit: int = 6,
        weight_div: float = 0.70,
        weight_spec_entropy: float = 0.15,
        weight_high_frac: float = 0.10,
        weight_rank_delta: float = 0.05,
        spec_entropy_floor: float = 0.75,
        high_frac_floor: float = 0.30,
    ):
        self.history = deque(maxlen=int(ma_window))
        self.state = ControllerState()

        self.TH_WARN = float(threshold_warn)
        self.TH_CRIT = float(threshold_crit)
        self.HOLD_WARN = int(hold_warn)
        self.HOLD_CRIT = int(hold_crit)

        self.W_DIV = float(weight_div)
        self.W_SPEC_ENT = float(weight_spec_entropy)
        self.W_HIGH_FRAC = float(weight_high_frac)
        self.W_RANK_DELTA = float(weight_rank_delta)

        self.SPEC_ENT_FLOOR = float(spec_entropy_floor)
        self.HIGH_FRAC_FLOOR = float(high_frac_floor)

    def reset(self) -> None:
        self.history.clear()
        self.state = ControllerState()

    def _score(self, diagnostics: Dict[str, Any]) -> float:
        div = float(diagnostics.get("divergence", 0.0))
        spectral = diagnostics.get("spectral", {}) or {}
        svd = diagnostics.get("svd", {}) or {}

        spec_entropy = float(spectral.get("spectral_entropy", 0.0))
        high_frac = float(spectral.get("high_frac", 0.0))
        eff_rank = float(svd.get("effective_rank", 0.0))

        rank_delta = 0.0
        if self.state.prev_effective_rank is not None:
            rank_delta = abs(eff_rank - self.state.prev_effective_rank)
        self.state.prev_effective_rank = eff_rank

        spec_term = max(0.0, spec_entropy - self.SPEC_ENT_FLOOR)
        high_term = max(0.0, high_frac - self.HIGH_FRAC_FLOOR)

        score = (
            self.W_DIV * div
            + self.W_SPEC_ENT * spec_term
            + self.W_HIGH_FRAC * high_term
            + self.W_RANK_DELTA * rank_delta
        )
        return float(max(0.0, min(2.0, score)))

    def update(self, diagnostics: Dict[str, Any]) -> ControllerState:
        raw = self._score(diagnostics)
        self.history.append(float(raw))
        avg = float(sum(self.history) / max(1, len(self.history)))

        self.state.raw_score = float(raw)
        self.state.avg_score = float(avg)

        if self.state.cooldown_counter > 0:
            self.state.cooldown_counter -= 1
            self.state.status = "COOLDOWN"
            return self.state

        if avg > self.TH_CRIT:
            self.state.status = "CRITICAL"
            self.state.cooldown_counter = int(self.HOLD_CRIT)
            return self.state

        if avg > self.TH_WARN:
            self.state.status = "WARNING"
            self.state.cooldown_counter = int(self.HOLD_WARN)
            return self.state

        self.state.status = "STABLE"
        return self.state
