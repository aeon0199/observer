from __future__ import annotations

from dataclasses import dataclass

from .controller import ControllerState


@dataclass
class ScalingPolicy:
    scale_warn: float = 0.90
    scale_crit: float = 0.75

    def __post_init__(self):
        self.scale_warn = min(1.0, max(0.0, float(self.scale_warn)))
        self.scale_crit = min(1.0, max(0.0, float(self.scale_crit)))
        if self.scale_crit >= self.scale_warn:
            self.scale_crit = max(0.0, self.scale_warn - 0.05)

    def next_scale(self, controller_state: ControllerState) -> float:
        status = str(controller_state.status)

        if status == "CRITICAL":
            return float(self.scale_crit)
        if status in ("WARNING", "COOLDOWN"):
            return float(self.scale_warn)
        return 1.0
