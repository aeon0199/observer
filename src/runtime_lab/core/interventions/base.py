from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Intervention(ABC):
    name: str = "intervention"

    @abstractmethod
    def apply(self, hidden_state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
