from __future__ import annotations

from typing import Optional

import torch

from runtime_lab.core.interventions.base import Intervention


class HiddenInterventionHook:
    def __init__(self, intervention: Optional[Intervention] = None):
        self.intervention = intervention
        self.active = False
        self.last_hidden_pre: Optional[torch.Tensor] = None
        self.last_hidden_post: Optional[torch.Tensor] = None

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs_out = hs

        hidden_last_pre = hs[:, -1, :].detach()

        if self.active and self.intervention is not None:
            hs_out = self.intervention.apply(hs)

        hidden_last_post = hs_out[:, -1, :].detach()

        self.last_hidden_pre = hidden_last_pre.cpu()
        self.last_hidden_post = hidden_last_post.cpu()

        if isinstance(output, tuple):
            return (hs_out,) + output[1:]
        return hs_out

    def set_active(self, active: bool) -> None:
        self.active = bool(active)

    def reset(self) -> None:
        self.active = False
        self.last_hidden_pre = None
        self.last_hidden_post = None


class HiddenCaptureHook:
    def __init__(self):
        self.last_hidden: Optional[torch.Tensor] = None

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        self.last_hidden = hs[:, -1, :].detach().cpu()
        return output

    def reset(self) -> None:
        self.last_hidden = None
