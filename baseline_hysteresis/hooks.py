# v1/hooks.py

"""
Telemetry Hooks
---------------
Forward hooks for capturing hidden states and logits.
"""

import torch


class HiddenStateCapture:
    """Hook that captures the final-token hidden state."""
    
    def __init__(self):
        self.captured = None

    def __call__(self, module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        self.captured = hs[:, -1, :].detach().cpu()

    def reset(self):
        self.captured = None


class LogitCapture:
    """Hook that captures final-token logits."""
    
    def __init__(self):
        self.captured = None

    def __call__(self, module, input, output):
        self.captured = output[:, -1, :].detach().cpu()

    def reset(self):
        self.captured = None


class HookManager:
    """Registers and manages hooks on decoder-only HF models."""

    def __init__(self, model):
        self.model = model
        self.hidden_hook = HiddenStateCapture()
        self.logit_hook = LogitCapture()
        self.handles = []

    def register(self):
        """Attach hooks to final transformer block and lm_head."""
        try:
            final_block = self.model.model.layers[-1]
            h1 = final_block.register_forward_hook(self.hidden_hook)
            self.handles.append(h1)
        except Exception as e:
            print(f"[V1] ERROR attaching hidden-state hook: {e}")

        try:
            lm_head = self.model.lm_head
            h2 = lm_head.register_forward_hook(self.logit_hook)
            self.handles.append(h2)
        except Exception as e:
            print(f"[V1] ERROR attaching logit hook: {e}")

        print("[V1] Hooks registered.")

    def reset(self):
        self.hidden_hook.reset()
        self.logit_hook.reset()

    def cleanup(self):
        for h in self.handles:
            h.remove()
        self.handles = []