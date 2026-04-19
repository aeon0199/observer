from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from runtime_lab.core.diagnostics.manager import DiagnosticsManager
from runtime_lab.core.interventions.base import Intervention
from runtime_lab.core.model.layers import resolve_layer_index, resolve_probe_layers, resolve_transformer_layers
from runtime_lab.core.runtime.events import RuntimeEvent
from runtime_lab.core.runtime.hooks import HiddenCaptureHook, HiddenInterventionHook
from runtime_lab.core.sampling import sample_token_id


@dataclass
class PrefillState:
    prompt: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_len: int
    past_key_values: Any
    logits: torch.Tensor
    next_token_id: int
    resolved_layer_idx: int
    hidden_pre: Optional[torch.Tensor]
    hidden_post: Optional[torch.Tensor]


@dataclass
class StepResult:
    t: int
    consumed_token_id: int
    consumed_token_text: str
    predicted_next_token_id: int
    logits: torch.Tensor
    hidden_pre: Optional[torch.Tensor]
    hidden_post: Optional[torch.Tensor]
    diagnostics: Dict[str, Any]
    event: RuntimeEvent
    past_key_values: Any


class RuntimeEngine:
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        layer_idx: int,
        diagnostics_manager: Optional[DiagnosticsManager] = None,
        intervention: Optional[Intervention] = None,
        probe_layers: Optional[list[int]] = None,
        mode: str = "runtime",
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.mode = str(mode)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)

        self.layers = resolve_transformer_layers(model)
        self.num_layers = len(self.layers)
        self.resolved_layer_idx = resolve_layer_index(layer_idx, self.num_layers)
        self.probe_resolved = resolve_probe_layers(probe_layers or [], self.num_layers)

        self.diagnostics_manager = diagnostics_manager
        self.intervention_hook = HiddenInterventionHook(intervention=intervention)
        self.capture_hooks: Dict[int, HiddenCaptureHook] = {}
        self._capture_handles = []

        self._main_handle = self.layers[self.resolved_layer_idx].register_forward_hook(self.intervention_hook)

        for raw_idx, resolved_idx in self.probe_resolved.items():
            if resolved_idx == self.resolved_layer_idx:
                continue
            cap = HiddenCaptureHook()
            self.capture_hooks[int(raw_idx)] = cap
            self._capture_handles.append(self.layers[resolved_idx].register_forward_hook(cap))

    def close(self) -> None:
        try:
            self._main_handle.remove()
        except Exception:
            pass
        for handle in self._capture_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._capture_handles.clear()

    def reset(self) -> None:
        self.intervention_hook.reset()
        for cap in self.capture_hooks.values():
            cap.reset()
        if self.diagnostics_manager is not None:
            self.diagnostics_manager.reset()

    def prefill(
        self,
        prompt: str,
        intervention_active: bool = False,
    ) -> PrefillState:
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        prompt_len = int(input_ids.shape[1])

        self.reset()
        self.intervention_hook.set_active(bool(intervention_active))

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )

        logits = out.logits[:, -1, :].detach().cpu()
        next_token_id = sample_token_id(logits, self.temperature, self.top_p, self.top_k)

        return PrefillState(
            prompt=prompt,
            input_ids=input_ids.detach().cpu(),
            attention_mask=attention_mask.detach().cpu(),
            prompt_len=prompt_len,
            past_key_values=out.past_key_values,
            logits=logits,
            next_token_id=next_token_id,
            resolved_layer_idx=self.resolved_layer_idx,
            hidden_pre=self.intervention_hook.last_hidden_pre,
            hidden_post=self.intervention_hook.last_hidden_post,
        )

    def step(
        self,
        t: int,
        consumed_token_id: int,
        prompt_len: int,
        past_key_values: Any,
        intervention_active: bool = False,
    ) -> StepResult:
        self.intervention_hook.set_active(bool(intervention_active))
        for cap in self.capture_hooks.values():
            cap.reset()

        token_tensor = torch.tensor([[int(consumed_token_id)]], device=self.device)
        attn = torch.ones((1, prompt_len + t + 1), device=self.device, dtype=torch.long)

        with torch.no_grad():
            out = self.model(
                input_ids=token_tensor,
                attention_mask=attn,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        logits = out.logits[:, -1, :].detach().cpu()
        predicted_next_token_id = sample_token_id(logits, self.temperature, self.top_p, self.top_k)

        hidden_pre = self.intervention_hook.last_hidden_pre
        hidden_post = self.intervention_hook.last_hidden_post

        layer_states: Dict[int, torch.Tensor] = {}
        for raw_idx, cap in self.capture_hooks.items():
            if cap.last_hidden is not None:
                layer_states[int(raw_idx)] = cap.last_hidden
        for raw_idx, resolved_idx in self.probe_resolved.items():
            if resolved_idx == self.resolved_layer_idx and hidden_post is not None:
                layer_states[int(raw_idx)] = hidden_post

        diagnostics = {}
        if self.diagnostics_manager is not None and hidden_post is not None:
            diagnostics = self.diagnostics_manager.step(hidden_post, layer_states=layer_states)

        pre_norm = float(hidden_pre.norm().item()) if hidden_pre is not None else 0.0
        post_norm = float(hidden_post.norm().item()) if hidden_post is not None else 0.0
        delta_norm = (
            float((hidden_post - hidden_pre).norm().item())
            if hidden_pre is not None and hidden_post is not None
            else 0.0
        )

        token_text = self.tokenizer.decode([int(consumed_token_id)], skip_special_tokens=False)

        event = RuntimeEvent(
            t=int(t),
            consumed_token_id=int(consumed_token_id),
            consumed_token_text=token_text,
            predicted_next_token_id=int(predicted_next_token_id),
            resolved_layer_idx=int(self.resolved_layer_idx),
            hidden_pre_norm=pre_norm,
            hidden_post_norm=post_norm,
            hidden_delta_norm=delta_norm,
            diagnostics=diagnostics,
            intervention_active=bool(intervention_active),
            mode=self.mode,
        )

        return StepResult(
            t=int(t),
            consumed_token_id=int(consumed_token_id),
            consumed_token_text=token_text,
            predicted_next_token_id=int(predicted_next_token_id),
            logits=logits,
            hidden_pre=hidden_pre,
            hidden_post=hidden_post,
            diagnostics=diagnostics,
            event=event,
            past_key_values=out.past_key_values,
        )
