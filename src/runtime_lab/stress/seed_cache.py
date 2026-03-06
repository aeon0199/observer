from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from runtime_lab.core.model.cache_utils import clone_past_key_values, compute_cache_fingerprint
from runtime_lab.core.model.layers import resolve_layer_index, resolve_transformer_layers
from runtime_lab.core.runtime.hooks import HiddenCaptureHook


@dataclass
class SeedCache:
    past_key_values: Any
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    seq_len: int
    fingerprint: str
    next_token_logits: torch.Tensor
    seed_hidden: torch.Tensor
    seed_hidden_available: bool
    requested_layer_idx: int
    resolved_layer_idx: int

    def clone(self) -> "SeedCache":
        return SeedCache(
            past_key_values=clone_past_key_values(self.past_key_values),
            input_ids=self.input_ids.clone(),
            attention_mask=self.attention_mask.clone(),
            seq_len=int(self.seq_len),
            fingerprint=str(self.fingerprint),
            next_token_logits=self.next_token_logits.clone(),
            seed_hidden=self.seed_hidden.clone(),
            seed_hidden_available=bool(self.seed_hidden_available),
            requested_layer_idx=int(self.requested_layer_idx),
            resolved_layer_idx=int(self.resolved_layer_idx),
        )


def build_seed_cache(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    intervention_layer: int,
) -> SeedCache:
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    layers = resolve_transformer_layers(model)
    num_layers = len(layers)
    resolved_layer_idx = resolve_layer_index(intervention_layer, num_layers)

    hook = HiddenCaptureHook()
    handle = layers[resolved_layer_idx].register_forward_hook(hook)

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
    finally:
        handle.remove()

    past_key_values = outputs.past_key_values
    fingerprint = compute_cache_fingerprint(past_key_values)
    next_token_logits = outputs.logits[:, -1, :].detach().cpu()

    seed_hidden = hook.last_hidden
    seed_hidden_available = isinstance(seed_hidden, torch.Tensor)
    if seed_hidden is None:
        seed_hidden = torch.zeros((1, 1), dtype=torch.float32)

    return SeedCache(
        past_key_values=past_key_values,
        input_ids=input_ids.detach().cpu(),
        attention_mask=attention_mask.detach().cpu(),
        seq_len=int(input_ids.shape[1]),
        fingerprint=fingerprint,
        next_token_logits=next_token_logits,
        seed_hidden=seed_hidden,
        seed_hidden_available=bool(seed_hidden_available),
        requested_layer_idx=int(intervention_layer),
        resolved_layer_idx=int(resolved_layer_idx),
    )
