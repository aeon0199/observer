# v2/cache.py

"""V2 — SeedCache (prompt-pass branchpoint)

For research-grade intervention experiments we want BASELINE and INTERVENTION
runs to start from an identical pre-generation internal state.

This module builds an immutable snapshot after processing the prompt:
- past_key_values (KV cache)
- next_token_logits (distribution for the first generated token)
- seed_hidden (final-token hidden state at the chosen intervention layer)

This mirrors the "branchpoint" rigor from V1/V1.5.
"""

import copy
from dataclasses import dataclass
from typing import Any

import torch


def clone_past_key_values(past_key_values: Any) -> Any:
    """Deep-clone a HF KV cache.

    HF cache formats vary by Transformers version / model:
    - legacy tuple-of-tuples
    - DynamicCache / other cache objects

    We want a clone that is safe for independent continuation.
    """
    if past_key_values is None:
        return None

    # 1) DynamicCache-like objects (best-effort manual clone)
    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        new_cache = None
        try:
            new_cache = past_key_values.__class__()
        except Exception:
            new_cache = None

        if new_cache is not None:
            try:
                new_cache.key_cache = [t.clone() for t in list(key_cache)]
                new_cache.value_cache = [t.clone() for t in list(value_cache)]

                # Preserve common metadata fields (best-effort).
                for attr in ("seen_tokens", "_seen_tokens"):
                    if hasattr(past_key_values, attr):
                        try:
                            setattr(new_cache, attr, getattr(past_key_values, attr))
                        except Exception:
                            pass

                return new_cache
            except Exception:
                # Fall through to deepcopy.
                pass

    # 2) Generic deep copy (works for legacy tuple caches and many objects)
    try:
        return copy.deepcopy(past_key_values)
    except Exception:
        pass

    # 3) Last-resort: recursively clone tensors in common containers
    def _clone(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.clone()
        if isinstance(x, tuple):
            return tuple(_clone(i) for i in x)
        if isinstance(x, list):
            return [_clone(i) for i in x]
        if isinstance(x, dict):
            return {k: _clone(v) for k, v in x.items()}
        return x

    return _clone(past_key_values)


def compute_cache_fingerprint(past_key_values: Any) -> str:
    """Compute a lightweight fingerprint for cache verification.

    Supports common HF cache formats:
    - legacy tuple-of-tuples: past_key_values[layer][0] is K
    - transformers DynamicCache: past_key_values.key_cache[0] is K
    """
    first_layer_k = None

    # Legacy: tuple-of-tuples
    try:
        first_layer_k = past_key_values[0][0]
    except Exception:
        first_layer_k = None

    # DynamicCache (best-effort)
    if first_layer_k is None:
        try:
            first_layer_k = past_key_values.key_cache[0]
        except Exception:
            first_layer_k = None

    if first_layer_k is None:
        return "unavailable"

    fingerprint_sum = float(first_layer_k.sum().item())
    fingerprint_mean = float(first_layer_k.mean().item())
    fingerprint_std = float(first_layer_k.std().item())

    return f"{fingerprint_sum:.6f}_{fingerprint_mean:.6f}_{fingerprint_std:.6f}"


def get_model_layers(model):
    """Best-effort access to the transformer block list."""
    # Most Llama/Qwen-style HF CausalLMs expose model.model.layers
    try:
        return model.model.layers
    except Exception:
        pass

    # Common alternatives
    try:
        return model.transformer.h
    except Exception:
        pass

    raise AttributeError(
        "[V2] Unsupported model structure: couldn't find transformer layers at "
        "model.model.layers or model.transformer.h"
    )


@dataclass
class SeedCache:
    """Immutable snapshot of model state after processing the prompt."""

    past_key_values: Any
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    seq_len: int
    fingerprint: str

    # For starting generation without re-running the prompt.
    next_token_logits: torch.Tensor
    seed_hidden: torch.Tensor

    def clone(self) -> "SeedCache":
        """Create a deep copy safe for independent continuation."""
        return SeedCache(
            past_key_values=clone_past_key_values(self.past_key_values),
            input_ids=self.input_ids.clone(),
            attention_mask=self.attention_mask.clone(),
            seq_len=self.seq_len,
            fingerprint=self.fingerprint,
            next_token_logits=self.next_token_logits.clone(),
            seed_hidden=self.seed_hidden.clone(),
        )


class _HiddenCaptureHook:
    """Capture final-token hidden state at a layer during a forward pass."""

    def __init__(self):
        self.captured = None

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        # Capture only the final token of the sequence.
        self.captured = hs[:, -1, :].detach().cpu().clone()
        return output


def build_seed_cache(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    intervention_layer: int,
) -> SeedCache:
    """Run the prompt once to build a pre-generation branchpoint."""

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    layers = get_model_layers(model)
    num_layers = len(layers)
    actual_layer = intervention_layer if intervention_layer >= 0 else num_layers + intervention_layer

    if not (0 <= actual_layer < num_layers):
        raise IndexError(
            f"[V2] intervention_layer out of range: {intervention_layer} (resolved to {actual_layer}, num_layers={num_layers})"
        )

    hook = _HiddenCaptureHook()
    handle = layers[actual_layer].register_forward_hook(hook)

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

    # Distribution for the first generated token.
    next_token_logits = outputs.logits[:, -1, :].detach().cpu().clone()

    seed_hidden = hook.captured
    if seed_hidden is None:
        # Keep pipeline robust if hooks are unavailable.
        seed_hidden = torch.zeros((1, 1), dtype=torch.float32)

    return SeedCache(
        past_key_values=past_key_values,
        input_ids=input_ids.detach().cpu(),
        attention_mask=attention_mask.detach().cpu(),
        seq_len=int(input_ids.shape[1]),
        fingerprint=fingerprint,
        next_token_logits=next_token_logits,
        seed_hidden=seed_hidden,
    )
