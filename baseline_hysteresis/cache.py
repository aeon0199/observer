# cot_runner/cot_cache.py

"""
COT Cache Management
--------------------
Handles creation and cloning of KV cache states for branching experiments.

The seed cache represents the model's state after processing prompt P,
before any generation. BASE and PERTURB runs can both start from
identical copies of this seed.
"""

import copy
import torch
from typing import Any, Optional
from dataclasses import dataclass


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


@dataclass
class SeedCache:
    """
    Immutable snapshot of model state after processing a prompt.

    This is the branching point for BASE and PERTURB runs (same pre-generation state).

    Notes:
    - past_key_values is the KV cache after the prompt tokens were processed.
    - next_token_logits is outputs.logits[:, -1, :] from that same prompt pass.
    - seed_hidden is the final-token hidden state captured during that prompt pass.
    """
    past_key_values: Any
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    seq_len: int
    fingerprint: str

    # For starting generation without re-running the prompt.
    next_token_logits: torch.Tensor
    seed_hidden: torch.Tensor

    def clone(self) -> 'SeedCache':
        """Create a deep copy safe for independent generation."""
        return SeedCache(
            past_key_values=clone_past_key_values(self.past_key_values),
            input_ids=self.input_ids.clone(),
            attention_mask=self.attention_mask.clone(),
            seq_len=self.seq_len,
            fingerprint=self.fingerprint,
            next_token_logits=self.next_token_logits.clone(),
            seed_hidden=self.seed_hidden.clone(),
        )


def compute_cache_fingerprint(past_key_values) -> str:
    """Compute a lightweight fingerprint for cache verification.

    Supports common HF cache formats:
    - legacy tuple-of-tuples: past_key_values[layer][0] is K
    - transformers DynamicCache: past_key_values.key_cache[0] is K

    Used to confirm BASE and PERTURB runs start from identical states.
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


def build_seed_cache(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    hook_mgr=None,
) -> SeedCache:
    """
    Build the seed cache by running a forward pass on the prompt.
    
    This processes the prompt through the model WITHOUT generating
    any new tokens, capturing the KV cache state that represents
    "the model has read P and is ready to continue."
    """
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    if hook_mgr is not None:
        try:
            hook_mgr.reset()
        except Exception:
            pass

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

    past_key_values = outputs.past_key_values
    fingerprint = compute_cache_fingerprint(past_key_values)

    # This is the model's next-token distribution after the full prompt.
    next_token_logits = outputs.logits[:, -1, :].detach().cpu()

    # Prefer hook-captured final-token hidden state if available.
    seed_hidden = None
    if hook_mgr is not None and getattr(hook_mgr, "hidden_hook", None) is not None:
        seed_hidden = getattr(hook_mgr.hidden_hook, "captured", None)
    if seed_hidden is None:
        # Fallback: use zeros (keeps pipeline robust if hooks are unavailable).
        # Runner will still compute metrics from logits.
        seed_hidden = torch.zeros((1, 1), dtype=torch.float32)

    print(f"[COT] Seed cache built. Fingerprint: {fingerprint}")
    print(f"[COT] Sequence length: {input_ids.shape[1]} tokens")

    return SeedCache(
        past_key_values=past_key_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        seq_len=input_ids.shape[1],
        fingerprint=fingerprint,
        next_token_logits=next_token_logits,
        seed_hidden=seed_hidden,
    )


def verify_cache_match(cache_a: SeedCache, cache_b: SeedCache) -> bool:
    """
    Verify two caches have identical fingerprints.
    """
    match = cache_a.fingerprint == cache_b.fingerprint
    if not match:
        print(f"[COT] WARNING: Cache fingerprint mismatch!")
        print(f"[COT]   Cache A: {cache_a.fingerprint}")
        print(f"[COT]   Cache B: {cache_b.fingerprint}")
    return match
