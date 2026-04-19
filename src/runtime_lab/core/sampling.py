"""Centralized token sampling + distribution-shift metrics.

Two concerns:

1. **Token selection**: `sample_token_id(logits, temperature, top_p, top_k)` —
   replaces hard `argmax` calls in the generation loop. Default temperature=0
   reproduces legacy greedy behavior exactly.

2. **Logit-level divergence**: `logit_kl(a, b)` — KL divergence in nats between
   two token logit distributions. Gives a *continuous* measure of how much a
   perturbation shifted the model's decision, even when argmax is invariant.
   This is what makes downstream advisories meaningful: a stress run can have
   token_match_rate=1.0 (no token flips) but logit_kl_mean=0.12 (perturbation
   had measurable effect).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def sample_token_id(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> int:
    """Return a single token id from `logits` (shape [V] or [..., V]).

    temperature <= 0 is treated as greedy (argmax). Otherwise logits are divided
    by temperature, optionally filtered by top_k / top_p, and sampled via
    multinomial.
    """
    if logits.dim() > 1:
        logits = logits.reshape(-1, logits.shape[-1])[-1]

    if temperature is None or temperature <= 0.0:
        return int(logits.argmax(dim=-1).item())

    x = logits.detach().float() / float(temperature)

    if top_k and top_k > 0 and top_k < x.numel():
        topk_vals, _ = torch.topk(x, int(top_k))
        threshold = topk_vals[-1]
        x = torch.where(x < threshold, torch.full_like(x, float("-inf")), x)

    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(x, descending=True)
        cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        keep = cum_probs <= top_p
        # Always keep at least one token
        keep[0] = True
        # Zero out those we drop (in sorted space)
        sorted_logits = torch.where(keep, sorted_logits, torch.full_like(sorted_logits, float("-inf")))
        # Scatter back
        x = torch.empty_like(x).fill_(float("-inf"))
        x.scatter_(0, sorted_idx, sorted_logits)

    probs = torch.softmax(x, dim=-1)
    idx = int(torch.multinomial(probs, num_samples=1).item())
    return idx


def logit_kl(a: torch.Tensor, b: torch.Tensor) -> float:
    """KL(P_a || P_b) in nats between two 1-D logit vectors.

    Uses log_softmax for numerical stability. Returns a Python float. Inputs
    must be the same vocab size; broadcasting is not supported.
    """
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    if a.numel() != b.numel():
        raise ValueError(f"logit_kl expects same vocab size, got {a.numel()} vs {b.numel()}")
    log_p = torch.log_softmax(a, dim=-1)
    log_q = torch.log_softmax(b, dim=-1)
    p = log_p.exp()
    # KL = sum p * (log_p - log_q)
    kl = torch.sum(p * (log_p - log_q)).item()
    # Clamp tiny negatives from numerical noise
    if kl < 0 and kl > -1e-6:
        kl = 0.0
    return float(kl)


def logit_jensen_shannon(a: torch.Tensor, b: torch.Tensor) -> float:
    """Jensen-Shannon divergence in nats; symmetric, bounded by ln(2)."""
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    p = torch.softmax(a, dim=-1)
    q = torch.softmax(b, dim=-1)
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp_min(1e-12))
    log_p = torch.log(p.clamp_min(1e-12))
    log_q = torch.log(q.clamp_min(1e-12))
    js = 0.5 * (p * (log_p - log_m)).sum() + 0.5 * (q * (log_q - log_m)).sum()
    return float(js.item())


def describe_sampling(temperature: float, top_p: float = 1.0, top_k: int = 0) -> str:
    if temperature is None or temperature <= 0.0:
        return "greedy (temperature=0)"
    parts = [f"temperature={temperature:g}"]
    if top_p and 0 < top_p < 1:
        parts.append(f"top_p={top_p:g}")
    if top_k and top_k > 0:
        parts.append(f"top_k={top_k}")
    return ", ".join(parts)
