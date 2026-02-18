# v1/runner.py

"""
V1.3 Runner — Branching Diagnostic with Continuous Hysteresis Test
-------------------------------------------------------------------
Three-stage observer loop with an explicit seed-cache branchpoint:
  1) Build SeedCache after ORIGINAL_QUESTION (pre-generation state).
  2) BASE:    Generate from SeedCache (baseline branch).
  3) PERTURB: Generate from the same SeedCache, after injecting Delta/instructions.
  4) REASK:   Continue from PERTURB's KV cache; minimal re-ask instruction (does not repeat prompt).

Tests whether the system returns toward baseline when re-asked,
while still carrying the perturbation in KV cache.
"""

import os
import time
import random
import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, Optional, Any

import torch
import numpy as np

try:
    from .model_loader import load_model, load_model_registry
    from .utils import basic_stats, svd_compress, save_json
    from .frames import COTFrame
    from .hooks import HookManager
    from .metrics import compute_cot_metrics, format_metrics_summary
    from .cache import build_seed_cache
except ImportError:  # pragma: no cover
    from model_loader import load_model, load_model_registry
    from utils import basic_stats, svd_compress, save_json
    from frames import COTFrame
    from hooks import HookManager
    from metrics import compute_cot_metrics, format_metrics_summary
    from cache import build_seed_cache

# NEW: plotting
try:
    from .plotter import generate_plots
except Exception:
    try:
        from plotter import generate_plots
    except Exception:
        generate_plots = None


class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"


DEFAULT_MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "128"))
RUNS_DIR = os.environ.get("RUNS_DIR", "runs")
VERSION = "1.3"


@dataclass
class StageTelemetry:
    text: str
    stats: Dict
    kv_cache: Optional[Any] = None
    seq_len: int = 0

    # Logits at the matched branchpoint before generation begins.
    # Used for distribution-shift metrics (e.g., JS divergence).
    context_logits: Optional[torch.Tensor] = None


def hash_config(config: dict) -> str:
    """Generate a short hash of experiment config for reproducibility verification."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def set_deterministic_state(seed: int):
    """Set all RNG sources."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_JS_EPS = 1e-12


def _softmax_np(x: np.ndarray) -> np.ndarray:
    """Stable softmax for 1D arrays."""
    x = x.astype(np.float64, copy=False)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + _JS_EPS)


def js_divergence_from_logits_bits(logits_a: Any, logits_b: Any) -> float:
    """Jensen–Shannon divergence in *bits* between next-token distributions.

    Inputs are logits tensors/arrays (any shape with vocab in the last dim).
    Output is a scalar where 0.0 means identical distributions.
    """
    if isinstance(logits_a, torch.Tensor):
        a = logits_a.detach().float().cpu().numpy()
    else:
        a = np.array(logits_a, dtype=np.float32)

    if isinstance(logits_b, torch.Tensor):
        b = logits_b.detach().float().cpu().numpy()
    else:
        b = np.array(logits_b, dtype=np.float32)

    a = a.reshape(-1)
    b = b.reshape(-1)

    p = _softmax_np(a)
    q = _softmax_np(b)
    m = 0.5 * (p + q)

    # Use log2 so the result is in bits.
    kl_pm = float(np.sum(p * (np.log2(p + _JS_EPS) - np.log2(m + _JS_EPS))))
    kl_qm = float(np.sum(q * (np.log2(q + _JS_EPS) - np.log2(m + _JS_EPS))))

    return 0.5 * (kl_pm + kl_qm)


def _greedy_append_tokens(
    model,
    tokenizer,
    hook_mgr: HookManager,
    generated_ids: torch.Tensor,
    past_key_values,
    next_token_logits: torch.Tensor,
    max_new_tokens: int,
) -> tuple[torch.Tensor, Any, torch.Tensor]:
    """Greedy-generate up to max_new_tokens, starting from next_token_logits.

    Important: this function *consumes* every generated token via a forward pass,
    so the returned KV cache corresponds to the full generated sequence.
    """
    if max_new_tokens <= 0:
        return generated_ids, past_key_values, next_token_logits

    last_logits = next_token_logits
    next_token = last_logits.argmax(dim=-1, keepdim=True)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Append token to the sequence.
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Consume token to advance cache and obtain next logits.
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            last_logits = outputs.logits[:, -1, :]

            # Stop after consuming EOS (cache now includes EOS).
            if int(next_token.item()) == int(tokenizer.eos_token_id):
                break

            next_token = last_logits.argmax(dim=-1, keepdim=True)

    return generated_ids, past_key_values, last_logits


def generate_from_seed_cache(
    model,
    tokenizer,
    device: torch.device,
    hook_mgr: HookManager,
    seed_cache,
    prefix_text: str = "",
    max_new_tokens: int = DEFAULT_MAX_NEW,
    return_kv_cache: bool = False,
) -> StageTelemetry:
    """Generate starting from a SeedCache (identical pre-generation state)."""
    hook_mgr.reset()

    generated_ids = seed_cache.input_ids.to(device).clone()
    past_key_values = seed_cache.past_key_values

    # Optionally process extra prefix tokens (e.g., delta/instructions) before generation.
    if prefix_text:
        # IMPORTANT: This is a continuation (we already have prompt tokens in KV cache),
        # so do NOT add BOS/EOS or other special tokens here.
        enc = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
        prefix_ids = enc["input_ids"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=prefix_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        # Matched context logits after delta/instruction prefix.
        context_logits = next_token_logits.detach().float().cpu()

        # Include the prefix tokens in the decoded text for this stage.
        generated_ids = torch.cat([generated_ids, prefix_ids], dim=-1)
    else:
        # Start generation from the prompt-pass logits.
        next_token_logits = seed_cache.next_token_logits.to(device)

        # Matched context logits after ORIGINAL_QUESTION.
        context_logits = seed_cache.next_token_logits.detach().float().cpu()

    generated_ids, past_key_values, last_logits = _greedy_append_tokens(
        model=model,
        tokenizer=tokenizer,
        hook_mgr=hook_mgr,
        generated_ids=generated_ids,
        past_key_values=past_key_values,
        next_token_logits=next_token_logits,
        max_new_tokens=max_new_tokens,
    )

    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    seq_len = int(generated_ids.shape[1])

    hidden = hook_mgr.hidden_hook.captured
    if hidden is None:
        hidden = seed_cache.seed_hidden

    logits = hook_mgr.logit_hook.captured[0] if hook_mgr.logit_hook.captured is not None else last_logits

    stats = basic_stats(hidden, logits)
    stats["svd"] = svd_compress(hidden)

    return StageTelemetry(
        text=decoded,
        stats=stats,
        kv_cache=(past_key_values if return_kv_cache else None),
        seq_len=seq_len,
        context_logits=context_logits,
    )


def generate_continuation(
    model,
    tokenizer,
    device: torch.device,
    hook_mgr: HookManager,
    new_prompt: str,
    prior_kv_cache: Any,
    max_new_tokens: int = DEFAULT_MAX_NEW,
) -> StageTelemetry:
    """
    Generate continuation using prior KV cache.

    The new_prompt is processed with attention to the prior context.

    max_new_tokens counts tokens generated AFTER new_prompt.
    """
    hook_mgr.reset()

    # IMPORTANT: This is a continuation on top of an existing KV cache.
    # Adding BOS/EOS (special tokens) here would corrupt the context.
    enc = tokenizer(new_prompt, return_tensors="pt", add_special_tokens=False)
    new_input_ids = enc["input_ids"].to(device)

    generated_ids = new_input_ids.clone()
    current_cache = prior_kv_cache

    with torch.no_grad():
        outputs = model(
            input_ids=new_input_ids,
            past_key_values=current_cache,
            use_cache=True,
            return_dict=True,
        )

    current_cache = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]

    # Matched context logits after the REASK instruction tokens.
    context_logits = next_token_logits.detach().float().cpu()

    generated_ids, current_cache, last_logits = _greedy_append_tokens(
        model=model,
        tokenizer=tokenizer,
        hook_mgr=hook_mgr,
        generated_ids=generated_ids,
        past_key_values=current_cache,
        next_token_logits=next_token_logits,
        max_new_tokens=max_new_tokens,
    )

    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    hidden = hook_mgr.hidden_hook.captured
    logits = hook_mgr.logit_hook.captured[0] if hook_mgr.logit_hook.captured is not None else last_logits

    stats = basic_stats(hidden, logits)
    stats["svd"] = svd_compress(hidden)

    return StageTelemetry(text=full_text, stats=stats, context_logits=context_logits)


def telemetry_to_reflection(stats: Dict) -> str:
    """Convert telemetry into reflection block (delta/perturbation)."""
    hidden_norm = stats["hidden_norm"]
    logit_norm = stats["logit_norm"]
    entropy = stats["entropy"]

    top_sv = None
    if stats.get("svd") and stats["svd"].get("singular_values"):
        top_sv = stats["svd"]["singular_values"][0]

    reflection = [
        "The following is a reflection on your internal reasoning state.",
        "",
        "<REFLECTION>",
        f"Entropy: {entropy:.6f}",
        f"Hidden norm: {hidden_norm:.4f}",
        f"Logit norm: {logit_norm:.4f}",
        f"Top singular values: {top_sv}",
        "Interpretation:",
        "- Low entropy means your next-token distribution is very sharp (high confidence).",
        "- Large jumps in hidden or logit norm between steps indicate strong internal modulation.",
        "- When re-asking, these values should return close to the original baseline.",
        "</REFLECTION>",
        "",
        "Use this reflection to stabilize your next reasoning step and avoid unnecessary shifts.",
    ]

    return "\n".join(reflection)


def run_branching_observer_loop(
    prompt: str, 
    model_key: str = None,
    seed: int = 42,
    max_new_tokens: int = DEFAULT_MAX_NEW,
) -> Dict:
    """
    Run the V1.3 branching observer loop.
    
    Structure:
      SEED:    Build SeedCache after ORIGINAL_QUESTION
      BASE:    Generate from SeedCache
      PERTURB: SeedCache + Delta/instructions → Generate (KV cache saved)
      REASK:   Continue from PERTURB KV cache + minimal re-ask instruction → Generate
    
    Tests whether re-asking returns the system toward baseline
    while the perturbation remains in KV cache.
    """
    
    print(f"\n{Color.MAGENTA}{'═' * 60}{Color.RESET}")
    print(f"{Color.MAGENTA}  V1.3 — Continuous Hysteresis Test{Color.RESET}")
    print(f"{Color.MAGENTA}{'═' * 60}{Color.RESET}\n")
    
    tokenizer, model, device, model_config = load_model(model_key=model_key)
    
    hook_mgr = HookManager(model)
    hook_mgr.register()
    
    os.makedirs(RUNS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, f"v1_run_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # ═══════════════════════════════════════════════════════
    # STAGE 1: BASE (Standalone Baseline)
    # ═══════════════════════════════════════════════════════
    
    print(f"{Color.CYAN}[V1.3] Stage 1 — BASE (standalone baseline){Color.RESET}")
    print(f"[V1.3] Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
    print(f"[V1.3] Seed: {seed}")

    # Define ORIGINAL_QUESTION once and reuse it for seed-cache + downstream stages.
    original_question = "ORIGINAL_QUESTION:\n" + prompt

    set_deterministic_state(seed)

    # Build a seed-cache snapshot after ORIGINAL_QUESTION.
    seed_cache = build_seed_cache(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=original_question,
        hook_mgr=hook_mgr,
    )

    base = generate_from_seed_cache(
        model=model,
        tokenizer=tokenizer,
        device=device,
        hook_mgr=hook_mgr,
        seed_cache=seed_cache.clone(),
        prefix_text="",
        max_new_tokens=max_new_tokens,
        return_kv_cache=False,
    )

    L_base_ctx = base.context_logits
    if L_base_ctx is None:
        raise RuntimeError("[V1.3] Missing BASE context_logits")
    
    base_frame = COTFrame(
        stage="base",
        token_text=base.text,
        timestamp=time.time(),
        hidden_norm=base.stats["hidden_norm"],
        entropy=base.stats["entropy"],
        topk=base.stats["topk"],
        svd=base.stats["svd"],
        logit_norm=base.stats["logit_norm"],
        extra={
            "context": "base_ctx_after_original_question",
            "js_units": "bits",
        },
    )
    
    save_json(os.path.join(run_dir, "frame_base.json"), base_frame.to_json())
    with open(os.path.join(run_dir, "output_base.txt"), "w", encoding="utf-8") as f:
        f.write(base.text)
    
    print(f"[V1.3] BASE complete.")
    print(f"       Entropy: {base.stats['entropy']:.6f}")
    print(f"       Hidden norm: {base.stats['hidden_norm']:.4f}")
    print(f"       Logit norm: {base.stats['logit_norm']:.4f}")
    
    # Build the delta from BASE's stats
    delta = telemetry_to_reflection(base.stats)
    
    with open(os.path.join(run_dir, "delta.txt"), "w", encoding="utf-8") as f:
        f.write(delta)
    
    # ═══════════════════════════════════════════════════════
    # STAGE 2: PERTURB (Perturbation Applied)
    # ═══════════════════════════════════════════════════════
    
    print(f"\n{Color.YELLOW}[V1.3] Stage 2 — PERTURB (perturbation applied){Color.RESET}")
    
    perturb_prefix = (
        "\n\n" + delta +
        "\n\nContinue your reasoning, using the reflection above to stay stable and consistent."
    )

    print(f"[V1.3] Perturb prefix length: {len(perturb_prefix)} chars")
    print(f"[V1.3] Seed: {seed}")

    set_deterministic_state(seed)

    perturb = generate_from_seed_cache(
        model=model,
        tokenizer=tokenizer,
        device=device,
        hook_mgr=hook_mgr,
        seed_cache=seed_cache.clone(),
        prefix_text=perturb_prefix,
        max_new_tokens=max_new_tokens,
        return_kv_cache=True,
    )

    L_perturb_ctx = perturb.context_logits
    if L_perturb_ctx is None:
        raise RuntimeError("[V1.3] Missing PERTURB context_logits")

    js_base_vs_perturb_ctx = js_divergence_from_logits_bits(L_base_ctx, L_perturb_ctx)
    
    perturb_frame = COTFrame(
        stage="perturb",
        token_text=perturb.text,
        timestamp=time.time(),
        hidden_norm=perturb.stats["hidden_norm"],
        entropy=perturb.stats["entropy"],
        topk=perturb.stats["topk"],
        svd=perturb.stats["svd"],
        logit_norm=perturb.stats["logit_norm"],
        extra={
            "context": "perturb_ctx_after_delta_prefix",
            "js_units": "bits",
            "js_base_vs_perturb_ctx": float(js_base_vs_perturb_ctx),
        },
    )
    
    save_json(os.path.join(run_dir, "frame_perturb.json"), perturb_frame.to_json())
    with open(os.path.join(run_dir, "output_perturb.txt"), "w", encoding="utf-8") as f:
        f.write(perturb.text)
    
    print(f"[V1.3] PERTURB complete.")
    print(f"       Entropy: {perturb.stats['entropy']:.6f}")
    print(f"       Hidden norm: {perturb.stats['hidden_norm']:.4f}")
    print(f"       Logit norm: {perturb.stats['logit_norm']:.4f}")
    
    if perturb.kv_cache is None:
        print(f"{Color.RED}[V1.3] ERROR: KV cache not captured from PERTURB{Color.RESET}")
        hook_mgr.cleanup()
        return {}
    
    print(f"[V1.3] PERTURB sequence length: {perturb.seq_len} tokens")
    print(f"[V1.3] KV cache captured for REASK stage")
    
    # ═══════════════════════════════════════════════════════
    # STAGE 3: REASK (Re-ask baseline while keeping PERTURB cache)
    # ═══════════════════════════════════════════════════════
    #
    # NOTE: We intentionally keep PERTURB's KV cache (so the delta remains in-memory).
    # The goal is to test whether the model can return toward the baseline answer
    # when explicitly re-asked to do so.
    
    print(f"\n{Color.GREEN}[V1.3] Stage 3 — REASK (re-ask baseline; keep PERTURB cache){Color.RESET}")
    print(f"[V1.3] Continuing from PERTURB's KV cache")
    print(f"[V1.3] New input: Minimal re-ask instruction only (does NOT repeat prompt)")
    print(f"[V1.3] Seed: {seed}")
    
    set_deterministic_state(seed)

    # Keep PERTURB's cache (so the perturbation remains in-memory), but do NOT
    # repeat the original prompt tokens. This minimizes confounds from simply
    # re-encoding the full question again.
    reask_prompt = (
        "Now ignore any reflection/perturbation above and answer ORIGINAL_QUESTION again, accurately."
    )
    
    reask = generate_continuation(
        model, tokenizer, device, hook_mgr,
        new_prompt=reask_prompt,
        prior_kv_cache=perturb.kv_cache,
        max_new_tokens=max_new_tokens,
    )

    L_reask_ctx = reask.context_logits
    if L_reask_ctx is None:
        raise RuntimeError("[V1.3] Missing REASK context_logits")

    js_base_vs_reask_ctx = js_divergence_from_logits_bits(L_base_ctx, L_reask_ctx)
    js_perturb_vs_reask_ctx = js_divergence_from_logits_bits(L_perturb_ctx, L_reask_ctx)
    
    reask_frame = COTFrame(
        stage="reask",
        token_text=reask.text,
        timestamp=time.time(),
        hidden_norm=reask.stats["hidden_norm"],
        entropy=reask.stats["entropy"],
        topk=reask.stats["topk"],
        svd=reask.stats["svd"],
        logit_norm=reask.stats["logit_norm"],
        extra={
            "context": "reask_ctx_after_reask_instruction",
            "js_units": "bits",
            "js_base_vs_reask_ctx": float(js_base_vs_reask_ctx),
            "js_perturb_vs_reask_ctx": float(js_perturb_vs_reask_ctx),
        },
    )
    
    save_json(os.path.join(run_dir, "frame_reask.json"), reask_frame.to_json())
    with open(os.path.join(run_dir, "output_reask.txt"), "w", encoding="utf-8") as f:
        f.write(reask.text)
    
    print(f"[V1.3] REASK complete.")
    print(f"       Entropy: {reask.stats['entropy']:.6f}")
    print(f"       Hidden norm: {reask.stats['hidden_norm']:.4f}")
    print(f"       Logit norm: {reask.stats['logit_norm']:.4f}")
    
    # ═══════════════════════════════════════════════════════
    # COMPUTE METRICS
    # ═══════════════════════════════════════════════════════
    
    cache_verified = seed_cache.fingerprint != "unavailable"

    metrics = compute_cot_metrics(
        before_stats=base.stats,
        after_stats=perturb.stats,
        revert_stats=reask.stats,
        cache_verified=cache_verified,
    )
    
    print(format_metrics_summary(metrics))
    
    # ═══════════════════════════════════════════════════════
    # RECOVERY ANALYSIS
    # ═══════════════════════════════════════════════════════
    
    print(f"\n{Color.BLUE}{'═' * 60}{Color.RESET}")
    print(f"{Color.BLUE}  RECOVERY ANALYSIS{Color.RESET}")
    print(f"{Color.BLUE}{'═' * 60}{Color.RESET}\n")
    
    def distance(a, b):
        return abs(a - b)
    
    # How far did PERTURB drift from BASE?
    drift_hidden = distance(base.stats["hidden_norm"], perturb.stats["hidden_norm"])
    drift_entropy = distance(base.stats["entropy"], perturb.stats["entropy"])
    drift_logit = distance(base.stats["logit_norm"], perturb.stats["logit_norm"])
    
    # How far is REASK from BASE? (residual)
    residual_hidden = distance(base.stats["hidden_norm"], reask.stats["hidden_norm"])
    residual_entropy = distance(base.stats["entropy"], reask.stats["entropy"])
    residual_logit = distance(base.stats["logit_norm"], reask.stats["logit_norm"])
    
    def recovery_ratio(drift, residual):
        if drift < 1e-9:
            return 1.0 if residual < 1e-9 else 0.0
        return max(0.0, 1.0 - (residual / drift))
    
    recovery_hidden = recovery_ratio(drift_hidden, residual_hidden)
    recovery_entropy = recovery_ratio(drift_entropy, residual_entropy)
    recovery_logit = recovery_ratio(drift_logit, residual_logit)
    
    print(f"  {Color.BOLD}Metric        BASE → PERTURB   BASE → REASK     Recovery{Color.RESET}")
    print("  " + "─" * 62)
    print(f"  hidden_norm   {drift_hidden:>14.4f}    {residual_hidden:>14.4f}    {recovery_hidden:>7.1%}")
    print(f"  entropy       {drift_entropy:>14.6f}    {residual_entropy:>14.6f}    {recovery_entropy:>7.1%}")
    print(f"  logit_norm    {drift_logit:>14.4f}    {residual_logit:>14.4f}    {recovery_logit:>7.1%}")
    
    avg_recovery = (recovery_hidden + recovery_entropy + recovery_logit) / 3
    
    print(f"\n  {Color.BOLD}Average Recovery: {avg_recovery:.1%}{Color.RESET}")
    
    if avg_recovery > 0.7:
        regime_desc = "ELASTIC — System returns toward baseline when re-asked"
        regime_color = Color.GREEN
    elif avg_recovery > 0.3:
        regime_desc = "PARTIAL — Some residual effect remains after re-ask"
        regime_color = Color.YELLOW
    else:
        regime_desc = "PLASTIC — Perturbation left a lasting mark after re-ask"
        regime_color = Color.RED
    
    print(f"  {regime_color}{regime_desc}{Color.RESET}")
    
    print(f"\n{Color.BLUE}{'═' * 60}{Color.RESET}")
    
    # ═══════════════════════════════════════════════════════
    # BUILD CONFIG & HASH
    # ═══════════════════════════════════════════════════════
    
    run_config = {
        "model": model_key or model_config.key,
        "prompt": prompt,
        "original_question": original_question,
        "seed": seed,
        "max_new_tokens": int(max_new_tokens),
    }
    
    config_hash = hash_config(run_config)
    
    print(f"\n[V1.3] Config hash: {config_hash}")
    
    # ═══════════════════════════════════════════════════════
    # TELEMETRY COMPARISON
    # ═══════════════════════════════════════════════════════
    
    print(f"\n{Color.BLUE}[V1.3] Telemetry Comparison:{Color.RESET}\n")
    
    def fmt(x): 
        return f"{x:.12g}"
    
    print(f"  {Color.BOLD}Metric        BASE             PERTURB          REASK{Color.RESET}")
    print("  " + "─" * 56)
    print(f"  hidden_norm   {fmt(base.stats['hidden_norm']):>16} "
          f"{fmt(perturb.stats['hidden_norm']):>16} "
          f"{fmt(reask.stats['hidden_norm']):>16}")
    print(f"  logit_norm    {fmt(base.stats['logit_norm']):>16} "
          f"{fmt(perturb.stats['logit_norm']):>16} "
          f"{fmt(reask.stats['logit_norm']):>16}")
    print(f"  entropy       {fmt(base.stats['entropy']):>16} "
          f"{fmt(perturb.stats['entropy']):>16} "
          f"{fmt(reask.stats['entropy']):>16}")
    
    # ═══════════════════════════════════════════════════════
    # SAVE SUMMARY
    # ═══════════════════════════════════════════════════════
    
    summary = {
        "version": VERSION,
        "timestamp": stamp,
        "config_hash": config_hash,
        "config": run_config,
        "continuous_run": True,
        "seed_cache": {
            "seq_len": int(seed_cache.seq_len),
            "fingerprint": seed_cache.fingerprint,
            "fingerprint_available": bool(seed_cache.fingerprint != "unavailable"),
        },
        "distribution_shift": {
            "js_units": "bits",
            "js_base_vs_perturb_ctx": float(js_base_vs_perturb_ctx),
            "js_base_vs_reask_ctx": float(js_base_vs_reask_ctx),
            "js_perturb_vs_reask_ctx": float(js_perturb_vs_reask_ctx),
        },
        "metrics": {
            "drift": metrics.drift,
            "hysteresis": metrics.hysteresis,
            "recovery": metrics.recovery,
            "regime": metrics.regime,
            "components": {
                "drift": {
                    "hidden": metrics.drift_hidden,
                    "entropy": metrics.drift_entropy,
                    "logit": metrics.drift_logit,
                    "svd": metrics.drift_svd
                },
                "hysteresis": {
                    "hidden": metrics.hysteresis_hidden,
                    "entropy": metrics.hysteresis_entropy,
                    "logit": metrics.hysteresis_logit,
                    "svd": metrics.hysteresis_svd
                }
            }
        },
        "recovery_analysis": {
            "drift": {
                "hidden": drift_hidden,
                "entropy": drift_entropy,
                "logit": drift_logit
            },
            "residual": {
                "hidden": residual_hidden,
                "entropy": residual_entropy,
                "logit": residual_logit
            },
            "recovery_ratio": {
                "hidden": recovery_hidden,
                "entropy": recovery_entropy,
                "logit": recovery_logit,
                "average": avg_recovery
            }
        },
        "telemetry": {
            "base": base.stats,
            "perturb": perturb.stats,
            "reask": reask.stats
        }
    }
    
    save_json(os.path.join(run_dir, "summary.json"), summary)

    if generate_plots is not None:
        try:
            paths = generate_plots(summary, run_dir)
            print("[V1.3] Plots generated:")
            for k, v in paths.items():
                print(f"  - {k}: {v}")
        except Exception as e:
            print(f"[V1.3] Plot generation failed (non-fatal): {e}")
    else:
        print("[V1.3] plotter.py not available; skipping plot generation.")

    hook_mgr.cleanup()

    print(f"\n{Color.GREEN}[V1.3] Run complete. Output saved to: {run_dir}{Color.RESET}\n")

    return summary


def main() -> None:
    """CLI entry point."""
    
    parser = argparse.ArgumentParser(description="V1.3 — Continuous Hysteresis Test")
    parser.add_argument("mode", choices=["observer", "interactive"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW)
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        print(f"\n{Color.CYAN}=== V1.3 Interactive ==={Color.RESET}")
        
        registry = load_model_registry("models.json")
        model_keys = list(registry["models"].keys())
        print("\nAvailable models:")
        for i, key in enumerate(model_keys, start=1):
            print(f"  {i}. {key}")
        
        model_choice = int(input("\nChoose model: "))
        model_key = model_keys[model_choice - 1]
        
        prompt = input("\nEnter prompt: ")
        
        print(f"\nModel: {model_key}")
        print(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Prompt: {prompt}")
        
        ok = input("\nRun? (y/n): ").strip().lower()
        if ok != "y":
            print("Aborted.")
            return
        
        run_branching_observer_loop(
            prompt=prompt,
            model_key=model_key,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )
    
    else:
        prompt = args.prompt or "Explain how airplanes fly in a clear, accurate way."
        run_branching_observer_loop(
            prompt=prompt,
            model_key=args.model,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
