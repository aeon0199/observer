from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from runtime_lab.config.schemas import HysteresisConfig
from runtime_lab.core.backend.loader import load_model_with_backend
from runtime_lab.core.io.artifacts import ensure_dir
from runtime_lab.core.io.hashing import hash_config
from runtime_lab.core.io.json import save_json
from runtime_lab.core.model.layers import resolve_transformer_layers
from runtime_lab.core.runtime.hooks import HiddenCaptureHook
from runtime_lab.stress.seed_cache import build_seed_cache


_JS_EPS = 1e-12


@dataclass
class StageTelemetry:
    text: str
    stats: Dict[str, Any]
    kv_cache: Optional[Any] = None
    seq_len: int = 0
    context_logits: Optional[torch.Tensor] = None


class _TokenwiseNoiseHook:
    """Forward hook that adds a seeded random direction to the last-token hidden
    state for a configurable window of generation steps. Magnitude is relative
    to the current hidden-state norm so the knob is model-agnostic.

    Used by the 'noise' hysteresis perturbation_mode to inject a real
    hidden-state perturbation during PERTURB that is then removed for REASK.
    """

    def __init__(self, magnitude: float, start: int, duration: int, seed: int):
        self.magnitude = float(magnitude)
        self.start = int(start)
        self.duration = int(duration)
        self.seed = int(seed)
        self._direction: Optional[torch.Tensor] = None
        self._step = 0
        self.active_steps: list = []

    def reset(self):
        self._step = 0
        self._direction = None
        self.active_steps = []

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor):
            return output

        self._step += 1
        in_window = (self._step >= self.start) and (self._step < self.start + self.duration)
        if not in_window:
            return output

        # Skip the large prefill pass (we only want to perturb single-token
        # generation steps, not the initial prompt forward).
        if hidden.dim() != 3 or hidden.shape[1] != 1:
            return output

        if self._direction is None:
            dim = hidden.shape[-1]
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed)
            vec = torch.randn(dim, generator=g, device="cpu", dtype=torch.float32)
            self._direction = vec / (vec.norm() + 1e-12)

        direction = self._direction.to(device=hidden.device, dtype=hidden.dtype)
        h_last = hidden[:, -1, :]
        h_norm = h_last.norm(dim=-1, keepdim=True)
        delta = direction.unsqueeze(0) * (self.magnitude * h_norm)

        new_hidden = hidden.clone()
        new_hidden[:, -1, :] = h_last + delta
        self.active_steps.append(int(self._step))

        if isinstance(output, tuple):
            return (new_hidden, *output[1:])
        return new_hidden


def _set_deterministic_state(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + _JS_EPS)


def js_divergence_from_logits_bits(logits_a: Any, logits_b: Any) -> float:
    if isinstance(logits_a, torch.Tensor):
        a = logits_a.detach().float().cpu().numpy().reshape(-1)
    else:
        a = np.array(logits_a, dtype=np.float32).reshape(-1)

    if isinstance(logits_b, torch.Tensor):
        b = logits_b.detach().float().cpu().numpy().reshape(-1)
    else:
        b = np.array(logits_b, dtype=np.float32).reshape(-1)

    p = _softmax_np(a)
    q = _softmax_np(b)
    m = 0.5 * (p + q)

    kl_pm = float(np.sum(p * (np.log2(p + _JS_EPS) - np.log2(m + _JS_EPS))))
    kl_qm = float(np.sum(q * (np.log2(q + _JS_EPS) - np.log2(m + _JS_EPS))))
    return 0.5 * (kl_pm + kl_qm)


def _topk_values(logits: torch.Tensor, k: int = 5) -> Dict[str, Any]:
    x = logits.detach().float().view(-1).cpu()
    if x.numel() == 0:
        return {"indices": [], "values": [], "probs": []}
    k = int(min(max(1, k), x.numel()))
    values, indices = torch.topk(x, k)
    probs = torch.softmax(x, dim=-1).gather(0, indices)
    return {
        "indices": [int(i) for i in indices.tolist()],
        "values": [float(v) for v in values.tolist()],
        "probs": [float(p) for p in probs.tolist()],
    }


def _svd_signature(hidden: torch.Tensor, k: int = 5) -> Dict[str, Any]:
    if not isinstance(hidden, torch.Tensor):
        return {"singular_values": []}
    x = hidden.detach().float().cpu()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    else:
        x = x.view(1, -1)
    try:
        s = torch.linalg.svdvals(x)
        s = s[: int(min(k, s.numel()))]
        return {"singular_values": [float(v) for v in s.tolist()]}
    except Exception as e:
        return {"singular_values": [], "error": str(e)}


def _tensor_norm(x: Optional[torch.Tensor]) -> float:
    if not isinstance(x, torch.Tensor):
        return 0.0
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def _tensor_entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.detach().float().view(-1), dim=-1)
    ent = -(probs * (probs + 1e-12).log()).sum()
    return float(ent.item())


def _stage_stats(hidden: torch.Tensor, logits: torch.Tensor) -> Dict[str, Any]:
    return {
        "hidden_norm": _tensor_norm(hidden),
        "logit_norm": _tensor_norm(logits),
        "entropy": _tensor_entropy(logits),
        "topk": _topk_values(logits, k=5),
        "svd": _svd_signature(hidden),
    }


def _greedy_append_tokens(
    model,
    tokenizer,
    device: torch.device,
    generated_ids: torch.Tensor,
    past_key_values: Any,
    next_token_logits: torch.Tensor,
    max_new_tokens: int,
    current_seq_len: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> tuple[torch.Tensor, Any, torch.Tensor]:
    from runtime_lab.core.sampling import sample_token_id

    if max_new_tokens <= 0:
        return generated_ids, past_key_values, next_token_logits

    last_logits = next_token_logits
    tok_id = sample_token_id(last_logits, temperature, top_p, top_k)
    next_token = torch.tensor([[tok_id]], device=device)

    with torch.no_grad():
        for step in range(int(max_new_tokens)):
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            attention_mask = torch.ones(
                (1, int(current_seq_len) + step + 1),
                device=device,
                dtype=torch.long,
            )
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            last_logits = outputs.logits[:, -1, :]

            if tokenizer.eos_token_id is not None and int(next_token.item()) == int(tokenizer.eos_token_id):
                break

            tok_id = sample_token_id(last_logits, temperature, top_p, top_k)
            next_token = torch.tensor([[tok_id]], device=device)

    return generated_ids, past_key_values, last_logits


def _generate_from_seed_cache(
    model,
    tokenizer,
    device: torch.device,
    capture_hook: HiddenCaptureHook,
    seed_cache,
    prefix_text: str = "",
    max_new_tokens: int = 128,
    return_kv_cache: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> StageTelemetry:
    capture_hook.reset()

    generated_ids = seed_cache.input_ids.to(device).clone()
    past_key_values = seed_cache.past_key_values
    current_seq_len = int(seed_cache.seq_len)

    if prefix_text:
        enc = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
        prefix_ids = enc["input_ids"].to(device)
        attention_mask = torch.ones(
            (1, int(current_seq_len) + int(prefix_ids.shape[1])),
            device=device,
            dtype=torch.long,
        )
        with torch.no_grad():
            outputs = model(
                input_ids=prefix_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        context_logits = next_token_logits.detach().float().cpu()
        generated_ids = torch.cat([generated_ids, prefix_ids], dim=-1)
        current_seq_len += int(prefix_ids.shape[1])
    else:
        next_token_logits = seed_cache.next_token_logits.to(device)
        context_logits = seed_cache.next_token_logits.detach().float().cpu()

    generated_ids, past_key_values, last_logits = _greedy_append_tokens(
        model=model,
        tokenizer=tokenizer,
        device=device,
        generated_ids=generated_ids,
        past_key_values=past_key_values,
        next_token_logits=next_token_logits,
        max_new_tokens=max_new_tokens,
        current_seq_len=current_seq_len,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    hidden = capture_hook.last_hidden
    if hidden is None and getattr(seed_cache, "seed_hidden_available", False):
        hidden = seed_cache.seed_hidden
    elif hidden is None:
        hidden = torch.zeros((1, 1), dtype=torch.float32)

    logits = last_logits.detach().float().cpu()
    stats = _stage_stats(hidden, logits)

    return StageTelemetry(
        text=decoded,
        stats=stats,
        kv_cache=(past_key_values if return_kv_cache else None),
        seq_len=int(generated_ids.shape[1]),
        context_logits=context_logits,
    )


def _generate_continuation(
    model,
    tokenizer,
    device: torch.device,
    capture_hook: HiddenCaptureHook,
    new_prompt: str,
    prior_kv_cache: Any,
    prior_seq_len: int,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> StageTelemetry:
    capture_hook.reset()

    enc = tokenizer(new_prompt, return_tensors="pt", add_special_tokens=False)
    new_input_ids = enc["input_ids"].to(device)
    current_seq_len = int(prior_seq_len) + int(new_input_ids.shape[1])
    attention_mask = torch.ones((1, current_seq_len), device=device, dtype=torch.long)

    with torch.no_grad():
        outputs = model(
            input_ids=new_input_ids,
            attention_mask=attention_mask,
            past_key_values=prior_kv_cache,
            use_cache=True,
            return_dict=True,
        )

    current_cache = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    context_logits = next_token_logits.detach().float().cpu()

    generated_ids = new_input_ids.clone()
    generated_ids, current_cache, last_logits = _greedy_append_tokens(
        model=model,
        tokenizer=tokenizer,
        device=device,
        generated_ids=generated_ids,
        past_key_values=current_cache,
        next_token_logits=next_token_logits,
        max_new_tokens=max_new_tokens,
        current_seq_len=current_seq_len,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    hidden = capture_hook.last_hidden
    if hidden is None:
        hidden = torch.zeros((1, 1), dtype=torch.float32)
    logits = last_logits.detach().float().cpu()
    stats = _stage_stats(hidden, logits)

    return StageTelemetry(
        text=full_text,
        stats=stats,
        seq_len=int(current_seq_len + max(0, int(generated_ids.shape[1]) - int(new_input_ids.shape[1]))),
        context_logits=context_logits,
    )


def _telemetry_to_reflection(stats: Dict[str, Any]) -> str:
    top_sv = None
    if stats.get("svd") and stats["svd"].get("singular_values"):
        top_sv = stats["svd"]["singular_values"][0]

    reflection = [
        "The following is a reflection on your internal reasoning state.",
        "",
        "<REFLECTION>",
        f"Entropy: {stats['entropy']:.6f}",
        f"Hidden norm: {stats['hidden_norm']:.4f}",
        f"Logit norm: {stats['logit_norm']:.4f}",
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


def _norm_distance(a: float, b: float) -> float:
    if a < 1e-10 and b < 1e-10:
        return 0.0
    return abs(a - b) / (max(a, b) + 1e-10)


def _entropy_distance(a: float, b: float) -> float:
    return abs(a - b)


def _svd_distance(svd_a: Dict[str, Any], svd_b: Dict[str, Any]) -> float:
    sv1 = np.array(svd_a.get("singular_values", []), dtype=np.float64)
    sv2 = np.array(svd_b.get("singular_values", []), dtype=np.float64)
    if len(sv1) == 0 or len(sv2) == 0:
        return 0.0
    min_len = min(len(sv1), len(sv2))
    sv1 = sv1[:min_len]
    sv2 = sv2[:min_len]
    norm = max(np.linalg.norm(sv1), np.linalg.norm(sv2))
    if norm < 1e-10:
        return 0.0
    return float(np.linalg.norm(sv1 - sv2) / norm)


def _compute_component_distances(stats_a: Dict[str, Any], stats_b: Dict[str, Any]) -> Dict[str, float]:
    d_hidden = _norm_distance(stats_a.get("hidden_norm", 0.0), stats_b.get("hidden_norm", 0.0))
    d_entropy = _entropy_distance(stats_a.get("entropy", 0.0), stats_b.get("entropy", 0.0))
    d_logit = _norm_distance(stats_a.get("logit_norm", 0.0), stats_b.get("logit_norm", 0.0))
    d_svd = _svd_distance(stats_a.get("svd", {}), stats_b.get("svd", {}))
    composite = (1.0 * d_hidden) + (1.0 * d_entropy) + (0.5 * d_logit) + (1.0 * d_svd)
    return {
        "hidden": float(d_hidden),
        "entropy": float(d_entropy),
        "logit": float(d_logit),
        "svd": float(d_svd),
        "composite": float(composite),
    }


def run_hysteresis_experiment(
    config: HysteresisConfig,
    registry_path: str = "models.json",
    runs_dir: Optional[str] = None,
    prebuilt_backend: Optional[Any] = None,
) -> Dict[str, Any]:
    _set_deterministic_state(config.seed)

    if prebuilt_backend is not None:
        backend_result = prebuilt_backend
    else:
        backend_result = load_model_with_backend(
            model_key=config.model_key,
            registry_path=registry_path,
            backend=config.backend,
            nnsight_remote=config.nnsight_remote,
            nnsight_device=config.nnsight_device,
        )
    tokenizer = backend_result.tokenizer
    model = backend_result.model
    device = backend_result.device
    model_cfg = backend_result.config
    model_id = config.model_key or getattr(model_cfg, "key", None) or "unknown"

    base_runs = ensure_dir(runs_dir or os.environ.get("RUNS_DIR", "runs"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(base_runs / f"hysteresis_run_{stamp}")

    original_question = f"{config.original_question_label}:\n{config.prompt}"
    run_config = {
        "mode": "hysteresis",
        "prompt": config.prompt,
        "original_question": original_question,
        "model": model_id,
        "backend": backend_result.backend,
        "backend_meta": dict(backend_result.backend_meta),
        "max_new_tokens": int(config.max_new_tokens),
        "seed": int(config.seed) if config.seed is not None else None,
        "original_question_label": str(config.original_question_label),
        "perturbation_mode": str(getattr(config, "perturbation_mode", "prompt")),
        "noise_layer": int(getattr(config, "noise_layer", -1)),
        "noise_magnitude": float(getattr(config, "noise_magnitude", 0.15)),
        "noise_start": int(getattr(config, "noise_start", 3)),
        "noise_duration": int(getattr(config, "noise_duration", 8)),
        "noise_seed": int(getattr(config, "noise_seed", 1234)),
    }
    cfg_hash = hash_config(run_config)

    layers = resolve_transformer_layers(model)
    num_layers = len(layers)
    capture_hook = HiddenCaptureHook()
    handle = layers[-1].register_forward_hook(capture_hook)

    # Resolve semantic noise_layer ("mid" etc.) against this model's depth.
    raw_noise_layer = getattr(config, "noise_layer", -1)
    try:
        from runtime_lab.cli._common import resolve_semantic_layer
        resolved_noise_layer = resolve_semantic_layer(raw_noise_layer, num_layers)
    except Exception:
        resolved_noise_layer = int(raw_noise_layer) if isinstance(raw_noise_layer, int) else num_layers // 2 - 1
    run_config["noise_layer_resolved"] = int(resolved_noise_layer)
    run_config["num_layers"] = int(num_layers)

    samp = {
        "temperature": float(getattr(config, "temperature", 0.0)),
        "top_p": float(getattr(config, "top_p", 1.0)),
        "top_k": int(getattr(config, "top_k", 0)),
    }
    run_config.update(samp)

    try:
        seed_cache = build_seed_cache(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=original_question,
            intervention_layer=-1,
        )

        base = _generate_from_seed_cache(
            model=model,
            tokenizer=tokenizer,
            device=device,
            capture_hook=capture_hook,
            seed_cache=seed_cache.clone(),
            max_new_tokens=int(config.max_new_tokens),
            **samp,
        )
        if base.context_logits is None:
            raise RuntimeError("Missing BASE context logits")

        perturbation_mode = str(getattr(config, "perturbation_mode", "prompt") or "prompt").lower()
        noise_handle = None
        delta = None

        if perturbation_mode == "noise":
            # Hidden-state perturbation path. Attach a noise-injection forward
            # hook to the resolved layer (semantic "mid" etc.), remove the
            # (final-layer) capture hook temporarily, re-register it AFTER the
            # noise hook — this guarantees the capture sees the post-perturbation
            # output when noise_layer == -1.
            noise_layer_idx = int(resolved_noise_layer)
            if noise_layer_idx < 0:
                noise_layer_idx = num_layers + noise_layer_idx
            noise_layer_idx = max(0, min(num_layers - 1, noise_layer_idx))

            noise_hook = _TokenwiseNoiseHook(
                magnitude=float(getattr(config, "noise_magnitude", 0.15)),
                start=int(getattr(config, "noise_start", 3)),
                duration=int(getattr(config, "noise_duration", 8)),
                seed=int(getattr(config, "noise_seed", 1234)),
            )
            handle.remove()
            noise_handle = layers[noise_layer_idx].register_forward_hook(noise_hook)
            handle = layers[-1].register_forward_hook(capture_hook)

            _set_deterministic_state(config.seed)
            perturb = _generate_from_seed_cache(
                model=model,
                tokenizer=tokenizer,
                device=device,
                capture_hook=capture_hook,
                seed_cache=seed_cache.clone(),
                prefix_text="",
                max_new_tokens=int(config.max_new_tokens),
                return_kv_cache=True,
                **samp,
            )
            perturb.stats["perturbation_mode"] = "noise"
            perturb.stats["noise_active_steps"] = list(noise_hook.active_steps)
            # Remove noise for REASK — perturbation is "off" from here on.
            noise_handle.remove()
            noise_handle = None
        else:
            delta = _telemetry_to_reflection(base.stats)
            perturb_prefix = (
                "\n\n"
                + delta
                + "\n\nContinue your reasoning, using the reflection above to stay stable and consistent."
            )
            _set_deterministic_state(config.seed)
            perturb = _generate_from_seed_cache(
                model=model,
                tokenizer=tokenizer,
                device=device,
                capture_hook=capture_hook,
                seed_cache=seed_cache.clone(),
                prefix_text=perturb_prefix,
                max_new_tokens=int(config.max_new_tokens),
                return_kv_cache=True,
                **samp,
            )
            perturb.stats["perturbation_mode"] = "prompt"

        if perturb.context_logits is None:
            raise RuntimeError("Missing PERTURB context logits")
        if perturb.kv_cache is None:
            raise RuntimeError("PERTURB stage did not retain KV cache")

        reask_prompt = (
            "Now ignore any reflection/perturbation above and answer ORIGINAL_QUESTION again, accurately."
        )
        _set_deterministic_state(config.seed)
        reask = _generate_continuation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            capture_hook=capture_hook,
            new_prompt=reask_prompt,
            prior_kv_cache=perturb.kv_cache,
            prior_seq_len=int(perturb.seq_len),
            max_new_tokens=int(config.max_new_tokens),
            **samp,
        )
        if reask.context_logits is None:
            raise RuntimeError("Missing REASK context logits")
    finally:
        handle.remove()

    drift = _compute_component_distances(base.stats, perturb.stats)
    hysteresis = _compute_component_distances(base.stats, reask.stats)

    # TD2 / F9 fix: recovery is CONCEPTUALLY UNDEFINED when the perturbation
    # didn't propagate (drift ≈ 0). Reporting a numeric ratio in that case
    # (the old `1 - H/(D+ε)` blows up to ±1000) mixes real recovery signal
    # with measurement artifacts. Flag and skip.
    DRIFT_MIN = 0.05
    perturbation_did_not_propagate = bool(drift["composite"] < DRIFT_MIN)

    if perturbation_did_not_propagate:
        recovery = None
        regime = "no-propagation"
    else:
        recovery = 1.0 - (hysteresis["composite"] / drift["composite"])
        if recovery > 0.8:
            regime = "elastic"
        elif recovery > 0.4:
            regime = "partial"
        elif recovery >= 0:
            regime = "plastic"
        else:
            regime = "runaway"

    drift_hidden = abs(base.stats["hidden_norm"] - perturb.stats["hidden_norm"])
    drift_entropy = abs(base.stats["entropy"] - perturb.stats["entropy"])
    drift_logit = abs(base.stats["logit_norm"] - perturb.stats["logit_norm"])

    residual_hidden = abs(base.stats["hidden_norm"] - reask.stats["hidden_norm"])
    residual_entropy = abs(base.stats["entropy"] - reask.stats["entropy"])
    residual_logit = abs(base.stats["logit_norm"] - reask.stats["logit_norm"])

    def _recovery_ratio(drift_value: float, residual_value: float) -> float:
        if drift_value < 1e-9:
            return 1.0 if residual_value < 1e-9 else 0.0
        return max(0.0, 1.0 - (residual_value / drift_value))

    recovery_hidden = _recovery_ratio(drift_hidden, residual_hidden)
    recovery_entropy = _recovery_ratio(drift_entropy, residual_entropy)
    recovery_logit = _recovery_ratio(drift_logit, residual_logit)
    avg_recovery = float((recovery_hidden + recovery_entropy + recovery_logit) / 3.0)

    js_base_vs_perturb_ctx = js_divergence_from_logits_bits(base.context_logits, perturb.context_logits)
    js_base_vs_reask_ctx = js_divergence_from_logits_bits(base.context_logits, reask.context_logits)
    js_perturb_vs_reask_ctx = js_divergence_from_logits_bits(perturb.context_logits, reask.context_logits)

    frame_base = {
        "stage": "base",
        "timestamp": datetime.now().timestamp(),
        "token_text": base.text,
        "hidden_norm": base.stats["hidden_norm"],
        "entropy": base.stats["entropy"],
        "topk": base.stats["topk"],
        "svd": base.stats["svd"],
        "logit_norm": base.stats["logit_norm"],
        "extra": {
            "context": "base_ctx_after_original_question",
            "js_units": "bits",
        },
    }
    frame_perturb = {
        "stage": "perturb",
        "timestamp": datetime.now().timestamp(),
        "token_text": perturb.text,
        "hidden_norm": perturb.stats["hidden_norm"],
        "entropy": perturb.stats["entropy"],
        "topk": perturb.stats["topk"],
        "svd": perturb.stats["svd"],
        "logit_norm": perturb.stats["logit_norm"],
        "extra": {
            "context": "perturb_ctx_after_delta_prefix",
            "js_units": "bits",
            "js_base_vs_perturb_ctx": float(js_base_vs_perturb_ctx),
        },
    }
    frame_reask = {
        "stage": "reask",
        "timestamp": datetime.now().timestamp(),
        "token_text": reask.text,
        "hidden_norm": reask.stats["hidden_norm"],
        "entropy": reask.stats["entropy"],
        "topk": reask.stats["topk"],
        "svd": reask.stats["svd"],
        "logit_norm": reask.stats["logit_norm"],
        "extra": {
            "context": "reask_ctx_after_reask_instruction",
            "js_units": "bits",
            "js_base_vs_reask_ctx": float(js_base_vs_reask_ctx),
            "js_perturb_vs_reask_ctx": float(js_perturb_vs_reask_ctx),
        },
    }

    summary = {
        "config_hash": cfg_hash,
        "mode": "hysteresis",
        "timestamp": stamp,
        "run_dir": str(run_dir),
        "model_id": model_id,
        "backend": backend_result.backend,
        "runtime": {
            "device": str(device),
            "resolved_dtype": backend_result.backend_meta.get("resolved_dtype"),
            "policy_notes": backend_result.backend_meta.get("policy_notes", []),
            "backend_meta": dict(backend_result.backend_meta),
        },
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
            "drift": float(drift["composite"]),
            "hysteresis": float(hysteresis["composite"]),
            "recovery": (None if recovery is None else float(recovery)),
            "regime": regime,
            "perturbation_did_not_propagate": bool(perturbation_did_not_propagate),
            "drift_min_threshold": float(DRIFT_MIN),
            "components": {
                "drift": drift,
                "hysteresis": hysteresis,
            },
        },
        "recovery_analysis": {
            "drift": {
                "hidden": float(drift_hidden),
                "entropy": float(drift_entropy),
                "logit": float(drift_logit),
            },
            "residual": {
                "hidden": float(residual_hidden),
                "entropy": float(residual_entropy),
                "logit": float(residual_logit),
            },
            "recovery_ratio": {
                "hidden": float(recovery_hidden),
                "entropy": float(recovery_entropy),
                "logit": float(recovery_logit),
                "average": float(avg_recovery),
            },
        },
        "telemetry": {
            "base": base.stats,
            "perturb": perturb.stats,
            "reask": reask.stats,
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "frame_base": str(Path(run_dir) / "frame_base.json"),
            "frame_perturb": str(Path(run_dir) / "frame_perturb.json"),
            "frame_reask": str(Path(run_dir) / "frame_reask.json"),
            "output_base": str(Path(run_dir) / "output_base.txt"),
            "output_perturb": str(Path(run_dir) / "output_perturb.txt"),
            "output_reask": str(Path(run_dir) / "output_reask.txt"),
            "delta": str(Path(run_dir) / "delta.txt"),
            "summary": str(Path(run_dir) / "summary.json"),
        },
    }

    save_json(str(Path(run_dir) / "frame_base.json"), frame_base)
    save_json(str(Path(run_dir) / "frame_perturb.json"), frame_perturb)
    save_json(str(Path(run_dir) / "frame_reask.json"), frame_reask)
    with open(Path(run_dir) / "output_base.txt", "w", encoding="utf-8") as f:
        f.write(base.text)
    with open(Path(run_dir) / "output_perturb.txt", "w", encoding="utf-8") as f:
        f.write(perturb.text)
    with open(Path(run_dir) / "output_reask.txt", "w", encoding="utf-8") as f:
        f.write(reask.text)
    with open(Path(run_dir) / "delta.txt", "w", encoding="utf-8") as f:
        if delta is not None:
            f.write(delta)
        else:
            f.write(
                "[noise perturbation mode — no prompt delta.\n"
                f" hook_layer={getattr(config, 'noise_layer', -1)}"
                f" magnitude={getattr(config, 'noise_magnitude', 0.15)} (relative)"
                f" start={getattr(config, 'noise_start', 3)}"
                f" duration={getattr(config, 'noise_duration', 8)}"
                f" seed={getattr(config, 'noise_seed', 1234)}]"
            )
    try:
        from runtime_lab.core.advisory import analyze as _analyze_advisory
        summary["advisory"] = _analyze_advisory("hysteresis", summary)
    except Exception as e:
        summary["advisory"] = {"error": f"advisory failed: {e}"}

    save_json(str(Path(run_dir) / "summary.json"), summary)

    print(f"[Runtime Lab][hysteresis] Run dir: {run_dir}")
    print(f"[Runtime Lab][hysteresis] Config hash: {cfg_hash}")
    print(f"[Runtime Lab][hysteresis] Regime: {regime}")
    print(
        "[Runtime Lab][hysteresis] Recovery ratios:"
        f" hidden={recovery_hidden:.1%} entropy={recovery_entropy:.1%} logit={recovery_logit:.1%}"
    )

    return summary
