from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

import torch

from runtime_lab.config.schemas import DiagnosticsConfig, StressConfig
from runtime_lab.core.backend.loader import load_model_with_backend
from runtime_lab.core.diagnostics.manager import DiagnosticsManager
from runtime_lab.core.interventions.factory import build_intervention
from runtime_lab.core.io.artifacts import ensure_dir
from runtime_lab.core.io.hashing import hash_config
from runtime_lab.core.io.json import save_json
from runtime_lab.core.runtime.engine import RuntimeEngine
from runtime_lab.core.trajectory.comparison import TrajectoryComparison
from runtime_lab.core.trajectory.state import TokenState, Trajectory, compute_entropy, compute_top1
from .seed_cache import SeedCache, build_seed_cache


def _set_deterministic_state(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        import numpy as np
        import random

        np.random.seed(int(seed))
        random.seed(int(seed))
    except Exception:
        pass

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _trajectory_from_seed_cache(
    model,
    tokenizer,
    device: torch.device,
    seed_cache: SeedCache,
    prompt: str,
    max_new_tokens: int,
    requested_layer_idx: int,
    resolved_layer_idx: int,
    intervention,
    intervention_start: int,
    intervention_end: int,
    model_id: str,
    diagnostics_manager: Optional[DiagnosticsManager] = None,
) -> tuple[Trajectory, str]:
    trajectory = Trajectory(
        prompt=prompt,
        model_id=model_id,
        intervention_layer_requested=int(requested_layer_idx),
        intervention_layer_resolved=int(resolved_layer_idx),
        intervention_start=int(intervention_start),
        intervention_end=int(intervention_end),
        intervention_type=getattr(intervention, "name", "none") if intervention is not None else "none",
    )

    if max_new_tokens <= 0:
        return trajectory, prompt

    engine = RuntimeEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        layer_idx=resolved_layer_idx,
        diagnostics_manager=diagnostics_manager,
        intervention=intervention,
        probe_layers=(diagnostics_manager.probe_layers if diagnostics_manager is not None else []),
        mode="stress",
    )

    try:
        if diagnostics_manager is not None:
            diagnostics_manager.reset()

        prompt_len = int(seed_cache.seq_len)
        past_key_values = seed_cache.past_key_values
        generated_ids: list[int] = []

        logits0 = seed_cache.next_token_logits
        hidden0 = seed_cache.seed_hidden if seed_cache.seed_hidden_available else None

        if isinstance(hidden0, torch.Tensor):
            trajectory._hidden_vecs.append(hidden0.to(dtype=torch.float16))
        else:
            trajectory._hidden_vecs.append(torch.zeros((1, 1), dtype=torch.float16))

        trajectory._logits.append(logits0.to(dtype=torch.float16))

        next_token_id = int(logits0.argmax(dim=-1).item())
        entropy = compute_entropy(logits0)
        top1_prob, top1_token = compute_top1(logits0)

        diagnostics0 = {}
        if diagnostics_manager is not None and hidden0 is not None:
            diagnostics0 = diagnostics_manager.step(hidden0, layer_states={int(requested_layer_idx): hidden0})

        state0 = TokenState(
            token_idx=0,
            token_id=next_token_id,
            token_text=tokenizer.decode([next_token_id]),
            hidden_norm=float(hidden0.norm().item()) if hidden0 is not None else 0.0,
            logit_norm=float(logits0.norm().item()),
            entropy=entropy,
            top1_prob=top1_prob,
            top1_token=top1_token,
            intervention_active=False,
            diagnostics=diagnostics0,
        )
        trajectory.add_state(state0)
        generated_ids.append(next_token_id)

        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None and next_token_id == int(eos):
            return trajectory, prompt + tokenizer.decode(generated_ids, skip_special_tokens=True)

        for t in range(1, max_new_tokens):
            consumed_token_id = int(generated_ids[-1])
            intervention_active = bool(intervention is not None and intervention_start <= t < intervention_end)

            step = engine.step(
                t=t,
                consumed_token_id=consumed_token_id,
                prompt_len=prompt_len,
                past_key_values=past_key_values,
                intervention_active=intervention_active,
            )

            past_key_values = step.past_key_values

            hidden_post = step.hidden_post
            if isinstance(hidden_post, torch.Tensor):
                trajectory._hidden_vecs.append(hidden_post.to(dtype=torch.float16))
            else:
                trajectory._hidden_vecs.append(torch.zeros((1, 1), dtype=torch.float16))

            trajectory._logits.append(step.logits.to(dtype=torch.float16))

            entropy = compute_entropy(step.logits)
            top1_prob, top1_token = compute_top1(step.logits)

            state = TokenState(
                token_idx=t,
                token_id=step.predicted_next_token_id,
                token_text=tokenizer.decode([step.predicted_next_token_id]),
                hidden_norm=step.event.hidden_post_norm,
                logit_norm=float(step.logits.norm().item()),
                entropy=entropy,
                top1_prob=top1_prob,
                top1_token=top1_token,
                intervention_active=bool(intervention_active),
                diagnostics=step.diagnostics,
            )
            trajectory.add_state(state)
            generated_ids.append(int(step.predicted_next_token_id))

            if eos is not None and int(step.predicted_next_token_id) == int(eos):
                break

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = prompt + generated_text
        return trajectory, full_text

    finally:
        engine.close()


def run_stress_experiment(
    config: StressConfig,
    registry_path: str = "models.json",
    runs_dir: Optional[str] = None,
    diagnostics_config: Optional[DiagnosticsConfig] = None,
    intervention_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if config.seed is not None:
        _set_deterministic_state(int(config.seed))

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

    run_config = {
        "mode": "stress",
        "prompt": config.prompt,
        "model": model_id,
        "backend": backend_result.backend,
        "backend_meta": {
            "remote": bool(backend_result.backend_meta.get("remote", False)),
            "device_map": backend_result.backend_meta.get("device_map", str(device)),
        },
        "max_new_tokens": int(config.max_new_tokens),
        "intervention_layer": int(config.intervention_layer),
        "intervention_type": str(config.intervention_type),
        "intervention_magnitude": float(config.intervention_magnitude),
        "intervention_start": int(config.intervention_start),
        "intervention_duration": int(config.intervention_duration),
        "seed": int(config.seed) if config.seed is not None else None,
        "with_diagnostics": bool(config.with_diagnostics),
    }

    cfg_hash = hash_config(run_config)

    base_runs = ensure_dir(runs_dir or os.environ.get("RUNS_DIR", "runs"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(base_runs / f"stress_run_{stamp}")

    intervention_end = int(config.intervention_start + config.intervention_duration)

    if config.intervention_duration > 0 and config.intervention_start < 1:
        raise ValueError("intervention_start must be >= 1 (token 0 comes from prompt-pass logits)")

    seed_cache = build_seed_cache(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=config.prompt,
        intervention_layer=config.intervention_layer,
    )

    run_config["seed_cache_fingerprint"] = seed_cache.fingerprint
    run_config["prompt_tokens"] = int(seed_cache.seq_len)
    run_config["resolved_layer_idx"] = int(seed_cache.resolved_layer_idx)

    diag_cfg = diagnostics_config or DiagnosticsConfig(
        enabled=bool(config.with_diagnostics),
        probe_layers=[int(config.intervention_layer), -1] if config.with_diagnostics else [],
    )

    baseline_diag = DiagnosticsManager(diag_cfg) if config.with_diagnostics else None
    intervention_diag = DiagnosticsManager(diag_cfg) if config.with_diagnostics else None

    intervention = build_intervention(
        config.intervention_type,
        magnitude=config.intervention_magnitude,
        seed=(config.seed or 42),
        **(intervention_kwargs or {}),
    )

    if config.seed is not None:
        _set_deterministic_state(int(config.seed))
    baseline_trajectory, baseline_text = _trajectory_from_seed_cache(
        model=model,
        tokenizer=tokenizer,
        device=device,
        seed_cache=seed_cache.clone(),
        prompt=config.prompt,
        max_new_tokens=int(config.max_new_tokens),
        requested_layer_idx=int(config.intervention_layer),
        resolved_layer_idx=int(seed_cache.resolved_layer_idx),
        intervention=None,
        intervention_start=-1,
        intervention_end=-1,
        model_id=model_id,
        diagnostics_manager=baseline_diag,
    )

    if config.seed is not None:
        _set_deterministic_state(int(config.seed))
    intervention_trajectory, intervention_text = _trajectory_from_seed_cache(
        model=model,
        tokenizer=tokenizer,
        device=device,
        seed_cache=seed_cache.clone(),
        prompt=config.prompt,
        max_new_tokens=int(config.max_new_tokens),
        requested_layer_idx=int(config.intervention_layer),
        resolved_layer_idx=int(seed_cache.resolved_layer_idx),
        intervention=intervention,
        intervention_start=int(config.intervention_start),
        intervention_end=int(intervention_end),
        model_id=model_id,
        diagnostics_manager=intervention_diag,
    )

    comparison = TrajectoryComparison(
        baseline=baseline_trajectory,
        intervention=intervention_trajectory,
    )
    comparison.compute_metrics()

    if abs(comparison.deviation_during) < 1e-9 and abs(comparison.final_distance) < 1e-9:
        regime = "NO_EFFECT"
    elif comparison.recovery_ratio > 0.8:
        regime = "ELASTIC"
    elif comparison.recovery_ratio > 0.4:
        regime = "PARTIAL"
    elif comparison.recovery_ratio > 0:
        regime = "PLASTIC"
    else:
        regime = "DIVERGENT"

    results = {
        "config_hash": cfg_hash,
        "timestamp": stamp,
        "mode": "stress",
        "config": run_config,
        "metrics": {
            "primary_metric": comparison.primary_metric,
            "deviation_during": comparison.deviation_during,
            "recovery_after": comparison.recovery_after,
            "final_distance": comparison.final_distance,
            "recovery_ratio": comparison.recovery_ratio,
            "convergence_rate": comparison.convergence_rate,
            "token_match_rate": comparison.token_match_rate,
            "first_token_divergence": comparison.first_token_divergence,
            "regime": regime,
            "summary": comparison.summary,
        },
        "trajectories": comparison.to_json(),
        "artifacts": {
            "run_dir": str(run_dir),
            "baseline_output": str(run_dir / "baseline_output.txt"),
            "intervention_output": str(run_dir / "intervention_output.txt"),
            "results_json": str(run_dir / "results.json"),
        },
    }

    with open(run_dir / "baseline_output.txt", "w", encoding="utf-8") as f:
        f.write(baseline_text)

    with open(run_dir / "intervention_output.txt", "w", encoding="utf-8") as f:
        f.write(intervention_text)

    save_json(str(run_dir / "results.json"), results)

    return results
