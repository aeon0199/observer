from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from runtime_lab.config.schemas import CommonRunConfig, DiagnosticsConfig
from runtime_lab.core.backend.loader import load_model_with_backend
from runtime_lab.core.diagnostics.manager import DiagnosticsManager
from runtime_lab.core.io.artifacts import ensure_dir
from runtime_lab.core.io.hashing import hash_config
from runtime_lab.core.io.json import save_json
from runtime_lab.core.runtime.engine import RuntimeEngine


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def run_observe_experiment(
    config: CommonRunConfig,
    registry_path: str = "models.json",
    runs_dir: Optional[str] = None,
    diagnostics_config: Optional[DiagnosticsConfig] = None,
) -> Dict[str, Any]:
    _set_seed(config.seed)

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

    diag_cfg = diagnostics_config or DiagnosticsConfig(
        enabled=True,
        probe_layers=[-1],
    )
    diagnostics = DiagnosticsManager(diag_cfg)

    engine = RuntimeEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        layer_idx=-1,
        diagnostics_manager=diagnostics,
        intervention=None,
        probe_layers=diag_cfg.probe_layers,
        mode="observe",
    )

    run_config = {
        "mode": "observe",
        "prompt": config.prompt,
        "model": model_id,
        "backend": backend_result.backend,
        "backend_meta": {
            "remote": bool(backend_result.backend_meta.get("remote", False)),
            "device_map": backend_result.backend_meta.get("device_map", str(device)),
        },
        "max_new_tokens": int(config.max_new_tokens),
        "seed": int(config.seed) if config.seed is not None else None,
    }
    cfg_hash = hash_config(run_config)

    base_runs = ensure_dir(runs_dir or os.environ.get("RUNS_DIR", "runs"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(base_runs / f"observe_run_{stamp}")

    events_path = Path(run_dir) / "events.jsonl"
    summary_path = Path(run_dir) / "summary.json"
    output_path = Path(run_dir) / "output.txt"
    config_path = Path(run_dir) / "config.json"

    save_json(str(config_path), {"config_hash": cfg_hash, "config": run_config})

    generated_text = ""
    events: list[Dict[str, Any]] = []

    n_tokens = 0
    sum_div = 0.0

    try:
        prefill = engine.prefill(config.prompt, intervention_active=False)
        pending_token_id = int(prefill.next_token_id)
        prompt_len = int(prefill.prompt_len)
        past_key_values = prefill.past_key_values
        eos = getattr(tokenizer, "eos_token_id", None)

        with open(events_path, "w", encoding="utf-8") as events_f:
            for t in range(int(config.max_new_tokens)):
                consumed_token_id = int(pending_token_id)

                step = engine.step(
                    t=t,
                    consumed_token_id=consumed_token_id,
                    prompt_len=prompt_len,
                    past_key_values=past_key_values,
                    intervention_active=False,
                )

                past_key_values = step.past_key_values
                pending_token_id = int(step.predicted_next_token_id)

                event = {
                    "t": int(step.t),
                    "token_id": int(step.consumed_token_id),
                    "token_text": step.consumed_token_text,
                    "predicted_next_token_id": int(step.predicted_next_token_id),
                    "hidden_pre_norm": float(step.event.hidden_pre_norm),
                    "hidden_post_norm": float(step.event.hidden_post_norm),
                    "hidden_delta_norm": float(step.event.hidden_delta_norm),
                    "diagnostics": step.diagnostics,
                    "mode": "observe",
                }

                events_f.write(json.dumps(event, ensure_ascii=False) + "\n")
                events.append(event)

                generated_text += step.consumed_token_text
                n_tokens += 1
                sum_div += float(step.diagnostics.get("divergence", 0.0))

                if eos is not None and int(consumed_token_id) == int(eos):
                    break

    finally:
        engine.close()

    full_text = config.prompt + generated_text
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    summary = {
        "config_hash": cfg_hash,
        "mode": "observe",
        "run_dir": str(run_dir),
        "model_id": model_id,
        "backend": backend_result.backend,
        "tokens": int(n_tokens),
        "avg_divergence": float(sum_div / max(1, n_tokens)),
        "artifacts": {
            "run_dir": str(run_dir),
            "events_path": str(events_path),
            "output_path": str(output_path),
            "summary_path": str(summary_path),
            "config_path": str(config_path),
        },
    }

    save_json(str(summary_path), summary)
    return summary
