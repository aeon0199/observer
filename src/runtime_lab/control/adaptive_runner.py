from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from runtime_lab.config.schemas import ControlConfig, DiagnosticsConfig
from runtime_lab.core.backend.loader import load_model_with_backend
from runtime_lab.core.diagnostics.manager import DiagnosticsManager, summarize_diagnostics_health
from runtime_lab.core.interventions.factory import build_intervention
from runtime_lab.core.interventions.scaling import DynamicScalingIntervention, ScaleState
from runtime_lab.core.io.artifacts import ensure_dir
from runtime_lab.core.io.hashing import hash_config
from runtime_lab.core.io.json import save_json
from runtime_lab.core.runtime.engine import RuntimeEngine
from runtime_lab.core.runtime.events import ControlEvent
from .controller import StabilityController
from .dashboard import generate_dashboard
from .policy import ScalingPolicy


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class Color:
    RESET = "\033[0m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"


def run_control_experiment(
    config: ControlConfig,
    registry_path: str = "models.json",
    runs_dir: Optional[str] = None,
    diagnostics_config: Optional[DiagnosticsConfig] = None,
    intervention_kwargs: Optional[Dict[str, Any]] = None,
    generate_dashboard_html: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    _set_seed(config.seed)

    if int(config.measure_layer) != int(config.act_layer):
        raise ValueError(
            "Control mode currently requires measure_layer == act_layer. "
            "Separate measurement and actuation hooks are not implemented yet."
        )

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
    run_dir = ensure_dir(base_runs / f"control_run_{stamp}")

    run_config = {
        "mode": "control",
        "prompt": config.prompt,
        "model": model_id,
        "backend": backend_result.backend,
        "backend_meta": dict(backend_result.backend_meta),
        "max_new_tokens": int(config.max_new_tokens),
        "measure_layer": int(config.measure_layer),
        "act_layer": int(config.act_layer),
        "intervention_type": str(config.intervention_type),
        "shadow": bool(config.shadow),
        "seed": int(config.seed) if config.seed is not None else None,
        "controller": asdict(config),
    }
    cfg_hash = hash_config(run_config)

    events_path = Path(run_dir) / "events.jsonl"
    summary_path = Path(run_dir) / "summary.json"
    output_path = Path(run_dir) / "output.txt"
    config_path = Path(run_dir) / "config.json"

    save_json(str(config_path), {"config_hash": cfg_hash, "config": run_config})

    diag_cfg = diagnostics_config or DiagnosticsConfig(
        enabled=True,
        probe_layers=[int(config.measure_layer), -1],
    )
    diagnostics = DiagnosticsManager(diag_cfg)

    controller = StabilityController(
        ma_window=int(config.ma_window),
        threshold_warn=float(config.threshold_warn),
        threshold_crit=float(config.threshold_crit),
        hold_warn=int(config.hold_warn),
        hold_crit=int(config.hold_crit),
    )

    policy = ScalingPolicy(
        scale_warn=float(config.scale_warn),
        scale_crit=float(config.scale_crit),
    )

    scale_state = ScaleState(1.0)

    if str(config.intervention_type).lower().strip() == "scaling":
        intervention = DynamicScalingIntervention(scale_state)
        intervention_name = intervention.name
    elif str(config.intervention_type).lower().strip() == "sae":
        intervention = build_intervention("sae", **(intervention_kwargs or {}))
        intervention_name = getattr(intervention, "name", "sae")
    else:
        raise ValueError("Control mode currently supports intervention_type in {'scaling', 'sae'}")

    engine = RuntimeEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        layer_idx=int(config.act_layer),
        diagnostics_manager=diagnostics,
        intervention=intervention,
        probe_layers=diag_cfg.probe_layers,
        mode="control",
        temperature=float(getattr(config, "temperature", 0.0)),
        top_p=float(getattr(config, "top_p", 1.0)),
        top_k=int(getattr(config, "top_k", 0)),
    )

    generated_text = ""
    events_cache: list[dict[str, Any]] = []

    n_tokens = 0
    n_warning = 0
    n_critical = 0
    n_cooldown = 0
    sum_raw_div = 0.0
    sum_avg_score = 0.0

    try:
        print(f"\n{Color.CYAN}{'=' * 72}{Color.RESET}")
        print(f"{Color.CYAN}Runtime Lab - Closed-Loop Control{Color.RESET}")
        print(f"{Color.CYAN}{'=' * 72}{Color.RESET}\n")
        print(f"{Color.BOLD}Run dir:{Color.RESET} {run_dir}")
        print(f"{Color.BOLD}Config hash:{Color.RESET} {cfg_hash}")
        print(f"{Color.BOLD}Model:{Color.RESET} {model_id}")
        print(f"{Color.BOLD}Backend:{Color.RESET} {backend_result.backend}")
        print(f"{Color.BOLD}Device:{Color.RESET} {backend_result.backend_meta.get('resolved_device', str(device))}")
        print(f"{Color.BOLD}DType:{Color.RESET} {backend_result.backend_meta.get('resolved_dtype', 'unknown')}")
        notes = backend_result.backend_meta.get("policy_notes", []) or []
        if notes:
            print(f"{Color.BOLD}Runtime policy:{Color.RESET} {'; '.join(str(note) for note in notes)}")
        print(f"{Color.BOLD}Intervention:{Color.RESET} {intervention_name}")
        print(f"{Color.BOLD}Shadow mode:{Color.RESET} {bool(config.shadow)}")
        print(f"{Color.BOLD}Prompt:{Color.RESET} {config.prompt}")
        print(f"{Color.DIM}{'-' * 86}{Color.RESET}")
        print(
            f" {'t':>3} | {'raw_div':>8} | {'avg_score':>9} | {'scale_used':>10} |"
            f" {'next_scale':>10} | {'status':<8} | token"
        )
        print(f"{Color.DIM}{'-' * 86}{Color.RESET}")

        prefill = engine.prefill(config.prompt, intervention_active=False)

        pending_token_id = int(prefill.next_token_id)
        prompt_len = int(prefill.prompt_len)
        past_key_values = prefill.past_key_values

        if prefill.hidden_post is not None:
            diagnostics0 = diagnostics.step(prefill.hidden_post, layer_states={int(config.measure_layer): prefill.hidden_post})
            ctl_state = controller.update(diagnostics0)
            scale_state.value = float(policy.next_scale(ctl_state))
        else:
            diagnostics0 = {}
            ctl_state = controller.state
            scale_state.value = 1.0

        eos = getattr(tokenizer, "eos_token_id", None)

        with open(events_path, "w", encoding="utf-8") as events_f:
            for t in range(int(config.max_new_tokens)):
                consumed_token_id = int(pending_token_id)

                scale_used = float(scale_state.value)
                intervention_active = bool((scale_used < 1.0) and (not config.shadow))

                step = engine.step(
                    t=t,
                    consumed_token_id=consumed_token_id,
                    prompt_len=prompt_len,
                    past_key_values=past_key_values,
                    intervention_active=intervention_active,
                )
                past_key_values = step.past_key_values
                pending_token_id = int(step.predicted_next_token_id)

                raw_div = float(step.diagnostics.get("divergence", 0.0))
                ctl_state = controller.update(step.diagnostics)
                next_scale = float(policy.next_scale(ctl_state))
                scale_state.value = float(next_scale)

                div_color = Color.GREEN
                if ctl_state.avg_score > controller.TH_WARN:
                    div_color = Color.YELLOW
                if ctl_state.avg_score > controller.TH_CRIT:
                    div_color = Color.RED

                status_color = Color.RESET
                if ctl_state.status == "CRITICAL":
                    status_color = Color.RED
                elif ctl_state.status == "WARNING":
                    status_color = Color.YELLOW
                elif ctl_state.status == "COOLDOWN":
                    status_color = Color.CYAN

                token_str = step.consumed_token_text.replace("\n", "\\n")
                print(
                    f" {t:3d} | {raw_div:8.4f} | {div_color}{ctl_state.avg_score:9.4f}{Color.RESET} |"
                    f" {scale_used:10.2f} | {next_scale:10.2f} |"
                    f" {status_color}{ctl_state.status:<8}{Color.RESET} | {token_str}"
                )

                ctrl_evt = ControlEvent(
                    t=int(step.t),
                    consumed_token_id=int(step.consumed_token_id),
                    consumed_token_text=step.consumed_token_text,
                    predicted_next_token_id=int(step.predicted_next_token_id),
                    resolved_layer_idx=int(step.event.resolved_layer_idx),
                    hidden_pre_norm=float(step.event.hidden_pre_norm),
                    hidden_post_norm=float(step.event.hidden_post_norm),
                    hidden_delta_norm=float(step.event.hidden_delta_norm),
                    diagnostics=step.diagnostics,
                    intervention_active=bool(intervention_active),
                    mode="control",
                    scale_used=float(scale_used),
                    next_scale=float(next_scale),
                    status=str(ctl_state.status),
                    shadow=bool(config.shadow),
                )

                evt = {
                    "t": ctrl_evt.t,
                    "token_id": ctrl_evt.consumed_token_id,
                    "token_text": ctrl_evt.consumed_token_text,
                    "predicted_next_token_id": ctrl_evt.predicted_next_token_id,
                    "raw_div": float(raw_div),
                    "avg_score": float(ctl_state.avg_score),
                    "control_score": float(ctl_state.raw_score),
                    "status": str(ctl_state.status),
                    "scale_used": float(scale_used),
                    "next_scale": float(next_scale),
                    "pre_hidden_norm": float(ctrl_evt.hidden_pre_norm),
                    "post_hidden_norm": float(ctrl_evt.hidden_post_norm),
                    "pre_post_delta_norm": float(ctrl_evt.hidden_delta_norm),
                    "diagnostics": ctrl_evt.diagnostics,
                    "shadow": bool(config.shadow),
                    "intervention_applied": bool(intervention_active),
                    "intervention_type": str(config.intervention_type),
                    "intervention_name": str(intervention_name),
                    "backend": backend_result.backend,
                }

                events_f.write(json.dumps(evt, ensure_ascii=False) + "\n")
                events_cache.append(evt)

                n_tokens += 1
                sum_raw_div += float(raw_div)
                sum_avg_score += float(ctl_state.avg_score)

                if ctl_state.status == "WARNING":
                    n_warning += 1
                elif ctl_state.status == "CRITICAL":
                    n_critical += 1
                elif ctl_state.status == "COOLDOWN":
                    n_cooldown += 1

                generated_text += step.consumed_token_text

                if eos is not None and int(consumed_token_id) == int(eos):
                    break

    finally:
        engine.close()

    full_text = config.prompt + generated_text
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    summary = {
        "config_hash": cfg_hash,
        "mode": "control",
        "run_dir": str(run_dir),
        "model_id": model_id,
        "backend": backend_result.backend,
        "runtime": {
            "device": str(device),
            "resolved_dtype": backend_result.backend_meta.get("resolved_dtype"),
            "policy_notes": backend_result.backend_meta.get("policy_notes", []),
            "backend_meta": dict(backend_result.backend_meta),
        },
        "intervention_type": str(config.intervention_type),
        "intervention_name": str(intervention_name),
        "shadow": bool(config.shadow),
        "tokens": int(n_tokens),
        "avg_raw_div_mean": float(sum_raw_div / max(1, n_tokens)),
        "avg_score_mean": float(sum_avg_score / max(1, n_tokens)),
        "diagnostics_health": summarize_diagnostics_health([event.get("diagnostics", {}) for event in events_cache]),
        "status_counts": {
            "WARNING": int(n_warning),
            "CRITICAL": int(n_critical),
            "COOLDOWN": int(n_cooldown),
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "events_path": str(events_path),
            "output_path": str(output_path),
            "summary_path": str(summary_path),
            "config_path": str(config_path),
        },
    }

    if generate_dashboard_html:
        try:
            dashboard_path = generate_dashboard(run_dir=Path(run_dir), events=events_cache, summary=summary)
            summary["artifacts"]["dashboard_path"] = str(dashboard_path)
        except Exception as e:
            summary["dashboard_error"] = str(e)

    try:
        from runtime_lab.core.advisory import analyze as _analyze_advisory
        # Pass config through so advise_control can check shadow flag.
        summary_for_advisor = dict(summary)
        summary_for_advisor["config"] = run_config
        summary["advisory"] = _analyze_advisory("control", summary_for_advisor)
    except Exception as e:
        summary["advisory"] = {"error": f"advisory failed: {e}"}

    save_json(str(summary_path), summary)

    print(f"{Color.DIM}{'-' * 72}{Color.RESET}")
    print(f"\n{Color.BOLD}Final Output:{Color.RESET}\n{full_text}\n")
    print(f"{Color.DIM}{'-' * 72}{Color.RESET}")
    print("Artifacts written:")
    print(f"  - {events_path}")
    print(f"  - {output_path}")
    print(f"  - {summary_path}")
    if summary.get("artifacts", {}).get("dashboard_path"):
        print(f"  - {summary['artifacts']['dashboard_path']}")

    artifacts = {
        "run_dir": str(run_dir),
        "events_path": str(events_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "config_hash": cfg_hash,
        "dashboard_path": summary.get("artifacts", {}).get("dashboard_path"),
    }

    return full_text, artifacts
