"""Observer warm-model daemon.

Loads one model once, then accepts experiment configs as JSON-lines on stdin.
Each line is dispatched to the matching runner with prebuilt_backend so the
4-second per-run model-load overhead vanishes.

Protocol (stdin → stdout):
    IN  : {"mode": "observe"|"stress"|"hysteresis"|"control", "config": {...}, ...}
    OUT : {"ok": true, "run_dir": "...", "summary_path": "..."}    on success
          {"ok": false, "error": "..."}                             on failure
    Every line of stdout is a single JSON object terminated by newline, so
    callers can read one response per request.

Usage:
    python scripts/observer_daemon.py --model qwen3-1.7b

This is launched from the dashboard's JobManager when warm-daemon mode is
enabled; stdin is fed JSON line-by-line, stdout is parsed back to pick up
the run_dir. The daemon stays alive across many requests so the model sits
in MPS memory the whole time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

# Keep the parent process's MPS context by forcing eager evaluation.
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# IMPORTANT: capture real stdout FIRST, before anything else writes to it.
# Experiment runners sprinkle print() statements that pollute the JSON
# protocol; we redirect sys.stdout to stderr so only our ack writes
# reach the parent's stdout pipe.
_ACK_OUT = sys.stdout
sys.stdout = sys.stderr


def _log(msg: str) -> None:
    # Daemon logs go to stderr so stdout stays pure JSON.
    print(f"[daemon] {msg}", file=sys.stderr, flush=True)


def _write_ack(payload: Dict[str, Any]) -> None:
    _ACK_OUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _ACK_OUT.flush()


def _serialize_summary(payload: Any) -> Dict[str, Any]:
    if is_dataclass(payload):
        return asdict(payload)
    if isinstance(payload, dict):
        return payload
    return {"raw": str(payload)}


def _build_config_observe(cfg: Dict[str, Any]):
    from runtime_lab.config.schemas import CommonRunConfig
    return CommonRunConfig(
        prompt=cfg["prompt"],
        model_key=cfg.get("model"),
        max_new_tokens=int(cfg.get("max_tokens", 64)),
        backend=cfg.get("backend", "hf"),
        seed=int(cfg.get("seed", 42)),
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        top_k=int(cfg.get("top_k", 0)),
    )


def _build_config_stress(cfg: Dict[str, Any]):
    from runtime_lab.config.schemas import StressConfig
    from runtime_lab.cli._common import resolve_semantic_layer
    layer = resolve_semantic_layer(cfg.get("layer", "mid"), None)
    return StressConfig(
        prompt=cfg["prompt"],
        model_key=cfg.get("model"),
        max_new_tokens=int(cfg.get("max_tokens", 64)),
        backend=cfg.get("backend", "hf"),
        seed=int(cfg.get("seed", 42)),
        intervention_layer=int(layer),
        intervention_type=str(cfg.get("intervention_type", "additive")),
        intervention_magnitude=float(cfg.get("magnitude", 0.15)),
        intervention_magnitude_relative=(not cfg.get("absolute_magnitude", False)),
        intervention_start=int(cfg.get("start", 5)),
        intervention_duration=int(cfg.get("duration", 10)),
        with_diagnostics=bool(cfg.get("with_diagnostics", True)),
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        top_k=int(cfg.get("top_k", 0)),
    )


def _build_config_hysteresis(cfg: Dict[str, Any]):
    from runtime_lab.config.schemas import HysteresisConfig
    from runtime_lab.cli._common import resolve_semantic_layer
    return HysteresisConfig(
        prompt=cfg["prompt"],
        model_key=cfg.get("model"),
        max_new_tokens=int(cfg.get("max_tokens", 128)),
        backend=cfg.get("backend", "hf"),
        seed=int(cfg.get("seed", 42)),
        perturbation_mode=str(cfg.get("perturbation_mode", "prompt")),
        noise_layer=resolve_semantic_layer(cfg.get("noise_layer", "mid"), None),
        noise_magnitude=float(cfg.get("noise_magnitude", 0.15)),
        noise_start=int(cfg.get("noise_start", 3)),
        noise_duration=int(cfg.get("noise_duration", 8)),
        noise_seed=int(cfg.get("noise_seed", 1234)),
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        top_k=int(cfg.get("top_k", 0)),
    )


def _build_config_control(cfg: Dict[str, Any]):
    from runtime_lab.config.schemas import ControlConfig
    return ControlConfig(
        prompt=cfg["prompt"],
        model_key=cfg.get("model"),
        max_new_tokens=int(cfg.get("max_tokens", 64)),
        backend=cfg.get("backend", "hf"),
        seed=int(cfg.get("seed", 42)),
        measure_layer=int(cfg.get("measure_layer", -1)),
        act_layer=int(cfg.get("act_layer", -1)),
        intervention_type=str(cfg.get("intervention_type", "scaling")),
        shadow=bool(cfg.get("shadow", False)),
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        top_k=int(cfg.get("top_k", 0)),
    )


def _run_request(request: Dict[str, Any], backend) -> Dict[str, Any]:
    mode = str(request.get("mode") or "").lower()
    registry_path = request.get("registry_path", "models.json")
    runs_dir = request.get("runs_dir")

    if mode == "observe":
        from runtime_lab.observe.runner import run_observe_experiment
        from runtime_lab.config.schemas import DiagnosticsConfig
        from runtime_lab.cli._common import resolve_probe_layers
        cfg = _build_config_observe(request)
        probe_layers = resolve_probe_layers(request.get("probe_layers", "auto"), None)
        diag = DiagnosticsConfig(enabled=True, probe_layers=probe_layers)
        summary = run_observe_experiment(
            config=cfg, registry_path=registry_path, runs_dir=runs_dir,
            diagnostics_config=diag, prebuilt_backend=backend,
        )
        return {"ok": True, "mode": mode, "summary": summary}

    if mode == "stress":
        from runtime_lab.stress.experiment import run_stress_experiment
        from runtime_lab.config.schemas import DiagnosticsConfig
        from runtime_lab.cli._common import resolve_probe_layers
        cfg = _build_config_stress(request)
        probe_layers = resolve_probe_layers(request.get("probe_layers", "auto"), None)
        if int(cfg.intervention_layer) not in probe_layers:
            probe_layers = [int(cfg.intervention_layer), *probe_layers]
        diag = DiagnosticsConfig(enabled=True, probe_layers=probe_layers)
        results = run_stress_experiment(
            config=cfg, registry_path=registry_path, runs_dir=runs_dir,
            diagnostics_config=diag, prebuilt_backend=backend,
        )
        return {"ok": True, "mode": mode, "summary": results}

    if mode == "hysteresis":
        from runtime_lab.hysteresis.runner import run_hysteresis_experiment
        cfg = _build_config_hysteresis(request)
        summary = run_hysteresis_experiment(
            config=cfg, registry_path=registry_path, runs_dir=runs_dir,
            prebuilt_backend=backend,
        )
        return {"ok": True, "mode": mode, "summary": summary}

    if mode == "control":
        from runtime_lab.control.adaptive_runner import run_control_experiment
        from runtime_lab.config.schemas import DiagnosticsConfig
        from runtime_lab.cli._common import resolve_probe_layers
        cfg = _build_config_control(request)
        probe_layers = resolve_probe_layers(request.get("probe_layers", "auto"), None)
        if int(cfg.measure_layer) not in probe_layers:
            probe_layers = [int(cfg.measure_layer), *probe_layers]
        diag = DiagnosticsConfig(enabled=True, probe_layers=probe_layers)
        summary = run_control_experiment(
            config=cfg, registry_path=registry_path, runs_dir=runs_dir,
            diagnostics_config=diag, prebuilt_backend=backend,
            generate_dashboard_html=bool(request.get("generate_dashboard_html", True)),
        )
        return {"ok": True, "mode": mode, "summary": summary}

    return {"ok": False, "error": f"unknown mode: {mode!r}"}


def _make_ack(run_dir: str) -> Dict[str, Any]:
    return {"ok": True, "run_dir": run_dir}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model key from models.json")
    ap.add_argument("--backend", default="hf")
    ap.add_argument("--registry-path", default="models.json")
    args = ap.parse_args()

    _log(f"loading model={args.model} backend={args.backend}...")
    from runtime_lab.core.backend.loader import load_model_with_backend
    backend_result = load_model_with_backend(
        model_key=args.model,
        registry_path=args.registry_path,
        backend=args.backend,
    )
    _log(f"model loaded · device={backend_result.device} · ready.")
    # Emit a hello line so clients know the daemon is alive.
    _write_ack({"event": "ready", "model": args.model, "device": str(backend_result.device)})

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
        except Exception as e:
            _write_ack({"ok": False, "error": f"bad json: {e}"})
            continue

        try:
            result = _run_request(request, backend_result)
            # Slim down — just run_dir + a flag; full summary is on disk.
            summary = result.get("summary") or {}
            ack = {
                "ok": result.get("ok", False),
                "mode": result.get("mode"),
                "run_dir": (summary.get("run_dir") or (summary.get("artifacts") or {}).get("run_dir")),
                "advisory": summary.get("advisory") or (summary.get("metrics") or {}).get("advisory"),
            }
            if not ack["ok"]:
                ack["error"] = result.get("error")
            _write_ack(ack)
        except Exception as e:
            tb = traceback.format_exc(limit=6)
            _write_ack({"ok": False, "error": str(e), "traceback": tb})

    _log("stdin closed. shutting down.")


if __name__ == "__main__":
    main()
