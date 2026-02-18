"""Unified adaptive-controller entrypoint for observe / stress / control modes.

This routes to the active runtime engines:
- observe: V2 baseline-only style run (intervention disabled)
- stress: V2 baseline vs fixed intervention experiment
- control: adaptive-controller closed-loop controller demo (with optional shadow mode)
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any


def _load_module_from_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"[AdaptiveController] Could not import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_v2_module(root: Path):
    return _load_module_from_path("adaptive_v2_intervention", root / "intervention_engine" / "intervention.py")


def _load_control_module(root: Path):
    return _load_module_from_path("adaptive_control_loop", root / "adaptive_controller" / "adaptive_loop.py")


def run_observe(args, root: Path):
    v2 = _load_v2_module(root)
    return v2.run_intervention_experiment(
        prompt=args.prompt,
        model_key=args.model,
        max_new_tokens=args.max_tokens,
        intervention_layer=args.layer,
        intervention_type="scaling",
        intervention_magnitude=1.0,
        intervention_start=1,
        intervention_duration=0,
        seed=args.seed,
        backend=args.backend,
        nnsight_remote=args.nnsight_remote,
        nnsight_device=args.nnsight_device,
        with_diagnostics=not args.no_diagnostics,
    )


def run_stress(args, root: Path):
    v2 = _load_v2_module(root)
    return v2.run_intervention_experiment(
        prompt=args.prompt,
        model_key=args.model,
        max_new_tokens=args.max_tokens,
        intervention_layer=args.layer,
        intervention_type=args.type,
        intervention_magnitude=args.magnitude,
        intervention_start=args.start,
        intervention_duration=args.duration,
        seed=args.seed,
        backend=args.backend,
        nnsight_remote=args.nnsight_remote,
        nnsight_device=args.nnsight_device,
        with_diagnostics=not args.no_diagnostics,
        sae_repo=args.sae_repo,
        sae_id=args.sae_id,
        sae_layer=args.sae_layer,
        sae_feature_idx=args.sae_feature_idx,
        sae_strength=args.sae_strength,
        sae_normalize=not args.sae_no_normalize,
        generate_dashboard_html=not args.no_dashboard,
    )


def run_control(args, root: Path):
    control_type = args.type
    if control_type == "additive":
        # Keep control mode ergonomic with parser default.
        control_type = "scaling"
    if control_type not in ("scaling", "sae"):
        raise ValueError("Control mode only supports --type scaling or --type sae.")
    ctl = _load_control_module(root)
    text, artifacts = ctl.run_adaptive_controller(
        prompt=args.prompt,
        model_key=args.model,
        max_new_tokens=args.max_tokens,
        layer_idx=args.layer,
        ma_window=args.ma_window,
        seed=args.seed,
        shadow=args.shadow,
        run_name=args.run_name,
        backend=args.backend,
        nnsight_remote=args.nnsight_remote,
        nnsight_device=args.nnsight_device,
        intervention_type=control_type,
        sae_repo=args.sae_repo,
        sae_id=args.sae_id,
        sae_layer=args.sae_layer,
        sae_feature_idx=args.sae_feature_idx,
        sae_strength=args.sae_strength,
        sae_normalize=not args.sae_no_normalize,
    )
    return {"text": text, "artifacts": artifacts}


def main():
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Adaptive-controller unified runner")
    parser.add_argument("mode", choices=["observe", "stress", "control"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default="Explain how airplanes fly in a clear, accurate way.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    # V2-mode options
    parser.add_argument("--type", default="additive", choices=["additive", "projection", "scaling", "sae"])
    parser.add_argument("--magnitude", type=float, default=2.0)
    parser.add_argument("--start", type=int, default=5)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--backend", default="hf", choices=["hf", "nnsight"])
    parser.add_argument("--nnsight-remote", action="store_true")
    parser.add_argument("--nnsight-device", default=None)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--sae-repo", default="apollo-research/llama-3.1-70b-sae")
    parser.add_argument("--sae-id", default=None)
    parser.add_argument("--sae-layer", type=int, default=None)
    parser.add_argument("--sae-feature-idx", type=int, default=0)
    parser.add_argument("--sae-strength", type=float, default=5.0)
    parser.add_argument("--sae-no-normalize", action="store_true")

    # Control mode options
    parser.add_argument("--ma-window", type=int, default=3)
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--run-name", default="adaptive_runner")
    parser.add_argument("--no-dashboard", action="store_true")

    args = parser.parse_args()

    if args.mode == "observe":
        run_observe(args, root)
    elif args.mode == "stress":
        run_stress(args, root)
    else:
        run_control(args, root)


if __name__ == "__main__":
    main()
