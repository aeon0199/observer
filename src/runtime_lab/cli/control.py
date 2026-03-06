from __future__ import annotations

import argparse

from runtime_lab.config.schemas import ControlConfig, DiagnosticsConfig
from runtime_lab.control.adaptive_runner import run_control_experiment


def add_control_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default="Explain how airplanes fly in a clear, accurate way.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--measure-layer", type=int, default=-1)
    parser.add_argument("--act-layer", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="hf", choices=["hf", "nnsight"])
    parser.add_argument("--nnsight-remote", action="store_true")
    parser.add_argument("--nnsight-device", default=None)
    parser.add_argument("--registry-path", default="models.json")
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--type", default="scaling", choices=["scaling", "sae"])
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--ma-window", type=int, default=3)
    parser.add_argument("--threshold-warn", type=float, default=0.55)
    parser.add_argument("--threshold-crit", type=float, default=0.85)
    parser.add_argument("--scale-warn", type=float, default=0.90)
    parser.add_argument("--scale-crit", type=float, default=0.75)
    parser.add_argument("--hold-warn", type=int, default=3)
    parser.add_argument("--hold-crit", type=int, default=6)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--sae-repo", default="apollo-research/llama-3.1-70b-sae")
    parser.add_argument("--sae-id", default=None)
    parser.add_argument("--sae-layer", type=int, default=None)
    parser.add_argument("--sae-feature-idx", type=int, default=0)
    parser.add_argument("--sae-strength", type=float, default=5.0)
    parser.add_argument("--sae-no-normalize", action="store_true")


def run_from_args(args) -> None:
    cfg = ControlConfig(
        prompt=args.prompt,
        model_key=args.model,
        max_new_tokens=args.max_tokens,
        backend=args.backend,
        nnsight_remote=args.nnsight_remote,
        nnsight_device=args.nnsight_device,
        seed=args.seed,
        measure_layer=args.measure_layer,
        act_layer=args.act_layer,
        intervention_type=args.type,
        shadow=bool(args.shadow),
        ma_window=args.ma_window,
        threshold_warn=args.threshold_warn,
        threshold_crit=args.threshold_crit,
        scale_warn=args.scale_warn,
        scale_crit=args.scale_crit,
        hold_warn=args.hold_warn,
        hold_crit=args.hold_crit,
    )

    diag_cfg = DiagnosticsConfig(
        enabled=True,
        probe_layers=[int(args.measure_layer), -1],
    )

    intervention_kwargs = {}
    if args.type == "sae":
        resolved_layer = args.sae_layer if args.sae_layer is not None else args.act_layer
        intervention_kwargs = {
            "repo_id": args.sae_repo,
            "sae_id": args.sae_id,
            "layer": None if args.sae_id is not None else int(resolved_layer),
            "feature_idx": int(args.sae_feature_idx),
            "strength": float(args.sae_strength),
            "normalize": bool(not args.sae_no_normalize),
            "device": args.nnsight_device,
            "name": f"sae_f{int(args.sae_feature_idx)}_s{float(args.sae_strength):.2f}",
        }

    run_control_experiment(
        config=cfg,
        registry_path=args.registry_path,
        runs_dir=args.runs_dir,
        diagnostics_config=diag_cfg,
        intervention_kwargs=intervention_kwargs,
        generate_dashboard_html=not args.no_dashboard,
    )
