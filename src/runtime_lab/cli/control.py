from __future__ import annotations

import argparse

from runtime_lab.config.schemas import ControlConfig, DiagnosticsConfig
from runtime_lab.control.adaptive_runner import run_control_experiment
from ._common import (
    add_probe_layers_arg,
    add_sampling_args,
    add_seed_sweep_arg,
    parse_seeds,
    resolve_probe_layers,
)
from ._sweep import run_sweep


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
    parser.add_argument("--type", default="scaling", choices=["scaling", "sae", "additive"])
    # Additive-controller tunables (E8/F18-F20). Magnitudes used when the
    # controller escalates. Conservative defaults — E6 showed mag=1.0
    # produced strong effect (logit_kl=10.40), so controller should start
    # well below that.
    parser.add_argument("--additive-warn-magnitude", type=float, default=0.3,
                        help="Additive intervention magnitude when controller is WARNING")
    parser.add_argument("--additive-crit-magnitude", type=float, default=0.6,
                        help="Additive intervention magnitude when controller is CRITICAL")
    parser.add_argument("--additive-seed", type=int, default=42,
                        help="Random direction seed for additive intervention")
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
    add_probe_layers_arg(parser)
    add_sampling_args(parser)
    add_seed_sweep_arg(parser)


def _run_one(args, seed: int):
    cfg = ControlConfig(
        prompt=args.prompt,
        model_key=args.model,
        max_new_tokens=args.max_tokens,
        backend=args.backend,
        nnsight_remote=args.nnsight_remote,
        nnsight_device=args.nnsight_device,
        seed=seed,
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
        temperature=float(getattr(args, "temperature", 0.0)),
        top_p=float(getattr(args, "top_p", 1.0)),
        top_k=int(getattr(args, "top_k", 0)),
    )
    # Attach additive-controller knobs as attrs (ControlConfig is a
    # dataclass that doesn't declare them; adaptive_runner reads via getattr).
    cfg.additive_warn_magnitude = float(getattr(args, "additive_warn_magnitude", 0.3))
    cfg.additive_crit_magnitude = float(getattr(args, "additive_crit_magnitude", 0.6))
    cfg.additive_seed = int(getattr(args, "additive_seed", 42))

    probe_layers = resolve_probe_layers(args.probe_layers, None)
    if int(args.measure_layer) not in probe_layers:
        probe_layers = [int(args.measure_layer), *probe_layers]

    diag_cfg = DiagnosticsConfig(enabled=True, probe_layers=probe_layers)

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

    return run_control_experiment(
        config=cfg,
        registry_path=args.registry_path,
        runs_dir=args.runs_dir,
        diagnostics_config=diag_cfg,
        intervention_kwargs=intervention_kwargs,
        generate_dashboard_html=not args.no_dashboard,
    )


def run_from_args(args) -> None:
    seeds = parse_seeds(getattr(args, "seeds", None))
    if not seeds:
        _run_one(args, int(args.seed))
        return
    run_sweep(
        mode="control",
        seeds=seeds,
        runs_dir=args.runs_dir,
        run_once=lambda s: _run_one(args, s),
        describe={
            "prompt": args.prompt,
            "model": args.model,
            "measure_layer": int(args.measure_layer),
            "act_layer": int(args.act_layer),
            "shadow": bool(args.shadow),
        },
    )
