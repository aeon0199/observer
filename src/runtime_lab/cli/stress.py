from __future__ import annotations

import argparse

from runtime_lab.config.schemas import DiagnosticsConfig, StressConfig
from runtime_lab.stress.experiment import run_stress_experiment


def add_stress_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default="Explain how airplanes fly in a clear, accurate way.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--type", default="additive", choices=["additive", "projection", "scaling", "sae"])
    parser.add_argument("--magnitude", type=float, default=2.0)
    parser.add_argument("--start", type=int, default=5)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="hf", choices=["hf", "nnsight"])
    parser.add_argument("--nnsight-remote", action="store_true")
    parser.add_argument("--nnsight-device", default=None)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--registry-path", default="models.json")
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--sae-repo", default="apollo-research/llama-3.1-70b-sae")
    parser.add_argument("--sae-id", default=None)
    parser.add_argument("--sae-layer", type=int, default=None)
    parser.add_argument("--sae-feature-idx", type=int, default=0)
    parser.add_argument("--sae-strength", type=float, default=5.0)
    parser.add_argument("--sae-no-normalize", action="store_true")


def run_from_args(args) -> None:
    cfg = StressConfig(
        prompt=args.prompt,
        model_key=args.model,
        max_new_tokens=args.max_tokens,
        backend=args.backend,
        nnsight_remote=args.nnsight_remote,
        nnsight_device=args.nnsight_device,
        seed=args.seed,
        intervention_layer=args.layer,
        intervention_type=args.type,
        intervention_magnitude=args.magnitude,
        intervention_start=args.start,
        intervention_duration=args.duration,
        with_diagnostics=not args.no_diagnostics,
    )

    diag_cfg = DiagnosticsConfig(
        enabled=not args.no_diagnostics,
        probe_layers=[int(args.layer), -1] if not args.no_diagnostics else [],
    )

    intervention_kwargs = {}
    if args.type == "sae":
        resolved_layer = args.sae_layer if args.sae_layer is not None else args.layer
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

    run_stress_experiment(
        config=cfg,
        registry_path=args.registry_path,
        runs_dir=args.runs_dir,
        diagnostics_config=diag_cfg,
        intervention_kwargs=intervention_kwargs,
    )
