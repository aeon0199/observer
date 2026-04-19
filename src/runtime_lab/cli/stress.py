from __future__ import annotations

import argparse

from runtime_lab.config.schemas import DiagnosticsConfig, StressConfig
from runtime_lab.stress.experiment import run_stress_experiment
from ._common import (
    add_probe_layers_arg,
    add_sampling_args,
    add_seed_sweep_arg,
    parse_seeds,
    resolve_probe_layers,
    resolve_semantic_layer,
)
from ._sweep import run_sweep


def add_stress_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default="Explain how airplanes fly in a clear, accurate way.")
    parser.add_argument("--max-tokens", type=int, default=64)
    # Default: 'mid' — resolves to n//2-1 at runtime. Final-layer (-1)
    # perturbations are frequently absorbed by confident greedy argmax.
    parser.add_argument("--layer", default="mid",
                        help="Transformer layer for intervention. Int (e.g. 14, -1) or semantic alias "
                             "(mid, mid-, mid+, late, early).")
    parser.add_argument("--type", default="additive", choices=["additive", "projection", "scaling", "sae"])
    # Default changed 2.0 -> 0.15 and interpreted as *fraction of hidden norm*.
    # The old default of 2.0 absolute was ~0 effect on hidden states with norm
    # ~3000; use --absolute-magnitude to opt into legacy absolute units.
    parser.add_argument("--magnitude", type=float, default=0.15,
                        help="Relative (fraction of hidden norm) by default.")
    parser.add_argument("--absolute-magnitude", action="store_true",
                        help="Interpret --magnitude as absolute L2 (legacy).")
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
    add_probe_layers_arg(parser)
    add_sampling_args(parser)
    add_seed_sweep_arg(parser)


def _run_one(args, seed: int):
    resolved_layer = resolve_semantic_layer(args.layer, None)

    cfg = StressConfig(
        prompt=args.prompt,
        model_key=args.model,
        max_new_tokens=args.max_tokens,
        backend=args.backend,
        nnsight_remote=args.nnsight_remote,
        nnsight_device=args.nnsight_device,
        seed=seed,
        intervention_layer=int(resolved_layer),
        intervention_type=args.type,
        intervention_magnitude=args.magnitude,
        intervention_magnitude_relative=(not args.absolute_magnitude),
        intervention_start=args.start,
        intervention_duration=args.duration,
        with_diagnostics=not args.no_diagnostics,
        temperature=float(getattr(args, "temperature", 0.0)),
        top_p=float(getattr(args, "top_p", 1.0)),
        top_k=int(getattr(args, "top_k", 0)),
    )

    probe_layers = resolve_probe_layers(args.probe_layers, None)
    # Make sure the intervention layer is covered so the delta shows up.
    if int(resolved_layer) not in probe_layers:
        probe_layers = [int(resolved_layer), *probe_layers]

    diag_cfg = DiagnosticsConfig(
        enabled=not args.no_diagnostics,
        probe_layers=probe_layers if not args.no_diagnostics else [],
    )

    intervention_kwargs = {}
    if args.type == "sae":
        sae_layer = args.sae_layer if args.sae_layer is not None else resolved_layer
        intervention_kwargs = {
            "repo_id": args.sae_repo,
            "sae_id": args.sae_id,
            "layer": None if args.sae_id is not None else int(sae_layer),
            "feature_idx": int(args.sae_feature_idx),
            "strength": float(args.sae_strength),
            "normalize": bool(not args.sae_no_normalize),
            "device": args.nnsight_device,
            "name": f"sae_f{int(args.sae_feature_idx)}_s{float(args.sae_strength):.2f}",
        }

    return run_stress_experiment(
        config=cfg,
        registry_path=args.registry_path,
        runs_dir=args.runs_dir,
        diagnostics_config=diag_cfg,
        intervention_kwargs=intervention_kwargs,
    )


def run_from_args(args) -> None:
    seeds = parse_seeds(getattr(args, "seeds", None))
    if not seeds:
        _run_one(args, int(args.seed))
        return
    run_sweep(
        mode="stress",
        seeds=seeds,
        runs_dir=args.runs_dir,
        run_once=lambda s: _run_one(args, s),
        describe={
            "prompt": args.prompt,
            "model": args.model,
            # Store as {raw, resolved}: raw preserves the original spec
            # (possibly "mid"), resolved is the int actually used.
            "layer": {
                "raw": args.layer,
                "resolved": resolve_semantic_layer(args.layer, None),
            },
            "type": args.type,
            "magnitude": float(args.magnitude),
            "relative": not args.absolute_magnitude,
            "start": int(args.start),
            "duration": int(args.duration),
        },
    )
