from __future__ import annotations

import argparse

from runtime_lab.config.schemas import CommonRunConfig, DiagnosticsConfig
from runtime_lab.observe.runner import run_observe_experiment
from ._common import (
    add_probe_layers_arg,
    add_sampling_args,
    add_seed_sweep_arg,
    parse_seeds,
    resolve_probe_layers,
)
from ._sweep import run_sweep


def add_observe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default="Explain how airplanes fly in a clear, accurate way.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="hf", choices=["hf", "nnsight"])
    parser.add_argument("--nnsight-remote", action="store_true")
    parser.add_argument("--nnsight-device", default=None)
    parser.add_argument("--registry-path", default="models.json")
    parser.add_argument("--runs-dir", default=None)
    add_probe_layers_arg(parser)
    add_sampling_args(parser)
    add_seed_sweep_arg(parser)


def _run_one(args, seed: int) -> dict:
    cfg = CommonRunConfig(
        prompt=args.prompt,
        model_key=args.model,
        max_new_tokens=args.max_tokens,
        backend=args.backend,
        nnsight_remote=args.nnsight_remote,
        nnsight_device=args.nnsight_device,
        seed=seed,
        temperature=float(getattr(args, "temperature", 0.0)),
        top_p=float(getattr(args, "top_p", 1.0)),
        top_k=int(getattr(args, "top_k", 0)),
    )
    diag_cfg = DiagnosticsConfig(
        enabled=True,
        probe_layers=resolve_probe_layers(args.probe_layers, None),
    )
    return run_observe_experiment(
        config=cfg,
        registry_path=args.registry_path,
        runs_dir=args.runs_dir,
        diagnostics_config=diag_cfg,
    )


def run_from_args(args) -> None:
    seeds = parse_seeds(getattr(args, "seeds", None))
    if not seeds:
        _run_one(args, int(args.seed))
        return
    run_sweep(
        mode="observe",
        seeds=seeds,
        runs_dir=args.runs_dir,
        run_once=lambda s: _run_one(args, s),
        describe={
            "prompt": args.prompt,
            "model": args.model,
            "max_tokens": int(args.max_tokens),
            "probe_layers": args.probe_layers,
        },
    )
