from __future__ import annotations

import argparse

from runtime_lab.config.schemas import CommonRunConfig, DiagnosticsConfig
from runtime_lab.observe.runner import run_observe_experiment


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


def run_from_args(args) -> None:
    cfg = CommonRunConfig(
        prompt=args.prompt,
        model_key=args.model,
        max_new_tokens=args.max_tokens,
        backend=args.backend,
        nnsight_remote=args.nnsight_remote,
        nnsight_device=args.nnsight_device,
        seed=args.seed,
    )
    diag_cfg = DiagnosticsConfig(enabled=True, probe_layers=[-1])
    run_observe_experiment(
        config=cfg,
        registry_path=args.registry_path,
        runs_dir=args.runs_dir,
        diagnostics_config=diag_cfg,
    )
