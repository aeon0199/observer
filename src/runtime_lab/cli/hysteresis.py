from __future__ import annotations

import argparse

from runtime_lab.config.schemas import HysteresisConfig
from runtime_lab.hysteresis.runner import run_hysteresis_experiment
from ._common import add_sampling_args, add_seed_sweep_arg, parse_seeds, resolve_semantic_layer
from ._sweep import run_sweep


def add_hysteresis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default="Explain how airplanes fly in a clear, accurate way.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="hf", choices=["hf", "nnsight"])
    parser.add_argument("--nnsight-remote", action="store_true")
    parser.add_argument("--nnsight-device", default=None)
    parser.add_argument("--registry-path", default="models.json")
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument(
        "--perturbation-mode",
        default="prompt",
        choices=["prompt", "noise"],
        help=(
            "'prompt': inject a synthetic <REFLECTION> block (legacy — measures "
            "prompt-contamination persistence). 'noise': apply a seeded additive "
            "perturbation to hidden states during PERTURB, then remove for "
            "REASK (measures true internal-dynamics hysteresis)."
        ),
    )
    # Default: mid-stack — final-layer perturbations are frequently absorbed
    # by confident greedy argmax without propagating.
    parser.add_argument("--noise-layer", default="mid",
                        help="Int or semantic alias (mid, mid-, mid+, late, early).")
    parser.add_argument("--noise-magnitude", type=float, default=0.3,
                        help="Fraction of hidden norm. 0.3 is a reasonable default — "
                             "strong enough to be visible, not so strong it's nonsense.")
    parser.add_argument("--noise-start", type=int, default=3)
    parser.add_argument("--noise-duration", type=int, default=8)
    parser.add_argument("--noise-seed", type=int, default=1234)
    add_sampling_args(parser)
    add_seed_sweep_arg(parser)


def _run_one(args, seed: int):
    cfg = HysteresisConfig(
        prompt=args.prompt,
        model_key=args.model,
        max_new_tokens=args.max_tokens,
        backend=args.backend,
        nnsight_remote=args.nnsight_remote,
        nnsight_device=args.nnsight_device,
        seed=seed,
        perturbation_mode=args.perturbation_mode,
        noise_layer=resolve_semantic_layer(args.noise_layer, None),
        noise_magnitude=args.noise_magnitude,
        noise_start=args.noise_start,
        noise_duration=args.noise_duration,
        noise_seed=args.noise_seed,
        temperature=float(getattr(args, "temperature", 0.0)),
        top_p=float(getattr(args, "top_p", 1.0)),
        top_k=int(getattr(args, "top_k", 0)),
    )
    return run_hysteresis_experiment(
        config=cfg,
        registry_path=args.registry_path,
        runs_dir=args.runs_dir,
    )


def run_from_args(args) -> None:
    seeds = parse_seeds(getattr(args, "seeds", None))
    if not seeds:
        _run_one(args, int(args.seed))
        return
    run_sweep(
        mode="hysteresis",
        seeds=seeds,
        runs_dir=args.runs_dir,
        run_once=lambda s: _run_one(args, s),
        describe={
            "prompt": args.prompt,
            "model": args.model,
            "perturbation_mode": args.perturbation_mode,
        },
    )
