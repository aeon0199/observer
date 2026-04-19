"""Shared CLI helpers.

Probe-layer parsing, seed-sweep iteration, semantic layer defaults, and
temperature-sampling args all live here so each mode's parser stays tight.

Semantic layer spec strings supported anywhere an int layer is accepted:

    "mid"   -> n // 2
    "mid-"  -> max(0, n//2 - 2)
    "mid+"  -> min(n-1, n//2 + 2)
    "late"  -> n - 1
    "early" -> max(0, n // 4 - 1)

These resolve at runtime (when the model is known) and the resolved value is
written back into the run summary so the LLM driving the next experiment can
see what "mid" actually was for this model.
"""
from __future__ import annotations

import argparse
import re
from typing import List, Optional, Union


SEMANTIC_LAYER_ALIASES = {"mid", "mid-", "mid+", "late", "early", "auto"}


def add_probe_layers_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--probe-layers",
        default="auto",
        help=(
            "Layers to instrument for diagnostics. 'auto' picks mid-stack + "
            "final: [n//4, n//2, 3n//4, -1]. Or pass comma-separated ints "
            "(e.g. '2,6,10,14'). Negative indices count from the end."
        ),
    )


def add_seed_sweep_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seeds",
        default=None,
        help=(
            "Run N seeds for aggregation. Either comma-separated "
            "('1,2,3,4,5') or a range ('0-9'). When omitted, a single run "
            "uses --seed."
        ),
    )


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature for token selection. 0 = greedy (default, "
            "legacy behavior). Use 0.7–1.0 to let logit-level perturbations "
            "actually flip tokens — essential for measuring stress/noise effects."
        ),
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling cutoff.")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling cap (0 = off).")


def parse_seeds(spec: Optional[str]) -> List[int]:
    if not spec:
        return []
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        m = re.match(r"^(-?\d+)\s*-\s*(-?\d+)$", spec)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            return list(range(a, b + 1))
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [int(p) for p in parts]


def resolve_semantic_layer(spec: Union[str, int], num_layers: Optional[int]) -> int:
    """Resolve a single layer spec. Accepts ints (passed through), negative
    indices (passed through), or semantic strings. If num_layers is None we
    defer with a best-effort fallback."""
    if isinstance(spec, int):
        return int(spec)
    s = str(spec).strip().lower()
    if s == "" or s.lstrip("-").isdigit():
        return int(s)
    if num_layers is None:
        # Fallback using negative indexing so the backend can still resolve.
        return {
            "mid": -14, "mid-": -16, "mid+": -12,
            "late": -1, "early": -22, "auto": -14,
        }.get(s, -1)
    n = int(num_layers)
    return {
        "mid":   max(0, n // 2 - 1),
        "mid-":  max(0, n // 2 - 3),
        "mid+":  min(n - 1, n // 2 + 1),
        "late":  n - 1,
        "early": max(0, n // 4 - 1),
        "auto":  max(0, n // 2 - 1),
    }.get(s, n - 1)


def resolve_probe_layers(spec: str, num_layers: Optional[int]) -> List[int]:
    """Resolve the --probe-layers argument against a concrete layer count.

    Callers that don't yet know num_layers should pass None and defer to
    runtime. For 'auto', we return a small default set using negative
    indices so the backend can apply them without knowing depth."""
    spec = (spec or "").strip()
    if spec == "" or spec.lower() == "auto":
        if num_layers is None:
            # Sensible defaults with negative indexing. For a 28-layer model
            # these map to depths ~25/50/75/~100%; for small models some may
            # collide, which the probe handles.
            return [-1, -7, -14, -21]
        n = int(num_layers)
        return sorted({max(0, n // 4 - 1), max(0, n // 2 - 1), max(0, 3 * n // 4 - 1), n - 1})
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [int(p) for p in parts]


def describe_layer_resolution(raw: Union[str, int], resolved: int, num_layers: Optional[int]) -> str:
    if isinstance(raw, str) and raw.lower() in SEMANTIC_LAYER_ALIASES:
        if num_layers is not None:
            return f"{raw} -> layer {resolved} (of {num_layers})"
        return f"{raw} (deferred)"
    return str(resolved)
