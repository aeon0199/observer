"""plotter.py — V1 Summary Plot Generator
----------------------------------------
Generates lab / investor-friendly plots from a V1-style summary dict
or summary.json.

V1 produces stage-level telemetry (BASE / PERTURB / REASK), not dense per-token
telemetry like V1.5. These plots therefore focus on stage comparisons.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

import numpy as np

# Prefer a non-interactive backend (safe for headless environments).
try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:
    pass

import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _safe_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _stages() -> list[str]:
    return ["base", "perturb", "reask"]


# ─────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────

def plot_stage_vitals(summary: Dict[str, Any], out_dir: str) -> str:
    """Plot hidden_norm / entropy / logit_norm across BASE/PERTURB/REASK."""
    tel = summary.get("telemetry", {}) or {}
    stages = _stages()

    hidden_norm = np.array([_safe_float(_get(_get(tel, s, {}), "hidden_norm")) for s in stages], dtype=float)
    entropy = np.array([_safe_float(_get(_get(tel, s, {}), "entropy")) for s in stages], dtype=float)
    logit_norm = np.array([_safe_float(_get(_get(tel, s, {}), "logit_norm")) for s in stages], dtype=float)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)

    def _bar(ax, values: np.ndarray, title: str, ylabel: str):
        ax.bar(stages, values)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        for i, v in enumerate(values):
            if np.isnan(v):
                continue
            ax.text(i, v, f"{v:.4g}", ha="center", va="bottom", fontsize=9)

    _bar(axes[0], hidden_norm, "V1 Stage Vitals — Hidden Norm", "hidden_norm")
    _bar(axes[1], entropy, "V1 Stage Vitals — Entropy", "entropy")
    _bar(axes[2], logit_norm, "V1 Stage Vitals — Logit Norm", "logit_norm")

    axes[2].set_xlabel("stage")

    _ensure_dir(out_dir)
    path = os.path.join(out_dir, "stage_vitals.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_distribution_shift(summary: Dict[str, Any], out_dir: str) -> str:
    """Plot JS divergence (bits) bars for matched context logits."""
    ds = summary.get("distribution_shift", {}) or {}

    labels = ["base→perturb", "base→reask", "perturb→reask"]
    values = np.array(
        [
            _safe_float(_get(ds, "js_base_vs_perturb_ctx")),
            _safe_float(_get(ds, "js_base_vs_reask_ctx")),
            _safe_float(_get(ds, "js_perturb_vs_reask_ctx")),
        ],
        dtype=float,
    )

    units = str(_get(ds, "js_units", "bits"))

    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)

    ax.bar(labels, values)
    ax.set_title(f"V1 Distribution Shift — JS Divergence ({units})")
    ax.set_ylabel(f"JS divergence ({units})")
    ax.grid(True, axis="y", alpha=0.25)

    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        ax.text(i, v, f"{v:.4g}", ha="center", va="bottom", fontsize=9)

    _ensure_dir(out_dir)
    path = os.path.join(out_dir, "distribution_shift_js.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_headline_metrics(summary: Dict[str, Any], out_dir: str) -> str:
    """Generate a single 'scorecard' image with key metrics for screenshots."""
    cfg = summary.get("config", {}) or {}
    metrics = summary.get("metrics", {}) or {}
    ds = summary.get("distribution_shift", {}) or {}
    seed = summary.get("seed_cache", {}) or {}

    model = str(_get(cfg, "model", ""))
    stamp = str(_get(summary, "timestamp", ""))
    config_hash = str(_get(summary, "config_hash", ""))

    drift = _safe_float(_get(metrics, "drift"))
    hysteresis = _safe_float(_get(metrics, "hysteresis"))
    recovery = _safe_float(_get(metrics, "recovery"))
    regime = str(_get(metrics, "regime", "")).upper()

    js_units = str(_get(ds, "js_units", "bits"))
    js_bp = _safe_float(_get(ds, "js_base_vs_perturb_ctx"))
    js_br = _safe_float(_get(ds, "js_base_vs_reask_ctx"))
    js_pr = _safe_float(_get(ds, "js_perturb_vs_reask_ctx"))

    fingerprint = str(_get(seed, "fingerprint", ""))
    fp_avail = bool(_get(seed, "fingerprint_available", False))

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    ax.axis("off")

    lines = [
        "V1 — Run Summary",
        "",
        f"Model: {model}",
        f"Timestamp: {stamp}",
        f"Config hash: {config_hash}",
        "",
        f"Seed cache fingerprint: {fingerprint}  (available={fp_avail})",
        "",
        "Core Metrics:",
        f"  Drift (D):      {drift:.6g}",
        f"  Hysteresis (H): {hysteresis:.6g}",
        f"  Recovery (R):   {recovery:.6g}",
        f"  Regime:         {regime}",
        "",
        f"Distribution Shift (JS, {js_units}):",
        f"  JS(base, perturb): {js_bp:.6g}",
        f"  JS(base, reask):   {js_br:.6g}",
        f"  JS(perturb, reask):{js_pr:.6g}",
    ]

    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)

    _ensure_dir(out_dir)
    path = os.path.join(out_dir, "headline_metrics.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def generate_plots(summary: Dict[str, Any], run_dir: str) -> Dict[str, Optional[str]]:
    """Generate all V1 plots into {run_dir}/plots and return their paths."""
    plots_dir = os.path.join(run_dir, "plots")

    out: Dict[str, Optional[str]] = {}
    out["stage_vitals"] = plot_stage_vitals(summary, plots_dir)
    out["distribution_shift_js"] = plot_distribution_shift(summary, plots_dir)
    out["headline_metrics"] = plot_headline_metrics(summary, plots_dir)

    return out


def load_summary_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate V1 plots from summary.json")
    parser.add_argument("summary_json", help="Path to runs/.../summary.json")
    args = parser.parse_args()

    summary = load_summary_json(args.summary_json)
    run_dir = os.path.dirname(os.path.abspath(args.summary_json))

    paths = generate_plots(summary, run_dir)
    print("Generated plots:")
    for k, v in paths.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
