"""Seed-sweep driver.

Runs a single-seed callable N times and writes an aggregated summary that
captures mean / std / per-seed values for every scalar metric surfaced in the
individual run summaries. Individual run artifacts are preserved; an additional
`sweep_<mode>_<timestamp>/` directory is written at the top level with the
aggregate and pointers back to each run.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


def _runs_dir(override: Optional[str]) -> Path:
    root = Path(override) if override else Path(os.environ.get("RUNS_DIR", "runs"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _scalar_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    """Flatten the summary into {dotted.path: float} for everything numeric."""
    out: Dict[str, float] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(node, list):
            # Only include if list of numbers — record mean & count.
            if node and all(isinstance(x, (int, float)) for x in node):
                try:
                    out[f"{prefix}.mean"] = float(sum(node) / len(node))
                    out[f"{prefix}.len"] = float(len(node))
                except Exception:
                    pass
        elif isinstance(node, bool):
            out[prefix] = 1.0 if node else 0.0
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            if math.isfinite(float(node)):
                out[prefix] = float(node)

    walk("", summary or {})
    return out


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, Any]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row.keys()})
    agg: Dict[str, Any] = {}
    for k in keys:
        vals = [row[k] for row in rows if k in row]
        if not vals:
            continue
        entry: Dict[str, Any] = {
            "mean": float(statistics.mean(vals)),
            "n": int(len(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
        }
        if len(vals) >= 2:
            entry["stdev"] = float(statistics.stdev(vals))
            entry["median"] = float(statistics.median(vals))
        agg[k] = entry
    return agg


def run_sweep(
    mode: str,
    seeds: Iterable[int],
    runs_dir: Optional[str],
    run_once: Callable[[int], Dict[str, Any]],
    describe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    seeds = [int(s) for s in seeds]
    base = _runs_dir(runs_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = base / f"sweep_{mode}_{stamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] mode={mode} seeds={seeds}")
    print(f"[sweep] sweep_dir={sweep_dir}")

    per_run: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, float]] = []

    for i, seed in enumerate(seeds, 1):
        print(f"[sweep] ({i}/{len(seeds)}) seed={seed}")
        try:
            result = run_once(seed) or {}
        except Exception as e:
            print(f"[sweep]   FAILED: {e}")
            per_run.append({"seed": seed, "error": str(e)})
            continue

        summary = result.get("summary", result)  # fall back to raw
        run_id = None
        artifacts = (summary or {}).get("artifacts") or {}
        run_dir = artifacts.get("run_dir")
        if run_dir:
            run_id = Path(run_dir).name

        per_run.append({
            "seed": seed,
            "run_id": run_id,
            "run_dir": run_dir,
            "status": "ok",
        })
        metric_rows.append(_scalar_metrics(summary or {}))

    aggregate = _aggregate(metric_rows)

    out = {
        "mode": mode,
        "timestamp": stamp,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "n_ok": sum(1 for r in per_run if r.get("status") == "ok"),
        "describe": describe or {},
        "per_run": per_run,
        "aggregate": aggregate,
        "sweep_dir": str(sweep_dir),
    }
    out_path = sweep_dir / "sweep.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[sweep] done → {out_path}")

    # Also write a narrow headline for the dashboard.
    headline = {
        "mode": mode,
        "n_seeds": len(seeds),
        "aggregate": {k: v for k, v in aggregate.items() if k.split(".")[-1] in {"mean", "stdev"}},
    }
    (sweep_dir / "headline.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")

    return out
