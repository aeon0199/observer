#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


TIMESTAMP_RE = re.compile(r"control_run_(\d{8})_(\d{6})")


@dataclass
class RunRecord:
    run_dir: Path
    run_name: str
    timestamp: str
    config: Dict[str, Any]
    controller: Dict[str, Any]
    summary: Dict[str, Any]
    events: List[Dict[str, Any]]
    tokens: int
    shadow: bool
    applied_count: int


@dataclass
class PairRecord:
    pair_id: str
    shadow_run: RunRecord
    active_run: RunRecord
    base_key: Tuple[Any, ...]
    variant_key: Tuple[Any, ...]
    rows: List[Dict[str, Any]]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_events(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _run_timestamp(name: str) -> str:
    match = TIMESTAMP_RE.match(name)
    if not match:
        return name
    return "".join(match.groups())


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    return None


def _flatten_numeric(prefix: str, payload: Any, out: Dict[str, float]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric(next_prefix, value, out)
        return
    val = _safe_float(payload)
    if val is not None:
        out[prefix] = val


def _token_features(token_text: str, step_idx: int) -> Dict[str, float]:
    stripped = token_text.strip()
    is_blank = 1.0 if stripped == "" else 0.0
    has_leading_space = 1.0 if token_text.startswith(" ") else 0.0
    is_punct = 1.0 if stripped and all(ch in string.punctuation for ch in stripped) else 0.0
    is_numeric = 1.0 if stripped.isdigit() and stripped else 0.0
    has_newline = 1.0 if "\n" in token_text else 0.0
    return {
        "step_idx": float(step_idx),
        "token_len": float(len(token_text)),
        "token_is_blank": is_blank,
        "token_has_leading_space": has_leading_space,
        "token_is_punct": is_punct,
        "token_is_numeric": is_numeric,
        "token_has_newline": has_newline,
        "token_is_initial": 1.0 if step_idx == 0 else 0.0,
    }


def _event_features(events: Sequence[Dict[str, Any]], idx: int) -> Dict[str, float]:
    """Extract per-step features for the branchpoint predictor.

    Must be called on SHADOW events — features from active runs include
    intervention-downstream fields that leak the label into the predictor.
    Circular features explicitly excluded: intervention_applied, scale_used,
    next_scale, controller_drift_norm, controller_ema_hidden_norm,
    controller_reference_hidden_norm, pre_post_delta_norm (this one directly
    measures the intervention delta when fired).
    """
    event = events[idx]
    features: Dict[str, float] = {}
    # Honest clean-trajectory features only.
    for key in (
        "raw_div",
        "avg_score",
        "control_score",
        "pre_hidden_norm",
        "post_hidden_norm",
    ):
        val = _safe_float(event.get(key))
        if val is not None:
            features[key] = val

    features.update(_token_features(str(event.get("token_text", "")), idx))

    if idx > 0:
        prev = events[idx - 1]
        for src_key, feat_key in (
            ("raw_div", "delta_raw_div"),
            ("avg_score", "delta_avg_score"),
            ("control_score", "delta_control_score"),
            ("pre_hidden_norm", "delta_pre_hidden_norm"),
        ):
            cur_val = _safe_float(event.get(src_key))
            prev_val = _safe_float(prev.get(src_key))
            if cur_val is not None and prev_val is not None:
                features[feat_key] = cur_val - prev_val

    diagnostics = event.get("diagnostics") or {}
    numeric_diag: Dict[str, float] = {}
    _flatten_numeric("diagnostics", diagnostics, numeric_diag)
    features.update(numeric_diag)
    return features


def _base_key(run: RunRecord) -> Tuple[Any, ...]:
    controller = run.controller
    return (
        run.config.get("prompt"),
        run.config.get("model"),
        run.config.get("backend"),
        run.config.get("max_new_tokens"),
        controller.get("temperature"),
        controller.get("top_p"),
        controller.get("top_k"),
        run.config.get("measure_layer"),
        run.config.get("act_layer"),
        run.config.get("seed"),
    )


def _variant_key(run: RunRecord) -> Tuple[Any, ...]:
    controller = run.controller
    return (
        run.config.get("intervention_type"),
        controller.get("additive_direction"),
        controller.get("additive_reference"),
        controller.get("additive_warn_magnitude"),
        controller.get("additive_crit_magnitude"),
        controller.get("additive_seed"),
        controller.get("threshold_warn"),
        controller.get("threshold_crit"),
        controller.get("hold_warn"),
        controller.get("hold_crit"),
        controller.get("scale_warn"),
        controller.get("scale_crit"),
    )


def _variant_label(run: RunRecord) -> str:
    controller = run.controller
    parts = [str(run.config.get("intervention_type"))]
    if run.config.get("intervention_type") == "additive":
        parts.append(str(controller.get("additive_direction")))
        parts.append(str(controller.get("additive_reference")))
        parts.append(f"mag={controller.get('additive_warn_magnitude')}/{controller.get('additive_crit_magnitude')}")
    else:
        parts.append(f"scale={controller.get('scale_warn')}/{controller.get('scale_crit')}")
    return "|".join(parts)


def _prompt_label(prompt: str, max_len: int = 36) -> str:
    prompt = " ".join(str(prompt).split())
    return prompt if len(prompt) <= max_len else prompt[: max_len - 3] + "..."


def _load_runs(runs_dir: Path) -> List[RunRecord]:
    records: List[RunRecord] = []
    for run_dir in sorted(runs_dir.glob("control_run_*")):
        cfg_path = run_dir / "config.json"
        summary_path = run_dir / "summary.json"
        events_path = run_dir / "events.jsonl"
        if not (cfg_path.exists() and summary_path.exists() and events_path.exists()):
            continue
        cfg_root = _load_json(cfg_path)
        summary = _load_json(summary_path)
        events = _load_events(events_path)
        config = cfg_root.get("config", {})
        controller = config.get("controller", {})
        applied_count = sum(1 for event in events if event.get("intervention_applied"))
        records.append(
            RunRecord(
                run_dir=run_dir,
                run_name=run_dir.name,
                timestamp=_run_timestamp(run_dir.name),
                config=config,
                controller=controller,
                summary=summary,
                events=events,
                tokens=int(summary.get("tokens", len(events))),
                shadow=bool(config.get("shadow")),
                applied_count=applied_count,
            )
        )
    return records


def _filter_selected(run: RunRecord, args: argparse.Namespace) -> bool:
    if args.model and run.config.get("model") != args.model:
        return False
    if args.prompt_contains and args.prompt_contains not in str(run.config.get("prompt", "")):
        return False
    if args.measure_layer is not None and int(run.config.get("measure_layer")) != int(args.measure_layer):
        return False
    if args.act_layer is not None and int(run.config.get("act_layer")) != int(args.act_layer):
        return False
    if args.intervention_type and not run.shadow and run.config.get("intervention_type") != args.intervention_type:
        return False
    if args.additive_direction and not run.shadow:
        if str(run.controller.get("additive_direction")) != args.additive_direction:
            return False
    if args.additive_reference and not run.shadow:
        if str(run.controller.get("additive_reference")) != args.additive_reference:
            return False
    return True


def _nearest_shadow(active: RunRecord, shadows: Sequence[RunRecord]) -> Optional[RunRecord]:
    if not shadows:
        return None
    active_ts = active.timestamp
    return min(
        shadows,
        key=lambda candidate: abs(int(candidate.timestamp) - int(active_ts)),
    )


def _build_pairs(records: Sequence[RunRecord], args: argparse.Namespace) -> Tuple[List[PairRecord], Dict[str, int], Dict[str, Any]]:
    counts = Counter()
    counts["total_control_runs"] = len(records)

    selected: List[RunRecord] = []
    for run in records:
        if not _filter_selected(run, args):
            counts["dropped_not_selected"] += 1
            continue
        if run.tokens < args.min_tokens:
            counts["dropped_eos_collapse"] += 1
            continue
        if run.shadow:
            if run.applied_count != 0:
                counts["dropped_shadow_dirty"] += 1
                continue
        else:
            if run.applied_count <= 0:
                counts["dropped_active_never_fired"] += 1
                continue
        selected.append(run)

    by_base: Dict[Tuple[Any, ...], Dict[str, List[RunRecord]]] = defaultdict(lambda: {"shadow": [], "active": []})
    for run in selected:
        kind = "shadow" if run.shadow else "active"
        by_base[_base_key(run)][kind].append(run)

    pairs: List[PairRecord] = []
    valid_pair_counter = 0
    variant_counter = Counter()
    base_counter = Counter()
    for base_key, bucket in by_base.items():
        shadows = sorted(bucket["shadow"], key=lambda run: run.timestamp)
        actives = sorted(bucket["active"], key=lambda run: run.timestamp)
        if not shadows:
            counts["dropped_no_matching_pair"] += len(actives)
            continue
        for active in actives:
            shadow = _nearest_shadow(active, shadows)
            if shadow is None:
                counts["dropped_no_matching_pair"] += 1
                continue
            variant_key = _variant_key(active)
            pair_id = f"pair_{valid_pair_counter:04d}"
            rows: List[Dict[str, Any]] = []
            for idx, (shadow_event, active_event) in enumerate(zip(shadow.events, active.events)):
                # IMPORTANT: features come from SHADOW (unperturbed) events.
                # Previous version read from active.events, which leaked
                # intervention-applied / scale-used / controller-drift-norm
                # into the predictor — those are all strictly zero on steps
                # where the controller didn't fire, so the classifier was
                # just relearning "flip iff controller fired" (circular).
                # Q1 asks: given a CLEAN-trajectory step, would a perturbation
                # here flip the argmax? That requires clean-trajectory features.
                features = _event_features(shadow.events, idx)
                rows.append(
                    {
                        "pair_id": pair_id,
                        "t": idx,
                        "flip": 1 if shadow_event.get("token_text") != active_event.get("token_text") else 0,
                        "shadow_token_text": shadow_event.get("token_text"),
                        "active_token_text": active_event.get("token_text"),
                        "shadow_run": shadow.run_name,
                        "active_run": active.run_name,
                        "variant_label": _variant_label(active),
                        "model": active.config.get("model"),
                        "measure_layer": active.config.get("measure_layer"),
                        "act_layer": active.config.get("act_layer"),
                        **features,
                    }
                )
            pairs.append(
                PairRecord(
                    pair_id=pair_id,
                    shadow_run=shadow,
                    active_run=active,
                    base_key=base_key,
                    variant_key=variant_key,
                    rows=rows,
                )
            )
            valid_pair_counter += 1
            variant_counter[_variant_label(active)] += 1
            base_counter[
                (
                    active.config.get("model"),
                    _prompt_label(active.config.get("prompt", "")),
                    active.config.get("seed"),
                    active.config.get("measure_layer"),
                    active.config.get("act_layer"),
                )
            ] += 1

    counts["selected_runs"] = len(selected)
    counts["valid_pairs"] = len(pairs)
    counts["valid_shadow_runs"] = sum(1 for run in selected if run.shadow)
    counts["valid_active_runs"] = sum(1 for run in selected if not run.shadow)
    meta = {
        "variant_counts": dict(variant_counter),
        "pair_groups": [
            {
                "model": model,
                "prompt": prompt,
                "seed": seed,
                "measure_layer": measure_layer,
                "act_layer": act_layer,
                "n_pairs": n_pairs,
            }
            for (model, prompt, seed, measure_layer, act_layer), n_pairs in sorted(base_counter.items())
        ],
    }
    return pairs, dict(counts), meta


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.zeros(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def _roc_auc_score(y_true: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = _average_ranks(s)
    sum_ranks_pos = float(np.sum(ranks[y == 1]))
    u = sum_ranks_pos - pos * (pos + 1) / 2.0
    return u / (pos * neg)


def _train_test_pair_split(pair_ids: Sequence[str], rows_by_pair: Dict[str, List[int]], y: np.ndarray, train_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    unique_pairs = list(pair_ids)
    for _ in range(64):
        rng.shuffle(unique_pairs)
        cut = max(1, min(len(unique_pairs) - 1, int(round(len(unique_pairs) * train_frac))))
        train_pairs = set(unique_pairs[:cut])
        test_pairs = set(unique_pairs[cut:])
        train_idx = [idx for pair in train_pairs for idx in rows_by_pair[pair]]
        test_idx = [idx for pair in test_pairs for idx in rows_by_pair[pair]]
        if not train_idx or not test_idx:
            continue
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(set(y_train.tolist())) >= 2 and len(set(y_test.tolist())) >= 2:
            return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)
    raise ValueError("Could not create a pair-level train/test split with both classes present.")


def _fit_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    l2_reg: float = 1e-3,
    learning_rate: float = 0.02,
    max_iter: int = 2500,
) -> Tuple[np.ndarray, np.ndarray]:
    weights = np.zeros(X_train.shape[1], dtype=float)
    bias = 0.0
    for _ in range(max_iter):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            logits = np.clip(X_train @ weights + bias, -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            error = probs - y_train
            grad_w = (X_train.T @ error) / len(y_train) + l2_reg * weights
        grad_b = float(np.mean(error))
        grad_norm = float(np.linalg.norm(grad_w))
        if grad_norm > 10.0:
            grad_w *= 10.0 / grad_norm
        grad_b = float(np.clip(grad_b, -1.0, 1.0))
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
        weights = np.clip(weights, -25.0, 25.0)
        bias = float(np.clip(bias, -25.0, 25.0))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        train_scores = 1.0 / (1.0 + np.exp(-np.clip(X_train @ weights + bias, -30.0, 30.0)))
        test_scores = 1.0 / (1.0 + np.exp(-np.clip(X_test @ weights + bias, -30.0, 30.0)))
    return train_scores, test_scores, weights


def _median_impute(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    train_imputed = np.where(np.isnan(train), medians, train)
    test_imputed = np.where(np.isnan(test), medians, test)
    return train_imputed, test_imputed


def _standardize(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    train_scaled = (train - mean) / std
    test_scaled = (test - mean) / std
    train_scaled = np.nan_to_num(train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    test_scaled = np.nan_to_num(test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    train_scaled = np.clip(train_scaled, -20.0, 20.0)
    test_scaled = np.clip(test_scaled, -20.0, 20.0)
    return train_scaled, test_scaled


def _analyze_pairs(pairs: Sequence[PairRecord], args: argparse.Namespace) -> Dict[str, Any]:
    if not pairs:
        raise ValueError("No valid shadow/active pairs found after filtering.")

    step_rows = [row for pair in pairs for row in pair.rows]
    if not step_rows:
        raise ValueError("Pairs were found, but there are no aligned step rows to analyze.")

    feature_names = sorted(
        {
            key
            for row in step_rows
            for key, value in row.items()
            if key
            not in {
                "pair_id",
                "flip",
                "shadow_token_text",
                "active_token_text",
                "shadow_run",
                "active_run",
                "variant_label",
                "model",
                "measure_layer",
                "act_layer",
            }
            and isinstance(value, (int, float))
        }
    )

    rows_by_pair: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(step_rows):
        rows_by_pair[str(row["pair_id"])].append(idx)

    y = np.asarray([int(row["flip"]) for row in step_rows], dtype=int)
    pair_ids = list(rows_by_pair.keys())
    train_idx, test_idx = _train_test_pair_split(pair_ids, rows_by_pair, y, args.train_frac, args.seed)

    univariate = []
    for feat in feature_names:
        scores = np.asarray(
            [
                float(row.get(feat)) if row.get(feat) is not None else np.nan
                for row in step_rows
            ],
            dtype=float,
        )
        valid_mask = ~np.isnan(scores)
        coverage = float(np.mean(valid_mask))
        auc = float("nan")
        if np.any(valid_mask) and len(set(y[valid_mask].tolist())) >= 2:
            auc = _roc_auc_score(y[valid_mask], scores[valid_mask])
        univariate.append(
            {
                "feature": feat,
                "coverage": coverage,
                "auroc": auc,
            }
        )
    univariate.sort(key=lambda row: (float("-inf") if math.isnan(row["auroc"]) else row["auroc"]), reverse=True)

    selected_features = [row["feature"] for row in univariate if row["coverage"] >= args.min_feature_coverage]
    if not selected_features:
        raise ValueError("No features met the minimum coverage threshold for the combined classifier.")

    X = np.asarray(
        [
            [float(step_rows[row_idx].get(feat)) if step_rows[row_idx].get(feat) is not None else np.nan for feat in selected_features]
            for row_idx in range(len(step_rows))
        ],
        dtype=float,
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_train, X_test = _median_impute(X_train, X_test)
    X_train, X_test = _standardize(X_train, X_test)
    train_scores, test_scores, weights = _fit_logistic_regression(X_train, y_train, X_test)

    classifier = {
        "train_auroc": _roc_auc_score(y_train, train_scores),
        "test_auroc": _roc_auc_score(y_test, test_scores),
        "n_train_rows": int(len(train_idx)),
        "n_test_rows": int(len(test_idx)),
        "n_features": int(len(selected_features)),
        "selected_features": selected_features,
        "feature_importances": [
            {"feature": feat, "weight": float(weight), "abs_weight": abs(float(weight))}
            for feat, weight in sorted(zip(selected_features, weights.tolist()), key=lambda item: abs(item[1]), reverse=True)
        ],
    }

    flip_rate = float(np.mean(y))
    first_flip_steps: List[int] = []
    variant_counts = Counter()
    for pair in pairs:
        variant_counts[pair.rows[0]["variant_label"]] += 1
        first_flip = next((row["t"] for row in pair.rows if row["flip"] == 1), None)
        if first_flip is not None:
            first_flip_steps.append(int(first_flip))

    return {
        "pair_count": len(pairs),
        "step_count": len(step_rows),
        "flip_rate": flip_rate,
        "first_flip_step_mean": (float(np.mean(first_flip_steps)) if first_flip_steps else None),
        "first_flip_step_median": (float(np.median(first_flip_steps)) if first_flip_steps else None),
        "variant_counts": dict(variant_counts),
        "univariate": univariate,
        "classifier": classifier,
    }


def _print_provenance(counts: Dict[str, int], meta: Dict[str, Any]) -> None:
    print("Provenance")
    print(f"Total control runs:              {counts.get('total_control_runs', 0)}")
    print(f"  dropped (not selected):        {counts.get('dropped_not_selected', 0)}")
    print(f"  dropped (EOS collapse):        {counts.get('dropped_eos_collapse', 0)}")
    print(f"  dropped (active never fired):  {counts.get('dropped_active_never_fired', 0)}")
    print(f"  dropped (dirty shadow):        {counts.get('dropped_shadow_dirty', 0)}")
    print(f"  dropped (no matching pair):    {counts.get('dropped_no_matching_pair', 0)}")
    print(f"Selected runs:                   {counts.get('selected_runs', 0)}")
    print(f"  valid shadow runs:             {counts.get('valid_shadow_runs', 0)}")
    print(f"  valid active runs:             {counts.get('valid_active_runs', 0)}")
    print(f"Valid (shadow, active) pairs:    {counts.get('valid_pairs', 0)}")
    print()
    if meta.get("variant_counts"):
        print("Pairs by active variant")
        for variant, n_pairs in sorted(meta["variant_counts"].items(), key=lambda item: (-item[1], item[0])):
            print(f"  {n_pairs:3d}  {variant}")
        print()
    if meta.get("pair_groups"):
        print("Pair groups (model × prompt × seed × measure × act)")
        for row in meta["pair_groups"]:
            print(
                "  "
                f"{row['model']} | seed={row['seed']} | measure={row['measure_layer']} | act={row['act_layer']} | "
                f"n={row['n_pairs']} | {row['prompt']}"
            )
        print()


def _print_analysis(results: Dict[str, Any], args: argparse.Namespace) -> None:
    print("Label distribution")
    print(f"Step rows analyzed:              {results['step_count']}")
    print(f"Valid pairs analyzed:            {results['pair_count']}")
    print(f"Overall flip rate:               {results['flip_rate']:.4f}")
    if results.get("first_flip_step_mean") is not None:
        print(f"First flip step mean / median:   {results['first_flip_step_mean']:.2f} / {results['first_flip_step_median']:.2f}")
    print()

    print("Feature-by-feature AUROC")
    for row in results["univariate"][: min(args.top_n, len(results["univariate"]))]:
        auc = row["auroc"]
        auc_str = "nan" if math.isnan(auc) else f"{auc:.4f}"
        print(f"  {auc_str:>7}  cov={row['coverage']:.2f}  {row['feature']}")
    print()

    clf = results["classifier"]
    print("Combined classifier")
    print(f"  train AUROC:                   {clf['train_auroc']:.4f}")
    print(f"  held-out test AUROC:           {clf['test_auroc']:.4f}")
    print(f"  train/test rows:               {clf['n_train_rows']} / {clf['n_test_rows']}")
    print(f"  features used:                 {clf['n_features']}")
    print("  top feature importances:")
    for row in clf["feature_importances"][: min(args.top_n, len(clf["feature_importances"]))]:
        print(f"    {row['weight']:+.4f}  {row['feature']}")
    print()

    verdict = "PASS" if clf["test_auroc"] >= args.stop_auroc else "FAIL"
    print("Verdict")
    print(
        f"  {verdict}: held-out AUROC {clf['test_auroc']:.4f} "
        f"{'>=' if verdict == 'PASS' else '<'} stop threshold {args.stop_auroc:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline branchpoint analysis for Observer control runs.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-contains", default=None)
    parser.add_argument("--measure-layer", type=int, default=None)
    parser.add_argument("--act-layer", type=int, default=None)
    parser.add_argument("--intervention-type", default="additive")
    parser.add_argument("--additive-direction", default=None)
    parser.add_argument("--additive-reference", default=None)
    parser.add_argument("--min-tokens", type=int, default=10)
    parser.add_argument("--min-valid-pairs", type=int, default=20)
    parser.add_argument("--min-feature-coverage", type=float, default=0.5)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--stop-auroc", type=float, default=0.8)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    records = _load_runs(runs_dir)
    pairs, counts, meta = _build_pairs(records, args)

    _print_provenance(counts, meta)
    if len(pairs) < args.min_valid_pairs:
        raise SystemExit(
            f"Only {len(pairs)} valid pairs found; minimum required is {args.min_valid_pairs}. "
            "M1.1 should stop here and queue M1.2."
        )

    results = _analyze_pairs(pairs, args)
    _print_analysis(results, args)

    if args.json_out:
        payload = {
            "filters": {
                "model": args.model,
                "prompt_contains": args.prompt_contains,
                "measure_layer": args.measure_layer,
                "act_layer": args.act_layer,
                "intervention_type": args.intervention_type,
                "additive_direction": args.additive_direction,
                "additive_reference": args.additive_reference,
                "min_tokens": args.min_tokens,
            },
            "provenance": counts,
            "meta": meta,
            "analysis": results,
        }
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(payload, indent=2))
        print()
        print(f"JSON summary written to {out_path}")


if __name__ == "__main__":
    main()
