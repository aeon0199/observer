"""Adaptive Controller — Closed-Loop Runtime Stability Controller

This is a standalone runner that composes the baseline/intervention engines.

Design goals:
- Use diagnostics as a streaming "instability" scalar.
- Apply a proportional damping intervention (scale last-token hidden state).
- Avoid import/package issues caused by folder names with spaces.
- Keep the token loop correct (no off-by-one): each printed line corresponds
  to the token that was just *consumed* by the model forward pass.

Note:
- This is a stability controller, not a proven hallucination detector.
  It can be used as an "instability brake" and then empirically calibrated.
"""

from __future__ import annotations

import sys
import os
import json
import hashlib
import importlib.util
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Tuple, Optional, Any, Dict
from datetime import datetime

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic module loading (avoid name collisions: model_loader.py exists in V1/V2)
# ─────────────────────────────────────────────────────────────────────────────

def _load_module_from_path(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing module file: {path}")

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_existing_dir(root: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find any of: {', '.join(candidates)}")


# ─────────────────────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────────────────────

class StabilityController:
    """Composite-score controller with moving average + cooldown.

    Uses divergence as the primary signal, with spectral/SVD terms as
    secondary indicators of instability.
    """

    def __init__(self, ma_window: int = 3):
        self.history = deque(maxlen=int(ma_window))
        self.cooldown_counter = 0
        self.active_scale = 1.0
        self._prev_eff_rank: Optional[float] = None

        # Thresholds over the smoothed control score.
        self.TH_WARN = 0.55
        self.TH_CRIT = 0.85

        # Scales and hold durations.
        self.SCALE_WARN = 0.90
        self.SCALE_CRIT = 0.75
        self.HOLD_WARN = 3
        self.HOLD_CRIT = 6

        # Term weights.
        self.W_DIV = 0.70
        self.W_SPEC_ENT = 0.15
        self.W_HIGH_FRAC = 0.10
        self.W_RANK_DELTA = 0.05

    def _score(self, diagnostics: Dict[str, Any]) -> float:
        div = float(diagnostics.get("divergence", 0.0))
        spectral = diagnostics.get("spectral", {}) or {}
        svd = diagnostics.get("svd", {}) or {}

        spec_entropy = float(spectral.get("spectral_entropy", 0.0))
        high_frac = float(spectral.get("high_frac", 0.0))
        eff_rank = float(svd.get("effective_rank", 0.0))

        rank_delta = 0.0
        if self._prev_eff_rank is not None:
            rank_delta = abs(eff_rank - self._prev_eff_rank)
        self._prev_eff_rank = eff_rank

        spec_term = max(0.0, spec_entropy - 0.75)
        high_term = max(0.0, high_frac - 0.30)
        score = (
            self.W_DIV * div
            + self.W_SPEC_ENT * spec_term
            + self.W_HIGH_FRAC * high_term
            + self.W_RANK_DELTA * rank_delta
        )
        return float(max(0.0, min(2.0, score)))

    def update(self, diagnostics: Dict[str, Any]) -> Tuple[float, float, float, str]:
        score = self._score(diagnostics)
        self.history.append(float(score))
        avg_score = float(sum(self.history) / max(1, len(self.history)))

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            status = "COOLDOWN" if self.active_scale < 1.0 else "STABLE"
            return float(self.active_scale), score, avg_score, status

        if avg_score > self.TH_CRIT:
            self.active_scale = float(self.SCALE_CRIT)
            self.cooldown_counter = int(self.HOLD_CRIT)
            return float(self.active_scale), score, avg_score, "CRITICAL"

        if avg_score > self.TH_WARN:
            self.active_scale = float(self.SCALE_WARN)
            self.cooldown_counter = int(self.HOLD_WARN)
            return float(self.active_scale), score, avg_score, "WARNING"

        self.active_scale = 1.0
        return 1.0, score, avg_score, "STABLE"


# ─────────────────────────────────────────────────────────────────────────────
# Hook / actuator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScaleState:
    value: float = 1.0


def make_scaling_intervention(scale_state: ScaleState):
    """Create a scaling intervention that reads the current scale state."""

    def _intervene(hidden_state: torch.Tensor) -> torch.Tensor:
        s = float(scale_state.value)
        if abs(s - 1.0) <= 1e-6:
            return hidden_state
        modified = hidden_state.clone()
        modified[:, -1, :] = modified[:, -1, :] * s
        return modified

    _intervene.__name__ = "dynamic_scaling"
    return _intervene


class DynamicInterventionHook:
    """Capture last-token hidden state and apply a pluggable intervention."""

    def __init__(self, intervention_fn):
        self.intervention_fn = intervention_fn
        self.active = False
        self.last_hidden: Optional[torch.Tensor] = None  # CPU tensor

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs_out = hs

        if self.active and self.intervention_fn is not None:
            hs_out = self.intervention_fn(hs)

        hidden_last = hs_out[:, -1, :].detach()
        self.last_hidden = hidden_last.cpu()
        if isinstance(output, tuple):
            return (hs_out,) + output[1:]
        return hs_out

    def set_active(self, active: bool) -> None:
        self.active = bool(active)

    def reset(self) -> None:
        self.active = False
        self.last_hidden = None


def _get_layers(model):
    """Best-effort access to the transformer block list."""
    for path in (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("model", "decoder", "layers"),
    ):
        cur = model
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok:
            return cur
    raise AttributeError(
        "Unsupported model structure: couldn't find transformer layers at common locations"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

class Color:
    RESET = "\033[0m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"


def _hash_config(config: dict) -> str:
    """Short stable hash for run reproducibility."""
    config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_adaptive_controller(
    prompt: str,
    model_key: str | None = None,
    max_new_tokens: int = 64,
    layer_idx: int = -1,
    ma_window: int = 3,
    seed: int | None = None,
    runs_dir: str | None = None,
    run_name: str | None = None,
    shadow: bool = False,
    backend: str = "hf",
    nnsight_remote: bool = False,
    nnsight_device: str | None = None,
    intervention_type: str = "scaling",
    sae_repo: str = "apollo-research/llama-3.1-70b-sae",
    sae_id: str | None = None,
    sae_layer: int | None = None,
    sae_feature_idx: int = 0,
    sae_strength: float = 5.0,
    sae_normalize: bool = True,
    generate_dashboard_html: bool = True,
) -> Tuple[str, dict[str, Any]]:
    """Run adaptive-controller loop.

    Returns:
      - full_text: prompt + generated
      - artifacts: paths + summary for downstream use

    Notes:
      - This is a research/demo runner.
      - If shadow=True, we log controller decisions but do not apply the hook.
    """

    # Locate code modules relative to this file.
    root = Path(__file__).resolve().parents[1]
    v2_dir = _resolve_existing_dir(root, ("intervention_engine",))
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

    v2_backend_path = v2_dir / "backend.py"
    v2_diag_bridge_path = v2_dir / "diagnostics_bridge.py"
    v2_sae_adapter_path = v2_dir / "sae_adapter.py"
    dashboard_path = root / "adaptive_controller" / "dashboard.py"

    v2_backend = _load_module_from_path("adaptive_v2_backend", v2_backend_path)
    v2_diag_bridge = _load_module_from_path("adaptive_v2_diag_bridge", v2_diag_bridge_path)
    v2_sae_adapter = _load_module_from_path("adaptive_v2_sae_adapter", v2_sae_adapter_path)
    adaptive_dashboard = _load_module_from_path("adaptive_dashboard_mod", dashboard_path)

    # Resolve run output directory.
    base_runs = Path(runs_dir or os.environ.get("RUNS_DIR", "runs"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = (run_name or "adaptive_demo").strip().replace(" ", "_")
    run_dir = base_runs / f"{safe_name}_{stamp}"
    _ensure_dir(run_dir)

    load_model_with_backend = v2_backend.load_model_with_backend
    DiagnosticsManager = v2_diag_bridge.DiagnosticsManager
    load_sae_decoder_vector = v2_sae_adapter.load_sae_decoder_vector
    make_sae_intervention = v2_sae_adapter.make_sae_intervention
    generate_dashboard = adaptive_dashboard.generate_dashboard

    # Config recorded for lab-facing reproducibility.
    run_config = {
        "mode": "control_demo",
        "shadow": bool(shadow),
        "model_key": model_key,
        "backend": str(backend),
        "nnsight_remote": bool(nnsight_remote),
        "nnsight_device": nnsight_device,
        "max_new_tokens": int(max_new_tokens),
        "layer_idx": int(layer_idx),
        "ma_window": int(ma_window),
        "seed": int(seed) if seed is not None else None,
        "prompt": prompt,
        "intervention_type": str(intervention_type),
        "sae": {
            "repo": sae_repo,
            "sae_id": sae_id,
            "sae_layer": int(sae_layer) if sae_layer is not None else None,
            "feature_idx": int(sae_feature_idx),
            "strength": float(sae_strength),
            "normalize": bool(sae_normalize),
        },
        "controller": {
            "th_warn": 0.55,
            "th_crit": 0.85,
            "scale_warn": 0.90,
            "scale_crit": 0.75,
        },
    }
    config_hash = _hash_config(run_config)

    events_path = run_dir / "events.jsonl"
    summary_path = run_dir / "summary.json"
    output_path = run_dir / "output.txt"

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"config_hash": config_hash, "config": run_config}, f, indent=2, ensure_ascii=False)

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    backend_result = load_model_with_backend(
        model_key=model_key,
        backend=backend,
        nnsight_remote=nnsight_remote,
        nnsight_device=nnsight_device,
    )
    tokenizer = backend_result.tokenizer
    model = backend_result.model
    device = backend_result.device
    loaded_cfg = backend_result.config
    model_id = model_key or getattr(loaded_cfg, "key", None) or getattr(loaded_cfg, "label", None) or "unknown"

    layers = _get_layers(model)
    target_layer = layers[layer_idx]

    probe_layers = list(dict.fromkeys([int(layer_idx), -1]))
    diagnostics_manager = DiagnosticsManager(enabled=True, probe_layers=probe_layers)
    controller = StabilityController(ma_window=ma_window)

    scale_state = ScaleState(1.0)
    if intervention_type == "scaling":
        intervention_fn = make_scaling_intervention(scale_state)
        intervention_name = "dynamic_scaling"
    elif intervention_type == "sae":
        resolved_layer = int(sae_layer) if sae_layer is not None else int(layer_idx)
        decoder_vector = load_sae_decoder_vector(
            repo_id=sae_repo,
            feature_idx=int(sae_feature_idx),
            layer=(None if sae_id is not None else resolved_layer),
            sae_id=sae_id,
            normalize=bool(sae_normalize),
            device=str(device),
        )
        intervention_fn = make_sae_intervention(
            decoder_vector=decoder_vector,
            strength=float(sae_strength),
            name=f"sae_f{int(sae_feature_idx)}_s{float(sae_strength):.2f}",
        )
        intervention_name = str(intervention_fn.__name__)
    else:
        raise ValueError("[AdaptiveController] intervention_type must be one of: scaling, sae")

    hook = DynamicInterventionHook(intervention_fn)
    handle = target_layer.register_forward_hook(hook)

    print(f"\n{Color.CYAN}{'=' * 72}{Color.RESET}")
    print(f"{Color.CYAN}Adaptive Controller — Closed-Loop Stability Controller{Color.RESET}")
    print(f"{Color.CYAN}{'=' * 72}{Color.RESET}\n")

    print(f"{Color.BOLD}Run dir:{Color.RESET} {run_dir}")
    print(f"{Color.BOLD}Config hash:{Color.RESET} {config_hash}")
    print(f"{Color.BOLD}Model:{Color.RESET} {model_id}")
    print(f"{Color.BOLD}Backend:{Color.RESET} {backend_result.backend}")
    if backend_result.backend == "nnsight":
        print(f"{Color.BOLD}NNsight remote:{Color.RESET} {bool(backend_result.backend_meta.get('remote', False))}")
    print(f"{Color.BOLD}Intervention:{Color.RESET} {intervention_name}")
    print(f"{Color.BOLD}Shadow mode:{Color.RESET} {bool(shadow)}")
    print(f"{Color.BOLD}Prompt:{Color.RESET} {prompt}")
    print(f"{Color.DIM}{'-' * 86}{Color.RESET}")
    print(
        f" {'t':>3} | {'raw_div':>8} | {'avg_score':>9} | {'scale_used':>10} |"
        f" {'next_scale':>10} | {'status':<8} | token"
    )
    print(f"{Color.DIM}{'-' * 86}{Color.RESET}")

    generated_text = ""

    # Simple summary stats.
    n_tokens = 0
    n_warning = 0
    n_critical = 0
    n_cooldown = 0
    sum_raw_div = 0.0
    sum_avg_score = 0.0
    events_cache: list[dict[str, Any]] = []

    try:
        # Open event stream.
        with open(events_path, "w", encoding="utf-8") as events_f:
            # Prompt pass (prefill)
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
            prompt_len = int(input_ids.shape[1])

            hook.reset()
            diagnostics_manager.reset()

            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )

            past_key_values = out.past_key_values
            logits0 = out.logits[:, -1, :]
            pending_token_id = int(logits0.argmax(dim=-1).item())

            # Warmup predictor/controller from prompt final hidden.
            if hook.last_hidden is not None:
                diagnostics0 = diagnostics_manager.step(hook.last_hidden, layer_states={int(layer_idx): hook.last_hidden})
                next_scale, control_score, avg_score, status = controller.update(diagnostics0)
                scale_state.value = float(next_scale)
            else:
                control_score = 0.0
                avg_score = 0.0
                status = "NOHOOK"

            # Consume generated tokens one-by-one.
            for t in range(int(max_new_tokens)):
                token_id = int(pending_token_id)

                # Apply the current scale during the forward pass that *processes* this token.
                scale_used = float(scale_state.value)
                # In shadow mode we never activate the hook; we only log decisions.
                hook.set_active((scale_used < 1.0) and (not shadow))

                token_tensor = torch.tensor([[token_id]], device=device)

                # Correct attention mask length: prompt + (t+1) generated tokens consumed.
                attn = torch.ones((1, prompt_len + t + 1), device=device, dtype=torch.long)

                with torch.no_grad():
                    out = model(
                        input_ids=token_tensor,
                        attention_mask=attn,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )

                past_key_values = out.past_key_values
                logits_next = out.logits[:, -1, :]

                # Measure divergence from the hidden state produced while processing token_id.
                if hook.last_hidden is not None:
                    hidden_norm = float(hook.last_hidden.norm().item())
                    diagnostics = diagnostics_manager.step(hook.last_hidden, layer_states={int(layer_idx): hook.last_hidden})
                    raw = float(diagnostics.get("divergence", 0.0))
                    next_scale, control_score, avg_score, status = controller.update(diagnostics)
                    next_scale = float(next_scale)
                else:
                    hidden_norm = 0.0
                    raw = 0.0
                    diagnostics = {}
                    next_scale = float(scale_state.value)
                    control_score = 0.0
                    avg_score = 0.0
                    status = "NOHOOK"

                # Pick the next token (greedy).
                pending_token_id = int(logits_next.argmax(dim=-1).item())

                # Update actuator state for the NEXT token.
                scale_state.value = float(next_scale)

                # Print token dashboard line aligned to the consumed token.
                div_color = Color.GREEN
                if avg_score > controller.TH_WARN:
                    div_color = Color.YELLOW
                if avg_score > controller.TH_CRIT:
                    div_color = Color.RED

                status_color = Color.RESET
                if status == "CRITICAL":
                    status_color = Color.RED
                elif status in ("WARNING",):
                    status_color = Color.YELLOW
                elif status in ("COOLDOWN",):
                    status_color = Color.CYAN

                token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                clean = token_str.replace("\n", "\\n")

                print(
                    f" {t:3d} | {float(raw):8.4f} | {div_color}{avg_score:9.4f}{Color.RESET} |"
                    f" {scale_used:10.2f} | {next_scale:10.2f} |"
                    f" {status_color}{status:<8}{Color.RESET} | {clean}"
                )

                # Log a single robust per-token event (JSONL).
                evt = {
                    "t": int(t),
                    "token_id": int(token_id),
                    "token_text": token_str,
                    "raw_div": float(raw),
                    "control_score": float(control_score),
                    "avg_score": float(avg_score),
                    "diagnostics": diagnostics,
                    "hidden_norm": float(hidden_norm),
                    "scale_used": float(scale_used),
                    "next_scale": float(next_scale),
                    "status": str(status),
                    "backend": backend_result.backend,
                    "intervention_type": str(intervention_type),
                    "intervention_name": str(intervention_name),
                    "intervention_applied": bool((scale_used < 1.0) and (not shadow)),
                }
                events_f.write(json.dumps(evt, ensure_ascii=False) + "\n")
                events_cache.append(evt)

                n_tokens += 1
                sum_raw_div += float(raw)
                sum_avg_score += float(avg_score)
                if status == "WARNING":
                    n_warning += 1
                elif status == "CRITICAL":
                    n_critical += 1
                elif status == "COOLDOWN":
                    n_cooldown += 1

                generated_text += token_str

                # Stop after consuming EOS.
                eos = getattr(tokenizer, "eos_token_id", None)
                if eos is not None and int(token_id) == int(eos):
                    break

                # Optional: small sleep to make CLI output readable.
                # time.sleep(0.01)

    finally:
        handle.remove()

        # Persist artifacts (even if the run errors mid-way).
        full_text = prompt + generated_text
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text)
        except Exception:
            pass

        avg_raw_div_mean = float(sum_raw_div / max(1, n_tokens))
        avg_score_mean = float(sum_avg_score / max(1, n_tokens))
        summary = {
            "config_hash": config_hash,
            "run_dir": str(run_dir),
            "model_id": model_id,
            "backend": backend_result.backend,
            "intervention_type": str(intervention_type),
            "intervention_name": str(intervention_name),
            "shadow": bool(shadow),
            "tokens": int(n_tokens),
            "avg_raw_div_mean": avg_raw_div_mean,
            "avg_score_mean": avg_score_mean,
            "status_counts": {
                "WARNING": int(n_warning),
                "CRITICAL": int(n_critical),
                "COOLDOWN": int(n_cooldown),
            },
        }
        if generate_dashboard_html:
            try:
                dashboard_path = generate_dashboard(run_dir=run_dir, events=events_cache, summary=summary)
                summary["dashboard_path"] = str(dashboard_path)
            except Exception as e:
                summary["dashboard_error"] = str(e)
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        print(f"{Color.DIM}{'-' * 72}{Color.RESET}")
        print(f"\n{Color.BOLD}Final Output:{Color.RESET}\n{full_text}\n")
        print(f"{Color.DIM}{'-' * 72}{Color.RESET}")
        print(f"Artifacts written:")
        print(f"  - {events_path}")
        print(f"  - {output_path}")
        print(f"  - {summary_path}")
        if summary.get("dashboard_path"):
            print(f"  - {summary['dashboard_path']}")

    artifacts = {
        "run_dir": str(run_dir),
        "events_path": str(events_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "config_hash": config_hash,
        "dashboard_path": summary.get("dashboard_path"),
    }

    return full_text, artifacts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive-controller demo: composite diagnostics + pluggable intervention hook")
    parser.add_argument("--prompt", type=str, default="Explain how airplanes fly in a clear, accurate way.")
    parser.add_argument("--model", type=str, default=None, help="Model key from intervention_engine/models.json (or omit for default)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--layer", type=int, default=-1, help="Layer index to hook (negative allowed)")
    parser.add_argument("--ma-window", type=int, default=3, help="Moving average window for controller")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=str, default=None, help="Base directory for run artifacts (default: $RUNS_DIR or ./runs)")
    parser.add_argument("--run-name", type=str, default="adaptive_demo", help="Prefix for run folder")
    parser.add_argument("--shadow", action="store_true", help="Log decisions but do not apply interventions")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "nnsight"])
    parser.add_argument("--nnsight-remote", action="store_true")
    parser.add_argument("--nnsight-device", type=str, default=None)
    parser.add_argument("--type", type=str, default="scaling", choices=["scaling", "sae"], help="Intervention type for control mode")
    parser.add_argument("--sae-repo", type=str, default="apollo-research/llama-3.1-70b-sae")
    parser.add_argument("--sae-id", type=str, default=None)
    parser.add_argument("--sae-layer", type=int, default=None)
    parser.add_argument("--sae-feature-idx", type=int, default=0)
    parser.add_argument("--sae-strength", type=float, default=5.0)
    parser.add_argument("--sae-no-normalize", action="store_true")
    parser.add_argument("--no-dashboard", action="store_true")

    args = parser.parse_args()

    run_adaptive_controller(
        prompt=args.prompt,
        model_key=args.model,
        max_new_tokens=args.max_new_tokens,
        layer_idx=args.layer,
        ma_window=args.ma_window,
        seed=args.seed,
        runs_dir=args.runs_dir,
        run_name=args.run_name,
        shadow=args.shadow,
        backend=args.backend,
        nnsight_remote=args.nnsight_remote,
        nnsight_device=args.nnsight_device,
        intervention_type=args.type,
        sae_repo=args.sae_repo,
        sae_id=args.sae_id,
        sae_layer=args.sae_layer,
        sae_feature_idx=args.sae_feature_idx,
        sae_strength=args.sae_strength,
        sae_normalize=not args.sae_no_normalize,
        generate_dashboard_html=not args.no_dashboard,
    )


# Backward-compat alias for older callers.
run_system4 = run_adaptive_controller
