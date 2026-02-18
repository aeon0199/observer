# v2/intervention.py

"""
V2 — Activation Intervention
----------------------------
Token-by-token generation with mid-inference activation intervention.
"""

import os
import random
import hashlib
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, Callable, List

import torch
import numpy as np

# Allow this file to be run either as part of a package (relative imports)
# or directly as a script from inside the folder (fallback imports).
try:
    from .model_loader import load_model_registry
    from .utils import save_json
    from .cache import SeedCache, build_seed_cache, get_model_layers
    from .backend import load_model_with_backend
    from .sae_adapter import load_sae_decoder_vector, make_sae_intervention
    from .trajectory import (
        Trajectory,
        TokenState,
        TrajectoryComparison,
        compute_entropy,
        compute_top1,
    )
    from .diagnostics_bridge import DiagnosticsManager
except ImportError:  # pragma: no cover
    from model_loader import load_model_registry
    from utils import save_json
    from cache import SeedCache, build_seed_cache, get_model_layers
    from backend import load_model_with_backend
    from sae_adapter import load_sae_decoder_vector, make_sae_intervention
    from trajectory import (
        Trajectory,
        TokenState,
        TrajectoryComparison,
        compute_entropy,
        compute_top1,
    )
    from diagnostics_bridge import DiagnosticsManager


class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"


VERSION = "2.3"
RUNS_DIR = os.environ.get("RUNS_DIR", "runs")


def hash_config(config: dict) -> str:
    """Generate a short hash of experiment config for reproducibility verification."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def set_deterministic_state(seed: int):
    """Set all RNG sources."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════
# INTERVENTION FUNCTIONS
# ══════════════════════════════════════════════════════════

def make_additive_intervention(magnitude: float = 1.0, seed: int = 42):
    """Add a random vector to hidden state."""
    intervention_vector = None
    
    def intervene(hidden_state: torch.Tensor) -> torch.Tensor:
        nonlocal intervention_vector
        
        if intervention_vector is None:
            # Use a local CPU RNG so we don't mutate global torch RNG mid-generation.
            dim = hidden_state.shape[-1]
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            vec = torch.randn(dim, generator=g, device="cpu", dtype=torch.float32)
            vec = vec / (vec.norm() + 1e-12) * float(magnitude)
            intervention_vector = vec.to(device=hidden_state.device, dtype=hidden_state.dtype)
        
        modified = hidden_state.clone()
        modified[:, -1, :] = modified[:, -1, :] + intervention_vector
        return modified
    
    intervene.__name__ = "additive"
    return intervene


def make_projection_intervention(subspace_dim: int = 10, seed: int = 42):
    """Project out a random subspace."""
    projection_matrix = None
    
    def intervene(hidden_state: torch.Tensor) -> torch.Tensor:
        nonlocal projection_matrix
        
        if projection_matrix is None:
            # Use a local CPU RNG so we don't mutate global torch RNG mid-generation.
            dim = hidden_state.shape[-1]
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            random_matrix = torch.randn(dim, subspace_dim, generator=g, device="cpu", dtype=torch.float32)
            random_matrix = random_matrix.to(device=hidden_state.device, dtype=hidden_state.dtype)
            Q, _ = torch.linalg.qr(random_matrix)
            projection_matrix = torch.eye(dim, device=hidden_state.device, dtype=hidden_state.dtype) - Q @ Q.T
        
        modified = hidden_state.clone()
        modified[:, -1, :] = modified[:, -1, :] @ projection_matrix
        return modified
    
    intervene.__name__ = "projection"
    return intervene


def make_scaling_intervention(scale: float = 0.5):
    """Scale hidden state magnitude."""
    def intervene(hidden_state: torch.Tensor) -> torch.Tensor:
        modified = hidden_state.clone()
        modified[:, -1, :] = modified[:, -1, :] * scale
        return modified
    
    intervene.__name__ = "scaling"
    return intervene


# ══════════════════════════════════════════════════════════
# HOOKS
# ══════════════════════════════════════════════════════════

class InterventionHook:
    """Hook that captures AND optionally modifies activations.

    Research-grade note:
    - We keep only the most recent capture here.
    - The trajectory object stores the per-token history.
    """

    def __init__(self, intervention_fn: Optional[Callable] = None):
        self.intervention_fn = intervention_fn
        self.active = False
        self.last_hidden: Optional[torch.Tensor] = None  # CPU tensor
        self.last_hidden_norm: float = 0.0

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output

        # Apply intervention first (if active), so captured hidden reflects the
        # actual computation that downstream layers / lm_head will see.
        hs_out = hs
        if self.active and self.intervention_fn is not None:
            hs_out = self.intervention_fn(hs)

        hidden_last = hs_out[:, -1, :].detach()
        self.last_hidden = hidden_last.cpu()
        self.last_hidden_norm = float(hidden_last.norm().item())

        if isinstance(output, tuple):
            return (hs_out,) + output[1:]
        return hs_out

    def set_active(self, active: bool):
        self.active = bool(active)

    def reset(self):
        self.active = False
        self.last_hidden = None
        self.last_hidden_norm = 0.0

    def get_last_state(self) -> Dict:
        out = {"hidden_norm": self.last_hidden_norm, "intervention_active": self.active}
        if self.last_hidden is not None:
            out["hidden"] = self.last_hidden
        return out


class LayerCaptureHook:
    """Capture final-token hidden state for diagnostic probe layers."""

    def __init__(self):
        self.last_hidden: Optional[torch.Tensor] = None

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        self.last_hidden = hs[:, -1, :].detach().cpu()
        return output

    def reset(self):
        self.last_hidden = None


# ══════════════════════════════════════════════════════════
# TRAJECTORY GENERATION
# ══════════════════════════════════════════════════════════

def generate_trajectory_from_seed_cache(
    model,
    tokenizer,
    device: torch.device,
    seed_cache: SeedCache,
    prompt: str,
    max_new_tokens: int,
    intervention_layer: int,
    intervention_fn: Optional[Callable] = None,
    intervention_start: int = -1,
    intervention_end: int = -1,
    model_id: str = "",
    diagnostics_manager: Optional[DiagnosticsManager] = None,
) -> Tuple[Trajectory, str]:
    """Generate tokens one-by-one, capturing a research-grade trajectory.

    Key correctness properties:
    - Starts from an explicit prompt-pass SeedCache (identical pre-generation state).
    - Does NOT re-insert the last prompt token (no off-by-one / duplication).
    - Uses the prompt-pass logits for token 0.
    - Applies interventions on the forward pass that *produces* the logits used to
      choose token t (same semantics as the original implementation).
    """

    trajectory = Trajectory(
        prompt=prompt,
        model_id=model_id,
        intervention_layer=intervention_layer,
        intervention_start=intervention_start,
        intervention_end=intervention_end,
        intervention_type=intervention_fn.__name__ if intervention_fn else "none",
    )

    if max_new_tokens <= 0:
        return trajectory, prompt

    layers = get_model_layers(model)
    num_layers = len(layers)
    actual_layer = intervention_layer if intervention_layer >= 0 else num_layers + intervention_layer

    if not (0 <= actual_layer < num_layers):
        raise IndexError(
            f"[V2] intervention_layer out of range: {intervention_layer} (resolved to {actual_layer}, num_layers={num_layers})"
        )

    intervention_hook = InterventionHook(intervention_fn)
    handle_hidden = layers[actual_layer].register_forward_hook(intervention_hook)

    probe_resolved: Dict[int, int] = {}
    capture_hooks: Dict[int, LayerCaptureHook] = {}
    capture_handles: List = []

    if diagnostics_manager is not None and diagnostics_manager.enabled:
        for probe_idx in diagnostics_manager.probe_layers:
            resolved = probe_idx if probe_idx >= 0 else num_layers + probe_idx
            if not (0 <= resolved < num_layers):
                continue
            probe_resolved[int(probe_idx)] = int(resolved)
            if resolved == actual_layer:
                continue
            capture_hook = LayerCaptureHook()
            capture_hooks[int(probe_idx)] = capture_hook
            capture_handles.append(layers[resolved].register_forward_hook(capture_hook))

    try:
        intervention_hook.reset()
        if diagnostics_manager is not None and diagnostics_manager.enabled:
            diagnostics_manager.reset()

        past_key_values = seed_cache.past_key_values
        prompt_len = int(seed_cache.seq_len)

        generated_ids: list[int] = []

        # t = 0 comes from the prompt-pass (seed) logits/hidden.
        logits0 = seed_cache.next_token_logits  # CPU tensor
        hidden0 = seed_cache.seed_hidden  # CPU tensor

        # Store internals for research-grade comparison.
        trajectory._hidden_vecs.append(hidden0.to(dtype=torch.float16))
        trajectory._logits.append(logits0.to(dtype=torch.float16))

        next_token_id = int(logits0.argmax(dim=-1).item())
        entropy = compute_entropy(logits0)
        top1_prob, top1_token = compute_top1(logits0)

        diagnostics0 = {}
        if diagnostics_manager is not None and diagnostics_manager.enabled:
            layer_states0: Dict[int, torch.Tensor] = {}
            for probe_idx, resolved in probe_resolved.items():
                if resolved == actual_layer and isinstance(hidden0, torch.Tensor):
                    layer_states0[int(probe_idx)] = hidden0
            diagnostics0 = diagnostics_manager.step(hidden0, layer_states=layer_states0)

        state0 = TokenState(
            token_idx=0,
            token_id=next_token_id,
            token_text=tokenizer.decode([next_token_id]),
            hidden_norm=float(hidden0.norm().item()) if isinstance(hidden0, torch.Tensor) else 0.0,
            logit_norm=float(logits0.norm().item()),
            entropy=entropy,
            top1_prob=top1_prob,
            top1_token=top1_token,
            intervention_active=False,
            diagnostics=diagnostics0,
        )
        trajectory.add_state(state0)
        generated_ids.append(next_token_id)

        if next_token_id == tokenizer.eos_token_id:
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            return trajectory, prompt + generated_text

        # t >= 1: forward pass with past_key_values.
        for t in range(1, max_new_tokens):
            # Intervention window refers to the token index being *chosen*.
            if t == intervention_start:
                intervention_hook.set_active(True)
            if t == intervention_end:
                intervention_hook.set_active(False)

            current_input = torch.tensor([[generated_ids[-1]]], device=device)

            # When providing past_key_values, attention_mask must match the full
            # sequence length (prompt + generated so far + current token).
            attention_mask = torch.ones((1, prompt_len + t), device=device, dtype=torch.long)

            for capture_hook in capture_hooks.values():
                capture_hook.reset()

            with torch.no_grad():
                outputs = model(
                    input_ids=current_input,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :].detach().cpu()

            last_state = intervention_hook.get_last_state()
            hidden_vec = last_state.get("hidden", None)
            hidden_norm = float(last_state.get("hidden_norm", 0.0))

            # Store internals for research-grade comparison.
            if isinstance(hidden_vec, torch.Tensor):
                trajectory._hidden_vecs.append(hidden_vec.to(dtype=torch.float16))
            else:
                trajectory._hidden_vecs.append(torch.zeros((1, 1), dtype=torch.float16))
            trajectory._logits.append(logits.to(dtype=torch.float16))

            next_token_id = int(logits.argmax(dim=-1).item())

            entropy = compute_entropy(logits)
            top1_prob, top1_token = compute_top1(logits)

            diagnostics = {}
            if diagnostics_manager is not None and diagnostics_manager.enabled:
                layer_states: Dict[int, torch.Tensor] = {}
                for probe_idx, resolved in probe_resolved.items():
                    if resolved == actual_layer:
                        if isinstance(hidden_vec, torch.Tensor):
                            layer_states[int(probe_idx)] = hidden_vec
                        continue
                    capture_hook = capture_hooks.get(int(probe_idx))
                    if capture_hook is not None and isinstance(capture_hook.last_hidden, torch.Tensor):
                        layer_states[int(probe_idx)] = capture_hook.last_hidden
                diagnostics = diagnostics_manager.step(hidden_vec, layer_states=layer_states)

            state = TokenState(
                token_idx=t,
                token_id=next_token_id,
                token_text=tokenizer.decode([next_token_id]),
                hidden_norm=hidden_norm,
                logit_norm=float(logits.norm().item()),
                entropy=entropy,
                top1_prob=top1_prob,
                top1_token=top1_token,
                intervention_active=bool(intervention_hook.active),
                diagnostics=diagnostics,
            )
            trajectory.add_state(state)
            generated_ids.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = prompt + generated_text

    finally:
        handle_hidden.remove()
        for handle in capture_handles:
            handle.remove()

    return trajectory, full_text


# ══════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════

def run_intervention_experiment(
    prompt: str,
    model_key: str = None,
    max_new_tokens: int = 64,
    intervention_layer: int = -1,
    intervention_type: str = "additive",
    intervention_magnitude: float = 10.0,
    intervention_start: int = 5,
    intervention_duration: int = 10,
    seed: int = 42,
    backend: str = "hf",
    nnsight_remote: bool = False,
    nnsight_device: Optional[str] = None,
    with_diagnostics: bool = True,
    diagnostics_probe_layers: Optional[List[int]] = None,
    sae_repo: str = "apollo-research/llama-3.1-70b-sae",
    sae_id: Optional[str] = None,
    sae_layer: Optional[int] = None,
    sae_feature_idx: int = 0,
    sae_strength: float = 5.0,
    sae_normalize: bool = True,
) -> Dict:
    """Run complete V2 intervention experiment."""
    
    print(f"\n{Color.MAGENTA}{'═' * 60}{Color.RESET}")
    print(f"{Color.MAGENTA}  V2 — Activation Intervention{Color.RESET}")
    print(f"{Color.MAGENTA}{'═' * 60}{Color.RESET}\n")
    
    backend_result = load_model_with_backend(
        model_key=model_key,
        backend=backend,
        nnsight_remote=nnsight_remote,
        nnsight_device=nnsight_device,
    )
    tokenizer = backend_result.tokenizer
    model = backend_result.model
    device = backend_result.device
    model_config = backend_result.config
    model_id = model_key or getattr(model_config, "key", None) or "unknown"
    
    os.makedirs(RUNS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, f"v2_run_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    selected_layer = int(intervention_layer)

    if intervention_type == "additive":
        intervention_fn = make_additive_intervention(magnitude=intervention_magnitude, seed=seed)
    elif intervention_type == "projection":
        intervention_fn = make_projection_intervention(subspace_dim=int(intervention_magnitude), seed=seed)
    elif intervention_type == "scaling":
        intervention_fn = make_scaling_intervention(scale=intervention_magnitude)
    elif intervention_type == "sae":
        if sae_layer is not None:
            selected_layer = int(sae_layer)
        decoder_vector = load_sae_decoder_vector(
            repo_id=sae_repo,
            feature_idx=int(sae_feature_idx),
            layer=(None if sae_id is not None else int(selected_layer)),
            sae_id=sae_id,
            normalize=bool(sae_normalize),
            device=str(device),
        )
        intervention_fn = make_sae_intervention(
            decoder_vector=decoder_vector,
            strength=float(sae_strength),
            name=f"sae_f{int(sae_feature_idx)}_s{float(sae_strength):.2f}",
        )
    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")
    
    intervention_end = intervention_start + intervention_duration

    # Research-grade semantic guard:
    # token 0 is chosen from the prompt-pass logits (no incremental forward pass),
    # so interventions cannot begin at token 0 in this runner.
    if intervention_duration > 0 and intervention_start < 1:
        raise ValueError(
            "[V2] intervention_start must be >= 1 (token 0 comes from prompt-pass logits)."
        )
    
    print(f"[V2] Model: {model_id}")
    print(f"[V2] Backend: {backend_result.backend}")
    if backend_result.backend == "nnsight":
        print(f"[V2] NNsight remote: {bool(backend_result.backend_meta.get('remote'))}")
    print(f"[V2] Intervention: {intervention_type} at layer {selected_layer}")
    print(f"[V2] Window: tokens {intervention_start} to {intervention_end}")
    if intervention_type == "sae":
        print(f"[V2] SAE repo/id: {sae_repo} / {sae_id or f'layer_{selected_layer}'}")
        print(f"[V2] SAE feature/strength: {int(sae_feature_idx)} / {float(sae_strength)}")
    else:
        print(f"[V2] Magnitude: {intervention_magnitude}")
    print(f"[V2] Diagnostics: {'ON' if with_diagnostics else 'OFF'}")
    
    # ═══════════════════════════════════════════════════════
    # BUILD PROMPT SEED CACHE (branchpoint)
    # ═══════════════════════════════════════════════════════

    set_deterministic_state(seed)
    seed_cache = build_seed_cache(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        intervention_layer=selected_layer,
    )

    print(f"[V2] Seed cache fingerprint: {seed_cache.fingerprint}")
    print(f"[V2] Prompt length:          {seed_cache.seq_len} tokens")

    baseline_diagnostics_manager = None
    intervention_diagnostics_manager = None
    diagnostics_error = None
    probe_layers = diagnostics_probe_layers or [int(selected_layer), -1]
    probe_layers = [int(i) for i in probe_layers]

    if with_diagnostics:
        try:
            baseline_diagnostics_manager = DiagnosticsManager(
                enabled=True,
                probe_layers=probe_layers,
            )
            intervention_diagnostics_manager = DiagnosticsManager(
                enabled=True,
                probe_layers=probe_layers,
            )
            print(f"[V2] Probe layers: {probe_layers}")
        except Exception as e:
            diagnostics_error = str(e)
            baseline_diagnostics_manager = None
            intervention_diagnostics_manager = None
            with_diagnostics = False
            print(f"{Color.YELLOW}[V2] Diagnostics disabled (init failed): {diagnostics_error}{Color.RESET}")

    # ═══════════════════════════════════════════════════════
    # BASELINE RUN
    # ═══════════════════════════════════════════════════════

    print(f"\n{Color.CYAN}[V2] Run 1 — Baseline (no intervention){Color.RESET}")

    set_deterministic_state(seed)

    baseline_trajectory, baseline_text = generate_trajectory_from_seed_cache(
        model=model,
        tokenizer=tokenizer,
        device=device,
        seed_cache=seed_cache.clone(),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        intervention_layer=selected_layer,
        intervention_fn=None,
        intervention_start=-1,
        intervention_end=-1,
        model_id=model_id,
        diagnostics_manager=baseline_diagnostics_manager,
    )

    print(f"[V2] Baseline complete. {len(baseline_trajectory)} tokens.")
    
    with open(os.path.join(run_dir, "baseline_output.txt"), "w", encoding="utf-8") as f:
        f.write(baseline_text)
    
    # ═══════════════════════════════════════════════════════
    # INTERVENTION RUN
    # ═══════════════════════════════════════════════════════
    
    print(f"\n{Color.YELLOW}[V2] Run 2 — Intervention (ON tokens {intervention_start}-{intervention_end}){Color.RESET}")
    
    set_deterministic_state(seed)

    intervention_trajectory, intervention_text = generate_trajectory_from_seed_cache(
        model=model,
        tokenizer=tokenizer,
        device=device,
        seed_cache=seed_cache.clone(),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        intervention_layer=selected_layer,
        intervention_fn=intervention_fn,
        intervention_start=intervention_start,
        intervention_end=intervention_end,
        model_id=model_id,
        diagnostics_manager=intervention_diagnostics_manager,
    )
    
    print(f"[V2] Intervention complete. {len(intervention_trajectory)} tokens.")
    
    with open(os.path.join(run_dir, "intervention_output.txt"), "w", encoding="utf-8") as f:
        f.write(intervention_text)
    
    # ═══════════════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════════════
    
    print(f"\n{Color.GREEN}[V2] Computing trajectory comparison...{Color.RESET}")
    
    comparison = TrajectoryComparison(
        baseline=baseline_trajectory,
        intervention=intervention_trajectory
    )
    comparison.compute_metrics()
    
    # ═══════════════════════════════════════════════════════
    # BUILD CONFIG & HASH
    # ═══════════════════════════════════════════════════════
    
    run_config = {
        "model": model_id,
        "backend": backend_result.backend,
        "backend_meta": {
            "remote": bool(backend_result.backend_meta.get("remote", False)),
            "device_map": backend_result.backend_meta.get("device_map", str(device)),
        },
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "intervention_layer": selected_layer,
        "intervention_type": intervention_type,
        "intervention_magnitude": intervention_magnitude,
        "intervention_start": intervention_start,
        "intervention_end": intervention_end,
        "seed": seed,
        "seed_cache_fingerprint": seed_cache.fingerprint,
        "prompt_tokens": int(seed_cache.seq_len),
        "with_diagnostics": bool(with_diagnostics),
        "diagnostics_probe_layers": probe_layers if with_diagnostics else [],
    }
    if intervention_type == "sae":
        run_config["sae"] = {
            "repo": sae_repo,
            "sae_id": sae_id or f"layer_{selected_layer}",
            "layer": int(selected_layer),
            "feature_idx": int(sae_feature_idx),
            "strength": float(sae_strength),
            "normalize": bool(sae_normalize),
        }
    if diagnostics_error is not None:
        run_config["diagnostics_error"] = diagnostics_error
    
    config_hash = hash_config(run_config)
    
    # ═══════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════
    
    print(f"\n{Color.BLUE}{'═' * 60}{Color.RESET}")
    print(f"{Color.BLUE}  V2 INTERVENTION RESULTS{Color.RESET}")
    print(f"{Color.BLUE}{'═' * 60}{Color.RESET}")
    print(f"")
    print(f"  Config hash:                   {config_hash}")
    print(f"  Primary metric:                {comparison.primary_metric}")
    print(f"  Deviation during intervention: {comparison.deviation_during:.6f}")
    print(f"  Recovery after removal:        {comparison.recovery_after:.6f}")
    print(f"  Final distance from baseline:  {comparison.final_distance:.6f}")
    print(f"  Recovery ratio:                {comparison.recovery_ratio:.4f}")
    print(f"  Convergence rate:              {comparison.convergence_rate:.6f}")
    print(f"  Token match rate:              {comparison.token_match_rate:.3f}")
    print(f"  First token divergence:        {comparison.first_token_divergence}")

    # Optional secondary summaries (present when internals were captured).
    try:
        if "logit_js_bits" in comparison.summary:
            js = comparison.summary["logit_js_bits"]
            print(f"  Logit JS (bits) mean_during:    {js.get('mean_during', 0.0):.6f}")
            print(f"  Logit JS (bits) mean_post:      {js.get('mean_post', 0.0):.6f}")
            print(f"  Logit JS (bits) final:          {js.get('final', 0.0):.6f}")
    except Exception:
        pass

    print(f"")
    
    # If the perturbation produced effectively no measurable deviation, don't
    # mislabel it as "DIVERGENT" just because recovery_ratio is 0/undefined.
    if abs(comparison.deviation_during) < 1e-9 and abs(comparison.final_distance) < 1e-9:
        regime = "NO_EFFECT"
        color = Color.CYAN
    elif comparison.recovery_ratio > 0.8:
        regime = "ELASTIC"
        color = Color.GREEN
    elif comparison.recovery_ratio > 0.4:
        regime = "PARTIAL"
        color = Color.YELLOW
    elif comparison.recovery_ratio > 0:
        regime = "PLASTIC"
        color = Color.RED
    else:
        regime = "DIVERGENT"
        color = Color.RED
    
    print(f"  {color}Regime: {regime}{Color.RESET}")
    print(f"")
    print(f"{Color.BLUE}{'═' * 60}{Color.RESET}")
    
    # ═══════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════
    
    results = {
        "version": VERSION,
        "timestamp": stamp,
        "config_hash": config_hash,
        "config": run_config,
        "metrics": {
            "primary_metric": comparison.primary_metric,
            "deviation_during": comparison.deviation_during,
            "recovery_after": comparison.recovery_after,
            "final_distance": comparison.final_distance,
            "recovery_ratio": comparison.recovery_ratio,
            "convergence_rate": comparison.convergence_rate,
            "token_match_rate": comparison.token_match_rate,
            "first_token_divergence": comparison.first_token_divergence,
            "regime": regime,
            "summary": comparison.summary,
        },
        "trajectories": comparison.to_json(),
    }
    
    save_json(os.path.join(run_dir, "results.json"), results)
    
    print(f"\n{Color.GREEN}[V2] Results saved to: {run_dir}{Color.RESET}\n")
    
    return results


def main():
    """CLI entry point."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="V2 — Activation Intervention")
    parser.add_argument("mode", choices=["run", "interactive"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--type", default="additive", choices=["additive", "projection", "scaling", "sae"])
    parser.add_argument("--magnitude", type=float, default=10.0)
    parser.add_argument("--start", type=int, default=5)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="hf", choices=["hf", "nnsight"])
    parser.add_argument("--nnsight-remote", action="store_true")
    parser.add_argument("--nnsight-device", default=None)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument(
        "--probe-layers",
        type=str,
        default="",
        help="Comma-separated layer indices for diagnostics probe (example: '-1,10,20')",
    )
    parser.add_argument("--sae-repo", default="apollo-research/llama-3.1-70b-sae")
    parser.add_argument("--sae-id", default=None)
    parser.add_argument("--sae-layer", type=int, default=None)
    parser.add_argument("--sae-feature-idx", type=int, default=0)
    parser.add_argument("--sae-strength", type=float, default=5.0)
    parser.add_argument("--sae-no-normalize", action="store_true")
    
    args = parser.parse_args()

    probe_layers = None
    if args.probe_layers.strip():
        probe_layers = [int(x.strip()) for x in args.probe_layers.split(",") if x.strip()]
    
    if args.mode == "interactive":
        print(f"\n{Color.CYAN}=== V2 Interactive ==={Color.RESET}")
        
        registry = load_model_registry("models.json")
        model_keys = list(registry["models"].keys())
        print("\nAvailable models:")
        for i, key in enumerate(model_keys, start=1):
            print(f"  {i}. {key}")
        model_choice = int(input("\nChoose model: "))
        model_key = model_keys[model_choice - 1]
        
        prompt = input("\nEnter prompt: ")
        
        print("\nIntervention settings:")
        intervention_type = input("Type (additive/projection/scaling/sae) [additive]: ").strip() or "additive"
        magnitude = float(input("Magnitude [10.0]: ").strip() or "10.0")
        start = int(input("Start token [5]: ").strip() or "5")
        duration = int(input("Duration [10]: ").strip() or "10")
        
        run_intervention_experiment(
            prompt=prompt,
            model_key=model_key,
            max_new_tokens=args.max_tokens,
            intervention_type=intervention_type,
            intervention_magnitude=magnitude,
            intervention_start=start,
            intervention_duration=duration,
            seed=args.seed,
            backend=args.backend,
            nnsight_remote=args.nnsight_remote,
            nnsight_device=args.nnsight_device,
            with_diagnostics=not args.no_diagnostics,
            diagnostics_probe_layers=probe_layers,
            sae_repo=args.sae_repo,
            sae_id=args.sae_id,
            sae_layer=args.sae_layer,
            sae_feature_idx=args.sae_feature_idx,
            sae_strength=args.sae_strength,
            sae_normalize=not args.sae_no_normalize,
        )
    
    else:
        prompt = args.prompt or "Explain how airplanes fly in a clear, accurate way."
        
        run_intervention_experiment(
            prompt=prompt,
            model_key=args.model,
            max_new_tokens=args.max_tokens,
            intervention_layer=args.layer,
            intervention_type=args.type,
            intervention_magnitude=args.magnitude,
            intervention_start=args.start,
            intervention_duration=args.duration,
            seed=args.seed,
            backend=args.backend,
            nnsight_remote=args.nnsight_remote,
            nnsight_device=args.nnsight_device,
            with_diagnostics=not args.no_diagnostics,
            diagnostics_probe_layers=probe_layers,
            sae_repo=args.sae_repo,
            sae_id=args.sae_id,
            sae_layer=args.sae_layer,
            sae_feature_idx=args.sae_feature_idx,
            sae_strength=args.sae_strength,
            sae_normalize=not args.sae_no_normalize,
        )


if __name__ == "__main__":
    main()
