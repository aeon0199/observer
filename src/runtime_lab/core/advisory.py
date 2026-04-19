"""Post-run advisory system.

Given a run summary (observe / stress / hysteresis / control), emit a structured
advisory block with:

- observations: enumerated facts the system saw
- likely_causes: hypotheses for each non-obvious observation
- next_actions: concrete, parameter-level suggestions the next LLM call can act on
- confidence: "high" / "medium" / "low"

This is the single most important LLM-facing feature in the repo: every run now
teaches the next run how to improve. Advisories are written into summary.json
under the top-level `advisory` key so the dashboard and any programmatic caller
can read them without re-deriving.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class Advisory:
    observations: List[str] = field(default_factory=list)
    likely_causes: List[str] = field(default_factory=list)
    next_actions: List[Dict[str, Any]] = field(default_factory=list)  # {label, cmd?, params?}
    flags: List[str] = field(default_factory=list)  # quick tags: "no-op", "degenerate", "good"
    confidence: str = "medium"
    summary_line: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-mode analyzers
# ---------------------------------------------------------------------------


def advise_observe(summary: Dict[str, Any]) -> Advisory:
    adv = Advisory()
    n_toks = int(summary.get("tokens") or 0)
    avg_div = _safe_float(summary.get("avg_divergence"))
    health = summary.get("diagnostics_health") or {}
    degraded = int(health.get("degraded_steps") or 0)

    if n_toks < 8:
        adv.observations.append(
            f"Only {n_toks} tokens generated — divergence/spectral probes need "
            "~8+ tokens to build meaningful windows."
        )
        adv.next_actions.append({
            "label": "Increase max-tokens for richer telemetry",
            "params": {"max_tokens": 64},
        })
        adv.flags.append("short-run")

    if avg_div is not None:
        if avg_div < 0.05:
            adv.observations.append(f"avg_divergence={avg_div:.4f} is very low — trajectory is near-linear.")
            adv.flags.append("quiet")
        elif avg_div > 1.5:
            adv.observations.append(f"avg_divergence={avg_div:.4f} is high — strong token-to-token surprise.")
            adv.flags.append("volatile")

    if degraded:
        adv.observations.append(f"{degraded}/{n_toks} diagnostic steps flagged degraded.")
        adv.likely_causes.append("Window warmup or numerical instability in predictor.")

    if not adv.observations:
        adv.observations.append("Clean run; telemetry nominal.")
        adv.flags.append("nominal")

    adv.summary_line = _fmt_summary(
        mode="observe",
        leads=[f"tokens={n_toks}", f"avg_div={_fmt(avg_div)}"],
        flags=adv.flags,
    )
    adv.confidence = "high" if n_toks >= 16 else "medium"
    return adv


def advise_stress(results: Dict[str, Any]) -> Advisory:
    """Stress stores metrics at top-level `metrics`, config under `config`."""
    adv = Advisory()
    metrics = results.get("metrics") or {}
    cfg = results.get("config") or {}

    regime = metrics.get("regime") or "UNKNOWN"
    token_match = _safe_float(metrics.get("token_match_rate"))
    first_div = _safe_float(metrics.get("first_token_divergence"))
    recovery = _safe_float(metrics.get("recovery_ratio"))
    primary = _safe_float(metrics.get("primary_metric"))
    logit_kl = _safe_float(metrics.get("logit_kl_mean_during"))  # new field if present

    layer = int(cfg.get("intervention_layer", -1))
    mag = _safe_float(cfg.get("intervention_magnitude"))
    mag_rel = bool(cfg.get("intervention_magnitude_relative", True))
    itype = str(cfg.get("intervention_type", "additive"))

    adv.observations.append(f"regime={regime}, token_match={_fmt(token_match)}, recovery_ratio={_fmt(recovery)}")

    # Diagnose no-op: tokens didn't flip at all
    if token_match is not None and token_match >= 0.999:
        adv.flags.append("no-op")
        adv.observations.append(
            "Intervention did NOT flip any generated tokens vs. baseline."
        )
        # Discriminate between "magnitude too small" and "wrong layer"
        hints: List[str] = []
        if layer in (-1, -2) or (isinstance(layer, int) and layer < 0 and layer > -3):
            hints.append(
                "Final-layer perturbations rarely flip greedy tokens — the LM head "
                "absorbs small changes into the argmax of a confident distribution."
            )
            adv.next_actions.append({
                "label": "Move intervention to mid-stack",
                "params": {"layer": "mid"},  # semantic default; CLI resolves to n//2
            })
        if mag is not None:
            if mag_rel and mag < 0.3:
                hints.append(f"magnitude={mag:.2f} (relative) is small — try 0.3–0.5.")
                adv.next_actions.append({
                    "label": "Increase relative magnitude",
                    "params": {"magnitude": 0.4},
                })
            elif not mag_rel:
                hints.append(f"absolute magnitude={mag:.2f} depends on activation scale; switch to relative.")
                adv.next_actions.append({
                    "label": "Use relative magnitude",
                    "params": {"magnitude": 0.3, "absolute_magnitude": False},
                })
        if logit_kl is not None and logit_kl > 0.05:
            hints.append(
                f"but logit-level KL={logit_kl:.3f} shows logits DID shift — the "
                "perturbation is working, greedy argmax is hiding it. Enable sampling."
            )
            adv.next_actions.append({
                "label": "Enable sampling so logit shifts become token shifts",
                "params": {"temperature": 0.8},
            })
        adv.likely_causes.extend(hints)
        adv.confidence = "high"
    else:
        if token_match is not None:
            adv.observations.append(f"Intervention flipped ~{(1 - token_match) * 100:.0f}% of tokens.")
        if regime == "ELASTIC" and recovery is not None and recovery > 0.8:
            adv.flags.append("good")
            adv.observations.append("System recovered well after intervention ended.")
        elif regime == "PLASTIC":
            adv.flags.append("persistent-effect")
            adv.observations.append("Intervention effect persisted after removal (plastic regime).")
        elif regime == "DIVERGENT":
            adv.flags.append("runaway")
            adv.observations.append("Trajectory did not recover — runaway divergence.")
        adv.confidence = "high"

    if first_div is not None and first_div < 1e-6:
        adv.observations.append(
            "first_token_divergence ≈ 0 — first perturbed token still matched baseline; "
            "confirms perturbation magnitude/layer effectively inert at first step."
        )

    adv.summary_line = _fmt_summary(
        mode=f"stress:{itype}@L{layer}",
        leads=[f"regime={regime}", f"match={_fmt(token_match)}", f"recovery={_fmt(recovery)}"],
        flags=adv.flags,
    )
    return adv


def advise_hysteresis(summary: Dict[str, Any]) -> Advisory:
    adv = Advisory()
    metrics = summary.get("metrics") or {}
    cfg = summary.get("config") or {}
    telemetry = summary.get("telemetry") or {}

    mode = str(cfg.get("perturbation_mode", "prompt"))
    regime = metrics.get("regime")
    drift = _safe_float(metrics.get("drift"))
    hysteresis = _safe_float(metrics.get("hysteresis"))
    recovery = _safe_float(metrics.get("recovery"))

    adv.observations.append(
        f"perturbation_mode={mode} regime={regime} drift={_fmt(drift)} "
        f"hysteresis={_fmt(hysteresis)} recovery={_fmt(recovery)}"
    )

    # Detect a true no-op — base and perturb telemetry identical
    base = telemetry.get("base") or {}
    perturb = telemetry.get("perturb") or {}
    same_hidden = (
        _safe_float(base.get("hidden_norm")) == _safe_float(perturb.get("hidden_norm"))
        and _safe_float(base.get("entropy")) == _safe_float(perturb.get("entropy"))
    )

    if mode == "noise" and same_hidden:
        adv.flags.append("noise-absorbed")
        adv.observations.append(
            "BASE and PERTURB stats are identical — the hidden-state perturbation "
            "did not propagate into the generated trajectory."
        )
        layer = int(cfg.get("noise_layer", -1))
        mag = _safe_float(cfg.get("noise_magnitude"))
        if layer in (-1, -2):
            adv.likely_causes.append(
                "noise_layer is near the output; perturbation doesn't cascade through "
                "subsequent layers and greedy argmax absorbs small shifts."
            )
            adv.next_actions.append({
                "label": "Use a mid-stack noise layer",
                "params": {"noise_layer": "mid"},
            })
        if mag is not None and mag < 0.3:
            adv.next_actions.append({
                "label": "Increase noise magnitude",
                "params": {"noise_magnitude": 0.4},
            })
        adv.next_actions.append({
            "label": "Enable sampling so logit shifts manifest as token shifts",
            "params": {"temperature": 0.8},
        })

    if mode == "prompt":
        adv.observations.append(
            "Note: 'prompt' mode measures prompt-contamination persistence, not "
            "internal-dynamics hysteresis. To test internal dynamics, use "
            "perturbation_mode=noise with a mid-stack layer."
        )
        adv.flags.append("prompt-contamination-test")

    if recovery is not None and recovery < -1.0:
        adv.observations.append(
            f"recovery={recovery:.2f} is strongly negative — REASK state diverged "
            "further from BASE than PERTURB did. Usually indicates context-persistence "
            "dominating (KV cache from PERTURB lingers across REASK)."
        )

    if regime == "elastic" and recovery and recovery > 0.8:
        adv.flags.append("good-recovery")

    adv.summary_line = _fmt_summary(
        mode=f"hysteresis:{mode}",
        leads=[f"regime={regime}", f"recovery={_fmt(recovery)}"],
        flags=adv.flags,
    )
    adv.confidence = "high"
    return adv


def advise_control(summary: Dict[str, Any]) -> Advisory:
    adv = Advisory()
    cfg = summary.get("config") or {}
    avg_div = _safe_float(summary.get("avg_raw_div_mean"))
    avg_score = _safe_float(summary.get("avg_score_mean"))
    status_counts = summary.get("status_counts") or {}

    shadow = bool(cfg.get("shadow"))
    tag = "shadow" if shadow else "active"
    adv.observations.append(
        f"mode={tag} avg_div={_fmt(avg_div)} avg_score={_fmt(avg_score)} status_counts={status_counts}"
    )

    if shadow:
        adv.flags.append("shadow")
        adv.next_actions.append({
            "label": "Run active counterpart to verify controller actually helps",
            "params": {"shadow": False},
        })
    else:
        adv.flags.append("active")
        crit = int(status_counts.get("CRITICAL") or status_counts.get("crit") or 0)
        warn = int(status_counts.get("WARN") or status_counts.get("warn") or 0)
        if crit == 0 and warn == 0 and (avg_div or 0) < 0.1:
            adv.flags.append("quiet-controller")
            adv.observations.append(
                "Controller never engaged (no WARN/CRIT triggers). Either the "
                "trajectory was stable (good), or the thresholds are too loose."
            )
        elif crit > 0:
            adv.observations.append(f"Controller fired CRIT {crit} time(s).")

    adv.summary_line = _fmt_summary(
        mode=f"control:{tag}",
        leads=[f"avg_div={_fmt(avg_div)}", f"avg_score={_fmt(avg_score)}"],
        flags=adv.flags,
    )
    adv.confidence = "medium"
    return adv


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def analyze(mode: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point.

    `mode` is one of observe/stress/hysteresis/control.
    `payload` is the summary (for observe/hysteresis/control) or the full results
    dict (for stress, which stores metrics at top-level not under `summary`).
    """
    mode = str(mode or "").lower()
    if mode == "observe":
        adv = advise_observe(payload)
    elif mode == "stress":
        adv = advise_stress(payload)
    elif mode == "hysteresis":
        adv = advise_hysteresis(payload)
    elif mode == "control":
        adv = advise_control(payload)
    else:
        adv = Advisory(observations=[f"No advisor for mode={mode!r}"])
    return adv.to_dict()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except Exception:
        return None


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1000 or abs(v) < 0.001:
        return f"{v:.3g}"
    return f"{v:.3f}"


def _fmt_summary(mode: str, leads: List[str], flags: List[str]) -> str:
    lead = " ".join(leads)
    flag_str = " ".join(f"[{f}]" for f in flags) if flags else ""
    return f"{mode} · {lead}{(' ' + flag_str) if flag_str else ''}"
