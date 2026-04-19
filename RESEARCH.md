# Observer Research Lab

_Living document. Every completed experiment gets an outcome line. Every new question gets a new backlog entry. The LLM agent reads this first (before touching anything else in a session) and updates it last (before returning to the human)._

Last updated: **2026-04-18** · Active agent: Claude Opus 4.7 (1M context)

---

## 0. How to use this doc (for the next LLM)

**Start of session**
1. Read §1 (Current state) to know where you are.
2. Read §2 (Findings log) so you don't rediscover things we already know.
3. Pick an experiment from §4 (Backlog) — highest-value unchecked one unless the human redirects.

**During the session**
- Mark experiments `in-progress` with your session date.
- Record every run_id under the experiment's `Runs` line — this creates a trail back to the raw data.
- Don't chase dead ends. If the advisory + 3 seeds agree the effect is null, record it and move on.

**End of session**
1. Fill in the `Outcome` line for anything you ran (one paragraph, grounded in numbers).
2. Add any follow-up experiments that fell out of your results to §4.
3. Update §1 (Current state) + §2 (Findings log) + §5 (Open hypotheses).
4. Leave a one-line `Next recommended action` for whoever comes next.

**Hard rules**
- **Evidence standard**: a finding needs ≥3 seeds AND ≥2 prompts before it's recorded under §2. Until then it's a "suggestive observation" at best.
- **When in doubt, check the advisory.** Every run writes `advisory` into `summary.json`. If the advisory flags `no-op`, don't pretend you got a result.
- **Don't touch production code without human sign-off.** Patching the repo mid-experiment to make numbers work is motivated reasoning.
- **Stop and report to the human when**: a finding contradicts §2, the controller behaves unexpectedly, you've spent >30 min debugging the same run, or you're tempted to change the methodology to produce a "better" result.

---

## 1. Current state

- **Active focus**: E8 MVP complete. Random-direction additive controller is net-zero (F21). Wiring works, actuator works, direction is the bottleneck. **Next: E8.5 with EMA-baseline drift opposition.**
- **Controller status**: **partial win** — additive controller wired and verified. But random direction is net-zero. Direction must oppose measured drift to reduce divergence. E8.5 will implement that.
- **Capabilities snapshot**: observe, stress, hysteresis (prompt + noise modes), control (shadow + active, currently scaling/sae only), seed sweeps, multi-layer probing, logit-KL, semantic layer defaults, advisory generator, spectral trajectory probe, warm-model daemon (~16× speedup). All verified 2026-04-19.
- **Next recommended action**: **E8.5 — drift-opposing controller.** Add EMA hidden-state tracker in control loop; on controller fire, compute `drift = h - h_ema`, inject `-β · drift/||drift|| · ||h||`. Re-run A/B suite. If active < shadow by >1 stdev on the same sourdough prompt, closed-loop control is empirically established.

---

## 2. Findings log

_What we've established, with evidence. Don't re-run these unless you suspect one is wrong. Format: F<n> — claim — evidence — confidence._

- **F1** — **The legacy spectral module was measuring neuron-axis noise, not trajectory structure.** Fixed 2026-04-18. The new `SpectralTrajectoryProbe` FFTs over the token-time axis and emits a `permutation_change` self-test ratio per step; verified non-zero (0.029 on a drifting synthetic signal, and real values across observe runs). Confidence: **high**.
- **F2** — **Final-layer perturbations (layer -1) are consistently absorbed by greedy argmax and fail to cascade.** Multiple runs showed `token_match_rate=1.0` and `drift=0` at layer -1 with relative magnitudes up to 0.4. Same perturbation at mid-stack (layer -14 of 28) produced measurable drift=1.85. Confidence: **high**. Implication: any "nothing happened" result at layer -1 is uninformative; retest at mid-stack.
- **F3** — ~~**Hidden-state hysteresis signal is real and non-trivial.**~~ **REVISED BY E2** ❌ The single-seed "recovery=0.476 partial" result did **not** replicate. E2 phase A (5 seeds × 2 prompts at L14/mag=0.3/temp=0.8) produced 8/10 "runaway" regimes, 1 "plastic", 1 "partial" (plus one seed with drift=0.002 giving recovery=-913 — metric blowup). Factual prompt recovery stdev=408 on n=5. The original F3 run was a seed-dependent fluke at these parameters. Confidence on the revised finding: **high** (n=10). Implication: the framework's "partial recovery" claim needs much more careful experimental work before being publishable.
- **F4** — **The closed-loop controller does NOT measurably stabilize trajectories.** Shadow vs active A/B on identical prompt+seed+sampling: shadow avg_raw_div=0.765 with 8 warnings + 1 critical; active avg_raw_div=**0.768** with 7 warnings + 1 critical. The controller triggers, applies scaling, but divergence is slightly *higher* in active mode. Confidence: **medium** (1 prompt, 1 seed — needs replication, but the direction is clear). Implication: the scaling intervention at current thresholds doesn't help. Investigate layer choice, scaling factors, and whether scaling is the right intervention type at all.
- **F5** — **Greedy decoding + identical seed produces identical tokens even when logits shift.** Observed logit_kl_mean_during=0.014 while token_match_rate=1.0 on a stress run. The `torch.multinomial` RNG is deterministic given the seed, so small logit shifts don't flip draws. Implication: sampling alone doesn't split trajectories; we need different branch seeds OR much larger temperature to get diverging token streams from small perturbations. Confidence: **high** (reproducible).
- **F6** — **Sampling with seed-sweep produces meaningful per-seed variance.** Observe sweep with seeds 0–2 and temperature=0.8 gave avg_divergence stdev=0.23 (range 0.32–0.75). With greedy it was stdev=0. Confirms sampling unlocks seed-variance analysis. Confidence: **high**.
- **F7** — **(2026-04-18 late) Integration-quality gap.** Independent audit (Codex) found three startup-time crashes we'd missed: control sweep broken (tuple vs dict return), stress sweep broken (`int("mid")` in describe), advisory misreading `WARN` vs `WARNING`. All three fixed and smoke-validated; repo still has no automated test layer that would have caught them. **Technical-debt item**: add a pytest smoke suite that exercises every CLI sweep path with a 2-seed fake model. Until then, treat any seeded sweep run as needing end-to-end verification after code changes.
- **F9** — **(E2 phase A) Hysteresis `recovery` metric is numerically unstable when `drift` is near zero.** Formula is `1 - hysteresis / (drift + 1e-8)`; when noise injection happens to produce `drift ≈ 0.002` (seed-dependent), recovery can explode to ±1000 even for modest `hysteresis`. Observed seed=3 factual: drift=0.002, hysteresis=2.027, recovery=-913. **Fix needed (TD2)**: if `drift < min_threshold` (e.g., 0.05), mark recovery as `None` / "perturbation-did-not-propagate" instead of computing a blown-up ratio. Confidence: **high**. Until fixed, treat negative-recovery values with skepticism — they may just be low-drift artifacts.
- **F13** — **(E2 phase C, 30 runs at L27, mag ∈ {0.5, 0.7, 1.0}) Signal-dominated regime IS achievable.** At (L27, mag=1.0, factual prompt): drift mean=1.96, stdev=0.37, **DSR=5.36** (first crossing of stability threshold in the repo). 3/5 runs in "partial" regime, mean recovery=+0.001 — bounded near-elastic boundary. This is the first reproducible hysteresis signal the repo has produced. Confidence: **high** (n=5). **This is the result the framework was built for.**
- **F14** — **(E2 phase C) Magnitude threshold for stability exists AND is prompt-dependent.** Factual DSR: 1.26 (mag=0.5) → 1.63 (0.7) → **5.36** (1.0) — crosses 2 between 0.7 and 1.0. Procedural DSR: 0.91 → 1.81 → 1.80 — plateaus around 1.8, doesn't cross 2 at mag=1.0. Either procedural generation is fundamentally noisier, or the threshold is >1.0. Confidence: **medium** (single prompt per class).
- **F15** — **(E2 phase C) Recovery-vs-magnitude trend is prompt-specific.** Factual mean recovery: -0.39 → -0.19 → +0.00 (smoothly moves toward elastic with more perturbation — counterintuitive but consistent). Procedural: +0.13 → -0.20 → -0.13 (no trend). Factual's smooth trend is evidence that magnitude is the right control knob for at least some prompt classes.
- **F17** — **(E6, 10 runs) Scaling intervention at L27 has ZERO effect.** `scaling@0.5` and `scaling@2.0` both produced `logit_kl_mean_during=0.0000` exactly AND `token_match_rate=1.000` across all 5 seeds each. The scaling factor is erased by the final RMSNorm before the LM head. **This is almost certainly why the controller (F4) didn't work** — its default `intervention_type=scaling` at `act_layer=-1` was a no-op. Confidence: **very high**.
- **F18** — **(E6) Additive is the right intervention class at L27.** Additive@1.0: `logit_kl_mean_during=10.40 ± 2.78`, DSR=3.73 (reproducible), token_match=0.145 (85% of tokens flipped). Mixed regimes PLASTIC/PARTIAL/DIVERGENT, showing genuine sensitivity without pure runaway. Confidence: **high**.
- **F19** — **(E6) Projection@64 is large but always destructive.** logit_kl=7.78 ± 4.05, DSR=1.92, token_match=0.20, regimes = all DIVERGENT. Keeping only 64/2048 dims wrecks the hidden state too aggressively — not a controllable intervention. Confidence: **high**.
- **F21** — **(E8 MVP) Random-direction additive controller has net-zero effect on avg_div.** A/B over 5 seeds on the F4 sourdough prompt: SHADOW avg_div=0.6732±0.174, ACTIVE_ADDITIVE avg_div=0.6728±0.176 (Δ=+0.0004). Per-seed: seed 2 IMPROVED (0.776→0.755, warnings 10→8); seeds 0 and 4 WORSENED; seed 3 controller never fired. The random-direction intervention helps when the seeded direction opposes drift and hurts when it aligns — they cancel in aggregate. **This confirms: the wiring is right, the actuator is correct (E6/F18), but direction is the missing piece.** Confidence: **high**.
- **F22** — **(E8 MVP re-confirmation of F17)** Active scaling vs shadow on the same prompt set: Δ=+0.0004 at 4 decimal precision. Scaling at L27 is definitively a no-op in a closed loop, matching F17's finding from stress mode. The closed loop with scaling is effectively always in shadow mode regardless of the flag. Confidence: **very high**.

- **F20** — **(E6 synthesis) Controller redesign path is now clear.** For Qwen3-1.7B at L27: use `intervention_type=additive` (not scaling), calibrate magnitude against the F13 drift envelope (1.96 ± 0.37), correct by injecting additive delta in the direction that *reduces* measured divergence. Current `control` CLI default is `scaling` which F17 proves is useless at -1. Either add an `additive` mode to control, or move `measure_layer` and `act_layer` to different points so scaling has a chance to act pre-norm.

- **F16** — **(E2 phase C) Per-seed "stable basins" exist.** Factual seed=2 produced identical drift=2.429 and recovery=+0.261 at BOTH mag=0.5 AND mag=0.7 — the 40% magnitude increase wasn't enough to escape that seed's token-choice basin. Only at mag=1.0 did seed=2 shift to drift=2.469, recovery=+0.726. Factual seed=3 was no-propagation at mag=0.5 and 0.7, then drift=1.835 at 1.0 — a per-seed propagation threshold between 0.7 and 1.0. Confidence: **medium** (n=1 observation each).

- **F11** — **(E2 phase B, 10 runs at L27 mag=0.5) H6 partially rejected: late-layer is not universally more stable.** Drift-stability ratio (DSR = mean/stdev across seeds; GPT called this "SNR" but it's 1/CV): factual went L14→L27: 0.89 → 1.26 (slight improvement); procedural went 1.40 → 0.91 (WORSE). Neither prompt crossed the DSR>2 "signal-dominated" threshold. But recovery aggregate DID cleanly improve — Phase A had an outlier-dominated mean of -183 (the -913 seed); Phase B has bounded, interpretable recovery numbers (-1.35 to +0.77) thanks to TD2. Procedural mean recovery shifted from -0.50 at L14 → +0.13 at L27 — real small positive shift. Implication: **prompt type interacts with layer choice**, and the seed-dominated regime F10 identified is NOT trivially escapable by moving layer + magnitude. Confidence: **high**.
- **F12** — **(TD2 works)** 1/10 Phase B runs (factual seed=3, drift=0.000) was correctly flagged `perturbation_did_not_propagate` and recovery marked undefined rather than computing a blown-up ratio. Aggregate stats are now trustworthy across seed populations. Confidence: **high**.

- **F10** — **(E2 phase A) At (L14, mag=0.3, temp=0.8), noise perturbation propagation is seed-dominated.** Drift varied 0.002 to 3.45 across 5 seeds on the same prompt. The perturbation either "catches" a decision-critical token or misses entirely, with no middle ground. This is why F3 was a single-seed fluke. **Implication**: noise-mode experiments need to either (a) perturb at a wider window (more duration), (b) perturb at a more sensitive layer (E1 peak was L27, not L14), or (c) raise magnitude substantially to overwhelm seed variance.

- **F8** — **(E1 outcome) Layer-sensitivity is monotonic-increasing toward the output at magnitude=0.3 + sampling.** Qwen3-1.7B, stress additive, 28 layers, sweep [2,7,14,21,27] × 2 prompts × 3 seeds. Per-layer `logit_kl_mean_during` (averaged across prompts): L2=2.39, L7=2.44, L14=3.77, L21=3.47, **L27=6.41**. Token-match (lower = more flipping): L2=0.58, L7=0.68, L14=0.45, L21=0.56, **L27=0.27**. **L27 is the peak-sensitivity layer on both logit and token metrics.** Confidence: **medium** — tight stdevs on factual prompt (0.011 ± 0.004 at L2) but huge stdevs on procedural prompt (4.78 ± 8.27 at L2), suggesting prompt type strongly modulates reproducibility. This partially **revises F2**: final-layer perturbations are absorbed at small magnitudes + greedy decoding, but dominate at moderate magnitudes + sampling. **Implication for controller**: L27 is a viable intervention point; L14 is second-best. Recovery at L27 is near-zero (partial) which is the "controllable regime" we want.

---

## 3. Open hypotheses

_Claims we want to test. Each should become an experiment in §4 (or get rejected)._

- **H1** (supports F3): Recovery ratio correlates with noise_layer depth — deeper injections produce more persistent residue. **Experiment**: E2.
- **H2** (supports/rejects F4): The controller's failure is due to wrong layer, not wrong principle. A controller that intervenes at the layer-sensitivity peak (from E1) will measurably help. **Experiment**: E8 (blocked on E1).
- **H3**: Divergence signal increases predictably with generation length in the absence of any perturbation; the slope depends on model and prompt type, not on "instability". **Experiment**: E3.
- **H4**: Divergence distributions have the same *shape* across models (gpt2, tinyllama, qwen3) even if they differ in absolute scale. **Experiment**: E4.
- **H5**: Prompt-mode hysteresis residue is systematically larger than noise-mode residue on the same prompts, because prompt contamination stays in the KV cache while noise-mode perturbation only propagates forward. **Experiment**: E5.
- **H6** (synthesis of F8 + F10): Late-layer perturbation is a better control surface than mid-layer. L27 has high leverage (logit_kl peak from F8) AND is likely less dominated by seed variance than L14 (F10). "Mid-layers: low leverage + high noise sensitivity → unstable signal detection. Late-layers: high leverage + direct logit impact → more reproducible control."  **Experiment**: E2 phase B (running now — stabilize by moving from L14 to L27 and raising magnitude).

---

## 4. Experiment backlog

_Status values: `pending` / `in-progress (session YYYYMMDD)` / `complete (YYYYMMDD)` / `abandoned` / `blocked`._

Ordered by data-value _right now_. Tackle top-down unless the human overrides.

### [x] E1 — Layer-sensitivity map *(complete 2026-04-18)*
- **Status**: complete
- **Runs**: 10 sweeps (`sweep_stress_*`), 30 total runs. Prompt × layer × 3 seeds. All successful (n_ok=3 each).
- **Outcome**: **Sensitivity rises monotonically toward the output.** L2=2.39 → L7=2.44 → L14=3.77 → L21=3.47 → L27=6.41 (logit_kl mean). **Peak at L27** — contradicts earlier F2 hunch that final-layer perturbations are absorbed; see F8 for the revised claim. Factual prompt produced tight stdevs (~3% of mean); procedural prompt produced huge stdevs (often >100% of mean), suggesting prompt type modulates reproducibility. Token-match rate at L27 was 0.27 (meaning 73% of tokens flipped) — plenty of signal for the controller to work with.
- **Follow-ups**:
  - **E1.5** (new, high priority): wider layer sampling — does the curve keep climbing monotonically, or is there a peak before L27? Test L24, L25, L26, L27 to pin the shape at the tail.
  - **E1.6** (new, medium priority): magnitude sweep at L27 — at what magnitude does recovery collapse? Currently `recovery` is -0.23 at L27 procedural (near zero = controllable). Map this.
  - Blocks: **E6** (intervention-type) can now run at L27. **E8** (controller redesign) should test L14 AND L27 as candidate intervention points.

### [x] E2 — Hysteresis noise sweep (the headline experiment) · COMPLETE 2026-04-19
- **Status**: A, B, and C all complete. Signal-dominated regime identified at (L27, mag=1.0, factual). F13 is the headline.
- **Phase A outcome** (L14 mag=0.3): F3 did not replicate. See F9, F10.
- **Phase B outcome** (L27 mag=0.5): H6 partially rejected. Factual slight stability improvement, procedural worse. See F11, F12.
- **Phase C outcome** (L27 mag sweep): **BREAKTHROUGH**. DSR=5.36 at (L27, mag=1.0, factual). See F13–F16.
- **What we now know**: magnitude threshold ~1.0 at L27 takes factual prompts into signal-dominated regime with ~partial-recovery hysteresis. Procedural plateaus around DSR=1.8 at same params — needs either higher mag, different layer, or a different methodology.
- **What comes next**: E6 (intervention-type comparison at the calibrated point) and E8 (controller redesign at L27 with drift envelope from F13).
- **Question**: Is partial-recovery hysteresis a consistent phenomenon, or was F3 a single-seed fluke?
- **Why it's high value**: this is the framework's *core research claim*. F3 showed it works on one seed; we need to prove it replicates.
- **Design**: `hysteresis perturbation_mode=noise`, temperature=0.8, max_tokens=64.
  - **Seeds**: 5 (`0-4`)
  - **Magnitudes**: 3 — `[0.2, 0.3, 0.5]`
  - **Noise layers**: 3 — `[early, mid, late]` (uses semantic defaults)
  - **Prompts**: 2 — same factual + procedural as E1.
  - Total: 5 × 3 × 3 × 2 = **90 runs**. Too large — split into phases.
  - **Phase A (decide first)**: 5 seeds × mid layer × mag=0.3 × 2 prompts = 10 runs.
  - **Phase B (if A shows signal)**: expand to full matrix or pruned subset based on A's shape.
- **Primary metric**: `recovery` across seeds. Is it clustered (consistent) or scattered (noise)?
- **Secondary**: `regime` distribution, `drift`/`hysteresis` scaling with magnitude and layer.
- **Expected runtime**: ~15 min for phase A; ~60–90 min for full phase B.
- **Expected payoff**: either a genuine research finding ("recovery ratio is model/layer/magnitude-dependent with pattern X") or a rejection of F3 as a fluke.
- **Runs**: _(filled as launched)_
- **Outcome**: _(filled when complete)_

### [ ] E3 — Length-controlled divergence baseline
- **Status**: pending
- **Question**: What does natural divergence-vs-length look like with no perturbation? What's the noise floor for every other experiment?
- **Why it's high value**: without this, we can't distinguish "intervention caused drift" from "sequences just drift over time." Gates the validity of every other number.
- **Design**: `observe` on clean prompts, temperature=0.8, long generation.
  - **Seeds**: 5 (`0-4`)
  - **max_tokens**: 128 (longer — we want to see the curve at scale)
  - **Prompts**: 3 — factual ("Explain how airplanes generate lift."), reasoning ("If a train leaves Chicago at 3pm going 60mph..."), creative ("Write a short poem about autumn.")
  - Total: 5 × 3 = **15 runs**
- **Primary metric**: per-token divergence curves; fit median + IQR envelope per prompt type. Save as `research/baselines/divergence_qwen3-1.7b.json` so future experiments can subtract it.
- **Expected runtime**: ~20 min.
- **Expected payoff**: a noise floor. Future advisory will flag "above expected baseline" as a real signal instead of trusting absolute numbers.
- **Runs**: _(filled as launched)_
- **Outcome**: _(filled when complete)_

### [ ] E4 — Cross-model divergence validation
- **Status**: pending
- **Question**: Does the divergence signal behave similarly across model families, or is it model-specific noise?
- **Why it's medium-high value**: answers "are we measuring something real or an artifact of one model?"
- **Design**: same prompts as E3 × 4 models: `gpt2`, `tinyllama`, `qwen3-1.7b`, `qwen3-4b` (if 4B fits — might OOM on 16GB). Drop 4B if needed.
  - **Seeds**: 3 each
  - **Prompts**: same 3 as E3
  - Total: 4 models × 3 prompts × 3 seeds = **36 runs**
- **Primary metric**: divergence distribution shape per model (normalized). If shapes line up, the signal is probably real. If they don't, the metric is fitting noise.
- **Expected runtime**: ~25–40 min.
- **Expected payoff**: credibility gate. Every paper claim hinges on this working.
- **Runs**: _(filled as launched)_
- **Outcome**: _(filled when complete)_

### [ ] E5 — Prompt-mode vs noise-mode hysteresis head-to-head
- **Status**: pending
- **Question**: Does context contamination (prompt-mode) leave more or less residue than internal perturbation (noise-mode)?
- **Why it's medium value**: clean finding if it works — tells us something about where "memory" lives in LLM inference.
- **Design**: 5 prompts × {prompt-mode, noise-mode} × 3 seeds. Fix everything else: max_tokens=64, mid-stack layer, magnitude=0.3 for noise, same reflection template for prompt.
  - Total: 5 × 2 × 3 = **30 runs**.
- **Primary metric**: paired recovery comparison (noise recovery minus prompt recovery per seed/prompt). If consistently positive or consistently negative, we have a finding.
- **Expected runtime**: ~25 min.
- **Expected payoff**: "Context sticks more than internal state" (or the inverse) — a cleanly defensible claim.
- **Runs**: _(filled as launched)_
- **Outcome**: _(filled when complete)_

### [x] E6 — Intervention-type comparison *(complete 2026-04-19)*
- **Status**: complete
- **Runs**: 20 stress runs at L27 factual prompt. 4 types × 5 seeds.
- **Outcome**: Additive wins by a wide margin. Scaling is completely absorbed by the final RMSNorm (F17). See F17–F20.
- **Ranked by logit_kl_during**:
  1. additive@1.0: 10.40 ± 2.78 (DSR=3.73) ← winner
  2. projection@64: 7.78 ± 4.05 (DSR=1.92, all DIVERGENT, too destructive)
  3. scaling@0.5: 0.00 (no effect)
  4. scaling@2.0: 0.00 (no effect)
- **Implication**: controller should use additive, not scaling, for L27 operation.

### [ ] E9 — Trajectory-deviation-over-time metric
- **Status**: pending
- **Question**: Instead of comparing BASE vs REASK at the endpoint only, can we track how the model's hidden-state trajectory deviates step-by-step from baseline during and after perturbation? The current `hysteresis` number is a single endpoint distance; a time-resolved version would let us see whether the trajectory recovers smoothly, diverges, or oscillates.
- **Design**: Extend the hysteresis runner to emit per-token events.jsonl during BASE, PERTURB, and REASK phases (currently only observe/stress/control do this). Compute per-step divergence from the BASE reference trajectory for each phase.
- **Why filed as its own experiment**: conceptually a new metric, not a fix to existing ones. Probably unlocks better regime classification (smoothly-recovering vs oscillating vs diverging).
- **Effort**: ~half-day, needs careful hook management across 3 phases.

### [ ] E7 — Temperature sensitivity
- **Status**: pending
- **Question**: How does divergence behave across temperatures 0 / 0.5 / 0.8 / 1.2? Does it scale predictably, or expose weird behavior?
- **Why it's lower value**: diagnostic / sanity-check, not generative.
- **Design**: `observe`, same prompt, 4 temps × 5 seeds = 20 runs.

### [x] TD1 — Warm-model daemon *(complete 2026-04-18)*
- **Status**: complete
- **Outcome**: `scripts/observer_daemon.py` built. Loads model once, reads JSON-lines from stdin, dispatches to any runner (observe/stress/hysteresis/control) via new `prebuilt_backend` kwarg. Measured speedup: **~16× on reused requests** (observe: 30s cold → 1.8s warm after first run). E2 phase A ran in 3.4 min (would have been ~15 min cold). Dashboard JobManager does NOT yet use the daemon — it still spawns subprocesses — but batch orchestrators (`/tmp/run_e2_phase_a.py` style) talk to it directly. Dashboard integration is the follow-up (TD3).

### [ ] TD2 — Fix recovery-metric numerical instability (F9)
- **Status**: pending
- **Why**: `recovery = 1 - hysteresis / (drift + 1e-8)` blows up when drift ≈ 0. In E2 phase A, seed=3 factual gave drift=0.002, recovery=-913. This pollutes all aggregate stats.
- **Fix**: in `hysteresis/runner.py`, when `drift_composite < 0.05` (tunable) emit `recovery=None` and a new flag `perturbation_did_not_propagate=True`. Advisory should read the flag and suggest a larger magnitude / different layer instead of reporting a recovery number.
- **Effort**: ~30 lines.

### [ ] TD3 — Dashboard daemon integration
- **Status**: pending
- **Why**: Dashboard JobManager still spawns fresh subprocesses. Batch orchestrators use the daemon directly.
- **Fix**: JobManager tracks one daemon per (model, backend). On launch, if running daemon matches payload.model, send to its stdin; else respawn. Tail daemon stderr as log events. Write ack to SSE.
- **Effort**: ~half-day.

### [~] E8 — Controller redesign (MVP complete, needs E8.5)
- **Status**: MVP complete 2026-04-19. See F21/F22.
- **MVP outcome**: additive intervention wired into control CLI + runner. A/B on sourdough prompt showed random-direction additive is net-zero vs shadow (Δ=+0.0004). Per-seed variance confirms the actuator works (seed 2 improved 2%, seeds 0/4 got worse). Random direction is the bottleneck.
- **Next — E8.5**: replace random direction with drift-opposing direction via EMA baseline.

### [ ] E8.5 — Drift-opposing additive controller (the real test)
- **Status**: pending (code + experiment)
- **Question**: does additive intervention pointed in the opposite direction of measured drift actually reduce avg_div?
- **Design**:
  - Add an EMA tracker to the control loop: `h_ema[t] = α · h_ema[t-1] + (1-α) · h_current[t]` with α≈0.9, warmed up over the first 3-5 tokens with `intervention_active=False`.
  - When controller fires (WARN or CRIT), compute `drift = h_current - h_ema`, and inject delta = `-β · (drift / ||drift||) · ||h_current||` (unit opposition × relative magnitude).
  - Re-run the E8 MVP A/B suite with this new controller.
- **Success criterion** (same as E8): ACTIVE avg_div < SHADOW avg_div by more than 1 shadow-stdev.
- **Fallback criterion**: even if aggregate doesn't cross stdev, per-seed wins should be NOT cancelled by per-seed losses (i.e., MOST seeds should improve, not just one).
- **Effort**: ~60 lines of new code in `adaptive_runner.py`, 10 lines in the additive intervention for signed-direction support, maybe a new `DriftOpposingAdditiveIntervention` class.

---

## 5. Sessions log

_One-line per session. Link to detailed journals in `research/session_*.md` when we start writing them._

- **2026-04-18** · Foundation session. Built advisory system, fixed spectral, added sampling + logit-KL, semantic layer defaults, noise-mode hysteresis, dashboard advisory display, capabilities endpoint, 5 recipes, seed-sweeps. Established F1–F6. Controller confirmed broken (F4). This research doc seeded.
- **2026-04-18 (late)** · Bug-fix pass after Codex audit. F7: three integration crashes fixed (control sweep tuple, stress sweep `int("mid")`, advisory `WARN` vs `WARNING`). Tech-debt flag: no smoke-test suite.
- **2026-04-18 (E1 session)** · Ran E1 (layer-sensitivity, 30 runs). **F8: peak sensitivity at L27 (final layer)**, monotonic increase from L2. Revised F2 — final-layer perturbations dominate at moderate magnitude + sampling, not absorbed. Next: E2 phase A (hysteresis replication) OR E1.5 (tail refinement).
- **2026-04-18 (daemon + E2A session)** · Built warm-model daemon (TD1 complete). Ran E2 phase A — 10 hysteresis runs at L14 mag=0.3. **F3 did not replicate** — only 1–2 of 10 runs showed partial-recovery; the rest were runaway, some with numerically-pathological recovery (F9). Added F9 (recovery metric instability) and F10 (seed-dominated perturbation at these params). Queued TD2 (recovery-metric fix) and TD3 (dashboard daemon integration). Daemon measured 16× speedup on reused requests.
- **2026-04-18 (TD2 + E2B session)** · Applied TD2: recovery returns `None` when drift<0.05, with `perturbation_did_not_propagate` flag; advisory reads the flag and suggests concrete next params. Added H6 (late-layer is better control surface) from GPT synthesis. Ran E2 phase B at L27 mag=0.5 × 5 seeds × 2 prompts. **H6 partially rejected** (F11): factual DSR improved 0.89→1.26, procedural DSR WORSENED 1.40→0.91. Neither crossed DSR>2 signal-dominated threshold. BUT recovery numbers are now interpretable post-TD2 (F12). Procedural recovery shifted positive at L27. Next step: E2 phase C — magnitude sweep to see if any magnitude escapes the seed-dominated regime.
- **2026-04-18 (E2C — breakthrough session)** · Ran E2 phase C, 30 runs, magnitude sweep at L27 with {0.5, 0.7, 1.0}. **BREAKTHROUGH**: at (L27, mag=1.0, factual) DSR=5.36 (first reproducible signal in the repo). Added F13 (signal-dominated regime achieved), F14 (prompt-dependent magnitude threshold), F15 (recovery-vs-magnitude trend is prompt-specific), F16 (per-seed stable basins). **Controller work is now unblocked.** Updated §1 next-recommended-action: E6 (intervention-type comparison at L27+mag=1.0) → E8 (controller redesign).
- **2026-04-19 (E6 — controller root-cause)** · Ran E6, 20 stress runs at L27 factual comparing additive/scaling/projection. **F17: scaling at L27 has ZERO effect** (logit_kl=0.0000 exactly, both at scale=0.5 and scale=2.0) because the final RMSNorm absorbs the scale factor. This EXPLAINS F4 — the controller was doing nothing. F18: additive@1.0 is the winner (kl=10.40, DSR=3.73, 85% tokens flipped). F19: projection@64 is too destructive. F20: controller redesign path is now concrete — use additive, not scaling. Next: E8 (actually code up the additive-based controller).
- **2026-04-19 (E8 MVP — partial win)** · Coded additive controller: new `DynamicAdditiveIntervention` + `MagnitudeState` classes, wired into `adaptive_runner.py`, added `--type additive` + `--additive-{warn,crit}-magnitude` CLI args, daemon updated. A/B on sourdough (5 seeds × 3 cells): **F21: random-direction additive has net-zero effect** (Δ=+0.0004 vs shadow). Per-seed: some improve, some worsen, averages cancel. Actuator confirmed (seed 2: warnings 10→8, div 0.776→0.755). **Direction is the missing piece.** F22: scaling-in-closed-loop also confirmed zero-effect. Next: E8.5 drift-opposing controller via EMA baseline.

---

## 6. Appendix: quick-start command snippets

_Copy-paste starters for each experiment. Each uses the LLM-friendly recipe layer where possible._

**Check current state before starting**
```bash
curl -s http://127.0.0.1:8899/api/capabilities | jq '.modes | keys, .recipes | keys'
curl -s http://127.0.0.1:8899/api/runs | jq '.runs[0:5] | .[] | {id, mode, headline}'
```

**E1 layer-sensitivity primitive (one layer)**
```bash
curl -X POST http://127.0.0.1:8899/api/launch -H "Content-Type: application/json" -d '{
  "mode":"stress","model":"qwen3-1.7b",
  "prompt":"What is the tallest mountain on Earth?",
  "max_tokens":48,"seeds":"0-2",
  "layer":7,"intervention_type":"additive","magnitude":0.3,
  "start":3,"duration":8,"temperature":0.8
}'
```

**E2 phase A primitive**
```bash
curl -X POST http://127.0.0.1:8899/api/launch -H "Content-Type: application/json" -d '{
  "mode":"hysteresis","model":"qwen3-1.7b",
  "prompt":"What is the tallest mountain on Earth?",
  "max_tokens":64,"seeds":"0-4",
  "perturbation_mode":"noise","noise_layer":"mid","noise_magnitude":0.3,
  "noise_start":3,"noise_duration":8,"temperature":0.8
}'
```

**E3 baseline primitive (one prompt)**
```bash
curl -X POST http://127.0.0.1:8899/api/launch -H "Content-Type: application/json" -d '{
  "mode":"observe","model":"qwen3-1.7b",
  "prompt":"Explain how airplanes generate lift.",
  "max_tokens":128,"seeds":"0-4","temperature":0.8
}'
```

**Reading results after a run**
```bash
curl -s http://127.0.0.1:8899/api/runs | jq '.runs[0]'
# or for a sweep:
LATEST=$(ls -t runs | grep sweep | head -1)
cat runs/$LATEST/sweep.json | jq '.aggregate'
```

---

_End of document. Update this before ending the session. The next LLM is counting on you._
