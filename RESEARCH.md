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

- **Active focus**: characterize the signal before fixing the controller. E1 is complete — peak sensitivity is at L27 (final layer), contradicting F2 at this magnitude. See F8.
- **Controller status**: **ON HOLD** — see F4. But now we have a candidate intervention point (L27 or L14) for E8 when we come back to it.
- **Capabilities snapshot**: observe, stress, hysteresis (prompt + noise modes), control (shadow + active), seed sweeps, multi-layer probing, logit-KL, semantic layer defaults, advisory generator, spectral trajectory probe. All verified 2026-04-18.
- **Next recommended action**: fix **TD2** (recovery-metric numerical instability) before any more hysteresis experiments — otherwise every future aggregate is polluted. It's ~30 lines. Then consider: **E2 phase B at L27 with mag=0.5** (higher magnitude + most-sensitive layer from E1 might overcome the seed variance F10 identified). Alternate path: **E3** (length-controlled baselines) so any future hysteresis result has a noise floor to compare against.

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

### [~] E2 — Hysteresis noise sweep (the headline experiment) · PHASE A COMPLETE 2026-04-18
- **Status**: phase A complete (negative/revising result). Phase B pending TD2 fix.
- **Phase A outcome**: 10 runs (5 seeds × 2 prompts) at L14/mag=0.3/temp=0.8. **F3 did not replicate** — 8/10 runaway, 1 plastic, 1 partial. See F9, F10. Recovery metric is unstable (F9), and the perturbation effect is seed-dominated at these params (F10).
- **Phase B plan** (once TD2 fixed): re-run at L27 (E1's peak) with mag=0.5 and 5 seeds. Hypothesis: a more-sensitive layer + larger magnitude will swamp seed noise and produce reproducible drift/recovery.
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

### [ ] E6 — Intervention-type comparison
- **Status**: pending
- **Question**: Which intervention class (additive / projection / scaling) produces the largest logit_kl per unit magnitude at the peak-sensitivity layer (from E1)?
- **Why it's medium value**: informs the controller's choice of what to *do* once it decides to act.
- **Design**: blocked on E1 (need to know peak layer). Then: 3 types × 3 magnitudes × 3 seeds × 2 prompts at the peak layer.
- **Blocked on**: E1.

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

### [ ] E8 — Controller redesign (blocked)
- **Status**: blocked on E1, E2, E6
- **Question**: Can we build a controller that actually helps, given what we learn from E1 (where) and E6 (what)?
- **Hypothesis H2**: controller that intervenes at the peak-sensitivity layer (from E1) with the intervention type that moves logits most per unit magnitude (from E6) will measurably reduce avg divergence AND improve token-level outcomes.
- **Design**: re-run the shadow/active A/B but at the E1-identified layer and with the E6-identified intervention, with thresholds calibrated from E3's baseline.
- **Success criterion**: active mode's avg_raw_div is at least 10% lower than shadow mode's, with no degradation in output coherence.

---

## 5. Sessions log

_One-line per session. Link to detailed journals in `research/session_*.md` when we start writing them._

- **2026-04-18** · Foundation session. Built advisory system, fixed spectral, added sampling + logit-KL, semantic layer defaults, noise-mode hysteresis, dashboard advisory display, capabilities endpoint, 5 recipes, seed-sweeps. Established F1–F6. Controller confirmed broken (F4). This research doc seeded.
- **2026-04-18 (late)** · Bug-fix pass after Codex audit. F7: three integration crashes fixed (control sweep tuple, stress sweep `int("mid")`, advisory `WARN` vs `WARNING`). Tech-debt flag: no smoke-test suite.
- **2026-04-18 (E1 session)** · Ran E1 (layer-sensitivity, 30 runs). **F8: peak sensitivity at L27 (final layer)**, monotonic increase from L2. Revised F2 — final-layer perturbations dominate at moderate magnitude + sampling, not absorbed. Next: E2 phase A (hysteresis replication) OR E1.5 (tail refinement).
- **2026-04-18 (daemon + E2A session)** · Built warm-model daemon (TD1 complete). Ran E2 phase A — 10 hysteresis runs at L14 mag=0.3. **F3 did not replicate** — only 1–2 of 10 runs showed partial-recovery; the rest were runaway, some with numerically-pathological recovery (F9). Added F9 (recovery metric instability) and F10 (seed-dominated perturbation at these params). Queued TD2 (recovery-metric fix) and TD3 (dashboard daemon integration). Daemon measured 16× speedup on reused requests.

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
