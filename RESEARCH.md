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

- **Active focus**: **The naive-controller arc is done.** E8.7 phase 2 (with Codex's decoupled-layer code change) falsifies the last remaining hypothesis — acting 1 or 2 layers back with measure held fixed at -1 makes things WORSE than shadow, not better. Synthesis finding F28: the divergence signal measures token-level surprise (natural prose artifacts), not dynamical instability. Every tested controller variant either does nothing, over-actuates, or hijacks one branchpoint into a different basin — none produces closed-loop stabilization.
- **Controller status**: **closed-loop stability control is not achievable with the current trigger signal on Qwen3-1.7B.** What we have demonstrated cleanly: (1) the instrumentation stack works end-to-end (F13: reproducible drift envelope at a calibrated operating point); (2) additive perturbation at final-layer can hijack a trajectory into a lower-divergence basin at certain branchpoints (F25: reproducible on seed 2 across all our tests); (3) the divergence signal and the concept of "instability" are not the same thing (F28). The project's legitimate story is now a negative result with clear follow-ups, not a stability controller.
- **Capabilities snapshot**: observe, stress, hysteresis (prompt + noise modes), control (shadow + active, scaling + additive/random + additive/opposing with `ema` or `anchor` references), seed sweeps, multi-layer probing, logit-KL, semantic layer defaults, advisory generator, spectral trajectory probe, warm-model daemon (~16× speedup). Additive controller settings now serialize into config hashes and sweep metadata, including reference mode. All verified 2026-04-19.
- **Next recommended action**: **Pivot.** The naive-control path is exhausted. Three legitimate next directions, pick one:
  - **(a) Reframe as interpretability research.** F25's branchpoint-hijacking phenomenon is real and reproducible. Study when/why it works, map how small final-layer perturbations open/close token-level branchpoints. This is a *different* paper — about trajectory basin structure, not stability control. No more controller work needed; just deeper analysis of existing data + targeted observe runs.
  - **(b) Build a new trigger signal.** Divergence-as-instability failed. Alternatives worth testing as observation-only features: (i) entropy × divergence conjunction, (ii) divergence *acceleration* (second derivative), (iii) prompt-calibrated thresholds from E3-style baselines, (iv) drift direction consistency (cosine of consecutive drift vectors). Any of these correlating with *downstream output failure* (rather than just high raw_div) would be a new kind of finding.
  - **(c) Accept the negative result, write it up, move on.** The project has a legitimate negative contribution. A paper saying "Observer-class divergence signals measure prose surprise not instability; closed-loop control over them fails for the documented reasons" is publishable and correct. This closes the repo as a research artifact.
  - Recommend **(a)**. Branchpoint-hijacking is unique and not yet well-studied, and it's what the data has actually shown us works.

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
- **F23** — **(E8.5, 20 runs) EMA drift-opposing additive helps, but not enough and not robustly enough.** Same sourdough A/B suite with four cells (shadow, active-scaling, active-additive-random, active-additive-opposing): SHADOW avg_div=`0.6732 ± 0.1743`; ACTIVE_ADDITIVE_OPPOSING avg_div=`0.6547 ± 0.1733`, an improvement of `0.0185` (2.74%, 0.106 shadow-stdevs). Warning count dropped from 25→18. But the gain is concentrated in seed 2 (`0.7755→0.6384`, warnings 10→5); seed 0 worsened (`0.9249→0.9485`) and hit 1 critical, seeds 1 and 4 regressed slightly, seed 3 never fired. **Conclusion: measured direction is materially better than random direction, but the current EMA baseline is not reliable enough to establish closed-loop stabilization.** Confidence: **medium**.
- **F24** — **(E8.6, 25 runs) Anchor reference is only marginally better than EMA; reference choice alone does not solve control.** Same sourdough suite, now with five cells: SHADOW, ACTIVE_SCALING, ACTIVE_ADDITIVE_RANDOM, ACTIVE_ADDITIVE_OPPOSING_EMA, ACTIVE_ADDITIVE_OPPOSING_ANCHOR. Anchor opposition produced avg_div=`0.6537 ± 0.1733`, vs EMA's `0.6547 ± 0.1733` and SHADOW's `0.6732 ± 0.1743`. That is only an extra `0.0010` improvement over EMA (total gain `0.0195`, 2.89%, 0.112 shadow-stdevs). Warnings stayed `18` and crits stayed `1`, identical to EMA. Per-seed pattern also stayed the same: seed 2 improved a bit more (`0.7755→0.6335` vs EMA's `0.6384`), seed 0 still regressed badly (`0.9249→0.9480`) and hit 1 critical, seed 1 regressed slightly, seed 4 regressed slightly, seed 3 never fired. **Conclusion: anchor is directionally better than EMA, but only by a hair; the next bottleneck is gating / control policy, not baseline choice by itself.** Confidence: **medium**.
- **F27** — **(E8.7 phase 2, 20 runs, decoupled measure/act) Layer-move hypothesis FALSIFIED.** After Codex's `6cd1757` code change lifted the `measure_layer == act_layer` gate, I reran the suite holding `measure_layer=-1` fixed (identical signal to shadow) while varying `act_layer ∈ {-1, -2, -3}`. All three active cells fail the success criterion:
  - ACT_L-1: Δ=+0.020 (+0.11σ), 1/5 seeds improve, 19 interventions total — reproduces F23/F24's opportunistic-flip pattern exactly
  - ACT_L-2: Δ=**−0.033** (**−0.19σ**), 1/5 improve, 22 interventions — **worse than shadow**; the controller fires 22 times and the trajectory gets noisier
  - ACT_L-3: Δ=−0.019 (−0.11σ), 0/5 improve, 25 interventions — also worse, more cascading error
  - The F26 confound (B/C runs firing ~once total despite measure moving) is gone: controllers fire 19-25 times, similar rates to shadow's WARN triggers. But intervening earlier in the residual stream doesn't stabilize — it accumulates drift through the remaining layers rather than opposing it.
  - **Closes out the "layer placement" hypothesis** (H6 alongside F11): neither final-layer nor mid-stack actuation produces closed-loop stabilization on this model with any of the magnitudes/directions/references tested. Confidence: **high**.
- **F29** — **(scope check on TinyLlama-1.1B-Chat) F25's claim splits: only Part A generalizes.** Ran the same (L=-1, additive, opposing+anchor, mag=0.3/0.6, temp=0.8) config on the sourdough prompt across 5 seeds × 2 cells. Results:
  - Seeds 0, 1 unusable — TinyLlama-Chat hit `</s>` EOS immediately after the raw prompt (expects chat templating we didn't apply). Not a controller artifact; model-specific termination.
  - Seeds 2, 3, 4 generated output AND the controller actually fired (9-10 interventions each). Active output differs from shadow on all three — confirms token flips happened (Part A of F25 generalizes across Qwen3 and Llama families).
  - **But** on all three active seeds, avg_div went UP vs shadow (seed 2: +0.06 / seed 3: ~0 / seed 4: +0.10), and output quality DEGRADED — seed 2 dropped all spaces ("`Pleaseincludeingredients,measurements,bakingtime`"), seed 4 similarly concatenated. The "flip lands in LOWER-divergence basin" half of F25 does NOT generalize.
  - Implication: **F25 was really two claims in a trench coat.** The mechanism ("additive at final layer flips tokens at close-margin branchpoints") is architecture-general. The consequence ("flip improves output") was Qwen3-specific basin luck on the sourdough prompt — the Qwen3 baseline on that prompt was an unusually degenerate numbered-list stub, and almost any perturbation-induced detour was an improvement. TinyLlama's sourdough baseline isn't degenerate the same way, so perturbations drop it into a worse basin.
  - For the interpretability write-up: the paper can claim "additive perturbation at L-1 flips tokens at predictable branchpoint conditions (top-2 logit margin low, near uncertainty peaks)" as an architecture-general finding. It CANNOT claim "perturbation improves output" without a per-model study of basin structure.
  - Confidence: **medium** (n=1 replication model, one prompt, 3 usable seeds). A third-model check (Gemma-3-1B-it or SmolLM) would move this to high confidence.
- **F28** — **(synthesis of F17-F27) Observer's divergence signal measures token-level surprise, not dynamical instability.** Cumulative evidence across 9 findings closes the naive-controller research arc:
  - The signal spikes on word-starts, numbered markers, punctuation transitions, new semantic units — events natural to coherent prose
  - Every tested intervention (scaling F17/F22, additive-random F21, additive-opposing-EMA F23, additive-opposing-anchor F24, larger magnitude F26A, earlier layers F27) either does nothing, over-actuates and derails, or hijacks one opportunistic branchpoint (F25)
  - Result: the framework's premise (divergence = destabilization that can be corrected) is not supported on Qwen3-1.7B
  - What IS reproducibly true: a well-timed additive perturbation at a confident-generation branchpoint can hijack the trajectory into a lower-divergence basin (F25 pattern on seed 2, reproduced ×4 now). This is perturbation-induced basin-hopping, not closed-loop control
  - **Publishable framing**: the project's legitimate story is now "we built instrumentation and found that the divergence signal doesn't measure what stability controllers need it to measure" — a negative result that clears a commonly-assumed path and points at a real open question (what WOULD be a good real-time destabilization signal?)
  - Confidence: **high**
- **F26** — **(E8.7, 20 runs) Measurement-layer confound invalidates variants B/C; only variant A is interpretable.** Suite: SHADOW (act/measure=-1), A_MAG_BUMP (act/measure=-1, mag=0.8/1.2), B_LAYER_BACK (act/measure=-2, mag=0.3/0.6), C_BOTH (act/measure=-2, mag=0.8/1.2). Aggregate looks like B/C crushed shadow: B=0.3806 (+1.68σ), C=0.3820 (+1.67σ), 5/5 seeds improved. **BUT**: counting `intervention_applied` per run, B fired total=1 across all 5 seeds, C fired total=1 — the controller almost never acted. Seed 3 produced *character-identical output text* in shadow and B despite avg_div dropping 40%. The "improvement" is entirely that `measure_layer=-2` reports a quieter signal than `measure_layer=-1` — we compared different quantities. **A_MAG_BUMP** is the only cell with real controller activity (total=23 interventions), and it FAILED the success criterion: aggregate regression (-0.03σ), only 1/5 seeds improved; seed 0's output was completely derailed ("are you well? I'm going to write it..." replacing "1. Prep the sourdough starter..."), seeds 1/2/4 regressed on divergence. **Verdict: magnitude bump at L=-1 is over-actuating; layer move was not actually tested due to measurement confound; can't fix the confound without code change to decouple `measure_layer` from `act_layer`.** Confidence: **high** (cleanly demonstrated by per-seed intervention counts and identical-text comparison).
- **F25** — **(E8.7-prep diagnostic) The F23/F24 "improvement" is NOT closed-loop stabilization — it's opportunistic single-token flipping at branchpoints.** Per-step trace comparison of seed 0 vs seed 2 on the E8.6 ACTIVE_ADDITIVE_OPPOSING_ANCHOR runs (run_ids `control_run_20260419_022012` vs `control_run_20260419_022146`, and `control_run_20260419_022023` vs `control_run_20260419_022154`):
  - **Seed 0**: controller fired 10 times; token output is **character-for-character identical** to shadow across all 48 tokens. The perturbations never crossed the LM-head argmax margin; all 10 "corrections" were effectively no-ops at the token level. The +0.023 avg_div "regression" is numerical jitter from perturbed hidden states that the argmax absorbs — active behaves as de-facto shadow with a tiny noise floor.
  - **Seed 2**: controller fired 5 times; output diverged at step 7 (" " → "Use") and then branched into a completely different attractor — a coherent recipe with ingredients (`100g flour, 100g water, 100g`) instead of shadow's degenerate numbered-list stub (`1. Prepare 2. Mix 3. Let ...`). Divergence dropped naturally because the new trajectory is semantically stabler, not because the controller kept pulling it back. One lucky early flip hijacked the trajectory into a lower-divergence basin.
  - **Implication**: at (act_layer=-1, mag=0.3/0.6, temp=0.8), the additive intervention is too weak to reliably flip argmax on a confident trajectory — it only flips tokens at pre-existing branchpoints where top-2 logits are close. Closed-loop control isn't happening; opportunistic branchpoint-hijacking is. Aggregate wins are one lucky flip per good seed, not continuous stabilization.
  - **Redirects E8.7**: Codex's planned "gating" would only reduce firing rate and thus reduce the opportunistic-flip rate — wouldn't make seed 0 controllable. Real E8.7 candidates: (a) raise magnitude to reliably flip argmax, (b) move act_layer back one step to preserve effect through final RMSNorm, (c) accept reality — "controller can redirect at branchpoints but not stabilize confident trajectories" as a framing.
  - **Confidence**: **high** (grounded in per-step trace + character-identical output text on seed 0).

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

### [~] E8 — Controller redesign (MVP complete; follow-up still needed)
- **Status**: additive controller complete through E8.6 on 2026-04-19. See F21/F22/F23/F24.
- **MVP outcome**: additive intervention wired into control CLI + runner. A/B on sourdough prompt showed random-direction additive is net-zero vs shadow (Δ=+0.0004). Per-seed variance confirms the actuator works (seed 2 improved 2%, seeds 0/4 got worse). Random direction is the bottleneck.
- **E8.5 outcome**: EMA drift-opposing direction improved aggregate avg_div and warning count, but not by enough and not across most seeds. The controller is direction-aware now; the remaining problem is reference quality / gating, not basic wiring.
- **E8.6 outcome**: anchor-based opposition is only marginally better than EMA. That means "pick a different reference baseline" is not enough by itself; the next lever is gating / policy.

### [~] E8.5 — Drift-opposing additive controller
- **Status**: complete 2026-04-19, but success criterion NOT met. See F23.
- **Question**: does additive intervention pointed in the opposite direction of measured drift actually reduce avg_div?
- **Design**:
  - Add an EMA tracker to the control loop: `h_ema[t] = α · h_ema[t-1] + (1-α) · h_current[t]` with α≈0.9, warmed up over the first 3-5 tokens with `intervention_active=False`.
  - When controller fires (WARN or CRIT), compute `drift = h_current - h_ema`, and inject delta = `-β · (drift / ||drift||) · ||h_current||` (unit opposition × relative magnitude).
  - Re-run the E8 MVP A/B suite with this new controller.
- **Outcome**: partial signal only. Aggregate improved by 2.74% and warnings fell 28%, but improvement was dominated by one seed and did not exceed 1 shadow-stdev. The fallback criterion also failed because some seeds still got worse.
- **Interpretation**: direction matters, but EMA baseline alone is not a stable enough reference.

### [~] E8.6 — Reference-quality follow-up for control
- **Status**: complete 2026-04-19, but success criterion NOT met. See F24.
- **Question**: can a better baseline / gating rule turn F23's partial signal into robust multi-seed improvement?
- **Design**:
  - Compare seed 0 vs seed 2 token-step traces to see when EMA opposition flips from helpful to harmful.
  - Try one stronger reference rule:
    - anchor-based opposition using a frozen mean of the first clean tokens, or
    - EMA opposition with drift-norm gating and/or longer warmup.
  - Keep the same sourdough suite and success criterion so results stay comparable.
- **Success criterion**: ACTIVE avg_div < SHADOW avg_div by >1 shadow-stdev AND no obvious "one seed wins, others regress" cancellation pattern.
- **Outcome**: anchor beat EMA, but only trivially (`0.6537` vs `0.6547`). Warnings/crits were unchanged (`18` warnings, `1` crit each). The core cancellation pattern remained.
- **Interpretation**: reference choice matters less than expected. The next experiment should focus on gating or a smarter first-fire policy, not another baseline swap.

### [ ] E8.7 — Gated drift-opposing controller
- **Status**: pending
- **Question**: can a simple gate stop the bad interventions without losing the seed-2 wins?
- **Design**:
  - Compare seed 0 vs seed 2 token-step traces from E8.6.
  - Add one explicit gate before firing additive control:
    - minimum drift-norm threshold, and/or
    - do not intervene on the first warning until instability persists.
  - Keep the same sourdough suite and compare against SHADOW, RANDOM, EMA, and ANCHOR baselines only if needed.
- **Success criterion**: ACTIVE avg_div < SHADOW avg_div by >1 shadow-stdev AND seed 0 no longer regresses into a critical while seed 2 still improves materially.

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
- **2026-04-19 (E8.5 — direction helps, but not enough)** · Fixed additive-controller provenance (config hashing + sweep metadata), added direction-aware additive control with EMA baseline opposition, and reran the sourdough A/B suite with 4 cells over 5 seeds using one warm backend. **F23:** ACTIVE_ADDITIVE_OPPOSING beat SHADOW on aggregate (`0.6547` vs `0.6732`, -2.74%) and cut warnings `25→18`, while ACTIVE_ADDITIVE_RANDOM stayed flat and ACTIVE_SCALING stayed dead. But the gain came mostly from seed 2; seed 0 regressed and hit 1 critical. **Interpretation**: direction matters and EMA opposition is better than random, but the reference estimate is still too fragile for a clean closed-loop win. Next: E8.6 reference-quality follow-up (anchor baseline or gated EMA).
- **2026-04-19 (E8.6 — anchor barely beats EMA)** · Added explicit additive reference modes (`ema` vs `anchor`) with provenance-safe config + CLI + daemon support, then reran the sourdough suite with 5 cells over 5 seeds on one warm backend. **F24:** anchor opposition (`0.6537`) beat EMA (`0.6547`) by only `0.0010`, with the same warnings (`18`) and crits (`1`). Seed 2 improved slightly more under anchor; seed 0 still regressed and hit 1 critical. **Interpretation**: better reference choice alone is not enough. Next: E8.7 gated drift-opposing control.

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
