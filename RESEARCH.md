# Observer Research Lab — Mapping Program

_Active living document. This is Phase 2 of the Observer project. The controller arc (Phase 1) is complete and archived in [RESEARCH_CONTROLLER.md](RESEARCH_CONTROLLER.md)._

Last updated: **2026-04-19** · Active agent: Claude Opus 4.7 (1M context) · Program start: **2026-04-19**

---

## 0. How to use this doc (for the next LLM)

**Start of session**
1. Read §1 (Current state) to know the active mapping question.
2. Skim §2 (Carry-over facts) — you don't need to re-derive these.
3. Open an experiment from §4 (Mapping backlog). Check what's unchecked with highest priority.
4. If the user redirects, follow the redirect explicitly and update §1.

**Historical context**
- This doc tracks the mapping program that started 2026-04-19. It is forward-looking by design.
- For *why* this program exists — the complete controller arc, findings F1–F29, experiments E1–E8.7 — see [RESEARCH_CONTROLLER.md](RESEARCH_CONTROLLER.md).
- Don't re-run experiments already recorded there. Evidence in the carry-over section below is a summary; the full record is in the archive.

**Discipline**
- Full protocol: [docs/RESEARCH_WORKFLOW.md](docs/RESEARCH_WORKFLOW.md)
- Evidence standard: strong findings need ≥3 seeds AND ≥2 prompts AND ≥2 models. Single-seed or single-model results are labeled as provisional.
- Before trusting any result, check provenance (config.json, events.jsonl, applied-count).
- Negative results are progress. Don't retroactively adjust success criteria.

**During the session**
- Mark experiments in-progress with today's date.
- Log run_ids under Outcome lines so future agents can re-inspect raw data.
- Stop when the stop condition is met — don't keep probing after the question is answered.

**End of session**
- Update §2 if a finding meets the evidence standard.
- Update §4 with outcomes on completed experiments, and new follow-ups.
- Update §5 with a one-line session entry.
- Update §1's "next recommended action" so the next agent can start cleanly.

---

## 1. Current state

- **North star**: *Observer's job is to map the geometry of trajectory sensitivity, persistence, and branchpoint behavior in real language-model generation.* This replaces the old "closed-loop stability control" framing, which was falsified on Qwen3-1.7B (see RESEARCH_CONTROLLER.md §2 F4/F17/F21/F27/F28).
- **Program structure**: three mapping questions, Q1–Q3 in §3. Each has a crisp stop condition so the work terminates cleanly.
- **Controller status**: **paused**, not dead. §4 defines the evidence that would bring controller research back to active status.
- **Instrument status**: fully operational. Warm daemon, sampling, seed sweeps, drift-opposing actuation, decoupled measure/act layers, per-step diagnostics — all verified and working (see RESEARCH_CONTROLLER.md TD1, §F17–F29).
- **Next recommended action**: **M1.2 — log top-2 logit margin and per-step logit entropy in control events.** M1.1 produced an honest within-architecture AUROC of 0.82 (passes) but cross-architecture AUROC of 0.77 (fails). Available features leak architecture identity; the architecture-invariant feature that directly measures argmax-margin (top-2 logit margin) isn't logged. Code change: ~15 lines in `adaptive_runner.py` + `observe/runner.py` to add `top2_margin` and `logit_entropy` to the per-step event dict. Owner: Codex per our division. Then rerun `scripts/analyze_branchpoints.py` with the same cross-model slice and see if AUROC clears 0.80.

---

## 2a. Mapping-program findings

- **F30** — **(M1.1 offline branchpoint analysis) Q1 partially answered: within-architecture flippability is predictable from hidden-state geometry, but cross-architecture generalization fails the stop bar.** Using existing control runs (~115 in `runs/`) with pair-level train/test split and shadow-features-only (after fixing a circular-feature leak in Codex's initial smoke test), held-out AUROCs:
  - Full archive, mixed variants (including random-direction that tautologically never flips): **0.94** — inflated
  - Qwen3-1.7B only, opposing, sourdough, L=-1: **0.82** — passes, but single-model
  - Qwen3-1.7B + TinyLlama, opposing, sourdough, L=-1: **0.77** — FAILS the 0.80 bar
  - Top predictive features (same across all three tests): `spectral.permutation_change` (trajectory-axis FFT response), `spectral.total_power` (higher = less flippable), `layer_stiffness.-1.elasticity` (low final-layer velocity = flippable), `svd.top1_energy_frac` (concentrated trajectory = flippable), `step_idx` (later tokens more flippable).
  - **Interpretation**: within a single architecture, features of the clean trajectory predict which tokens will flip under perturbation. Across architectures, these features leak architecture identity (hidden-state norms, stiffness scales differ between Qwen3 and TinyLlama) and don't transfer. The mechanism is general (F25 Part A) but the predictive signal in available features is per-architecture.
  - **What's missing**: top-2 logit margin and per-step logit entropy — both architecture-invariant and directly measure argmax-margin (the mechanistic definition of flippability). Neither is logged in current events.jsonl.
  - **Next (M1.2)**: code change to log those two features in control and observe event writers. Rerun analyzer. Expected to close the cross-model gap if logit-margin is indeed the right feature.
  - Analyzer: `scripts/analyze_branchpoints.py`. Artifacts on branch `codex/runtime-lab-ad` commit `1ec2c14`.
  - Confidence: **medium/high** on same-architecture claim, **pending** on cross-architecture until M1.2.

## 2. Carry-over facts (established in controller arc)

_These are motivating evidence for the mapping program. Full records in RESEARCH_CONTROLLER.md §2._

- **F13** (cited): reproducible drift operating point at (Qwen3-1.7B, L27, noise_mag=1.0, factual, temp=0.8) — drift 1.96 ± 0.37, DSR=5.36 across 5 seeds. The instrument can produce reproducible perturbation effects.
- **F17 / F22**: scaling at final layer is absorbed by the model's final RMSNorm. Zero effect on logits. Don't use scaling as an intervention class at L=-1.
- **F18**: additive at L=-1 with mag=1.0 produces reproducible decision-distribution shifts (logit_kl=10.40, DSR=3.73, 85% token flip rate on factual prompts).
- **F25 Part A** (**generalizes across Qwen3 + TinyLlama**): additive perturbation at final transformer layer can flip individual tokens at close-margin branchpoints. Architecture-general mechanism.
- **F25 Part B** (**Qwen3-specific**): the hijacked trajectory landing in a *lower-divergence* basin was specific to Qwen3's degenerate sourdough baseline. On TinyLlama, same mechanism drops trajectories into *worse* basins. Basin-hop direction is not a general law.
- **F27**: acting 1–2 layers back from the final layer produces WORSE control, not better. Mid-stack interventions cascade destructively. Layer-move hypothesis is falsified.
- **F28** (the big one): the divergence signal measures token-level prose surprise (word-starts, semantic transitions, punctuation) — not dynamical instability. The original controller trigger wasn't what we thought it was.

---

## 3. Mapping questions

Three questions, each a deliberate research program with a stop condition. Work serially on one question at a time; don't split attention.

### Q1 — Branchpoint geometry: when are tokens flippable?

**The claim we're trying to establish.** Given a token position during generation, predict whether a small additive perturbation at L=-1 will flip the argmax. Build a feature-based classifier from step features already in `events.jsonl`:
- top-2 logit margin
- step entropy
- position in sequence
- token type (word-start vs mid-word vs punctuation)
- hidden-state delta norm from prior step
- predicted next-token divergence

**Evidence standard**: ≥3 seeds, ≥2 prompts, ≥2 models. Held-out test set.

**Stop condition**: a feature-based rule achieves precision ≥0.7 AND recall ≥0.5 on held-out runs. If no single rule gets there, we can train a simple logistic regression / decision tree — still counts if AUROC ≥ 0.8.

**Why it matters**
- Clean interpretability finding on its own: "we can predict when perturbations will flip tokens."
- It's the direct path back to a smarter controller — predictable branchpoints = designable trigger.
- Matches where mechanistic interpretability research is going.

**Current status**: not started. M1.1 planned as the first step using existing data.

---

### Q2 — Perturbation propagation: how does an injected delta evolve?

**The claim we're trying to establish.** Given an additive injection of size `s` at layer `L`, measure the delta's L2 norm at every downstream layer up to L=-1. Does it decay, amplify, or get absorbed by specific norm layers?

**Evidence standard**: ≥5 seed replicates per `(injection_layer, intervention_type, magnitude)` cell. Multi-model replication for any "general" claim.

**Stop condition**: per-layer propagation curve that cleanly explains F27's "earlier layer = worse" finding. Specifically — the curve should predict which layers produce "bounded" cascade (delta stays bounded at each downstream layer) vs "destructive" cascade (delta amplifies or destroys hidden-state structure).

**Why it matters**
- Explains F27 mechanistically instead of treating it as an empirical fact.
- If any layer has bounded propagation, that's the principled layer for a future controller.
- Useful for any hidden-state intervention work (safety, steering, interpretability).

**Current status**: not started. Existing stress runs already have layer_stiffness data at multiple probe layers — partial data available for free.

---

### Q3 — Basin structure: when does a flip improve vs. degrade output?

**The claim we're trying to establish.** Given a token flip at branchpoint T, predict whether the resulting trajectory lands in a better or worse basin. F29 showed this is model-specific; we want to characterize the model-dependent features that determine it.

**Evidence standard**: ≥3 models, ≥10 prompts, ≥3 seeds per `(model, prompt)` cell.

**Stop condition**: a feature of the pre-flip generation (or the model itself) that predicts improve-vs-degrade with AUROC ≥ 0.7. Candidate features:
- Baseline output repetition score (is the model in a degenerate loop?)
- Baseline avg_divergence (how unstable is the trajectory anyway?)
- Prompt class (factual/procedural/creative/reasoning/code)
- Model family (Qwen vs Llama vs Gemma)
- Token position where flip lands (early in generation may matter more)

**Why it matters**
- Resolves the F25/F29 split — tells us when branchpoint-hijacking actually helps.
- If improvement predictors exist, the controller gains a meaningful trigger: "act only when expected-improvement > threshold."
- Either way: a complete interpretability story about perturbation-induced basin-hopping.

**Current status**: not started. F29 is a single data point (TinyLlama). Needs at least 2 more models.

---

## 4. Controller-return criteria

The controller thesis is paused, not dead. It returns to active status when **any two** of the following become true during mapping work:

1. **Q1 produces a branchpoint predictor with precision ≥ 0.7.** Means we have a real trigger (predictable flippable tokens) instead of blind `raw_div > threshold`.
2. **Q2 identifies a specific layer with bounded-cascade propagation.** Means we know where to act without destructive cascade (F27 problem solved).
3. **Q3 identifies a predictable class of prompts or models where flips tend to improve.** Means we know when firing the controller helps rather than harms.

If any two land, a controller redesign experiment is warranted. If none land during the mapping program, the interpretability findings stand alone and the controller stays on ice.

**What does NOT justify re-opening controller work**: more tuning of the existing `raw_div → scaling | random additive | EMA opposition` design. That entire search space is exhausted (RESEARCH_CONTROLLER.md F17–F27).

---

## 5. Mapping backlog

Order: do M1.1 first (free, might answer Q1 from existing data). Then M2.1 and M3.1 in parallel (low cost, leverage data). Only move to M1.2 / M2.2 / M3.2 if earlier results don't settle the question.

### [~] M1.1 — Offline branchpoint analysis from existing control runs *(partial pass, 2026-04-19)*
- **Outcome**: analyzer at `scripts/analyze_branchpoints.py` (Codex wrote, Claude fixed a circular-feature leak). Honest cross-model AUROC 0.77 (fails 0.80 bar); same-model AUROC 0.82 (passes). See F30.
- **Status**: complete on existing data. Gap identified: need top-2 logit margin + per-step logit entropy as architecture-invariant features. Queued as M1.2.
- **Archived task entry** (preserving original design for context):

### [ ] M1.1 — Offline branchpoint analysis from existing control runs (original task)
- **Question**: From our ~150 existing control runs, can we characterize which step-level features predict whether a token gets flipped?
- **Design**:
  - Iterate over every shadow/active pair we have in `runs/`
  - For each step, compute `flipped = shadow_token_text != active_token_text`
  - Extract per-step features from events.jsonl (raw_div, avg_score, hidden_post_norm, pre_post_delta_norm, ...)
  - Fit a simple classifier (logistic regression or decision tree, feature importances visible)
  - Hold out 20% of runs as test set
- **Stop condition**: AUROC ≥ 0.8 on held-out runs using features already in events.jsonl. If that fails, AUROC ≥ 0.7 with one or two additional re-extracted features (top-2 logit margin computed offline from saved logits if available).
- **Expected runtime**: no GPU, all offline. Writing the analyzer is ~1 hour.
- **Expected payoff**: either the Q1 predictor (if features are sufficient) or a clear specification of what per-step data we need to log in future runs for M1.2 to work.
- **Runs**: (no new runs — purely offline analysis)
- **Outcome**: _(filled when complete)_
- **Follow-ups**: _(filled when complete)_

### [ ] M1.2 — Add logit-margin / logit-entropy to control + observe events
- **Status**: queued 2026-04-19. Owner: Codex per code-change division.
- **Question**: Does adding architecture-invariant logit-level features close the cross-model gap that M1.1 left open (same-model AUROC 0.82, cross-model 0.77)?
- **Code change** (Codex):
  - In `control/adaptive_runner.py` and `observe/runner.py`, after computing `logits`, extract:
    - `top1_logit = float(logits.max().item())`
    - `top2_logit = float(logits.topk(2).values[-1].item())`
    - `top2_margin = top1_logit - top2_logit`
    - `logit_entropy = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum().item()`
  - Write these four floats into the per-step event dict alongside existing fields.
  - Keep backward-compat: don't break old event readers — new fields are additive.
  - ~15 lines per runner, ~30 total.
- **Experiment once code lands** (Claude):
  - Rerun a small suite (sourdough, 5 seeds, both Qwen3-1.7B and TinyLlama, opposing+anchor, mag=0.3/0.6). ~10 runs.
  - Re-run `scripts/analyze_branchpoints.py` on the new events.jsonl — top-2 margin should be the dominant predictor if the theory is right.
  - **Stop condition** (Q1): cross-model AUROC ≥ 0.80.
- **Expected payoff**: either Q1 fully answered with a clean architecture-invariant predictor, OR clean evidence that branchpoint flippability has real architecture-specific components beyond just logit margin (also a finding).

### [ ] M1.3 — Targeted flippability sweep (only if M1.2 insufficient)
- **Question**: If even with logit features the cross-model AUROC doesn't clear 0.80, run a larger targeted suite to see whether more data helps or the ceiling is structural.
- **Design**: 10 prompts × 3 magnitudes × 3 seeds × 2 models = 180 runs.
- **Expected runtime**: ~1 hour with daemon.
- **Blocked on**: M1.2 being insufficient.

### [ ] M2.1 — Per-layer propagation measurement from existing data
- **Question**: For the stress runs we already have at varying `intervention_layer`, extract the delta L2 norm at each probe_layer downstream of the injection. Build the propagation curve from existing events.jsonl layer_stiffness fields.
- **Design**: pure offline analysis of all existing stress runs.
- **Stop condition**: per-layer propagation curve for at least 2 intervention types at 3 injection layers on Qwen3-1.7B. If clean monotonic pattern, claim; otherwise needs M2.2.
- **Expected runtime**: ~30 min analysis.
- **Outcome**: _(filled when complete)_

### [ ] M2.2 — Synthetic-injection propagation sweep (only if M2.1 inconclusive)
- **Question**: If events.jsonl doesn't have enough probe_layer coverage, run a controlled observe suite with synthetic injection: seed an additive delta at chosen layer, hook-capture hidden states at every subsequent layer.
- **Design**: needs a small code change in the observe runner to allow "inject at L, capture at all downstream" mode. 3 injection layers × 3 magnitudes × 3 seeds × 2 models.
- **Code**: Codex territory. ~40 lines in observe/runner.py + a new orchestrator.
- **Blocked on**: M2.1 being insufficient + Codex code change.

### [ ] M3.1 — Third-model scope check for F25 Part A
- **Question**: Does the token-flip mechanism replicate on a third model family (Gemma-3-1b-it or SmolLM3-3B)?
- **Design**: replicate the F29 scope check setup — (L=-1, additive, opposing+anchor, mag=0.3/0.6, temp=0.8), 5 seeds × 2 cells (shadow/active) × 1 well-chosen prompt that doesn't hit EOS on instruct-tuned models.
- **Stop condition**: ≥3 of 5 seeds show clear token flips when active. If yes, F25 Part A is 3-model confirmed.
- **Expected runtime**: ~5 min with daemon (after model download if not cached).
- **Outcome**: _(filled when complete)_

### [ ] M3.2 — Prompt-class × model basin mapping
- **Question**: For each (model, prompt_class) cell, does the active intervention land in a better or worse basin on average? What features of the baseline predict the outcome?
- **Design**: 3 models × 5 prompt classes (factual / procedural / creative / reasoning / code) × 3 seeds × 2 cells = 90 control runs.
- **Stop condition**: one feature of the baseline that predicts improve-vs-degrade with AUROC ≥ 0.7.
- **Expected runtime**: ~30 min with daemon.
- **Blocked on**: M3.1 succeeding (no point doing class mapping if Part A doesn't replicate on a third model).

---

## 6. Sessions log (Mapping phase)

- **2026-04-19 · Program defined.** Pivoted from controller-focused to mapping-focused. Three questions (Q1 branchpoint geometry, Q2 propagation, Q3 basin structure) with stop conditions. Controller-return criteria specified. Next: M1.1.
- **2026-04-19 · M1.1 ran, partial pass.** Codex wrote `scripts/analyze_branchpoints.py`. Claude caught + fixed a circular-feature leak (features from active events included intervention_applied, scale_used, etc. — tautological). With honest shadow-features, held-out AUROC 0.82 same-model, 0.77 cross-model. F30 added. M1.1 can't fully close Q1 without architecture-invariant features; M1.2 (logit-margin + logit-entropy logging) is queued as next code task for Codex.

_(For sessions covering the controller arc 2026-02 through 2026-04-19, see [RESEARCH_CONTROLLER.md](RESEARCH_CONTROLLER.md) §5.)_

---

## 7. Quick-start snippets

**Orient before starting**
```bash
# see the status of both docs
head -40 RESEARCH.md RESEARCH_CONTROLLER.md

# current state via API
curl -s http://127.0.0.1:8899/api/capabilities | jq '.modes | keys'
curl -s http://127.0.0.1:8899/api/runs | jq '.runs[0:5] | .[] | {id, mode, headline}'
```

**Run the daemon (required for any active experiment)**
```bash
cd ~/observer
source .venv/bin/activate
python scripts/observer_daemon.py --model qwen3-1.7b
```

**Orchestrator pattern for a new mapping experiment**
See `/tmp/run_e87_phase2.py` or `/tmp/run_f25_scope_check.py` in the controller arc — same pattern works:
  1. spawn daemon as subprocess (stdin=PIPE, stdout=PIPE)
  2. read "ready" line from stdout
  3. send JSON-line request, read JSON-line ack
  4. record run_id + summary metrics + event-level data
  5. aggregate per-cell, print verdict table with stop-condition check

**Offline analysis pattern for M1.1 / M2.1**
Read `runs/*/events.jsonl`, extract per-step features, pair shadow vs active by matching seed + config, aggregate into a DataFrame / dict-of-lists, fit a scikit-learn classifier or hand-compute thresholds. No daemon needed.

---

_End of document. Update before ending the session. The next LLM is counting on you._
