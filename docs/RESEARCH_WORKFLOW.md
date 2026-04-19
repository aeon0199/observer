# Observer Research Workflow

This is the default workflow for experiment-driven work in `observer`.

The goal is simple: make one clear change, run one clear experiment, record one honest result.

## 1. Orient Before Acting

Open [RESEARCH.md](../RESEARCH.md) first.

Read:
- `§1 Current state`
- `§2 Findings log`
- `§4 Backlog`

Do not start coding or running experiments until you know:
- what is already established
- what is still open
- what exact question this session is answering

Also remember: each run's `summary.json` includes an `advisory` block with `observations`, `likely_causes`, `next_actions`, and `flags`. That advisory is part of the handoff between sessions and between agents, so read it when picking work up cold.

If the human redirects, follow the redirect explicitly. Otherwise, pick the highest-value unchecked item in `§4`.

## 2. Write the Question Down

Frame the session as one sentence before doing work.

Examples:
- `Does drift-opposing additive intervention reduce avg_div vs shadow?`
- `Does anchor-based reference outperform EMA opposition on the sourdough prompt?`

Write the success criterion before the run.

Examples:
- `ACTIVE avg_div < SHADOW avg_div by >1 shadow-stdev`
- `Most seeds improve, not just the mean`

If there is no crisp question and no crisp success criterion, the run is probably not ready.

## 3. Separate Build Work From Experiment Work

If code changes are needed:
- make them first
- keep them scoped to one logical change
- verify they compile before running experiments

Do not mix exploratory code edits, experiment execution, and interpretation in one sloppy batch.

## 4. Use the Warm Daemon

For repeated experiments, use `scripts/observer_daemon.py` so the model loads once.

One gotcha: the daemon redirects `stdout` so JSON acknowledgements are not polluted by runner printouts. If an orchestrator expects human-readable logs on `stdout`, it will get confused. Treat structured daemon responses as the real protocol and read diagnostic noise from `stderr` / saved artifacts instead.

Experiment orchestration scripts should:
- start the daemon once
- send JSON-line requests
- print per-run progress with clear labels
- print aggregate metrics at the end
- include run ids for every completed run
- compute reproducibility stats such as DSR when making stability claims

Prefer temporary orchestrators for one-off suites. Keep permanent repo scripts only if they will be reused.

## 5. Check Provenance Before Interpretation

Before trusting any result, inspect the saved artifacts and confirm the run actually recorded what you think you ran.

Check:
- `config.json`
- `summary.json`
- run id / run directory
- intervention type
- layer(s)
- seed(s)
- controller settings
- backend / model identity

If the saved config does not fully capture the meaningful knobs, fix that before claiming a result.

## 6. Interpret Honestly

Always look at per-seed behavior before writing the aggregate story.

Rules:
- a mean hiding large variance is not a strong finding
- a single-seed win is an anecdote, not a conclusion
- `perturbation_did_not_propagate` means the run was uninformative, not stable
- if advisory flags say `no-op` or `noise-absorbed`, treat the result with skepticism and verify from raw metrics

Use the advisory as a helper, not as the ground truth. Raw metrics and saved artifacts win.

## 7. Use Evidence Tiers

Not every result needs the same claim strength.

- `Provisional finding`: useful directional result, but narrower evidence. Label it lower confidence.
- `Strong finding`: should usually have multi-seed support and at least some cross-prompt validation.

As a rule of thumb:
- never promote a single-seed result into the main findings log as if it is settled
- broader claims should usually have at least `>=3` seeds and `>=2` prompts

If a result is helpful but still narrow, record it honestly as provisional rather than pretending it is final.

## 8. Update RESEARCH.md Every Session

This is mandatory.

Update:
- `§1 Current state`
- `§2 Findings log`
- `§4 Backlog`
- `§5 Sessions log`

The advisory written into each run's `summary.json` is also part of the handoff surface. When a run changes what the next agent should do, make sure the `RESEARCH.md` update and the run advisory tell the same story.

When adding a finding:
- give it an `F##`
- state the claim clearly
- include actual numbers
- include confidence
- include implication / what it means
- mention the run ids when possible

If a new result revises an older one, mark that explicitly.

## 9. Git Hygiene

Commit each logical code change cleanly.

Default branch workflow:
- commit changes to the working branch
- push the working branch
- merge or fast-forward `main` only when the state is stable enough to be the public face of the repo

Do not treat `main` as a scratchpad for mid-iteration research code.

## 9.5. Know When To Stop Iterating

Do not keep tuning the same experiment after it has already taught you the core thing.

As a rule of thumb:
- if the same fundamental result reproduces across two close variants, move to the next backlog question
- if the remaining work is parameter polish rather than hypothesis resolution, stop and advance
- if you need a different question to make progress, write that new question down explicitly instead of endlessly extending the old one

## 10. Report Like a Research Partner

When reporting back:
- use tables for numbers when possible
- say what worked
- say what failed
- say what the failure implies
- recommend the next experiment clearly

Negative results are good research output if they narrow the search space.

## Hard Stops

Pause and report to the human if:
- a new result appears to contradict the findings log
- you are tempted to change code mid-experiment to improve the numbers
- you have spent more than 30 minutes debugging the same run
- the work has drifted away from the queued question
- a destructive git operation seems necessary

## Default Mindset

This repo is not trying to prove the controller was already correct.

The point of the research loop is to discover:
- where intervention matters
- which intervention class works
- what magnitude regime is real
- what counts as a real signal vs noise
- what kind of controller the system actually needs

That means ruling out bad ideas is progress, not failure.
