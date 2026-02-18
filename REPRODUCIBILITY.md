# Reproducibility Checklist

## Required for Public Claims

1. Pin commit hash in every figure/table caption.
2. Report model key, backend, seed, and intervention settings.
3. Include at least 3 seeds for each claimed comparison.
4. Report mean + confidence interval (not only best run).
5. Publish raw `results.json` artifacts used for plots.

## Standard Run Metadata

Each V2 run writes:

- `config_hash`
- `seed_cache_fingerprint`
- `config` with intervention + backend fields
- `metrics.summary` and per-run regime labels

Adaptive-controller runs write:

- `events.jsonl` with per-token diagnostics and control decisions
- `summary.json` with status counts and aggregate control metrics
- optional `dashboard.html` artifact (when plotly/pandas are installed)

Baseline-hysteresis runs write:

- stage frames (`frame_base.json`, `frame_perturb.json`, `frame_reask.json`)
- `summary.json` with hysteresis/recovery metrics
- distribution-shift (JS bits) comparisons across contexts

## Suggested Reporting

1. Include at least one baseline-hysteresis protocol result for protocol comparison.
2. Include at least one intervention-engine result from the same model/backend family.
3. Include at least one adaptive-controller run with per-token event traces.

## Public Release Packaging

1. Tag a release (e.g., `v0.1.0`).
2. Upload run artifacts under `artifacts/` or a release asset.
3. Update `CITATION.cff` with real repository URL + release date.
4. (Optional) connect repo to Zenodo for DOI minting.
