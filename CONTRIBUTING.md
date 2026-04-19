# Contributing

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional research extras:

```bash
pip install -r requirements-optional.txt
```

## Pull Request Guidelines

1. Keep changes scoped to one subsystem when possible.
2. Include a short reproducibility note in PR description:
   - command used
   - seed(s)
   - key output paths
3. Do not merge claims without multi-seed results.
4. Keep CLI behavior backward compatible unless discussed first.

## Research Workflow

For experiment-driven work, use the repo workflow note:

- [docs/RESEARCH_WORKFLOW.md](docs/RESEARCH_WORKFLOW.md)

It covers:
- how to orient from `RESEARCH.md`
- how to define success criteria before runs
- how to verify provenance before interpretation
- how to update the research log after each session

## Quick Checks

```bash
python -m py_compile baseline_hysteresis_v1/runner.py
python -m py_compile v1.5/V1.5_runner.py
python -m py_compile intervention_engine_v1.5_v2/intervention.py
python -m py_compile intervention_engine_v1.5_v2/backend.py
python -m py_compile intervention_engine_v1.5_v2/sae_adapter.py
python -m py_compile adaptive_controller_system4/adaptive_loop.py
python -m py_compile adaptive_controller_system4/adaptive_runner.py
python -m py_compile adaptive_controller_system4/dashboard.py
```
