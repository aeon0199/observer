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

## Quick Checks

```bash
python -m py_compile baseline_hysteresis/runner.py
python -m py_compile intervention_engine/intervention.py
python -m py_compile intervention_engine/backend.py
python -m py_compile intervention_engine/sae_adapter.py
python -m py_compile adaptive_controller/adaptive_loop.py
python -m py_compile adaptive_controller/adaptive_runner.py
python -m py_compile adaptive_controller/dashboard.py
```
