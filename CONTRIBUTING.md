# Contributing

Contributions that improve correctness, reproducibility, documentation, testing, or experimental transparency are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
```

Run a small deterministic smoke test before opening a pull request:

```bash
veriweave --offline --max-tasks 2 --methods "VeriWeave-Core,VeriWeave-VITA-PRO"
```

## Pull requests

- Keep each pull request focused on one coherent change.
- Add or update tests for behavioral changes.
- Update the README and architecture documentation when interfaces or methodology change.
- Do not commit generated graph exports, virtual environments, caches, or experiment outputs.
- Explain any changes to prompts, budgets, metrics, selection weights, or benchmark labels.
- Preserve backward compatibility where practical, or document the migration clearly.

## Research integrity

- Never present offline fallback outputs as model-backed experimental results.
- Do not tune on the final held-out benchmark.
- Report negative findings and failed runs alongside successful results.
- Distinguish official external baselines from local approximations.
- Retain the run manifest and task-level metrics for reported experiments.

## Commit style

Use concise, imperative commit messages, for example:

```text
Add provenance selection regression tests
Clarify paper-valid execution rules
Fix temporal replay for superseded clauses
```
