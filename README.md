# Human Activity Recognition (HAR)

Unified, Hydra-configured training pipeline for pose-to-IMU regression + activity classification.

## Current Layout (trimmed)
```
conf/                # Hydra config root (data/, model/, experiment/, optim/, trainer/)
experiments/
  run_experiment.py  # Single Lightning-based entrypoint
src/
  data.py            # Dataloader factory (by dataset name)
  lightning_module.py# HARLightningModule implementation
  models/            # Regressor / FeatureExtractor / ActivityClassifier
  datasets/          # Dataset implementations (mmfit + others)
docs/
legacy/              # (Will be removed; no longer used in training)
datasets/            # Actual data directory (subject folders, not tracked)
```

## Key Ideas
* Single command entrypoint (`experiments/run_experiment.py`)
* Configuration-first (Hydra): compose + override at CLI
* Single Lightning training backend (legacy removed)
* Multi-loss objective (MSE + alpha * classification + beta * feature-similarity)
* Automatic run directory creation per launch (Hydra) under `experiments/outputs/DATE-TIME/`

## Install
```bash
conda env create -f environment.yml
```

## Environments
the code is intended to run either on `local` machine or `cluster computing`. when you're in `local` then be sure to first activate the conda env using:
```bash
conda activate har
```

## Data
Place dataset subjects under the path configured in `conf/env/*.yaml` (e.g. `env/local.yaml` sets `data_dir`).
For MM-Fit expect a structure like:
```
<data_dir>/mm-fit/w01/...
```
If your subjects are directly under `<data_dir>/w01`, adjust or set `data.data_dir` explicitly.

## Run (Basic)
```bash
python experiments/run_experiment.py experiment=scenario2
```
Common overrides:
```bash
python experiments/run_experiment.py trainer.epochs=50 optim.lr=5e-4 \
  model.regressor.sequence_length=256 experiment.alpha=1.5
```

Short smoke test (2 epochs):
```bash
python experiments/run_experiment.py trainer.epochs=2
```

## Fast Debug Run
Use the tiny debug split to validate end-to-end behavior quickly:
```bash
python experiments/run_experiment.py data=mmfit_debug experiment=debug trainer.epochs=2
```
What this does:
* Restricts subjects to 1 per split
* Optionally limits samples per split (`debug_limit_per_split`)
* Runs only 2 epochs (override)
* Computes macro F1 (torchmetrics) at epoch end

Ultra-fast plumbing smoke (no checkpoints / early stopping):
```bash
python experiments/run_experiment.py data=mmfit_debug experiment=debug trainer.epochs=1 trainer.enable_checkpointing=false trainer.early_stopping.enabled=false
```

## Determinism / Seeding
Global seed configured in `conf/conf.yaml` under `seed:`. Override per run:
```bash
python experiments/run_experiment.py seed=42
```

## Config Anatomy
Root defaults file (`conf/conf.yaml`) defines a `defaults:` list specifying which groups load (data, model components, experiment, optim, trainer, etc.). You override any leaf via dotted syntax.

Accepted dataset name keys inside `conf/data/*.yaml`: `name`, `dataset_name`, or `data_name` (current MM-Fit file uses `data_name`).

Example overrides:
* Change dataset split file: `data=mmfit` (switch group member)
* Adjust learning rate: `optim.lr=3e-4`
* Reduce epochs: `trainer.epochs=10`

Hydra stores the fully resolved config in each run dir; we also write `results.json` containing final metrics.

## Results & Artifacts
Each run creates: `experiments/outputs/<timestamp>/results.json` plus checkpoints (if enabled in `trainer.checkpoint.enabled`).

`results.json` now contains:
* `config`: fully resolved Hydra config
* `history`: placeholder dict (future per-epoch aggregation)
* `final_metrics`:
  - `train_loss_last`: last logged train loss (if tracked in future)
  - `val_loss_last`: validation loss after final epoch
  - `val_f1_last`: validation macro F1 after final epoch
  - `best_val_f1`: best macro F1 observed across epochs
  - `best_val_loss`: best monitored validation loss (from checkpoint callback)

## Roadmap (Brief)
Planned near-term improvements (see `ROADMAP.md`): debug subset config, W&B logging, Optuna sweeps, torchmetrics F1, logging refinements.

## Weights & Biases Logging (Optional)
Enable W&B logging via Hydra overrides:
```bash
python experiments/run_experiment.py trainer.logger=wandb trainer.wandb.enabled=true trainer.wandb.project=har
```
Additional useful overrides:
```bash
# Offline mode (no network)
python experiments/run_experiment.py trainer.logger=wandb trainer.wandb.enabled=true trainer.wandb.mode=offline

# Add grouping / tags
python experiments/run_experiment.py trainer.logger=wandb trainer.wandb.enabled=true \
  trainer.wandb.group=scenario2 trainer.wandb.tags='["mmfit","alpha1.0"]'
```
If `wandb` is not installed the run will continue without a logger and print a warning.
Environment variable alternative:
```bash
export WANDB_API_KEY=...  # prior to launching
```
Logged items:
* Scalars: train/val losses, val_f1, best_val_f1, learning rate (if added later)
* Config: alpha, beta, lr, weight_decay, epochs, dataset
* Final summary metrics (also in `results.json`).

## Packaging & Imports (Why `import src` Works Here)
This repository intentionally exposes a top-level package named `src` (because `src/__init__.py` exists). Normally, "src layout" projects nest the real package (e.g., `src/har/`) and you would import `har`. We kept `src` directly for lightweight research iteration.

Historical issue encountered:
* Initial `pyproject.toml` used auto-discovery with:
  ```toml
  [tool.setuptools]
  package-dir = {"" = "src"}
  [tool.setuptools.packages.find]
  where = ["src"]
  include = ["*"]
  ```
* Setuptools searched **inside** `src/` for subpackages and did not register the root directory itself as a package named `src`.
* Result: `pip install -e .` produced metadata but `import src` failed when running scripts from outside the repo root.

Fix applied:
```toml
[tool.setuptools]
packages = ["src"]
```
This explicitly instructs setuptools to install the `src/` directory as the package `src`.

Key points / how to avoid future confusion:
1. Editable install must be run *after* activating the target conda env: `conda activate har && pip install -e .`.
2. If you change back to auto-discovery, `import src` will likely break again.
3. To migrate to a conventional layout later: move code into `src/har/`, rename in `pyproject.toml` to `packages = ["har"]`, and update imports (`from src.` → `from har.`).
4. Tests include a pytest `pythonpath = ["."]` safeguard, but with the explicit package list it’s not strictly required for runtime.

Troubleshooting checklist if `ModuleNotFoundError: No module named 'src'` reappears:
* Confirm installation: `pip show har` (or whatever `[project].name` is).
* Inspect site-packages link: `python -c "import src, inspect; print(src.__file__)"`.
* Ensure no stale wheel in a different environment (reactivate env, reinstall).

This section documents the rationale so we do not repeat the previous trial-and-error phase.

