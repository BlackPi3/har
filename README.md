# Human Activity Recognition (HAR)

Unified, Hydra-configured training pipeline for pose-to-IMU regression + activity classification.

## Current Layout (trimmed)
```
conf/                # Hydra config root (data/, model/, experiment/, optim/, trainer/)
experiments/
  run_trial.py       # Canonical single-trial entrypoint (Hydra)
  run_hpo.py         # Native Optuna orchestrator (per-trial subprocess)
  slurm_hpo.sh       # SLURM launcher for the orchestrator (scratch-friendly)
src/
  data.py            # Dataloader factory (by dataset name)
  train_scenario2.py # Manual Trainer (current production flow)
  lightning_unused.py# HARLightningModule implementation (unused placeholder)
  models/            # Regressor / FeatureExtractor / ActivityClassifier
  datasets/          # Dataset implementations (mmfit + others)
docs/
legacy/              # (Will be removed; no longer used in training)
datasets/            # Actual data directory (subject folders, not tracked)
```

## Key Ideas
* Single command entrypoint (`experiments/run_trial.py`)
* Single HPO entrypoint (`experiments/run_hpo.py`)
* Configuration-first (Hydra): compose + override at CLI
* Single manual training flow (Lightning module kept for future use)
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
python experiments/run_trial.py scenario=scenario2
```
Scenario defaults to `scenario2`; pass `scenario=<name>` to swap flows or overrides bundled hyperparameters.
Common overrides:
```bash
python experiments/run_trial.py trainer.epochs=50 optim.lr=5e-4 \
  model.regressor.sequence_length=256 scenario.alpha=1.5
```

Short smoke test (2 epochs):
```bash
python experiments/run_trial.py trainer.epochs=2
```

## Fast Debug Run
Use the tiny debug split to validate end-to-end behavior quickly:
```bash
python experiments/run_trial.py data=mmfit_debug scenario=debug trainer.epochs=2
```
What this does:
* Restricts subjects to 1 per split
* Runs only 2 epochs (override)
* Computes macro F1 (torchmetrics) at epoch end

Ultra-fast plumbing smoke (no checkpoints / early stopping):
```bash
python experiments/run_trial.py data=mmfit_debug scenario=debug trainer.epochs=1 trainer.enable_checkpointing=false trainer.early_stopping.enabled=false
```

Faster epochs without changing the dataset size (use a fraction of batches):
```bash
python experiments/run_trial.py trainer.epochs=2 trainer.limit_train_batches=0.1 trainer.limit_val_batches=0.25
```

## Determinism / Seeding
Global seed configured in `conf/conf.yaml` under `seed:`. Override per run:
```bash
python experiments/run_trial.py seed=42
```

## Config Anatomy
Root defaults file (`conf/conf.yaml`) defines a `defaults:` list specifying which groups load (data, model components, optim, trainer, etc.). Individual scenarios (e.g. `conf/scenario/scenario2.yaml`) now act as composite overlays: they gather the loss weights, optimiser settings, regressor/feature-extractor knobs, and a few data conveniences in one editable spot. When you launch with `scenario=scenario2`, those overrides are merged on top of the shared defaults.

Accepted dataset name keys inside `conf/data/*.yaml`: `name`, `dataset_name`, or `dataset_name` (current MM-Fit file uses `dataset_name`).

Example overrides:
* Switch dataset split: `data=mmfit`
* Swap scenario (and all bundled knobs): `scenario=debug`
* Quick tweak on top of the scenario: `optim.lr=3e-4` or `model.regressor.fc_hidden=128`
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
python experiments/run_trial.py trainer.logger=wandb trainer.wandb.enabled=true trainer.wandb.project=har
```
Additional useful overrides:
```bash
# Offline mode (no network)
python experiments/run_trial.py trainer.logger=wandb trainer.wandb.enabled=true trainer.wandb.mode=offline

# Add grouping / tags
python experiments/run_trial.py trainer.logger=wandb trainer.wandb.enabled=true \
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

## Hyperparameter Optimization (Optuna)
Use the HPO orchestrator to run sequential Optuna trials. The search space is defined only via YAML files under `conf/hpo/` (Hydra sweeper-style). Pass `--space-config conf/hpo/<name>.yaml` or set `HPO=<name>` to auto-pick `conf/hpo/$HPO.yaml`.

### Sweep outputs (SLURM)
When launched via `experiments/slurm_hpo.sh`, all sweep artifacts are written into a single directory under the repository:

- Root: `<project>/experiments/hpo/<study_name>/`
  - Per‑trial Hydra run dirs: `trials/trial_<N>/...`
  - Optuna SQLite database: `<study_name>.db`
  - Aggregates: `trials.csv`, `best.json`

Environment overrides:

- `HPO` (default: `scenario2_mmfit`) – used to derive the study name
- `STUDY_NAME` (defaults to `$HPO`)
- `OUTPUT_ROOT` (defaults to `$PROJECT_ROOT/experiments/hpo/$STUDY_NAME`)
- `STORAGE` (defaults to `$OUTPUT_ROOT/$STUDY_NAME.db`)

Example submit:

```bash
HPO=scenario2_mmfit N_TRIALS=50 \
OVERRIDES="env=remote data=mmfit trainer.epochs=30" \
sbatch experiments/slurm_hpo.sh
```

Quick facts:
- Search space is provided by `conf/hpo/*.yaml` (Hydra sweeper-style). The orchestrator parses `interval`, `tag(log, interval)`, `choice`, and `range` into Optuna distributions.
- Output root defaults to `experiments/hpo/<study_name>/` unless `--output-root` is provided
- Resumable: Passing the same `--study-name` and `--storage` resumes the study
- Objective: `--metric {val_f1|val_loss}` with `--direction {maximize|minimize}`
- Reproducibility: fixed `--seed` is propagated to all trials

- Search space: `conf/hpo/scenario2_mmfit.yaml`
  - Canonical window length: `data.sensor_window_length` (models inherit via interpolation)
  - Core knobs: `optim.lr` (log), `optim.weight_decay` (log), `scenario.alpha`, `scenario.beta`
  - Model FE capacity: `model.feature_extractor.n_filters`, `model.feature_extractor.filter_size`
  - Throughput: `data.batch_size` (coarse steps)

- Quick debug search (tiny split, short epochs)
  ```bash
  conda activate har
  python experiments/run_hpo.py \
    --n-trials 10 \
    --metric val_f1 --direction maximize \
    --study-name mmfit_sc2_debug \
    --space-config conf/hpo/scenario2_mmfit.yaml \
    data=mmfit_debug trainer.epochs=5
  ```

- Typical search (full split)
  ```bash
  conda activate har
  python experiments/run_hpo.py \
    --n-trials 50 \
    --metric val_f1 --direction maximize \
    --study-name mmfit_sc2 \
    --storage experiments/hpo/mmfit_sc2/mmfit_sc2.db \
    --space-config conf/hpo/scenario2_mmfit.yaml \
    data=mmfit trainer.epochs=30
  ```

- Resume a study (same name + storage)
  ```bash
  conda activate har
  python experiments/run_hpo.py \
    --n-trials 25 \
    --metric val_f1 --direction maximize \
    --study-name mmfit_sc2 \
    --storage experiments/hpo/mmfit_sc2/mmfit_sc2.db \
    --space-config conf/hpo/scenario2_mmfit.yaml \
    data=mmfit trainer.epochs=30
  # Adds 25 more trials onto the existing study
  ```

- With W&B logging (optional; forwarded to each trial)
  ```bash
  conda activate har
  python experiments/run_hpo.py \
    --n-trials 10 \
    --metric val_f1 --direction maximize \
    --study-name mmfit_sc2_wandb \
    --space-config conf/hpo/scenario2_mmfit.yaml \
    data=mmfit_debug trainer.epochs=5 \
    trainer.logger=wandb trainer.wandb.enabled=true \
    trainer.wandb.project=har trainer.wandb.group=optuna-sc2
  ```

- Outputs and artifacts
  - Per‑trial run dirs: `<output_root>/trials/trial_XXX/` (default repo path: `experiments/hpo/<study_name>/trials/trial_XXX/`) with a full Hydra config and `results.json`
  - Study summary: `<output_root>/best.json` and `<output_root>/trials.csv`
  - CLI prints best metric and params on completion

- Fairness guidelines (recommended)
  - Keep the HPO budget constant across models: `--n-trials`, `trainer.epochs`, early‑stopping policy
  - Do not tune `num_workers` in HPO; set via env config: `conf/env/local.yaml`, `conf/env/remote.yaml`
  - Use subject‑independent splits and never include test data during HPO

- Re‑run best config for testing
  1) Read `<output_root>/best.json` (default: `experiments/hpo/<study_name>/best.json`) and copy the `best_params` as Hydra overrides.
  2) Train and evaluate with multiple seeds:
  ```bash
  # Example using printed best params (replace with your values)
  python experiments/run_trial.py \
    scenario=scenario2 data=mmfit \
    optim.lr=3e-4 optim.weight_decay=1e-4 \
    data.sensor_window_length=256 \
    model.feature_extractor.n_filters=16 model.feature_extractor.filter_size=5 \
    scenario.alpha=1.2 scenario.beta=0.3 \
    trainer.epochs=50 seed=0
  # Repeat with seed=1..4 and report mean ± std
  ```

### Running on SLURM (cluster)
If your cluster provides a container or preinstalled PyTorch image and no conda, use the SLURM script:

```bash
# Edit SBATCH resources inside if needed; override via env vars as shown
HPO=mmfit_sc2 N_TRIALS=50 \
OUTPUT_ROOT=$PROJECT_ROOT/experiments/hpo/mmfit_sc2 \
STORAGE=$OUTPUT_ROOT/mmfit_sc2.db \
OVERRIDES="env=remote data=mmfit trainer.epochs=30" \
sbatch experiments/slurm_hpo.sh
```

Notes:
- No conda required. The script installs the project in the container with `pip install -e .` if needed.
- Set `CONTAINER_IMAGE` to use a Slurm containerized job: `CONTAINER_IMAGE=docker://pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` (or your cluster image path).
- By default outputs and the Optuna DB are placed under `experiments/hpo/<study>/`. Adjust `OUTPUT_ROOT`/`STORAGE` as needed.

To run a single long-form training job with the tuned hyperparameters, use the companion script (defaults shown):

```bash
BEST_OVERRIDES="optim.lr=0.0026421464 optim.weight_decay=0.00018897 scenario.alpha=10 scenario.beta=10 data.sensor_window_length=400 data.stride_seconds=0.1 model.feature_extractor.n_filters=24 model.feature_extractor.filter_size=7" \
RUN_LABEL=apex \
EPOCHS=200 SEED=0 \
sbatch experiments/slurm_best.sh
```

Set `BEST_OVERRIDES` to the tuned hyperparameters you want to run. The script passes those overrides to `experiments.run_trial` and stores artifacts under `experiments/best_run/<scenario>/<dataset>/<timestamp>/` (plots, metrics CSV/JSON, resolved config, checkpoints). Override `ENV_NAME`, `DATA_NAME`, `SCENARIO_NAME`, `EPOCHS`, `SEED`, or `RUN_DIR` as needed; `RUN_LABEL` just tags the log filenames.

### Single entrypoints and legacy scripts
- Preferred entrypoints:
  - Training: `experiments/run_trial.py` (module form: `python -m experiments.run_trial ...`)
  - HPO: `experiments/run_hpo.py`
- Legacy compatibility: `experiments/run.py` still works for single runs but `run_trial.py` is the canonical entrypoint going forward.
- The older sweeper-based SLURM script is deprecated. Use `experiments/slurm_hpo.sh` instead. A shim `slurm_sweep.sh` may exist but simply forwards to the new script.

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
* Ensure no stale wheel in a different environment (reactivate env, reinstall). If needed, run as a module: `python -m experiments.run_trial ...`.

This section documents the rationale so we do not repeat the previous trial-and-error phase.

# Experiments with Hydra + PyTorch Lightning

This repository runs experiments via a single Hydra entrypoint that builds dataloaders and Lightning modules, handles logging/checkpointing, and writes results.json per run.

## Installation

- Python 3.9+
- PyTorch + PyTorch Lightning
- Hydra and OmegaConf

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .            # required so `src/` is importable
```

If you run without editable install and see `ModuleNotFoundError: No module named 'src'`, either install with `pip install -e .` or invoke as a module: `python -m experiments.run_trial ...`.

## Data

- Configure the base data directory via `env.data_dir` in Hydra config (conf/).
- The dataset path used by the runner is: `<env.data_dir>/<data.dataset_name>`.
- If the directory doesn’t exist, the script will warn and proceed.

## Running an experiment

Two equivalent ways:

```bash
# Module form (works without editable install)
python -m experiments.run_trial data=mmfit_debug trainer.epochs=2

# Script form (requires pip install -e . so `src/` is importable)
python experiments/run_trial.py scenario=scenario2 trainer.epochs=5 optim.lr=5e-4
```

Common overrides:

- Device selection: `device=auto|cuda|mps|cpu` (auto prefers CUDA, then macOS MPS, else CPU).
- Seed: `seed=42` (used for all RNGs via src.config.set_seed).
- Data: `data.dataset_name=mmfit` and optionally `env.data_dir=/absolute/path`.
- Training: `trainer.epochs=100 trainer.patience=10`
- Optimizer: `optim.lr=1e-3 optim.weight_decay=1e-4`
- Scenario knobs: `scenario.alpha=... scenario.beta=...`
- Window length used by data factory: `data.sensor_window_length=256`

Hydra run directory: each run executes in its own working directory (printed at start) and writes outputs there.

## Models expected by config

The runner builds three components from `cfg.model`:

- `model.regressor`
  - `input_channels, num_joints, sequence_length`
- `model.feature_extractor`
  - `n_filters, filter_size, n_dense, input_channels, window_size, drop_prob, pool_size`
- `model.classifier`
  - `f_in, n_classes`

These are used to construct:
- `src.models.Regressor`, `FeatureExtractor`, `ActivityClassifier`

## Logging and checkpointing

Configure via `trainer` in Hydra config:

- Early stopping (optional):
  - `trainer.early_stopping.enabled=true`
  - `trainer.early_stopping.monitor=val_loss` (or your metric)
  - `trainer.early_stopping.mode=min|max`
  - `trainer.early_stopping.patience=10`
- Checkpoints (optional):
  - `trainer.checkpoint.enabled=true`
  - `trainer.checkpoint.monitor=val_loss`
  - `trainer.checkpoint.mode=min|max`
  - `trainer.checkpoint.save_top_k=1`
  - `trainer.checkpoint.dirpath=checkpoints`
  - `trainer.checkpoint.filename=epoch{epoch}-val{val_loss:.4f}`

Weights & Biases (optional):
- Enable either with `trainer.logger=wandb` or `trainer.wandb.enabled=true`
- Optional fields: `trainer.wandb.{project,entity,group,name,tags,mode}`
  - Set `trainer.wandb.mode=offline` to avoid network use.

## Outputs

At the end of training, the runner writes `results.json` in the Hydra run directory with:

```json
{
  "config": { ...resolved config... },
  "history": {
    "train_loss": [],
    "val_loss": [],
    "train_f1": [],
    "val_f1": []
  },
  "final_metrics": {
    "train_loss_last": null,
    "val_loss_last": <float or null>,
    "val_f1_last": <float or null>,
    "best_val_f1": <float or null>,
    "best_val_loss": <float or null>
  }
}
```

Notes:
- `best_val_loss` is taken from the ModelCheckpoint callback if enabled and monitoring that metric.
- `best_val_f1` is tracked by the LightningModule if exposed as `best_val_f1`.

## Device notes

- `device=auto` picks CUDA if available, else macOS MPS, else CPU.
- On Apple Silicon, ensure a recent PyTorch build for MPS support.

## Troubleshooting

- `ModuleNotFoundError: No module named 'src'`
  - Run `pip install -e .` in your environment or use `python -m experiments.run_trial ...`.
- No data found
  - Ensure `env.data_dir` and `data.dataset_name` point to an existing folder.
- W&B not installed
  - Either install `pip install wandb` or disable the logger (`trainer.logger=null` or `trainer.wandb.enabled=false`).
