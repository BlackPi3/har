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

## Roadmap (Brief)
Planned near-term improvements (see `ROADMAP.md`): debug subset config, W&B logging, Optuna sweeps, torchmetrics F1, logging refinements.

