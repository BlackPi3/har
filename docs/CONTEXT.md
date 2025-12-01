# Project Brief

- This is a research project on improving Human Activity Recognition (HAR) performance across multiple datasets (currently MM-Fit and UTD, with room to add more).
- Data lives under `datasets/`. Hydra-style configs live under `conf/` and describe datasets, trials, trainers, and HPO spaces.
- Main entrypoints:
  - Single experiment: `python -m experiments.run_trial ...`
  - Hyperparam search: `python -m experiments.run_hpo ...`
  - Config knobs get passed to `run_trial` and `run_hpo` scripts to control behavior (dataset selection, model/trainer settings, etc.).
- `src/` contains the backbone Python code: PyTorch datasets/dataloaders, preprocessing pipelines, and training logic (`train_scenario2.py`).
- Cluster launchers (`experiments/slurm_trial.sh`, `experiments/slurm_hpo.sh`) wrap the same entrypoints for SLURM environments.
- For local runs/tests, activate the `har` conda env first (`conda activate har`).

## TODO

- Normalize the remaining config sections (trainer/optim/model) so only the grouped blocks exist in `resolved_config`—currently only `data` has been fully cleaned up.
