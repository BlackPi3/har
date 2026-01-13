# Project Brief

- This is a research project on improving Human Activity Recognition (HAR) performance across multiple datasets (currently MM-Fit, UTD and NTU)
- Data lives under `datasets/`. Hydra-style configs live under `conf/` and describe datasets, trials, trainers, and HPO spaces.
- Main entrypoints:
  - Single experiment: `python -m experiments.run_trial ...`
  - Hyperparam search: `python -m experiments.run_hpo ...`
  - Evaluation on test set: `python -m experiments.run_eval ...`
  - Config knobs get passed to `run_trial`, `run_hpo`, `run_eval` scripts to control behavior (dataset selection, model/trainer settings, etc.).
- `src/` contains the backbone Python code: PyTorch datasets/dataloaders, preprocessing pipelines, training logic (`train_scenario2.py`), models (`classifier.py`, `discriminator.py`, `regressor`)
- Cluster launchers (`experiments/slurm_trial.sh`, `experiments/slurm_hpo.sh`, `experiments/slurm_eval.sh`) wrap the same entrypoints for SLURM environments.
- For local runs/tests, activate the `har` conda env first (`conda activate har`).
- There are multiple training pipelines that go under the names: `scneario2`, `scenario22`, `scenario23`, `scenario24`, `scneario25`, `scenario3`, `scenario4`, `scenairo42`. All these pipeline's have the same goal of improving macro F1 and accuracy metrics on unseen data in HAR tasks. the general idea in all these pipellines is that we train our models using real and simulated accelerometer data. the simulated accelerometer data come from pose sequence. 
- for each scenairo we have a dedicated config under `conf/trial/` and `conf/hpo/`. 
- we do our hyperparameters optimization (HPO) in 3 passes (`conf/hpo/`). each pass optimizes a subset of parameters, which are loss weights + dataset-specific, regularization and model capacity params.

## experiments architectures

- Baseline (config name scenario2): each batch yields `(pose, acc, labels)`; pose feeds the regressor (`pose2imu`) to produce `sim_acc`, then both `acc` (real) and `sim_acc` (simulated) pass through the feature extractor (`fe`) and classifier (`ac`). Losses combine MSE on `sim_acc` vs `acc`, optional feature-similarity on features, and activity CE on real and/or simulated logits per `trainer.losses` toggles.
- MSE loss ablation (config name scenario22): same as baseline, but the `trainer.losses.mse` is disabled and `trainer.gamma` weight is set to 0.
- Similarity loss ablation (config name scenario25): same as baseline, but the `trainer.losses.feature_similarity` is set disabled and `trainer.beta` is set to 0.
- Shared classifier (config name scenario24): enable `trainer.separate_feature_extractors` to route simulated IMU through `fe_sim` while real IMU stays on `fe`; similarity loss compares `fe_sim(sim_acc)` to `fe(acc)` when turned on.
- Shared representation (config name scenario23): enable `trainer.separate_classifiers` so real features go to `ac` and simulated features go to `ac_sim`; activity losses can include real, simulated, or both depending on `trainer.losses.activity_*`.
- Secondary pose-only dataset (config name scenario3): scenario2 plus a secondary pose-only dataset; shares regressor and feature extractor but uses a dedicated secondary classifier. Goal: test whether more pose data improves generalization. enable `trainer.secondary.enabled` to draw `pose, label` batches from `secondary_train`, run them through `pose2imu` → `fe_sim` and a dedicated classifier (default `ac_secondary`, overridable via `trainer.secondary.classifier_key`), and add that CE loss scaled by `trainer.secondary.loss_weight`.
- Feature discriminator (config name scenario4): baseline plus discriminator component. enable `trainer.adversarial.enabled` and set `trainer.adversarial.discriminator.input_type` to `features`. `real_feat` and `sim_feat` go through this discriminator. the discriminator distinguishes real and fake features.
- Signal discriminator (config name scenario42): baseline plus discriminator component. enable `trainer.adversarial.enabled` and set `trainer.adversarial.discriminator.input_type` to `signal`. `acc` and `sim_acc` go through this discriminator. the discriminator distinguishes real and fake signals.

## HPO Configuration Structure

Base trial configs live in `conf/trial/`:
- `scenario2_utd.yaml` - UTD dataset baseline
- `scenario2_mmfit.yaml` - MMFit dataset baseline
and so on.

HPO configs in `conf/hpo/` reference a base trial and apply scenario-specific overrides:
- `scenario2_utd.yaml`, `scenario2_mmfit.yaml` - baseline HPO
- `scenario22_utd.yaml`, `scenario22_mmfit.yaml` - no MSE (gamma=0)
- `scenario23_utd.yaml`, `scenario23_mmfit.yaml` - separate classifiers
- `scenario24_utd.yaml`, `scenario24_mmfit.yaml` - separate feature extractors
- `scenario3_utd.yaml`, `scenario3_mmfit.yaml` - secondary NTU dataset
and so on.

Each HPO config specifies:
- `trial:` which base trial config to use
- `trainer:` scenario-specific overrides (epochs, patience, flags)
- `hydra.sweeper.params:` search space

## TODO

- Normalize the remaining config sections (trainer/optim/model) so only the grouped blocks exist in `resolved_config`—currently only `data` has been fully cleaned up.
