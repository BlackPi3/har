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

## Scenario2 training paths

- Base: each batch yields `(pose, acc, labels)`; pose feeds the regressor (`pose2imu`) to produce `sim_acc`, then both `acc` (real) and `sim_acc` (simulated) pass through the feature extractor (`fe`) and classifier (`ac`). Losses combine MSE on `sim_acc` vs `acc`, optional feature-similarity on features, and activity CE on real and/or simulated logits per `trainer.losses` toggles.
- Dual feature extractors: enable `trainer.separate_feature_extractors` to route simulated IMU through `fe_sim` while real IMU stays on `fe`; similarity loss compares `fe_sim(sim_acc)` to `fe(acc)` when turned on.
- Dual classifiers: enable `trainer.separate_classifiers` so real features go to `ac` and simulated features go to `ac_sim`; activity losses can include real, simulated, or both depending on `trainer.losses.activity_*`.
- Secondary pose-only dataset: enable `trainer.secondary.enabled` to draw `pose, label` batches from `secondary_train`, run them through `pose2imu` → `fe_sim` and a dedicated classifier (default `ac_secondary`, overridable via `trainer.secondary.classifier_key`), and add that CE loss scaled by `trainer.secondary.loss_weight`.
- Branch gating: `trainer.losses.activity_real` / `activity_sim` let you train only the real branch or only the simulated branch; turning both off yields a pure regression/feature-alignment run (MSE + optional feature similarity). You can also disable MSE or feature similarity individually via `trainer.losses.mse` and `trainer.losses.feature_similarity`.
- Freezing components: `trainer.trainable_modules` can freeze `pose2imu`, `fe`, or `ac` (and any duplicated variants), effectively turning the path into evaluation-only for those modules while others keep training.

## Training scenarios under test

- `scenario2` (baseline): train regressor, feature extractor, and classifier jointly on the normal path described above.
- `scenario22`: same as scenario2 but set `trainer.gamma=0` to drop the regression/MSE term.
- `scenario23`: scenario2 with separate classifiers for real vs simulated features (`trainer.separate_classifiers=true`).
- `scenario24`: scenario2 with separate feature extractors for real vs simulated IMU streams (`trainer.separate_feature_extractors=true`) while sharing the classifier.
- `scenario3`: scenario2 plus a secondary pose-only dataset; shares regressor and feature extractor but uses a dedicated secondary classifier. Goal: test whether more pose data improves generalization.
- `scenario4`: scenario2 plus an adversarial discriminator that distinguishes real vs simulated accelerometer data to push the regressor/generator to produce better simulated IMU.

## TODO

- Normalize the remaining config sections (trainer/optim/model) so only the grouped blocks exist in `resolved_config`—currently only `data` has been fully cleaned up.
