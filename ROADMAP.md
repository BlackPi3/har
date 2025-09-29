# Project Roadmap

Tracking upcoming implementation tasks and enhancements. This list mirrors the current state after moving to a unified Hydra entrypoint with optional Lightning backend.

## 1. Modularize Lightning
- [ ] Extract inline `HARLightningModule` class from `experiments/run_experiment.py` -> `src/lightning_module.py`.
- [ ] Add docstring & type hints; expose constructor accepting `cfg` and pre-built model dict.
- [ ] Import it in the runner (cleaner diff & testability).

## 2. Config-Driven Lightning Trainer
- [ ] Extend `conf/trainer_backend/lightning.yaml` with: `max_epochs`, `accelerator`, `devices`, `precision`, `deterministic`, `gradient_clip_val`, `enable_checkpointing`, `early_stopping` block.
- [ ] Map those keys in the runner to `pl.Trainer` arguments.
- [ ] Add optional EarlyStopping & ModelCheckpoint callback configs.

## 3. Checkpointing & Metrics
- [ ] Introduce `ModelCheckpoint` (monitor: `val_loss`, save_top_k=1).
- [ ] Save path relative to Hydra run dir (e.g. `checkpoints/`).
- [ ] Add torchmetrics `MulticlassF1Score` (macro) replacing manual confusion matrix code.

## 4. Debug Dataset Config
- [ ] Create `conf/data/mmfit_debug.yaml` inheriting from base mmfit: smaller `train_subjects`, `debug_subset=true`, low limit, maybe `batch_size=8`.
- [ ] Document usage: `data=mmfit_debug trainer.epochs=2`.

## 5. W&B Integration
- [ ] Add optional `logging=wandb` config group (API key expected in env var).
- [ ] Implement `WandbLogger` only when selected; fallback to `False` otherwise.
- [ ] Log: losses, val_f1, learning rate, hyperparameters.

## 6. Hyperparameter Optimization (Optuna)
- [ ] Create `experiments/run_hpo.py` using Optuna study (objective wraps Hydra composition + Lightning training).
- [ ] Define search space via config (e.g. `hpo/space/*.yaml`).
- [ ] Persist study to `experiments/studies/<name>.db` (SQLite) + export summary JSON.

## 7. Parity & Deprecation Plan
- [ ] Run a controlled comparison (fixed seed) between legacy & lightning for a short schedule.
- [ ] Document any metric drift; once acceptable, mark legacy backend deprecated in README and optionally remove after grace period.

## 8. Reproducibility & Seeds
- [ ] Add `conf/repro.yaml` (seed, deterministic toggles, cudnn flags) & include in defaults.
- [ ] Expose `repro.deterministic=false` override for speed runs.

## 9. Mixed Precision & Performance
- [ ] Allow `precision=16` or `bf16-mixed` in lightning config (guard if unsupported by device).
- [ ] Add timing logs (epoch duration) when `experiment.profile=true`.

## 10. Code Quality / Tests
- [ ] Add unit test for LightningModule forward + a single train/val step using a synthetic batch.
- [ ] Add config composition test for every group member (parametrize over data, trainer_backend).

## 11. Metrics & Reporting
- [ ] Extend `results.json` with: `best_val_loss`, `best_epoch`, `val_f1`.
- [ ] Optionally write `final_config.yaml` for copy/paste reproduction.

## 12. Dataset Path Robustness
- [ ] Auto-detect if subjects are nested under a subfolder (e.g. `mm-fit/`) and adjust internally with a warning.
- [ ] Validate that at least one subject loads; otherwise raise a clear error early.

## 13. Documentation Updates
- [ ] Expand README with minimal example for W&B + Optuna once implemented.
- [ ] Add diagram (models + data flow) in `docs/`.

## 14. Cleanup
- [ ] Remove unused legacy config loading helpers once Lightning path stabilized.
- [ ] Prune dead code / unused imports flagged by linter.

## 15. Optional Future Work
- [ ] Domain adaptation hooks (re-enable discriminator path under config flag).
- [ ] Support additional datasets via same interface (MHAD / NTU) with dedicated config files.
- [ ] Add simple inference script producing per-activity predictions and exporting CSV.

---
Prioritize (1) Modularize Lightning, (2) Config-driven Trainer, (3) Checkpoint + torchmetrics, then proceed toward logging & HPO.
