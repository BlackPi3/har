# Project Roadmap (Updated)

Status legend: ✅ Done | 🟡 Partial / In Progress | 🔜 Planned | ⏭️ Deferred / Reconsider

## 1. Lightning Modularization ✅
`HARLightningModule` extracted to `src/lightning_module.py` with constructor using `cfg`, `ns`, and model dict. Runner imports it.

## 2. Config-Driven Trainer ✅
`conf/trainer/default.yaml` supplies epochs, accelerator, devices, precision, deterministic flag, gradient clipping, checkpoint & early stopping blocks. Runner maps these to `pl.Trainer`.

## 3. Checkpointing & Core Metrics ✅
ModelCheckpoint + EarlyStopping integrated. Macro F1 via `torchmetrics.MulticlassF1Score` (streamlined batch-wise). Best F1 tracked.

## 4. Debug Dataset Config ✅
`conf/data/mmfit_debug.yaml` with reduced subjects, sample caps, smaller batch size. Documented in README.

## 5. W&B Integration 🟡
Optional via trainer overrides. Logging scalars + final metrics works. Missing (future): learning rate logs, artifact upload, automatic git hash.

## 6. Hyperparameter Optimization (Optuna) ✅
Implemented unified Optuna orchestrator `experiments/run_optuna.py` with YAML search space (`conf/hpo/scenario2_mmfit.yaml`). Trials call the single-run entrypoint, write `results.json`, and the study exports `best.json` + `trials.csv`. README includes quickstart and resume instructions. Future enhancements (optional): pruning/samplers selection, multi-objective, and distributed sweeps.

## 7. Legacy Backend Parity ⏭️ (Deprecated Strategy)
Legacy code retained under `legacy/` but parity comparison intentionally skipped—Lightning accepted as canonical.

## 8. Reproducibility & Seeds 🟡
Global `seed` in root config; deterministic flag present. No separate `repro.yaml`; cudnn benchmark/flags not explicitly exposed yet.

## 9. Mixed Precision & Performance 🔜
Config supports `precision` key; no bf16/amp benchmarking or timing hooks yet.

## 10. Code Quality / Tests 🟡
Added one sanity test (`tests/test_sanity.py`). Need: per-component shape tests, config composition test, reproducibility test.

## 11. Metrics & Reporting 🟡
`results.json` now has last + best val metrics (loss/F1). Missing: per-epoch history population, best epoch index, optional `final_config.yaml` dump.

## 12. Dataset Path Robustness 🔜
No auto-detection or validation yet (e.g. missing subjects, overlap). Planned: early sanity checks + warnings.

## 13. Documentation 🟡
README updated (debug run, W&B, packaging rationale, Optuna/HPO quickstart). Still pending: architecture / data-flow diagram.

## 14. Cleanup 🔜
Legacy folder still present; unused helpers not pruned. The `experiments/scenario2/` directory is legacy (pre-unified HPO) and can be removed or migrated (keep analysis utilities if needed). Plan: remove after confirming no regressions rely on legacy assets.

## 15. Optional / Future Work 🔜
- Domain adaptation / discriminator reintroduction under a flag
- Additional datasets (MHAD / NTU) aligned to current interface
- Simple inference/export script (CSV predictions)
- Model artifact registry (W&B or local)
// Results aggregation CLI moved to High-Priority Next Sprint

---
## High-Priority Next Sprint (Proposed)
1. Per-epoch history capture (loss, F1, lr) + include in `results.json`.
2. Learning rate logging + W&B integration.
3. Results aggregation script -> CSV summary of runs.
4. W&B artifact upload for best checkpoint (if logger active).
5. Dataset validation (no split leakage, subject existence) with clear errors.

## Medium-Term
- Optuna enhancements (pruning strategies, sampler comparison, multi-objective, distributed trials).
- Mixed precision benchmarking (fp16 / bf16 where supported).
- Additional unit tests (config composition + reproducibility).

## Deferred / Revisit Later
- Full legacy parity removal (when confident no fallback needed).
- Discriminator / advanced adaptation features.

---
Last updated: 2025-10-05 (update this timestamp when editing roadmap).
