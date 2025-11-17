# Research Protocol for Cross-Dataset HAR (MMFiT ↔ UTD)

This document specifies the experimental design, data alignment policy, evaluation protocol, and reproducibility checklist for studying transfer between MMFiT (continuous streams) and UTD (isolated clips).

## 1) Problem framing

- Objective: Learn activity representations using pose→IMU regression + activity classification, and study transfer from MMFiT to UTD and vice versa.
- Mismatch: MMFiT is continuous (includes idle); UTD is segmented action-only clips with variable durations.

## 2) Data alignment policy

- Fixed window length W and stride S for both datasets to standardize model input.
- Window generation:
  - MMFiT: sliding windows over the continuous stream; label by center-frame (or majority) with tolerance.
  - UTD: slide within each clip; for short clips (T < W), emit one padded window and a boolean mask; for long clips, use same stride S.
- Padding and masks:
  - Use mask-aware batching (`src/datasets/common/collate.py::mask_collate`). Batches return `(pose, acc, label, mask)`; models may optionally use the mask for temporal pooling.
- Normalization:
  - Per-dataset channel-wise normalization (mean/std) computed on the respective training split to reduce domain-induced scale shifts.

## 3) Training regimens

- Pretrain→Finetune:
  1) Pretrain on MMFiT (idle included or filtered per ablation) with fixed W,S.
  2) Finetune on UTD; drop/ignore idle class.
  3) Evaluate on UTD held-out subjects.
- Joint training (optional):
  - Mix datasets with a dataset-aware sampler to balance minibatches and prevent MMFiT dominance.
- Domain regularization (optional):
  - Feature normalization layers, temporal jittering, light Gaussian noise.
  - Ablation: domain-adversarial loss or CORAL-style alignment.

## 4) Evaluation protocols

- Splits:
  - MMFiT: use existing subject splits in `conf/data/mmfit.yaml`.
- UTD: use Leave-One-Subject-Out (LOSO) or pre-defined splits; report macro-F1 and per-class F1.
- Metrics:
  - Macro-F1 (primary), Accuracy, Confusion Matrix; (optional) Calibration (ECE) and AUROC per class.
- Statistical testing:
  - Report mean±std over 5 seeds; paired t-test or Wilcoxon signed-rank across seeds for key comparisons.

## 5) Ablations (minimum set)

- Window length W ∈ {64, 128, 256, 300}; stride S ∈ {W/4, W/2}.
2) UTD short-clip handling: pad+mask vs. center-crop vs. simple upsample.
3) Idle handling: include vs. exclude during MMFiT pretraining.
4) Normalization: per-dataset vs. global.

## 6) Reproducibility checklist

- Seeds: set Python/NumPy/Torch seeds; fix cuDNN determinism where applicable.
- Configs: all knobs controlled via YAML in `conf/`; save resolved config per run (Hydra already writes `.hydra`).
- Environment: freeze with `environment.yml`/`requirements.txt` and record `torch`, `numpy` versions.
- Logging: persist metrics per epoch; save best checkpoint keyed by `val_loss` and track `val_f1`.
- Artifacts: store confusion matrices and per-class F1 as JSON for later aggregation.

## 7) Implementation notes

- Use `src/datasets/common/collate.py::mask_collate` when batching UTD windows or any variable-length data.
- For MMFiT, current dataset returns fixed-length windows; behavior remains unchanged.
- Consider adding an explicit `stride` to data configs and mapping indices accordingly for both datasets.

## 8) Reporting

- Provide a single summary table with macro-F1 (mean±std across seeds) for each setting.
- Include learning curves and per-class F1 bar plots.

## 9) Pitfalls to avoid

- Data leakage via overlapping windows across train/val/test splits (respect subject boundaries).
- Distribution shift from idle over-representation—ensure class/dataset-balanced sampling.
- Inconsistent window policies between training and evaluation.
