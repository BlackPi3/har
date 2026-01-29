# Thesis Plan: Pose-to-IMU Simulation for Human Activity Recognition

## Overview

This thesis investigates using simulated accelerometer signals derived from skeleton pose data to augment real IMU data for Human Activity Recognition (HAR). Multiple training scenarios are explored to find optimal strategies for leveraging synthetic data.

---

## Chapter 3: Methodology ✓ DRAFT COMPLETE

> **Status**: First draft written in `thesis/3-method.tex`
> **Figures**: TODO placeholders inserted; need to create with SciSpace

### 3.1 Problem Formulation ✓
- Task definition: pose-to-IMU regression + activity classification
- Formal notation: $\mathcal{D} = \{(\mathbf{p}_i, \mathbf{a}_i, y_i)\}$
- Joint training objective motivation
- Evaluation protocol (real-only at inference)

### 3.2 System Architecture ✓
- Two-stream design: real path and simulated path
- Three components: Regressor $R$, Feature Extractor $F$, Classifier $C$
- Component interactions and gradient flow
- Weight sharing in baseline configuration

### 3.3 Loss Function Design ✓
- **Classification loss (α)**: CE on both real and sim paths (same coefficient)
- **Feature similarity loss (β)**: Cosine similarity between z_real and z_sim
- **Regression loss (γ)**: MSE between sim_acc and real_acc signals
- Total objective: `L_total = α·L_cls + β·L_similarity + γ·L_regression`

### 3.4 Training Scenarios ✓
| Thesis Name | Code | Description |
|-------------|------|-------------|
| Scenario 2.1 (Baseline) | scenario2 | Shared F and C; all losses enabled |
| Scenario 2.2 | scenario22 | γ=0 (no regression loss) |
| Scenario 2.3 | scenario25 | β=0 (no feature similarity) |
| Scenario 2.4 | scenario23 | Separate C and C_sim |
| Scenario 2.5 | scenario24 | Separate F and F_sim |
| Scenario 3 | scenario3 | Secondary NTU dataset |
| Scenario 4.1 | scenario4 | Adversarial on features with GRL |
| Scenario 4.2 | scenario42 | Adversarial on raw accelerometer |

### 3.5 Adversarial Learning (Feature & Signal Discriminators) ✓
- Gradient Reversal Layer (GRL) for end-to-end training
- Feature-level: D operates on z_real vs z_sim (affects F)
- Signal-level: D operates on a vs ã (affects R)
- Adversarial loss weighted by λ_adv

### 3.6 Auxiliary Pose Data ✓
- NTU RGB+D as secondary pose-only source
- Dedicated classifier C_aux for secondary data
- Loss weighted by λ_aux

---

## Chapter 4: Implementation ✓ DRAFT COMPLETE

> **Status**: Written in `thesis/4-implementation.tex`
> **Structure**: Uses tables for hyperparameters, algorithms for HPO/eval pipelines

### 4.1 Model Architectures ✓
- **Feature Extractor (FE)**: 1D CNN backbone (Table: hyperparameters)
- **Activity Classifier (AC)**: MLP head (Table: hyperparameters)
- **Pose-to-IMU Regressor**: TCN-based (Table: hyperparameters)
- **Discriminators**: Feature-level MLP + Signal-level CNN (Tables: hyperparameters)

### 4.2 Training Pipeline ✓
- Two computational paths (real and simulated)
- Learning rate schedule: warmup + plateau reduction
- Early stopping on validation F1

### 4.3 Hyperparameter Optimization ✓
- **Framework**: Optuna with TPE sampler
- **3-Pass Strategy**:
  1. Pass 1: Loss weights (α, β, γ) + data params
  2. Pass 2: Regularization (lr, weight_decay, dropout)
  3. Pass 3: Capacity (hidden_units, embedding_dim)
- **Top-K Validation**: Top-10 configs × 5 seeds per pass
- **Final Evaluation**: Best config × 10 seeds with full epochs
- Algorithms included for duplicate handling, top-K repeats, final eval

### 4.4 Reproducibility ✓
- Seed control, config snapshots, checkpoint management

---

## Chapter 5: Experimental Setup ✓ DRAFT COMPLETE

> **Status**: Written in `thesis/5-experimental-setup.tex`
> **Structure**: Compact paragraphs (no deep subsection nesting)

### 5.1 Datasets (paragraphs) ✓
- **UTD-MHAD**: 8 subjects, 21 actions, 50 Hz, pose + IMU
- **MM-Fit**: 21 subjects, 11 exercises, 100 Hz, pose + IMU
- **NTU RGB+D**: 40 subjects, 60 actions, pose only (secondary)
- Summary table included

### 5.2 Preprocessing (paragraphs) ✓
- Skeleton normalization (hip-centered, scale-invariant)
- Temporal alignment (interpolation)
- Accelerometer standardization
- Sliding window segmentation
- Joint selection (3 arm joints)

### 5.3 Evaluation Protocol (paragraphs) ✓
- **Metric**: Macro F1 only (handles class imbalance)
- **HPO Selection**: 3 passes, top-10 × 5 seeds each
- **Final Evaluation**: Best config × 10 seeds, full epochs

> **Note**: "Experimental Scenarios" section removed — redundant with Chapter 3. Scenarios defined in methodology, results shown in Chapter 6.

---

## Chapter 6: Results & Analysis ✓ DRAFT COMPLETE

> **Status**: Draft in `thesis/6-results.tex`
> **Purpose**: Present experimental findings ("what happened and why")
> **Structure**: Paragraphs with inline analysis, no deep subsection nesting

- Main results table (F1 for all scenarios × datasets)
- Loss ablation analysis: γ=0 catastrophic failure, β=0 modest degradation
- Architecture variant analysis: separate classifiers (2.4), separate FEs (2.5)
- Auxiliary data analysis (Scenario 3): per-class F1, domain overlap tables (UTD-NTU, MMFit-NTU)
- Adversarial training analysis (Scenarios 4.1, 4.2): discriminator dynamics figures
- HPO insights: optimal configurations table
- Figures: waveform comparisons, classifier confidence, discriminator accuracy, per-class F1

---

## Chapter 7: Discussion ✓ DRAFT COMPLETE

> **Status**: Written in `thesis/7-discussion.tex`
> **Purpose**: Broader interpretation ("what it means + what's next")
> **Structure**: `\paragraph{}` style, no deep subsection nesting

### Findings (paragraphs, no subsections)
- **Dataset Size**: UTD benefits from variants, MM-Fit degrades; HPO with regularization/capacity tuning could not recover MM-Fit performance
- **Domain Gap**: Signal-level adversarial works (UTD), feature-level doesn't; similarity loss may already handle feature alignment; adversarial actively degrades MM-Fit (competing gradients)
- **Regression Loss**: Necessary anchor; without it regressor exploits unconstrained freedom; shared-weight architecture means bad sim data actively corrupts real representations
- **Auxiliary Data**: Domain compatibility > quantity; NTU clip structure more similar to UTD than MM-Fit

### Limitations and Future Work (paragraphs, no subsections)
- **Baseline Performance**: Slightly below prior work (zolfaghari2024sensor), unresolved
- **Generalizability**: Only two primary datasets tested
- **Adversarial Variants**: GRL vs WGAN-GP confounded with feature-level vs signal-level; need cross-combinations
- **Auxiliary Data**: NTU format confounds overlap analysis; need MM-Fit-compatible secondary dataset; Scenarios 3+4 untested

---

## Chapter 8: Conclusion ✓ DRAFT COMPLETE

> **Status**: Written in `thesis/8-conclusion.tex`
> **Structure**: Four flowing paragraphs, no sections/subsections. Focus on adversarial training as main goal.

- Summary: framework, eight scenarios, three-pass HPO, two datasets
- Adversarial results: signal-level WGAN-GP improved UTD-MHAD (+3.3%), feature-level GRL degraded both; improvement did not transfer to MM-Fit
- Dataset size divergence: variants help data-scarce UTD, hurt data-sufficient MM-Fit; regression loss essential; auxiliary data conditional on domain compatibility
- Research questions answered inline (no Q&A format): simulation helps conditionally, no single best scenario, adversarial helps on small dataset only

---

## Appendices

### A. Hyperparameter Configurations
- Full HPO search spaces for all scenarios
- Best configs found per scenario × dataset

### B. Additional Results
- Per-class F1 scores
- Learning curves

### C. Implementation Details
- Code repository structure
- Reproduction instructions
- Computational resources used

---

## Checklist Before Submission

- [ ] All scenarios have completed HPO (3 passes)
- [ ] Final eval runs with 10 seeds per best config
- [ ] Figures: waveform comparisons, discriminator dynamics, confidence plots
- [ ] Tables formatted consistently (Scenario 2.1, 2.2, etc. naming)
- [ ] Code cleaned and documented for release
