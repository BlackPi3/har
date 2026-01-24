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
| Baseline | scenario2 | Shared F and C; all losses enabled |
| Ablation: Effect of MSE Loss | scenario22 | γ=0 (no regression loss) |
| Ablation: Effect of Similarity Loss | scenario25 | β=0 (no feature similarity) |
| Shared Representation, Separate Classifiers | scenario23 | Separate C and C_sim |
| Separate Representations, Shared Classifier | scenario24 | Separate F and F_sim |
| Auxiliary Pose Data | scenario3 | Secondary NTU dataset |
| Feature-level Discriminator | scenario4 | Adversarial on features with GRL |
| Signal-level Discriminator | scenario42 | Adversarial on raw accelerometer |

> Note (naming): in the thesis text, these are referred to as Scenario 2.1--2.5 for narrative order: `2.1→scenario2`, `2.2→scenario22`, `2.3→scenario25`, `2.4→scenario23`, `2.5→scenario24` (and `4.1→scenario4`, `4.2→scenario42`).

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

## Chapter 4: Implementation

### 4.1 Model Architectures
- **Feature Extractor (FE)**: Convolutional backbone
  - Configurable: kernel_size, base_filters, embedding_dim
  - Dropout for regularization
- **Activity Classifier (AC)**: MLP with hidden layers
  - Input: FE embeddings
  - Output: class logits
- **Pose-to-IMU Regressor**: TCN-based architecture
  - Joint-wise processing blocks
  - Temporal aggregation block
  - FC projection to accelerometer dimensions
- **Discriminator**: Simple MLP (Scenario 4)
  - Input: FE embeddings
  - Output: real/fake probability
  - GRL applied before discriminator

### 4.2 Training Pipeline
- Hydra-based configuration management
- Trial configs: `conf/trial/scenario*.yaml`
- Trainer: `src/train_scenario2.py` (unified for all scenarios)
- Config knobs for enabling scenario-specific features

### 4.3 Hyperparameter Optimization
- **Framework**: Optuna with TPE sampler
- **Storage**: SQLite for persistence and resume
- **3-Pass Strategy**:
  1. Pass 1: Loss weights (α, β, γ) + data params (window, stride, batch)
  2. Pass 2: Regularization (lr, weight_decay, dropout, label_smoothing)
  3. Pass 3: Capacity (hidden_units, embedding_dim, kernel sizes)
- **Rationale**: Hierarchical search reduces combinatorial explosion
- **Duplicate handling**: Re-run with different seeds, report running mean
- **Top-K repeats**: Best configs validated with multiple seeds

### 4.4 Reproducibility
- Seed control for data splits, initialization, sampling
- Config snapshots saved with each HPO study
- Checkpoint saving and evaluation pipeline

---

## Chapter 5: Experimental Setup

### 5.1 Datasets
- **UTD-MHAD**: 8 subjects, 27 actions, inertial + skeleton
  - 50 Hz sampling rate
  - Leave-subjects-out splits
- **MM-Fit**: 10 subjects, 11 exercises, wearable + video
  - 100 Hz sampling rate
  - Workout-based splits
- **NTU-RGB+D** (secondary): 40 subjects, 60 actions, skeleton only
  - 50/100 Hz versions for matching primary datasets

### 5.2 Preprocessing
- Skeleton normalization (hip-centered, scale-invariant)
- Accelerometer standardization (per-dataset statistics)
- Sliding window segmentation (configurable window/stride)
- Joint selection for accelerometer simulation

### 5.3 Evaluation Protocol
- **Validation**: Used for early stopping and HPO
- **Test**: Held-out subjects, evaluated after HPO completion
- **Metrics**: 
  - Macro F1 (primary metric, handles class imbalance)
  - Accuracy, Precision, Recall
- **Reporting**: Mean ± std over N seeds (N=10 for final eval)

### 5.4 Baselines
- Real-only: Train classifier only on real_acc (no simulation)
- Pose-only: Train on sim_acc without real data
- Simple augmentation: Noise injection, time warping

---

## Chapter 6: Results & Analysis

### 6.1 Main Results
- Table: Test F1 for all scenarios × datasets
- Statistical comparison (paired t-test or Wilcoxon)
- Best scenario per dataset

### 6.2 Ablation Studies
- **Loss weight ablations**:
  - α=0 (no real classification) vs α>0
  - β=0 (no consistency) vs β>0
  - γ=0 (no MSE) vs γ>0 — captured by scenario 22
- **Architecture ablations**:
  - Shared vs separate FE (scenario 2 vs 24)
  - Shared vs separate classifier (scenario 2 vs 23)
- **Data ablations**:
  - Effect of window size
  - Effect of stride (data augmentation via overlap)
  - Effect of batch size

### 6.3 Scenario 3 Analysis (Secondary Data)
- Does more pose diversity help?
- Effect of `loss_weight` on primary vs secondary performance
- Does NTU hurt or help when primary dataset is small?

### 6.4 Scenario 4 Analysis (Adversarial)
- **Discriminator dynamics**:
  - Plot D accuracy over training epochs
  - Expected: starts high (easy to distinguish), drops as FE improves
  - Healthy training: D accuracy stabilizes around 50% (fooled)
- **Feature visualization**:
  - t-SNE of real vs sim features before/after adversarial training
  - Should show: domain gap closes with adversarial loss
- **Ablations**:
  - `adversarial.weight` sensitivity: 0, 0.1, 0.2, 0.5
  - GRL lambda: 0.5 vs 1.0
  - With vs without β (does adversarial replace consistency loss?)
- **Failure modes**:
  - Mode collapse: generator produces constant features
  - Discriminator domination: D wins, FE can't learn
  - How to detect and mitigate

### 6.5 HPO Insights
- Which hyperparameters matter most? (Optuna importance)
- Correlation analysis: does small batch help? why?
- Optimal loss weight ratios across datasets

### 6.6 Confusion Matrices
- Per-scenario confusion matrices on test set
- Which actions are confused? Why?
- Does simulation help for specific action types?

### 6.7 Qualitative Analysis
- Visualize sim_acc vs real_acc waveforms for sample actions
- Where does regression succeed/fail?
- Correlation with action dynamics (fast vs slow movements)

---

## Chapter 7: Discussion

### 7.1 Key Findings
- Summarize which scenario works best and why
- Role of each loss component
- When does simulation help vs hurt?

### 7.2 Adversarial Learning Insights
- Competition dynamics: G vs D balance
- Why feature-level discrimination over signal-level?
- Connection to domain adaptation literature (DANN, ADDA)
- Alternative losses: Wasserstein, hinge, least-squares GAN

### 7.3 Limitations
- Dataset-specific tuning required
- Computational cost of HPO
- Skeleton quality dependence
- Generalization to unseen sensor placements

### 7.4 Practical Recommendations
- Guidelines for practitioners:
  - When to use simulation-augmented training
  - Which scenario to start with
  - HPO budget recommendations

---

## Chapter 8: Conclusion

### 8.1 Summary of Contributions
1. Systematic comparison of training scenarios for pose-to-IMU simulation
2. Adversarial domain adaptation for bridging sim-to-real gap
3. Multi-pass HPO strategy for efficient hyperparameter search
4. Open-source implementation with reproducible configs

### 8.2 Answers to Research Questions
- RQ1: Does simulated accelerometer data improve HAR? → Yes/No/Depends
- RQ2: Which training scenario is most effective? → Scenario X
- RQ3: Does adversarial learning help close the domain gap? → Analysis

### 8.3 Future Work
- Extension to other sensor modalities (gyroscope, magnetometer)
- Cross-dataset generalization (train on UTD, test on MM-Fit)
- Real-time deployment considerations
- Integration with video-based HAR systems

---

## Appendices

### A. Hyperparameter Configurations
- Full HPO search spaces for all scenarios
- Best configs found per scenario × dataset

### B. Additional Results
- Extended tables with all metrics
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
- [ ] Statistical significance tests computed
- [ ] t-SNE visualizations generated (especially for Scenario 4)
- [ ] Discriminator accuracy curves plotted
- [ ] Confusion matrices generated
- [ ] sim_acc vs real_acc waveform plots
- [ ] Optuna importance analysis extracted
- [ ] All figures have proper captions and are referenced
- [ ] Tables formatted consistently
- [ ] Code cleaned and documented for release
