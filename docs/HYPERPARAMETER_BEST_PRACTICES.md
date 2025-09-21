# Hyperparameter Management Best Practices

## 🎯 Storage Strategy (Recommended)

### Hierarchical Configuration
```
configs/
├── base.yaml                    # Safe defaults + cluster_overrides section
├── scenario2.yaml               # Scenario-specific overrides (shared local+cluster)
└── hyperparameter_search/       # Systematic search configs
    ├── coarse_search.yaml       # Wide parameter ranges
    ├── fine_search.yaml         # Narrow ranges around best
    └── bayesian_config.yaml     # Advanced optimization
```

### Configuration Inheritance
1. **Base** → Safe, literature-based defaults
2. **Experiment** → Override for specific scenarios
3. **Automatic environment overrides** → Injected from `base.yaml.cluster_overrides` if SLURM detected
4. **Search** → Systematic exploration ranges

## 📊 Hyperparameter Selection Process

### Phase 1: Literature Baseline
```yaml
# Based on paper/domain knowledge
lr: 1e-3
batch_size: 64
optimizer: Adam
weight_decay: 1e-4
```

### Phase 2: Coarse Grid Search
```bash
# Test order-of-magnitude ranges
LR_VALUES=(1e-4 1e-3 1e-2)
BATCH_SIZES=(32 64 128)
REGULARIZATION=(0.0 1e-4 1e-3)
```

### Phase 3: Fine-Tuning
```bash
# Narrow around best from coarse search
LR_VALUES=(5e-4 7e-4 1e-3 1.5e-3)  # If 1e-3 was best
```

### Phase 4: Advanced Methods
- **Random Search**: Often outperforms grid search
- **Bayesian Optimization**: Efficient for expensive experiments
- **Population-based Training**: For dynamic schedules

## 🔬 Search Strategies

### 1. Grid Search (Systematic)
```python
# Good for few parameters, interpretable
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
batch_sizes = [32, 64, 128]
# Total: 4 × 3 = 12 experiments
```

### 2. Random Search (Efficient)
```python
# Better for many parameters
import random
lr = random.uniform(1e-5, 1e-2)      # Log scale better
batch_size = random.choice([32, 64, 128, 256])
```

### 3. Bayesian Optimization (Smart)
```python
# Using Optuna example
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    # Train model and return validation loss
    return val_loss
```

## 📈 Parameter Importance Ranking

### Critical (Search First)
1. **Learning Rate** - Most important, search log scale
2. **Batch Size** - Affects convergence and memory
3. **Architecture** - Model capacity

### Important (Search Second)  
4. **Regularization** - Weight decay, dropout
5. **Optimizer** - Adam vs SGD vs others
6. **Learning Rate Schedule** - Decay strategy

### Fine-Tuning (Search Last)
7. **Momentum** - For SGD
8. **Beta parameters** - For Adam
9. **Epsilon** - Numerical stability

## 🛠️ Implementation Recommendations

### For HAR Project Specifically:

#### 1. Create Hyperparameter Search Config
```yaml
# configs/hyperparameter_search/coarse_search.yaml
search_space:
  lr: [1e-4, 5e-4, 1e-3, 5e-3]
  batch_size: [64, 128, 256]
  alpha: [0.5, 1.0, 2.0]  # Loss coefficient
  hidden_dim: [64, 128, 256]
  
search_method: "grid"  # or "random", "bayesian"
n_trials: 36  # 4 × 3 × 3 × 3
budget_hours: 12  # Max time per trial
```

#### 2. Enhanced Sweep Script
```bash
#!/bin/bash
# submit_hyperparameter_search.sh

SEARCH_CONFIG=${1:-"coarse_search"}
N_PARALLEL=${2:-4}

#SBATCH --array=0-35%4  # 36 trials, max 4 parallel
#SBATCH --time=12:00:00  # Longer for search

# Load search configuration
python run_hyperparameter_search.py \
    --search-config configs/hyperparameter_search/${SEARCH_CONFIG}.yaml \
    --trial-id $SLURM_ARRAY_TASK_ID
```

#### 3. Results Tracking
```python
# Save hyperparameter results
results = {
    'hyperparameters': hyperparams,
    'metrics': {
        'val_loss': final_val_loss,
        'val_accuracy': final_val_acc,
        'train_time': training_time,
        'convergence_epoch': best_epoch
    },
    'config_hash': config_hash,  # For reproducibility
    'timestamp': datetime.now(),
    'git_commit': get_git_commit()
}
```

## 📊 Analysis & Selection

### 1. Performance Metrics
- **Primary**: Validation loss/accuracy
- **Secondary**: Training time, convergence speed
- **Tertiary**: Model size, inference speed

### 2. Visualization
```python
# Plot hyperparameter importance
import matplotlib.pyplot as plt
import seaborn as sns

# Heatmap of lr vs batch_size
results_df.pivot_table(
    values='val_loss', 
    index='lr', 
    columns='batch_size'
)
```

### 3. Selection Criteria
```python
# Multi-objective selection
def score_hyperparams(result):
    val_loss = result['val_loss']
    train_time = result['train_time']
    
    # Weighted combination
    score = (1 - val_loss) * 0.8 + (1/train_time) * 0.2
    return score
```

## 🎯 Recommendations for Your HAR Project

### 1. Start with Coarse Search
```bash
# Test these ranges first
LR: [1e-4, 5e-4, 1e-3, 5e-3]
BATCH_SIZE: [64, 128, 256]  # Cluster can handle larger
ALPHA: [0.5, 1.0, 2.0]      # Loss coefficient balance
```

### 2. Use Environment-Specific Defaults
```yaml
# Local (fast iteration)
epochs: 50
batch_size: 64

# Cluster (thorough training)  
epochs: 200
batch_size: 128
```

### 3. Track Everything
- Git commit hash
- Random seeds
- Full configuration
- System info (GPU type, etc.)
- Training curves, not just final metrics

This approach gives you systematic, reproducible hyperparameter optimization! 🚀