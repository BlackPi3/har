# Human Activity Recognition (HAR) Project

This repository contains a modular Human Activity Recognition system focused on pose-to-IMU regression and activity classification. The project uses a research-oriented structure with modular components for reproducible experiments.

## Project Structure

```
├── src/                          # Core modular codebase
│   ├── config.py                 # Configuration loading and merging
│   ├── data.py                   # Dataset factory and dataloader creation
│   ├── train.py                  # Training pipeline and Trainer class
│   ├── inference.py              # Inference utilities
│   ├── datasets/                 # Dataset implementations
│   │   ├── common/               # Shared dataset utilities
│   │   │   ├── base_dataset.py   # Base HAR dataset class
│   │   │   ├── samplers.py       # Custom sampling strategies
│   │   │   └── utils.py          # Common dataset utilities
│   │   └── mmfit/                # MMFit dataset implementation
│   │       ├── dataset.py        # MMFit dataset class
│   │       ├── factory.py        # Dataset factory functions
│   │       ├── loaders.py        # Data loading utilities
│   │       └── constants.py      # Dataset constants
│   └── models/                   # Model implementations
│       ├── classifiers.py        # Activity classifiers
│       ├── discriminator.py      # Domain discriminators
│       ├── regressor.py          # Pose-to-IMU regression models
│       └── tcn.py                # Temporal Convolutional Networks
├── experiments/                  # Self-contained experiments
│   └── scenario2/                # Example experiment
│       ├── configs/              # Experiment-specific configs
│       ├── notebooks/            # Experiment notebooks
│       ├── outputs/              # Experiment outputs
│       └── run_experiment.py     # CLI experiment runner
├── configs/                      # Base configuration files
│   ├── base.yaml                 # Base configuration
│   └── scenario2.yaml            # Example experiment config
├── datasets/                     # Legacy dataset implementations
├── legacy/                       # Archived legacy code
└── data/                         # Data directory (gitignored)
```

## Key Features

- **Modular Architecture**: Clean separation between datasets, models, training, and configuration
- **Research-Focused**: Self-contained experiments with isolated configs and outputs
- **Reproducible**: Configuration-driven experiments with seed control
- **Multiple Datasets**: Support for MMFit, MHAD, NTU, and adversarial datasets
- **Flexible Training**: Multi-model training with pose-to-IMU regression and activity classification

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt  # or use conda/poetry as preferred

# Ensure data directory exists
mkdir -p data/
```

### 2. Run an Experiment

**Option A: Using Jupyter Notebook (for exploration)**
```bash
cd experiments/scenario2/notebooks/
jupyter notebook scenario2.ipynb
```

**Option B: Using CLI (for reproducible runs)**
```bash
python experiments/scenario2/run_experiment.py --config configs/scenario2.yaml --seed 42
```

### 3. Configuration

Experiments use hierarchical YAML configuration:
- `configs/base.yaml`: Base settings (data paths, model architectures, training parameters)
- `experiments/*/configs/*.yaml`: Experiment-specific overrides

Example configuration structure:
```yaml
# Base config
device: mps
dataset_name: mmfit
batch_size: 64
lr: 1e-3

models:
  regressor:
    input_channels: 51
    num_joints: 17
    sequence_length: 200
  classifier:
    f_in: 512
    n_classes: 12
```

## Dataset Support

### MMFit Dataset
- **Purpose**: Pose-to-IMU regression and activity recognition
- **Data**: 3D pose sequences → accelerometer data
- **Splits**: Subject-based train/val/test splits
- **Activities**: 12 fitness activities (squats, lunges, etc.)

### Adding New Datasets
1. Implement dataset class inheriting from `BaseHARDataset`
2. Create factory function in `src/datasets/{dataset}/factory.py`
3. Add dataset name to `src/data.py` factory dispatcher
4. Update configuration files as needed

## Models

- **Regressor**: TCN-based pose-to-IMU regression
- **FeatureExtractor**: Shared feature extraction from IMU data  
- **ActivityClassifier**: Activity classification from extracted features
- **Discriminator**: Domain adversarial training (optional)

## Development Guidelines

### Adding New Experiments
1. Create `experiments/{experiment_name}/` directory
2. Add experiment-specific configs in `configs/`
3. Create Jupyter notebook in `notebooks/`
4. Implement CLI runner script if needed

### Code Organization
- Keep dataset-specific logic in `src/datasets/{dataset}/`
- Use `src/datasets/common/` for shared utilities
- Configuration should be declarative and hierarchical
- All experiments should be reproducible with fixed seeds

## Legacy Code

The `legacy/` directory contains archived implementations that have been refactored:
- `scenario2_legacy.ipynb` → `experiments/scenario2/notebooks/scenario2.ipynb`
- Monolithic dataset files → modular `src/datasets/` structure

## Contributing

1. Follow the modular architecture principles
2. Add tests for new components in `tests/`
3. Update documentation when adding new features
4. Use type hints and docstrings for public APIs

## License

[Add your license information here]
