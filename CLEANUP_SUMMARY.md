# HAR Project - Clean Structure Summary

## Final Directory Structure

```
code/
├── src/                          # Core reusable components
│   ├── models/                   # Model architectures
│   ├── data.py                   # DataLoader factory (fixed imports)
│   ├── train.py                  # Training pipeline
│   ├── config.py                 # Configuration utilities
│   ├── inference.py              # Model inference utilities
│   └── datasets/                 # Dataset classes and utilities (modular)
│       ├── common/               # Shared utilities (samplers, base classes)
│       ├── mmfit/                # MMFit-specific modules
│       ├── mhad.py              # MHAD dataset
│       ├── ntu.py               # NTU dataset
│       └── adv.py               # Adversarial dataset
├── data/                         # Raw datasets (gitignored)
│   ├── mm-fit/                   # MMFit data files
│   ├── UTD_MHAD/                 # MHAD data files
│   └── processed/                # Generated/cached data
├── experiments/                  # Experiment-specific code
│   └── scenario2/                # Pose-to-IMU regression experiment
│       ├── configs/              # Experiment configs
│       │   └── scenario2.yaml    # Experiment-specific config
│       ├── notebooks/            # Analysis notebooks
│       │   └── scenario2.ipynb   # WORKING notebook (cleaned)
│       ├── outputs/              # Results, plots, logs
│       └── run_experiment.py     # CLI experiment runner
├── scripts/                      # Preprocessing and utilities
│   ├── *_preprocess.ipynb       # Preprocessing notebooks
│   └── data_gen/                # Data generation scripts
├── configs/                      # Global/shared configs
│   ├── base.yaml                # Base configuration
│   └── scenario2.yaml           # (moved to experiments/)
├── legacy/                       # Legacy code (optional)
├── legacy_datasets/              # Old datasets directory (moved)
└── .gitignore                    # Updated to ignore data/, outputs/, etc.
```

## ✅ Issues Fixed:

1. **Import Conflicts**: Removed conflicting `src/data/` directory, kept `src/data.py`
2. **Duplicate Notebooks**: Removed old `notebooks/scenario2.ipynb`, kept experiment version
3. **Duplicate Cells**: Cleaned up duplicate cells in experiment notebook
4. **Outdated Configs**: Updated paths in configs for new structure
5. **Directory Conflicts**: Moved old `datasets/` to `legacy_datasets/`
6. **Git Ignore**: Updated `.gitignore` to exclude data files and outputs
7. **Old Files**: Removed duplicate `run_experiment.py` and `outputs/scenario2/`

## ✅ Working Components:

1. **Notebook**: `experiments/scenario2/notebooks/scenario2.ipynb` works correctly
2. **Imports**: All `src.*` imports function properly
3. **Data Loading**: MMFit dataset loads successfully (8400/2800/2800 train/val/test)
4. **Modular Structure**: Clean separation between common and dataset-specific code
5. **Experiment Isolation**: Each experiment has its own directory

## Usage Examples:

**Notebook development:**
```bash
cd experiments/scenario2/notebooks/
jupyter lab scenario2.ipynb
```

**CLI experiment:**
```bash
cd experiments/scenario2/
python run_experiment.py --epochs 10 --lr 1e-4
```

**Create new experiment:**
```bash
cp -r experiments/scenario2 experiments/scenario3
# Modify configs and run
```

## Research Benefits:

- **Clean imports**: No more circular dependencies or conflicts
- **Experiment isolation**: Each experiment is self-contained
- **Reproducible**: Configs and outputs saved together
- **Scalable**: Easy to add new datasets or experiments
- **Git-friendly**: Large data files ignored, only code tracked

The project is now clean, organized, and ready for productive research! 🎉
