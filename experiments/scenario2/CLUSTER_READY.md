# Scenario 2 - Cluster Deployment Summary

## 🎯 Overview
Your Scenario 2 experiment is now ready for cluster deployment! The setup includes:

## 📁 Files Created

### Core Scripts
- **`run_experiment.py`** - Main experiment runner (equivalent to notebook functionality)
- **`test_setup.py`** - Quick validation script to test the setup

### SLURM Job Scripts
- **`submit_slurm.sh`** - Single experiment submission script
- **`submit_sweep.sh`** - Hyperparameter sweep (12 jobs: 4 learning rates × 3 alpha values)

### Configuration
- **`configs/scenario2.yaml`** - Standard configuration for local development
- **`configs/scenario2_cluster.yaml`** - Cluster-optimized configuration with:
  - Cluster data path: `/netscratch/zolfaghari/data/mm-fit/`
  - Larger batch size (128 vs 64)
  - More epochs (200 vs 100)
  - Larger model architecture
  - Full dataset usage

### Setup & Documentation
- **`setup_cluster.sh`** - Environment validation script
- **`README_CLUSTER.md`** - Comprehensive deployment guide
- **`requirements.txt`** - Python dependencies (at project root)

## ✅ Validation Results
The test script confirms:
- ✓ Configuration loading works
- ✓ Path resolution is correct
- ✓ Data loading works (700 samples per split)
- ✓ Models import successfully
- ✓ Device detection works (MPS locally, CUDA on cluster)

## 🚀 How to Deploy on Cluster

### 1. Transfer Files
Copy the entire project to your cluster, ensuring the data structure is replicated:
```
# Cluster data structure (same as local):
/netscratch/zolfaghari/data/mm-fit/w00/
/netscratch/zolfaghari/data/mm-fit/w01/
... (all subject directories)

# Project structure:
code/
├── data/                 # Local data directory (for reference)
├── src/                  # Core modules
├── configs/              # Base configurations  
└── experiments/scenario2/ # This directory
```

### 2. Single Job Submission
```bash
cd experiments/scenario2
sbatch submit_slurm.sh
```

### 3. Hyperparameter Sweep
```bash
cd experiments/scenario2  
sbatch submit_sweep.sh
```

## 📊 Expected Output Structure
```
experiments/scenario2/outputs/
├── 20250908-HHMMSS/           # Single runs
│   ├── results.json           # Training history + metrics
│   └── config_clean.json      # Configuration used
└── sweep_lr1e-3_alpha1.0_*/   # Sweep runs
    ├── results.json
    └── config_clean.json
```

## 🔧 Key Features

### Automatic Environment Detection
- Detects SLURM environment via `$SLURM_JOB_ID`
- Automatically uses CUDA when available on cluster
- Falls back to MPS on Mac, CPU elsewhere

### Path Resolution
- Handles relative paths correctly regardless of execution context
- Works both locally and on cluster

### Flexible Configuration
- Command-line overrides: `--epochs`, `--lr`, `--seed`
- Multiple configuration files for different environments
- Easy parameter sweeps

### Robust Error Handling
- Graceful fallbacks for missing data
- Clear error messages for troubleshooting

## 🎛️ Customization Options

### SLURM Parameters
Edit `#SBATCH` directives in scripts:
- `--partition=gpu` - Change partition  
- `--gres=gpu:v100:1` - Request specific GPU
- `--mem=64G` - Increase memory
- `--time=12:00:00` - Extend time limit

### Sweep Parameters  
Modify arrays in `submit_sweep.sh`:
```bash
LR_VALUES=(1e-4 5e-4 1e-3 5e-3)
ALPHA_VALUES=(0.5 1.0 2.0)
```

## 📋 Next Steps
1. **Test locally**: Run `python test_setup.py` to validate
2. **Transfer to cluster**: Copy entire project directory
3. **Submit job**: Use `sbatch submit_slurm.sh` for single run
4. **Monitor**: Check outputs with `squeue -u $USER`

Your setup is complete and ready for production cluster runs! 🎉
