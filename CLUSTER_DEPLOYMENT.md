# Cluster Deployment Guide

## Overview
This document describes how to deploy HAR experiments on the cluster infrastructure, with specific configurations for the cluster environment.

## Environment Differences

### Local Development
- **Base Data Path**: `./data/`
- **MM-Fit Dataset**: `./data/mm-fit/`
- **Environment**: Conda environment `har`
- **Device**: MPS (Apple Silicon) or CPU
- **Purpose**: Development, debugging, visualization

### Cluster Production
- **Base Data Path**: `/netscratch/zolfaghari/data/`
- **MM-Fit Dataset**: `/netscratch/zolfaghari/data/mm-fit/`
- **Environment**: Pre-installed container/image
- **Device**: CUDA GPUs
- **Purpose**: Production training, parameter sweeps

## Data Directory Structure

### Local Structure
```
./data/
├── mm-fit/           # MM-Fit dataset
│   ├── w00/
│   │   ├── w00_labels.csv
│   │   ├── w00_pose_3d_upsample_normal.npy
│   │   └── ...
│   ├── w01/
│   └── ... (w02-w20)
├── UTD_MHAD/         # Other datasets
└── processed/        # Processed data
```

### Cluster Structure (Same as Local)
```
/netscratch/zolfaghari/data/
├── mm-fit/           # MM-Fit dataset (same structure)
│   ├── w00/
│   │   ├── w00_labels.csv
│   │   ├── w00_pose_3d_upsample_normal.npy
│   │   └── ...
│   ├── w01/
│   └── ... (w02-w20)
├── UTD_MHAD/         # Other datasets  
└── processed/        # Processed data
```

## Configuration Files

### Data Path Configuration
- **Base config** (`configs/base.yaml`): Uses `./data/mm-fit/` for local development
- **Local scenario** (`configs/scenario2.yaml`): Inherits from base (local path) 
- **Cluster scenario** (`configs/scenario2_cluster.yaml`): Overrides with `/netscratch/zolfaghari/data/mm-fit/`

### Path Resolution Logic
The system uses different configurations for different environments:
- **Local**: `scenario2.yaml` → inherits `./data/mm-fit/` from base.yaml
- **Cluster**: `scenario2_cluster.yaml` → explicitly sets `/netscratch/zolfaghari/data/mm-fit/`

## Deployment Steps

### 1. Prepare Data
Ensure the data directory structure is replicated on the cluster:
```bash
# Cluster should have the same structure as local:
/netscratch/zolfaghari/data/mm-fit/    # (same as ./data/mm-fit/ locally)
/netscratch/zolfaghari/data/UTD_MHAD/  # (same as ./data/UTD_MHAD/ locally)  
/netscratch/zolfaghari/data/processed/ # (same as ./data/processed/ locally)
```

### 2. Transfer Code
Copy the project directory to the cluster, maintaining the structure:
```
code/
├── src/                    # Core modules
├── configs/               # Base configurations
└── experiments/scenario2/ # Experiment files
```

### 3. Submit Jobs

#### Single Experiment
```bash
cd experiments/scenario2
sbatch submit_slurm.sh
```

#### Parameter Sweep
```bash
cd experiments/scenario2
sbatch submit_sweep.sh
```

## Cluster-Specific Features

### Automatic Environment Detection
- Detects SLURM environment via `$SLURM_JOB_ID`
- Automatically uses cluster data path
- Selects CUDA when available
- Optimizes worker processes for cluster

### Resource Allocation
```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
```

### Output Management
- Results saved to timestamped directories
- Job logs separated by job ID
- Configuration snapshots preserved

## Troubleshooting

### Common Issues
1. **Data not found**: Verify data path `/netscratch/zolfaghari/data/mm-fit/`
2. **Permission issues**: Check read access to data directory
3. **GPU allocation**: Ensure GPU partition is available
4. **Memory errors**: Reduce batch size in configuration

### Monitoring Jobs
```bash
# Check job status
squeue -u $USER

# View job output
tail -f outputs/JOBID_scenario2.out

# Check completed jobs
sacct -u $USER
```

## Performance Optimizations

### Cluster Configuration
- Batch size: 128 (vs 64 local)
- Workers: 8 (vs 2-4 local)  
- Full dataset: No subset sampling
- Extended training: 200 epochs

### Memory Management
- Larger models enabled on cluster
- Efficient data loading with multiple workers
- GPU memory optimization for larger batches

## Results Structure
```
experiments/scenario2/outputs/
├── 20250908-143022/          # Single runs
│   ├── results.json          # Training history
│   └── config_clean.json     # Configuration used
└── sweep_lr1e-3_alpha1.0_*/  # Parameter sweep
    ├── results.json
    └── config_clean.json
```

## Validation
Before deploying to cluster, validate locally:
```bash
# Test configuration loading
python test_setup.py

# Quick training test
python run_experiment.py --epochs 1
```
