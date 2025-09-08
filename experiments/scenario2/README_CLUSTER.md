# HAR Scenario 2 - Cluster Deployment Guide

This directory contains everything needed to run Scenario 2 experiments on a cluster with SLURM.

## Files Overview

- `run_experiment.py` - Main experiment runner (equivalent to the notebook)
- `configs/scenario2.yaml` - Standard configuration
- `configs/scenario2_cluster.yaml` - Cluster-optimized configuration
- `submit_slurm.sh` - Single experiment SLURM job script
- `submit_sweep.sh` - Hyperparameter sweep SLURM job script
- `setup_cluster.sh` - Setup and validation script

## Quick Start

1. **Setup and validate environment:**
   ```bash
   chmod +x setup_cluster.sh
   ./setup_cluster.sh
   ```

2. **Run single experiment:**
   ```bash
   sbatch submit_slurm.sh
   ```

3. **Run hyperparameter sweep:**
   ```bash
   sbatch submit_sweep.sh
   ```

## Configuration

### Cluster Detection
The system automatically detects cluster environment via SLURM environment variables and:
- Uses CUDA when available
- Optimizes number of workers
- Uses full dataset (no subset for debugging)
- Uses cluster-specific data path: `/netscratch/zolfaghari/data/mm-fit/`

### Key Configuration Differences

**Local (notebook/debugging):**
- Small epochs (5-10) for quick testing
- Smaller batch size (64)
- Subset of data for faster iteration
- MPS support on Mac
- Data path: `./data/mm-fit/` (from base config)

**Cluster:**
- Full training epochs (200+)
- Larger batch size (128)
- Full dataset
- GPU-optimized settings
- Data path: `/netscratch/zolfaghari/data/mm-fit/` (cluster override)

## SLURM Scripts

### submit_slurm.sh
- Single experiment run
- 4 hour time limit
- 1 GPU, 32GB RAM, 8 CPUs
- Uses cluster configuration (`scenario2_cluster.yaml`)

### submit_sweep.sh
- Hyperparameter sweep (12 configurations)
- 6 hour time limit per job
- Maximum 4 concurrent jobs
- Sweeps learning rate and alpha values
- Uses cluster configuration

## Customization

### Adjusting SLURM Parameters
Edit the `#SBATCH` directives in the scripts:
```bash
#SBATCH --partition=gpu        # Change partition name
#SBATCH --gres=gpu:v100:1     # Request specific GPU type
#SBATCH --mem=64G             # Increase memory
#SBATCH --time=12:00:00       # Extend time limit
```

### Adding Command Line Options
The experiment runner supports several options:
```bash
python run_experiment.py \
    --base-config ../../configs/base.yaml \
    --exp-config ../../configs/scenario2_cluster.yaml \
    --epochs 500 \
    --lr 5e-4 \
    --seed 123 \
    --output-dir custom_output
```

### Environment Modules
Uncomment and adjust module loading in SLURM scripts:
```bash
module load python/3.11
module load cuda/11.8
module load pytorch/1.13
```

## Output Structure

Results are saved to timestamped directories in `outputs/`:
```
outputs/
├── 20250908-143022/          # Single run
│   ├── results.json          # Training history and metrics
│   └── config_clean.json     # Used configuration
└── sweep_lr1e-3_alpha1.0_*/  # Sweep runs
    ├── results.json
    └── config_clean.json
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View job output (while running)
tail -f outputs/JOBID_scenario2.out

# Check completed jobs
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed
```

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch_size in config
2. **Job timeout**: Increase time limit or reduce epochs
3. **Module not found**: Check PYTHONPATH and module loading

### Testing Locally
```bash
# Quick test with small epochs
python run_experiment.py --epochs 2

# Test cluster config locally
python run_experiment.py --exp-config ../../configs/scenario2_cluster.yaml --epochs 2
```

### Data Issues
Ensure data is accessible from compute nodes:
- Check data directory path in config
- Verify file permissions
- Consider copying to local scratch if needed
