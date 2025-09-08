#!/bin/bash

# HAR Project - Scenario 2 Cluster Setup Script
# This script helps set up the environment for running on a cluster

set -e  # Exit on any error

echo "=== HAR Scenario 2 Cluster Setup ==="

# Check if we're on a cluster
if [[ -n "${SLURM_JOB_ID}" || -n "${SLURM_CPUS_ON_NODE}" ]]; then
    echo "✓ Detected SLURM environment"
    ON_CLUSTER=true
else
    echo "! Not on SLURM cluster - setting up for local testing"
    ON_CLUSTER=false
fi

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"

# Check Python environment
echo "=== Checking Python Environment ==="
if [[ "$ON_CLUSTER" == "true" ]]; then
    echo "On cluster - using pre-installed environment"
else
    echo "Local development - using conda environment 'har'"
    echo "Activate with: conda activate har"
fi
python --version
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"

# Check data directory
echo "=== Checking Data ==="
DATA_DIR="${PROJECT_ROOT}/data/mm-fit"
if [[ -d "${DATA_DIR}" ]]; then
    echo "✓ Data directory found: ${DATA_DIR}"
    echo "Available subjects: $(ls -1 "${DATA_DIR}" | grep '^w' | wc -l | tr -d ' ')"
else
    echo "✗ Data directory not found: ${DATA_DIR}"
    echo "Please ensure data is available before running experiments"
fi

# Test configuration loading
echo "=== Testing Configuration ==="
cd "${SCRIPT_DIR}"
python -c "
import sys
sys.path.append('${PROJECT_ROOT}')
from src.config import load_config
cfg = load_config('../../configs/base.yaml', '../../configs/scenario2.yaml')
print(f'✓ Configuration loaded successfully')
print(f'  Device: {cfg.device}')
print(f'  Cluster: {cfg.cluster}')
print(f'  Dataset: {cfg.dataset_name}')
print(f'  Batch size: {cfg.batch_size}')
"

# Create necessary directories
echo "=== Creating Directories ==="
mkdir -p outputs
mkdir -p logs
echo "✓ Created output directories"

# Make scripts executable
chmod +x submit_slurm.sh
chmod +x submit_sweep.sh
echo "✓ Made SLURM scripts executable"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To run on cluster:"
echo "  1. Single run:     sbatch submit_slurm.sh"
echo "  2. Parameter sweep: sbatch submit_sweep.sh"
echo ""
echo "To test locally:"
echo "  python run_experiment.py --epochs 5"
echo ""
