#!/bin/bash
#SBATCH --job-name=har_hypersearch
#SBATCH --output=outputs/hyperparameter_search/%A_%a.out
#SBATCH --error=outputs/hyperparameter_search/%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Usage: sbatch --array=0-23 submit_hyperparameter_search.sh coarse_search
# Usage: sbatch --array=0-19 submit_hyperparameter_search.sh fine_search

SEARCH_CONFIG=${1:-"coarse_search"}

echo "=== Hyperparameter Search Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Search Config: $SEARCH_CONFIG"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR}"

# Set up environment variables
export PYTHONPATH="${SLURM_SUBMIT_DIR}:$PYTHONPATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create output directories
mkdir -p outputs/hyperparameter_search

# Print environment info
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run hyperparameter search trial
echo "Running hyperparameter search trial ${SLURM_ARRAY_TASK_ID}..."

python run_hyperparameter_search.py \
    --search-config "../../configs/hyperparameter_search/${SEARCH_CONFIG}.yaml" \
    --trial-id $SLURM_ARRAY_TASK_ID \
    --output-dir "outputs/hyperparameter_search/${SEARCH_CONFIG}_${SLURM_JOB_ID}"

echo "Trial completed at: $(date)"