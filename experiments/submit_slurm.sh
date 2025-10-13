#!/usr/bin/env bash

# Usage: sbatch submit_slurm.sh <gpu_type> <dataset> [lr] [alpha] [epochs]
# Example: sbatch submit_slurm.sh v100 mmfit 1e-3 1.0 200

# Parse command line arguments
GPU_TYPE=${1:-"1"}           # Default: any GPU
DATASET=${2:-"mmfit"}        # Default: mmfit
LR=${3:-""}                  # Optional: learning rate
ALPHA=${4:-""}               # Optional: alpha parameter  
EPOCHS=${5:-""}              # Optional: epochs

#SBATCH --job-name=har_scenario2
#SBATCH --output=outputs/%j_scenario2.out
#SBATCH --error=outputs/%j_scenario2.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Set GPU requirement based on parameter
if [[ "$GPU_TYPE" != "1" ]]; then
    export SBATCH_GRES="gpu:${GPU_TYPE}:1"
else
    export SBATCH_GRES="gpu:1"
fi

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME" 
echo "GPU Type: $GPU_TYPE"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Dataset: $DATASET"
echo "Parameters: LR=$LR, ALPHA=$ALPHA, EPOCHS=$EPOCHS"
echo "Start time: $(date)"

# Load modules (adjust based on your cluster)
# module load python/3.11
# module load cuda/11.8

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR}"

# Activate virtual environment if using one
# source venv/bin/activate

# Set up environment variables
export PYTHONPATH="${SLURM_SUBMIT_DIR}:$PYTHONPATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print environment info
echo "Python path: $PYTHONPATH"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Create outputs directory
mkdir -p outputs

# Build command line arguments
# Cluster-specific adjustments (data_dir, epochs, batch_size) are applied automatically
# via cluster_overrides in base.yaml when SLURM env vars are detected.
CMD_ARGS="--base-config ../../configs/base.yaml --exp-config ../../configs/scenario2.yaml"

# Add dataset override if specified
if [[ "$DATASET" != "mmfit" ]]; then
    CMD_ARGS="$CMD_ARGS --dataset $DATASET"
fi

# Add optional hyperparameters if provided
if [[ -n "$LR" ]]; then
    CMD_ARGS="$CMD_ARGS --lr $LR"
fi

if [[ -n "$ALPHA" ]]; then
    CMD_ARGS="$CMD_ARGS --alpha $ALPHA"  
fi

if [[ -n "$EPOCHS" ]]; then
    CMD_ARGS="$CMD_ARGS --epochs $EPOCHS"
fi

# Run the experiment with cluster configuration
echo "Starting experiment..."
echo "Command: python run_experiment.py $CMD_ARGS"
python run_experiment.py $CMD_ARGS

echo "Job completed at: $(date)"
