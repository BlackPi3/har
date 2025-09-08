#!/bin/bash
#SBATCH --job-name=har_scenario2
#SBATCH --output=outputs/%j_scenario2.out
#SBATCH --error=outputs/%j_scenario2.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
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

# Run the experiment with cluster configuration
echo "Starting experiment..."
python run_experiment.py \
    --base-config ../../configs/base.yaml \
    --exp-config ../../configs/scenario2_cluster.yaml

echo "Job completed at: $(date)"
