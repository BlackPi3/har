#!/bin/bash
#SBATCH --job-name=har_s2_sweep
#SBATCH --output=outputs/%j_sweep_%A_%a.out
#SBATCH --error=outputs/%j_sweep_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-11%4  # Run 12 jobs, max 4 concurrent

# Hyperparameter sweep configurations
LR_VALUES=(1e-4 5e-4 1e-3 5e-3)
ALPHA_VALUES=(0.5 1.0 2.0)

# Calculate parameter combination
LR_IDX=$((SLURM_ARRAY_TASK_ID / 3))
ALPHA_IDX=$((SLURM_ARRAY_TASK_ID % 3))

LR=${LR_VALUES[$LR_IDX]}
ALPHA=${ALPHA_VALUES[$ALPHA_IDX]}

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Parameters: LR=$LR, ALPHA=$ALPHA"
echo "Start time: $(date)"

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR}"

# Set up environment (no conda activation needed on cluster)
export PYTHONPATH="${SLURM_SUBMIT_DIR}:$PYTHONPATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create outputs directory
mkdir -p outputs

# Run experiment with specific hyperparameters; cluster overrides auto-applied from base.yaml
python run_experiment.py \
    --base-config ../../configs/base.yaml \
    --exp-config ../../configs/scenario2.yaml \
    --lr $LR \
    --epochs 200 \
    --seed $((42 + SLURM_ARRAY_TASK_ID)) \
    --output-dir outputs/sweep_lr${LR}_alpha${ALPHA}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "Job completed at: $(date)"
