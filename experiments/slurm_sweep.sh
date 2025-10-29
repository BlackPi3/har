#!/usr/bin/env bash
#SBATCH -J har-sweep
#SBATCH -p A100-40GB
#SBATCH -o /home/zolfaghari/experiments/log/slurm-%x-%j.out
#SBATCH -e /home/zolfaghari/experiments/log/slurm-%x-%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH -t 24:00:00

set -euo pipefail

# --- User knobs (override via env) ---
PROJECT_ROOT=/home/$USER/har
CONTAINER_IMAGE=/netscratch/$USER/images/har.sqsh

# Unified output root and study directory inside scratch
STUDY_NAME=${STUDY_NAME:-scenario2_mmfit}
N_TRIALS=${N_TRIALS:-2}
OUTPUT_ROOT=${OUTPUT_ROOT:-/netscratch/$USER/experiments/output}
STUDY_DIR="$OUTPUT_ROOT/$STUDY_NAME"
mkdir -p "$STUDY_DIR"

# Optuna study database lives inside the same study directory
STUDY_DB="$STUDY_DIR/$STUDY_NAME.db"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-workdir="$PROJECT_ROOT" \
  --container-mounts="$PROJECT_ROOT":"$PROJECT_ROOT",/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro \
  python -m experiments.run -m \
    scenario=scenario2 data=mmfit hpo=scenario2_mmfit \
    hydra/sweeper=optuna \
    hydra.sweeper.storage=sqlite:////$STUDY_DB \
  hydra.sweeper.n_trials=$N_TRIALS \
  hydra.sweeper.study_name=$STUDY_NAME \
  hydra.sweep.dir=$STUDY_DIR \
  hydra.sweep.subdir=trial_\${hydra.job.num} \
    trainer.epochs=10 \
    seed=0