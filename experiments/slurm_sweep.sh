#!/usr/bin/env bash
#SBATCH -J har-sweep
#SBATCH -p A100-40GB
#SBATCH -o netscratch/zolfaghari/experiments/log/slurm-%x-%j.out
#SBATCH -e netscratch/zolfaghari/experiments/log/slurm-%x-%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH -t 24:00:00

set -euo pipefail

# --- User knobs (override via env) ---
PROJECT_ROOT=/home/$USER/har
CONTAINER_IMAGE=/netscratch/$USER/images/har.sqsh

# Optuna study database path (for persistence/resume across runs)
STUDY_DB=${STUDY_DB:-/netscratch/$USER/experiments/optuna/scenario2_mmfit.db}
# Ensure parent directory exists
mkdir -p "$(dirname "$STUDY_DB")"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-workdir="$PROJECT_ROOT" \
  --container-mounts="$PROJECT_ROOT":"$PROJECT_ROOT",/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro \
  python -m experiments.run_trial -m \
    scenario=scenario2 data=mmfit hpo=scenario2_mmfit \
    hydra/sweeper=optuna \
    hydra.sweeper.storage=sqlite:////$STUDY_DB \
    trainer.epochs=10 \
    seed=0