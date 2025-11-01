#!/usr/bin/env bash
#SBATCH -J har-sweep
#SBATCH -p L40S
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH -t 00:30:00

# GPUs: RTX3090, V100-32GB, L40S / RTXA6000, A100-40GB / A100-PCI

set -euo pipefail
set -x

# --- User knobs (override via env) ---
PROJECT_ROOT=/home/zolfaghari/har
CONTAINER_IMAGE=/netscratch/zolfaghari/images/har.sqsh

# Number of Optuna trials (override with N_TRIALS=... when submitting)
N_TRIALS=${N_TRIALS:-2}

# HPO space to use (override with HPO=... when submitting)
HPO=${HPO:-scenario2_mmfit}

# Ensure log directory exists (use a single, consistent user for scratch paths)
export SCRATCH_USER=${SCRATCH_USER:-$USER}
mkdir -p /netscratch/zolfaghari/experiments/log
# Pre-create per-study output dir so Optuna SQLite parent exists
mkdir -p "/netscratch/zolfaghari/experiments/output/$HPO"
LOGDIR=/netscratch/zolfaghari/experiments/log
mkdir -p "$LOGDIR"

# --- Dynamic timestamp (e.g., 251025-5pm) ---
DATESTAMP=$(date +%d%m%y-%I%P)   # %I = 12-hour, %P = am/pm (e.g., 5pm)

# --- Redirect stdout/stderr manually ---
exec > >(tee -a "$LOGDIR/${DATESTAMP}.out")
exec 2> >(tee -a "$LOGDIR/${DATESTAMP}.err" >&2)

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-workdir="$PROJECT_ROOT" \
  --container-mounts="$PROJECT_ROOT":"$PROJECT_ROOT",/netscratch/zolfaghari:/netscratch/zolfaghari,/ds:/ds:ro \
  \
  python -m experiments.run -m \
    env=remote \
    scenario=scenario2 \
  data=mmfit \
    trainer.epochs=2 \
    +hpo=$HPO \
    hydra/sweeper=optuna \
    hydra.sweeper.n_trials=$N_TRIALS \
    seed=0