#!/usr/bin/env bash
#SBATCH -J har-optuna
#SBATCH -p L40S
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH -t 00:45:00

# Container + project paths
PROJECT_ROOT=${PROJECT_ROOT:-/home/zolfaghari/har}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/netscratch/zolfaghari/images/har.sqsh}

########################################
# HPO and config selection
########################################
# Config-driven study name: mirror previous sweeper behavior where HPO config defined the study name
HPO=${HPO:-scenario2_mmfit}
N_TRIALS=${N_TRIALS:-10}
# Derive STUDY_NAME from selected HPO config unless explicitly provided
STUDY_NAME=${STUDY_NAME:-$HPO}
OUTPUT_ROOT=${OUTPUT_ROOT:-/netscratch/$USER/experiments/output/$STUDY_NAME}
STORAGE=${STORAGE:-$OUTPUT_ROOT/$STUDY_NAME.db}

# Hydra overrides passed to experiments.run_trial (space-separated tokens)
# Keep only what is required for a single trial; no unused flags
OVERRIDES=${OVERRIDES:-env=remote data=mmfit scenario=scenario2 trainer.epochs=5}

# Logs
set -euo pipefail
set -x
mkdir -p "$OUTPUT_ROOT"
LOGDIR=/netscratch/$USER/experiments/log
mkdir -p "$LOGDIR"

# Optuna + Hydra diagnostics
export HYDRA_FULL_ERROR=1

# Timestamped logs
DATESTAMP=$(date +%d%m%y-%I%P)
exec > >(tee -a "$LOGDIR/${DATESTAMP}.out")
exec 2> >(tee -a "$LOGDIR/${DATESTAMP}.err" >&2)

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-workdir="$PROJECT_ROOT" \
  --container-mounts="$PROJECT_ROOT":"$PROJECT_ROOT",/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro \
  bash -lc '
    set -euo pipefail
    # If the image already has the project installed, this is a no-op
    python -m pip install --user -e . -q || true
    python -m experiments.run_optuna \
      --n-trials '"$N_TRIALS"' \
      --study-name '"$STUDY_NAME"' \
      --storage '"$STORAGE"' \
      --metric val_f1 --direction maximize \
      --output-root '"$OUTPUT_ROOT"' \
      -- -- $OVERRIDES
  '
