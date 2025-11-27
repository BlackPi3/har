#!/usr/bin/env bash
#SBATCH -J har-hpo
#SBATCH -p RTX3090
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=40G
#SBATCH -t 3-00:00:00

# Container + project paths
PROJECT_ROOT=${PROJECT_ROOT:-/home/zolfaghari/har}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/netscratch/zolfaghari/images/har.sqsh}

########################################
# HPO configuration (study via conf/hpo)
########################################
# Defaults (override via environment variables as needed)
HPO=${HPO:-scenario2_utd}
N_TRIALS=${N_TRIALS:-1728}
SPACE_CONFIG=${SPACE_CONFIG:-conf/hpo/$HPO.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-/netscratch/zolfaghari/experiments/hpo/$HPO}
STORAGE=${STORAGE:-$OUTPUT_ROOT/$HPO.db}

# Explicit run-time config (no aggregated overrides)
ENV_NAME=${ENV_NAME:-remote}
# EPOCHS=${EPOCHS:-50} # --epochs $EPOCHS
LOG_ROOT=${LOG_ROOT:-/netscratch/zolfaghari/experiments/log}

# Logs
set -euo pipefail
mkdir -p "$OUTPUT_ROOT"
mkdir -p "$LOG_ROOT"
timestamp=$(date +%y%m%d_%H%M)
LOG_STEM="$LOG_ROOT/hpo_${timestamp}"
LOG_OUT="${LOG_STEM}.out"
LOG_ERR="${LOG_STEM}.err"
echo "Stdout: $LOG_OUT"
echo "Stderr: $LOG_ERR"

# Optuna + Hydra diagnostics
export HYDRA_FULL_ERROR=1

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-workdir="$PROJECT_ROOT" \
  --container-mounts="$PROJECT_ROOT":"$PROJECT_ROOT",/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro \
  bash -lc "
    set -euo pipefail
    python -m experiments.run_hpo \
      --n-trials $N_TRIALS \
      --storage '$STORAGE' \
      --output-root '$OUTPUT_ROOT' \
      --space-config '$SPACE_CONFIG' \
      --env $ENV_NAME \
  " >"$LOG_OUT" 2>"$LOG_ERR"
