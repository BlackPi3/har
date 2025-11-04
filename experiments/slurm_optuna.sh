#!/usr/bin/env bash
#SBATCH -J har-optuna
#SBATCH -p L40S
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00

# Container + project paths
PROJECT_ROOT=${PROJECT_ROOT:-/home/zolfaghari/har}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/netscratch/zolfaghari/images/har.sqsh}

########################################
# HPO configuration (study via conf/hpo)
########################################
# Defaults (override via environment variables as needed)
HPO=${HPO:-scenario2_mmfit}
N_TRIALS=${N_TRIALS:-60}
SPACE_CONFIG=${SPACE_CONFIG:-conf/hpo/$HPO.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-/netscratch/zolfaghari/experiments/hpo/$HPO}
STORAGE=${STORAGE:-$OUTPUT_ROOT/$HPO.db}

# Explicit run-time config (no aggregated overrides)
ENV_NAME=${ENV_NAME:-remote}
DATA_NAME=${DATA_NAME:-mmfit}
EPOCHS=${EPOCHS:-50}

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
  bash -lc "
    set -euo pipefail
    python -m experiments.run_optuna \
      --n-trials $N_TRIALS \
      --storage '$STORAGE' \
      --output-root '$OUTPUT_ROOT' \
      --space-config '$SPACE_CONFIG' \
      --env $ENV_NAME \
      --data $DATA_NAME \
      --epochs $EPOCHS
  "
