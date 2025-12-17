#!/usr/bin/env bash
#SBATCH -J har-eval
#SBATCH -p RTXA6000
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00

# Container + project paths
PROJECT_ROOT=${PROJECT_ROOT:-/home/zolfaghari/har}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/netscratch/zolfaghari/images/har.sqsh}

########################################
# Eval configuration (override via env)
########################################
# STUDY_NAME must match the HPO run directory under experiments/hpo/
STUDY_NAME=${STUDY_NAME:-scenario2_utd}
ENV_NAME=remote
REPEAT_COUNT=${REPEAT_COUNT:-5}
HPO_ROOT=${HPO_ROOT:-/netscratch/$USER/experiments/hpo/$STUDY_NAME}
EPOCHS=${EPOCHS:-}
LOG_ROOT=${LOG_ROOT:-/netscratch/zolfaghari/experiments/log/eval}

# Logs
set -euo pipefail
mkdir -p "$LOG_ROOT"
timestamp=$(date +%y%m%d_%H%M)
LOG_STEM="$LOG_ROOT/eval_${timestamp}_${STUDY_NAME}"
LOG_OUT="${LOG_STEM}.out"
LOG_ERR="${LOG_STEM}.err"
echo "Stdout: $LOG_OUT"
echo "Stderr: $LOG_ERR"

export HYDRA_FULL_ERROR=1

RUN_CMD="python -m experiments.run_eval --study-name \"$STUDY_NAME\" --env \"$ENV_NAME\" --repeat-count \"$REPEAT_COUNT\" --hpo-root \"$HPO_ROOT\""
if [[ -n "$EPOCHS" ]]; then
  RUN_CMD+=" --epochs \"$EPOCHS\""
fi

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-workdir="$PROJECT_ROOT" \
  --container-mounts="$PROJECT_ROOT":"$PROJECT_ROOT",/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro \
  bash -lc "
    set -euo pipefail
    echo Running: $RUN_CMD
    $RUN_CMD
  " >"$LOG_OUT" 2>"$LOG_ERR"
