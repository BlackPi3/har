#!/usr/bin/env bash
#SBATCH -J sc2_mmfit_p1
#SBATCH -p RTX3090
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=40G
#SBATCH -t 03-00:00:00

########################################
# Eval configuration (override via env)
########################################
# STUDY_NAME: name for this eval run (used for output directory)
# HPO_NAME: name of the HPO study to evaluate (may differ from STUDY_NAME)
STUDY_NAME=${STUDY_NAME:-sc2_mmfit_p1}
REPEAT_COUNT=${REPEAT_COUNT:-5}

# Container + project paths
PROJECT_ROOT=${PROJECT_ROOT:-/home/zolfaghari/har}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/netscratch/zolfaghari/images/har.sqsh}

ENV_NAME=remote
HPO_ROOT=${HPO_ROOT:-/netscratch/$USER/experiments/hpo/$STUDY_NAME}
OUTPUT_ROOT=${OUTPUT_ROOT:-/netscratch/$USER/experiments/eval/$STUDY_NAME}
EPOCHS=${EPOCHS:-}
LOG_ROOT=${LOG_ROOT:-/netscratch/$USER/experiments/log/eval}

# Logs
set -euo pipefail
mkdir -p "$OUTPUT_ROOT"
mkdir -p "$LOG_ROOT"
timestamp=$(date +%y%m%d_%H%M)
LOG_STEM="$LOG_ROOT/eval_${timestamp}_${STUDY_NAME}"
LOG_OUT="${LOG_STEM}.out"
LOG_ERR="${LOG_STEM}.err"
echo "Stdout: $LOG_OUT"
echo "Stderr: $LOG_ERR"
echo "Output: $OUTPUT_ROOT"

# Unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

RUN_CMD="python -u -m experiments.run_eval --study-name \"$STUDY_NAME\" --env \"$ENV_NAME\" --repeat-count \"$REPEAT_COUNT\" --hpo-root \"$HPO_ROOT\" --output-root \"$OUTPUT_ROOT\""
if [[ -n "$EPOCHS" ]]; then
  RUN_CMD+=" --epochs \"$EPOCHS\""
fi

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-workdir="$PROJECT_ROOT" \
  --container-mounts="$PROJECT_ROOT":"$PROJECT_ROOT",/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro \
  bash -lc "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    echo Running: $RUN_CMD
    $RUN_CMD
  " >"$LOG_OUT" 2>"$LOG_ERR"
