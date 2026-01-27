#!/usr/bin/env bash
#SBATCH -J har-trial
#SBATCH -p RTXA6000
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=40G
#SBATCH -t 06:00:00

########################################
# Trial configuration (override via env)
########################################
TRIAL_NAME=${TRIAL_NAME:-scenario2_utd}
EPOCHS=${EPOCHS:-}  # Optional: override trainer.epochs

# Container + project paths
PROJECT_ROOT=${PROJECT_ROOT:-/home/zolfaghari/har}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/netscratch/zolfaghari/images/har.sqsh}

ENV_NAME=remote
OUTPUT_ROOT=${OUTPUT_ROOT:-/netscratch/zolfaghari/experiments/trial/$TRIAL_NAME}
LOG_ROOT=${LOG_ROOT:-/netscratch/zolfaghari/experiments/log/trial}

# Logs
set -euo pipefail
mkdir -p "$OUTPUT_ROOT"
mkdir -p "$LOG_ROOT"
timestamp=$(date +%y%m%d_%H%M)
LOG_STEM="$LOG_ROOT/trial_${timestamp}_${TRIAL_NAME}"
LOG_OUT="${LOG_STEM}.out"
LOG_ERR="${LOG_STEM}.err"
echo "Stdout: $LOG_OUT"
echo "Stderr: $LOG_ERR"
echo "Output: $OUTPUT_ROOT"

# Unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Assemble overrides
BASE_OVERRIDES=(
  "env=$ENV_NAME"
  "trial=$TRIAL_NAME"
  "save_artifacts=true"
  "save_checkpoints=true"
)

# Optional overrides
[[ -n "$EPOCHS" ]] && BASE_OVERRIDES+=("trainer.epochs=$EPOCHS")

RUN_COMMAND="python -u -m experiments.run_trial"
for arg in "${BASE_OVERRIDES[@]}"; do
  RUN_COMMAND+=" $(printf '%q' "$arg")"
done

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-workdir="$PROJECT_ROOT" \
  --container-mounts="$PROJECT_ROOT":"$PROJECT_ROOT",/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro \
  bash -lc "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export RUN_TRIAL_DIR='$OUTPUT_ROOT'
    echo Running: $RUN_COMMAND
    echo Output dir: \$RUN_TRIAL_DIR
    eval \"$RUN_COMMAND\"
  " >"$LOG_OUT" 2>"$LOG_ERR"
