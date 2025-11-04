#!/usr/bin/env bash
#SBATCH -J har-best
#SBATCH -p L40S
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00

# Container + project paths
PROJECT_ROOT=${PROJECT_ROOT:-/home/zolfaghari/har}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/netscratch/zolfaghari/images/har.sqsh}

########################################
# Best-run configuration (override via env)
########################################
ENV_NAME=${ENV_NAME:-remote}
DATA_NAME=${DATA_NAME:-mmfit}
SCENARIO_NAME=${SCENARIO_NAME:-scenario2}
EPOCHS=${EPOCHS:-200}
SEED=${SEED:-0}
BEST_OVERRIDES=${BEST_OVERRIDES:-}
RUN_LABEL=${RUN_LABEL:-}

# Output layout: experiments/best_run/<scenario>/<data>/<timestamp>/
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_ROOT/experiments/best_run/$SCENARIO_NAME/$DATA_NAME}
RUN_STAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR=${RUN_DIR:-$OUTPUT_ROOT/$RUN_STAMP}

# Logs
set -euo pipefail
set -x
mkdir -p "$OUTPUT_ROOT" "$RUN_DIR"
LOGDIR=/netscratch/$USER/experiments/log
mkdir -p "$LOGDIR"

export HYDRA_FULL_ERROR=1

DATESTAMP=$(date +%d%m%y-%I%P)
exec > >(tee -a "$LOGDIR/${DATESTAMP}_${RUN_LABEL}.out")
exec 2> >(tee -a "$LOGDIR/${DATESTAMP}_${RUN_LABEL}.err" >&2)

# Assemble Hydra overrides
BASE_OVERRIDES=(
  "env=$ENV_NAME"
  "data=$DATA_NAME"
  "scenario=$SCENARIO_NAME"
  "trainer.epochs=$EPOCHS"
  "seed=$SEED"
  "hydra.run.dir=$RUN_DIR"
)

if [[ -n "$BEST_OVERRIDES" ]]; then
  read -r -a EXTRA_OVERRIDES <<< "$BEST_OVERRIDES"
  BASE_OVERRIDES+=("${EXTRA_OVERRIDES[@]}")
fi

RUN_COMMAND="python -m experiments.run_trial"
for arg in "${BASE_OVERRIDES[@]}"; do
  RUN_COMMAND+=" $(printf '%q' "$arg")"
done

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-workdir="$PROJECT_ROOT" \
  --container-mounts="$PROJECT_ROOT":"$PROJECT_ROOT",/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro \
  bash -lc "
    set -euo pipefail
    echo Running: $RUN_COMMAND
    eval \"$RUN_COMMAND\"
    printf '%s\\n' \"$RUN_DIR\" > \"$OUTPUT_ROOT/latest_run.txt\"
  "
