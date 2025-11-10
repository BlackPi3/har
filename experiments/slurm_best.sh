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
TRIAL_NAME=${TRIAL_NAME:-scenario2_mmfit}
EPOCHS=${EPOCHS:-200}
SEED=${SEED:-0}
# Optional extra Hydra overrides appended verbatim to the run command.
# Example:
#   export BEST_OVERRIDES="optim.lr=1e-4 trainer.objective.metric=val_mse"
#   sbatch experiments/slurm_best.sh
BEST_OVERRIDES=${BEST_OVERRIDES:-}
RUN_DIR=${RUN_DIR:-$PROJECT_ROOT/experiments/best_run/$TRIAL_NAME}

# Logs
set -euo pipefail
mkdir -p "$RUN_DIR"

export HYDRA_FULL_ERROR=1

# Assemble overrides
BASE_OVERRIDES=(
  "env=$ENV_NAME"
  "data=$DATA_NAME"
  "trial=$TRIAL_NAME"
  "trainer.epochs=$EPOCHS"
  "seed=$SEED"
  "run.dir=$RUN_DIR"
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
  "
