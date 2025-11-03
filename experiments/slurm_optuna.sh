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
# Choose mode: HPO (study via conf/hpo) or single trial (scenario via conf/scenario)
HPO=${HPO:-scenario2_mmfit}
SCENARIO=${SCENARIO:-scenario2}
N_TRIALS=${N_TRIALS:-10}

if [ -n "$HPO" ] && [ -n "$SCENARIO" ]; then
  echo "Error: Set either HPO=<name> (study) or SCENARIO=<name> (single trial), not both." >&2
  exit 2
fi

# Common defaults
OVERRIDES=${OVERRIDES:-env=remote data=mmfit trainer.epochs=5}

# HPO defaults
if [ -n "$HPO" ]; then
  OUTPUT_ROOT=${OUTPUT_ROOT:-/netscratch/$USER/experiments/output/$HPO}
  STORAGE=${STORAGE:-$OUTPUT_ROOT/$HPO.db}
  SPACE_CONFIG=${SPACE_CONFIG:-conf/hpo/$HPO.yaml}
fi

# Single-trial defaults
if [ -n "$SCENARIO" ]; then
  OUTPUT_ROOT=${OUTPUT_ROOT:-/netscratch/$USER/experiments/output/$SCENARIO}
fi

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
    if [ -n "'"$HPO"'" ]; then
      # HPO/study mode: require conf/hpo/$HPO.yaml
      if [ ! -f "'"$SPACE_CONFIG"'" ]; then
        echo "Missing space config: '"$SPACE_CONFIG"'" >&2
        exit 3
      fi
      mkdir -p '"$OUTPUT_ROOT"'
      python -m experiments.run_optuna \
        --n-trials '"$N_TRIALS"' \
        --storage '"$STORAGE"' \
        --output-root '"$OUTPUT_ROOT"' \
        --space-config '"$SPACE_CONFIG"' \
        -- -- $OVERRIDES
    elif [ -n "'"$SCENARIO"'" ]; then
      # Single trial mode: run the scenario directly
      RUN_DIR='"$OUTPUT_ROOT"'/'"$SCENARIO"'_'"$DATESTAMP"'
      mkdir -p '"$OUTPUT_ROOT"'
      python -m experiments.run_trial \
        hydra.run.dir='"$RUN_DIR"' \
        $OVERRIDES \
        scenario='"$SCENARIO"'
    else
      echo "Error: Set HPO=<name> for study or SCENARIO=<name> for single trial." >&2
      exit 2
    fi
  '
