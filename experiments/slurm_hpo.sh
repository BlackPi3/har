#!/usr/bin/env bash
#SBATCH -J har-hpo
#SBATCH -p RTXA6000
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=40G
#SBATCH -t 03:00:00

# Container + project paths
PROJECT_ROOT=${PROJECT_ROOT:-/home/zolfaghari/har}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/netscratch/zolfaghari/images/har.sqsh}

########################################
# HPO configuration (study via conf/hpo)
########################################
# Defaults (override via environment variables as needed)
HPO_SPACE=${HPO_SPACE:-scenario2_utd} # scenario2_utd | scenario2_mmfit
STUDY_NAME=${STUDY_NAME:-$sc2_utd_coarse}  # sc2_utd_coarse | sc2_utd_fine | sc2_mmfit_coarse | sc2_mmfit_fine
N_TRIALS=${N_TRIALS:-100} # 100-150
SEARCH_MODE=${SEARCH_MODE:-coarse}  # coarse | fine

SPACE_CONFIG=${SPACE_CONFIG:-conf/hpo/$HPO_SPACE.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-/netscratch/zolfaghari/experiments/hpo/$STUDY_NAME}
STORAGE=${STORAGE:-$OUTPUT_ROOT/$STUDY_NAME.db}
ENV_NAME=remote
LOG_ROOT=${LOG_ROOT:-/netscratch/zolfaghari/experiments/log/hpo}
RUN_TOPK=${RUN_TOPK:-1}
TOPK_BASE_SEED=${TOPK_BASE_SEED:-0}

# Logs
set -euo pipefail
mkdir -p "$OUTPUT_ROOT"
mkdir -p "$LOG_ROOT"
timestamp=$(date +%y%m%d_%H%M)
LOG_STEM="$LOG_ROOT/hpo_${timestamp}_${STUDY_NAME}"
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
      --study-name $STUDY_NAME \
      --storage '$STORAGE' \
      --output-root '$OUTPUT_ROOT' \
      --space-config '$SPACE_CONFIG' \
      --search-mode $SEARCH_MODE \
      --env $ENV_NAME
    if [[ \"$RUN_TOPK\" == \"1\" && \"$SEARCH_MODE\" == \"fine\" ]]; then
      python -m experiments.run_topk \
        --study-name $HPO_SPACE \
        --space-config '$SPACE_CONFIG' \
        --topk-source-root '$OUTPUT_ROOT' \
        --env $ENV_NAME \
        --base-seed $TOPK_BASE_SEED
    fi
  " >"$LOG_OUT" 2>"$LOG_ERR"
