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
HPO=${HPO:-scenario2_utd}
# Guidance: number of trials vs search space size (categorical combos)
# - Tiny space (<1e3 combos): consider GridSampler, set N_TRIALS to full grid if affordable.
# - Small space (~1e3–1e4 combos): TPE with 50–200 trials often suffices; increase if runs are cheap.
# - Large space (>1e4 combos): TPE/random with a fixed budget (e.g., 100–300) and monitor convergence.
# Adjust N_TRIALS accordingly for your budget and space size.
N_TRIALS=${N_TRIALS:-64}

SPACE_CONFIG=${SPACE_CONFIG:-conf/hpo/$HPO.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-/netscratch/zolfaghari/experiments/hpo/$HPO}
STORAGE=${STORAGE:-$OUTPUT_ROOT/$HPO.db}
# Explicit run-time config (no aggregated overrides)
ENV_NAME=${ENV_NAME:-remote}
# EPOCHS=${EPOCHS:-50} # --epochs $EPOCHS
LOG_ROOT=${LOG_ROOT:-/netscratch/zolfaghari/experiments/log/hpo}
RUN_TOPK=${RUN_TOPK:-1}
TOPK_BASE_SEED=${TOPK_BASE_SEED:-0}

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
      --env $ENV_NAME
    if [[ \"$RUN_TOPK\" == \"1\" ]]; then
      python -m experiments.run_topk \
        --study-name $HPO \
        --space-config '$SPACE_CONFIG' \
        --topk-source-root '$OUTPUT_ROOT' \
        --env $ENV_NAME \
        --base-seed $TOPK_BASE_SEED
    fi
  " >"$LOG_OUT" 2>"$LOG_ERR"
