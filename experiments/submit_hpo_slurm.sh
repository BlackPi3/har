#!/usr/bin/env bash
# Submit Optuna HPO on SLURM. Edit SBATCH fields to match your cluster.
# Usage:
#   sbatch experiments/submit_hpo_slurm.sh
#
# You can also override any variable below via environment, e.g.:
#   STUDY_NAME=my_study N_TRIALS=100 OUTPUT_ROOT=/netscratch/$USER/har/hpo \
#   sbatch experiments/submit_hpo_slurm.sh

#SBATCH -J har-hpo
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
#SBATCH -p gpu                 # EDIT: partition/queue name
#SBATCH --gres=gpu:1           # EDIT: number/type of GPUs
#SBATCH -c 8                   # EDIT: CPU cores
#SBATCH --mem=32G              # EDIT: memory
#SBATCH -t 24:00:00            # EDIT: walltime

# Optional: if your cluster supports containerized jobs via Slurm
# Uncomment and set your image or pass via env CONTAINER_IMAGE
# #SBATCH --container-image=docker://pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

set -euo pipefail

# --- Config (overridable via env) ---
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STUDY_NAME=${STUDY_NAME:-mmfit_sc2}
N_TRIALS=${N_TRIALS:-50}
METRIC=${METRIC:-val_f1}
DIRECTION=${DIRECTION:-maximize}
SEED=${SEED:-0}

# Absolute scratch paths recommended on cluster
OUTPUT_ROOT=${OUTPUT_ROOT:-/netscratch/$USER/experiments/output/hpo}
STORAGE=${STORAGE:-/netscratch/$USER/experiments/optuna/${STUDY_NAME}.db}

# Search space file in repo (can be overridden)
SEARCH_SPACE=${SEARCH_SPACE:-$PROJECT_ROOT/conf/hpo/scenario2_mmfit.yaml}

# Hydra overrides forwarded to each trial (adjust env/data_dir as needed)
OVERRIDES=${OVERRIDES:-"env=remote data=mmfit experiment=scenario2 trainer.epochs=30"}

# If you want to install the package into the job environment, set to 1.
# If 0, we'll rely on PYTHONPATH pointing to repo root (works for this repo).
USE_PIP_INSTALL=${USE_PIP_INSTALL:-0}

# --- Runtime ---
cd "$PROJECT_ROOT"

# If running in a container, you can wrap the command with srun --container-image
RUN_PREFIX=()
if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
  RUN_PREFIX=(srun --container-image="$CONTAINER_IMAGE")
fi

# Ensure parent dirs exist (run_optuna also creates subdirs per-trial)
mkdir -p "$(dirname "$STORAGE")"
mkdir -p "$OUTPUT_ROOT"

# Optional editable install; otherwise ensure repo is on PYTHONPATH
if [[ "$USE_PIP_INSTALL" == "1" ]]; then
  python3 -m pip install --user -e .
else
  export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
fi

# Launch HPO (sequential trials in a single job)
"${RUN_PREFIX[@]}" python3 experiments/run_optuna.py \
  --n-trials "$N_TRIALS" \
  --metric "$METRIC" --direction "$DIRECTION" \
  --study-name "$STUDY_NAME" \
  --storage "$STORAGE" \
  --output-root "$OUTPUT_ROOT" \
  --search-space "$SEARCH_SPACE" \
  --seed "$SEED" \
  $OVERRIDES

# Notes:
# - To RESUME, submit again with the same STUDY_NAME and STORAGE; Optuna will append trials.
# - For PARALLEL sweeps across multiple jobs, use a proper RDB (Postgres/MySQL) instead of SQLite
#   or keep concurrency low; SQLite has limited write concurrency.
# - If your cluster uses Singularity instead of Slurm's container plugin, you can replace RUN_PREFIX with:
#     singularity exec --nv <image.sif> python3 ...
