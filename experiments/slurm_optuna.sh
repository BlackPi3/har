#!/usr/bin/env bash
#SBATCH -J har-optuna
#SBATCH -o netscratch/zolfaghari/experiments/log/slurm-%x-%j.out
#SBATCH -e netscratch/zolfaghari/experiments/log/slurm-%x-%j.err
#SBATCH -p A100-40GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH -t 24:00:00

set -euo pipefail

# --- User knobs (override via env) ---
PROJECT_ROOT=/home/$USER/har
CONTAINER_IMAGE=/netscratch/$USER/images/har.sqsh

EXPERIMENT=${EXPERIMENT:-scenario2_mmfit}
N_TRIALS=${N_TRIALS:-10}
METRIC=${METRIC:-val_f1}
DIRECTION=${DIRECTION:-maximize}
SEED=${SEED:-0}
SEARCH_SPACE=${SEARCH_SPACE:-$PROJECT_ROOT/conf/hpo/$EXPERIMENT.yaml}
OVERRIDES=${OVERRIDES:-"env=remote data=mmfit experiment=scenario2 trainer.epochs=30"}

OUTPUT_ROOT=${OUTPUT_ROOT:-/netscratch/$USER/experiments/output/hpo}
STORAGE=${STORAGE:-/netscratch/$USER/experiments/optuna/${EXPERIMENT}.db}

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$STORAGE")" "$OUTPUT_ROOT"

# Avoid CPU oversubscription (recommended on Pegasus)
# export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 USE_OPENMP=1  # 

# Run your Optuna launcher (does multiple trials sequentially)
srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-workdir="$(pwd)" \
  --container-mounts="$(pwd)":"$(pwd)",/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro \
  python3 experiments/run_optuna.py \
    --n-trials "$N_TRIALS" \
    --metric "$METRIC" --direction "$DIRECTION" \
    --study-name "$EXPERIMENT" \
    --storage "$STORAGE" \
    --output-root "$OUTPUT_ROOT" \
    --search-space "$SEARCH_SPACE" \
    --seed "$SEED" \
    $OVERRIDES