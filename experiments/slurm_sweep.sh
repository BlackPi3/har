#!/usr/bin/env bash
#SBATCH -J har-sweep
#SBATCH -p A100-40GB
#SBATCH -o /netscratch/zolfaghari/experiments/log/slurm-%x-%j.out
#SBATCH -e /netscratch/zolfaghari/experiments/log/slurm-%x-%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00

set -euo pipefail
set -x

# --- User knobs (override via env) ---
PROJECT_ROOT=/home/$USER/har
CONTAINER_IMAGE=/netscratch/$USER/images/har.sqsh

# Number of Optuna trials (override with N_TRIALS=... when submitting)
N_TRIALS=${N_TRIALS:-2}

# HPO space to use (override with HPO=... when submitting)
HPO=${HPO:-scenario2_mmfit}

# Ensure log directory exists
mkdir -p /netscratch/$USER/experiments/log
# Pre-create per-study output dir so Optuna SQLite parent exists
mkdir -p "/netscratch/$USER/experiments/output/$HPO"

# Sanity checks (optional): enable with DEBUG=1
DEBUG=${DEBUG:-0}
## Limit enroot/unsquashfs parallelism to reduce peak RAM during container extract
export ENROOT_MAX_PROCESSORS=${ENROOT_MAX_PROCESSORS:-1}
if [ "${DEBUG:-0}" != "0" ]; then
  echo "[DEBUG] Host: $(hostname)  User: $USER  Date: $(date)"
  echo "[DEBUG] PROJECT_ROOT=$PROJECT_ROOT"
  echo "[DEBUG] CONTAINER_IMAGE=$CONTAINER_IMAGE"
  echo "[DEBUG] N_TRIALS=$N_TRIALS"
  echo "[DEBUG] HPO=$HPO"
fi

if [ ! -f "$CONTAINER_IMAGE" ]; then
  echo "Error: container image not found: $CONTAINER_IMAGE" >&2
  exit 1
fi

if [ "${DEBUG:-0}" != "0" ]; then
  echo "[DEBUG] Running Python package preflight inside container..."
  srun \
    --container-image="$CONTAINER_IMAGE" \
    --container-workdir="$PROJECT_ROOT" \
    --container-mounts="$PROJECT_ROOT":"$PROJECT_ROOT",/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro \
    python - <<'PY'
import sys, platform
print('Python:', sys.version)
print('Platform:', platform.platform())
try:
    import hydra
    print('hydra OK')
except Exception as e:
    print('hydra import failed:', e)
try:
    import optuna
    print('optuna OK')
except Exception as e:
    print('optuna import failed:', e)
PY
fi

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-workdir="$PROJECT_ROOT" \
  --container-mounts="$PROJECT_ROOT":"$PROJECT_ROOT",/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro \
  python -m experiments.run -m \
    env=remote scenario=scenario2 data=mmfit +hpo=$HPO \
    hydra/sweeper=optuna \
    hydra.sweeper.n_trials=$N_TRIALS \
    trainer.epochs=10 \
    seed=0