#!/usr/bin/env bash
set -euo pipefail

# --- config ---
PARTITION="A100-80GB"                       # change if needed (e.g., H100, H200, RTX3090)
BASE_IMG="/enroot/nvcr.io_nvidia_pytorch_25.06-py3.sqsh"
WORK_DIR="/home/zolfaghari/har"
SAVE_IMG="/netscratch/$USER/images/har.sqsh"
TIME="02:00:00"
GPUS=1
CPUS_PER_GPU=8
MEM="64G"                                   # ensure enough RAM for image modify/save
# -------------

mkdir -p "$(dirname "$SAVE_IMG")"

# run from your project root (has requirements.txt and pyproject.toml)
srun -p "$PARTITION" --time="$TIME" \
  --gpus="$GPUS" --cpus-per-gpu="$CPUS_PER_GPU" --mem="$MEM" \
  --container-image="$BASE_IMG" \
  --container-workdir="$WORK_DIR" \
  --container-mounts="$WORK_DIR":"$WORK_DIR",/netscratch/$USER:/netscratch/$USER \
  --container-save="$SAVE_IMG" \
  bash -lc '
    python -m pip install --upgrade pip wheel setuptools &&
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi &&
    pip install -e .
  '

echo "Saved image -> $SAVE_IMG"