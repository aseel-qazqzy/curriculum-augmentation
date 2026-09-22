#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job script — Augmentation Difficulty Validation (isolated diagnostic)
# University HPC Cluster (CUDA) + local MacBook Pro (MPS/CPU) compatible
#
# Does NOT touch ETS/LPS/EGS, static mixing, existing baselines, or any
# existing config/checkpoint/result file. Runs
# experiments/run_augmentation_difficulty.py only; outputs go to
# results/augmentation_difficulty/.
#
# Usage on cluster:
#   sbatch scripts/run_augmentation_difficulty.sh
#
# Usage locally (runs the same command without SLURM):
#   bash scripts/run_augmentation_difficulty.sh
#
# Edit the CONFIG section below before submitting.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=aug_difficulty
#SBATCH --output=results/augmentation_difficulty/slurm_%j.log
#SBATCH --error=results/augmentation_difficulty/slurm_%j.err
#SBATCH --partition=gpu                  # adjust to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=asseel7723@gmail.com

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET="cifar100"
MODEL="resnet18"
SEEDS="42 123 456"
WARMUP_EPOCHS=10          # matches lps_min_epochs in experiments/config.py
PROBE_EPOCHS=1
FIXED_STRENGTH=0.7        # matches FIXED_STRENGTH in augmentations/policies.py
BATCH_SIZE=128
LR=0.1
WEIGHT_DECAY=0.0005
VAL_SPLIT=0.1
NUM_WORKERS=4             # cluster has real cores; use 0 for local Mac runs
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Detect environment ────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if command -v squeue &>/dev/null; then
    RUNNING_ON_CLUSTER=1
else
    RUNNING_ON_CLUSTER=0
fi

# ── Load modules (cluster only) ───────────────────────────────────────────────
if [ "$RUNNING_ON_CLUSTER" -eq 1 ]; then
    # Adjust module names to match your cluster — check with: module avail
    module purge
    module load python/3.11
    module load cuda/12.1        # or cuda/11.8 — match your PyTorch build
    module load cudnn/8.9
fi

# ── Activate virtual environment ──────────────────────────────────────────────
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
elif [ -d "$HOME/venvs/curriculum_aug" ]; then
    source "$HOME/venvs/curriculum_aug/bin/activate"
else
    echo "ERROR: No virtual environment found. Create one first." >&2
    exit 1
fi

# ── Verify PyTorch sees the GPU ───────────────────────────────────────────────
python - <<'EOF'
import torch
if torch.cuda.is_available():
    print(f"CUDA: {torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}")
elif torch.backends.mps.is_available():
    print(f"MPS (Apple Silicon)  |  PyTorch {torch.__version__}")
else:
    print(f"CPU only  |  PyTorch {torch.__version__}")
EOF

# ── Run ───────────────────────────────────────────────────────────────────────
mkdir -p results/augmentation_difficulty

echo "Starting: python -m experiments.run_augmentation_difficulty ..."
python -m experiments.run_augmentation_difficulty \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --seeds $SEEDS \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --probe_epochs "$PROBE_EPOCHS" \
    --fixed_strength "$FIXED_STRENGTH" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --val_split "$VAL_SPLIT" \
    --num_workers "$NUM_WORKERS"
