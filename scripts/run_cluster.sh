#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job script — Curriculum Augmentation Experiments
# University HPC Cluster (CUDA) + local MacBook Pro (MPS/CPU) compatible
#
# Usage on cluster:
#   sbatch scripts/run_cluster.sh
#
# Usage locally (runs the same command without SLURM):
#   bash scripts/run_cluster.sh
#
# Edit the CONFIG section below before submitting.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=curriculum_aug
#SBATCH --output=results/logs/slurm_%j.log
#SBATCH --error=results/logs/slurm_%j.err
#SBATCH --partition=gpu                  # adjust to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=asseel7723@gmail.com

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET="cifar10"          # cifar10 | cifar100 | tiny_imagenet
MODEL="resnet18"
AUGMENTATION="tiered_curriculum"   # none | standard | randaugment | tiered_curriculum
EPOCHS=100
BATCH_SIZE=128
LR=0.1
EXPERIMENT_NAME="${DATASET}_${MODEL}_${AUGMENTATION}"

# Curriculum-specific
FIXED_STRENGTH=0.7
TIER_T1=33
TIER_T2=66

# Set to 1 to enable AMP (float16) — auto-disabled on CPU/MPS, cluster only
USE_AMP=1
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
    # Cluster venv in home dir (install once with: python -m venv ~/venvs/curriculum_aug)
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

# ── Build argument list ───────────────────────────────────────────────────────
mkdir -p results/logs

ARGS=(
    --dataset        "$DATASET"
    --model          "$MODEL"
    --augmentation   "$AUGMENTATION"
    --epochs         "$EPOCHS"
    --batch_size     "$BATCH_SIZE"
    --lr             "$LR"
    --experiment_name "$EXPERIMENT_NAME"
    --fixed_strength "$FIXED_STRENGTH"
    --tier_t1        "$TIER_T1"
    --tier_t2        "$TIER_T2"
)

if [ "$USE_AMP" -eq 1 ]; then
    ARGS+=(--use_amp)
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Starting: python experiments/train_baseline.py ${ARGS[*]}"
python experiments/train_baseline.py "${ARGS[@]}" 2>&1 | tee "results/logs/${EXPERIMENT_NAME}.log"
