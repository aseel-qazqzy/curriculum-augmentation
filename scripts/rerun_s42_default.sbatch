#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Re-train the DEFAULT seed-42 ETS / LPS models (WRN-28-10, CIFAR-100, 19-op,
# cosine, 100 ep) into a separate checkpoint folder.
#
# Why: the seed-42 files in checkpoints/ were overwritten on 2026-06-07 by the
# ETS tier-boundary ablation (t1/t2 = 0.33/0.66) and the LPS window ablation
# (lps_window = 10), which save under the same experiment name. The t-SNE and
# Grad-CAM figures need the default-config models.
#
#   sbatch --export=SCHED=ets scripts/rerun_s42_default.sbatch
#   sbatch --export=SCHED=lps scripts/rerun_s42_default.sbatch
#
# Output: checkpoints_s42_default/  (never touches checkpoints/)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=s42default
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/slurm_s42default_%x_%j.log

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate curraug
mkdir -p results/logs checkpoints_s42_default

: "${SCHED:?set SCHED=ets|lps}"

python experiments/train_baseline.py \
    --model wideresnet --dataset cifar100 --epochs 100 \
    --augmentation tiered_curriculum --tier_schedule "$SCHED" \
    --tier_t1 0.20 --tier_t2 0.45 \
    --lps_tau 0.02 --lps_window 5 --lps_min_epochs 10 \
    --fixed_strength 0.7 --op_pool 19 \
    --mix_mode both --mix_prob 0.5 \
    --optimizer sgd --lr 0.1 --scheduler cosine --warmup_epochs 5 \
    --batch_size 128 --label_smoothing 0.0 \
    --val_split 0.1 --seed 42 --use_amp \
    --checkpoint_dir checkpoints_s42_default \
    --log_dir results/logs
