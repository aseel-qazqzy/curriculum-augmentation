#!/bin/bash
#SBATCH --job-name=s42_default_ets_lps
#SBATCH --output=results/logs/%x_%j.log
#SBATCH --error=results/errors/%x_%j.err
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# Re-train the DEFAULT seed-42 ETS and LPS models (WRN-28-10, CIFAR-100, 19-op, cosine, 100 ep).
# The seed-42 files in checkpoints/ were overwritten on 2026-06-07 by the ETS tier-boundary
# ablation (0.33/0.66) and the LPS window ablation (W=10). These runs go to checkpoints_s42_default/
# so nothing in checkpoints/ is touched. Needed for the t-SNE and Grad-CAM figures.
#
# Submit from ~/curriculum-augmentation:   sbatch run_s42_default.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate curraug
mkdir -p checkpoints_s42_default

# ── Tiered ETS, 100 epochs, seed 42 (default boundaries 0.20 / 0.45) ──────────
echo "Running tiered_ets seed 42, 100 epochs, cosine (default config)..." && python -m experiments.train_baseline --dataset cifar100 --model wideresnet --augmentation tiered_curriculum --tier_schedule ets --tier_t1 0.20 --tier_t2 0.45 --epochs 100 --scheduler cosine --warmup_epochs 5 --lr 0.1 --fixed_strength 0.7 --op_pool 19 --mix_mode both --mix_prob 0.5 --label_smoothing 0.0 --val_split 0.1 --use_amp --seed 42 --checkpoint_dir checkpoints_s42_default

# ── Tiered LPS, 100 epochs, seed 42 (default window 5) ────────────────────────
echo "Running tiered_lps seed 42, 100 epochs, cosine (default config)..." && python -m experiments.train_baseline --dataset cifar100 --model wideresnet --augmentation tiered_curriculum --tier_schedule lps --tier_t1 0.20 --tier_t2 0.45 --lps_tau 0.02 --lps_window 5 --lps_min_epochs 10 --epochs 100 --scheduler cosine --warmup_epochs 5 --lr 0.1 --fixed_strength 0.7 --op_pool 19 --mix_mode both --mix_prob 0.5 --label_smoothing 0.0 --val_split 0.1 --use_amp --seed 42 --checkpoint_dir checkpoints_s42_default
