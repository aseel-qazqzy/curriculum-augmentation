#!/bin/bash 
#SBATCH --job-name=svhn_full_ets_tier_wideresnet_all_seeds_dev
#SBATCH --output=results/logs/%x_%j.log     
#SBATCH --error=results/errors/%x_%j.err
#SBATCH --mail-user=
#SBATCH --mail-type=ALL                                                                              
#SBATCH --partition=STUD                
#SBATCH --gres=gpu:1

source ~/miniconda3/etc/profile.d/conda.sh  
conda activate curraug


# #------------ Static mixing, 200 epochs ------

# echo "Running static_mixing seed 42, 200 epochs, cosine..." && python -m experiments.train_baseline --dataset svhn --model wideresnet --augmentation static_mixing --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 42
# echo "Running static_mixing seed 123, 100 epochs, cosine..." && python -m experiments.train_baseline --dataset cifar100 --model wideresnet --augmentation static_mixing --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 123
# echo "Running static_mixing seed 456, 100 epochs, cosine..." && python -m experiments.train_baseline --dataset cifar100 --model wideresnet --augmentation static_mixing --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 456
# echo "Running static_mixing seed 3407, 100 epochs, cosine..." && python -m experiments.train_baseline --dataset cifar100 --model wideresnet --augmentation static_mixing --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 3407
# echo "Running static_mixing seed 1024, 100 epochs, cosine..." && python -m experiments.train_baseline --dataset cifar100 --model wideresnet --augmentation static_mixing --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 1024

# ── Tiered ETS, 200 epochs ───────────────────────────────────────────────
echo "Running full tiered_ets seed 42, 200 epochs, cosine..." && python -m experiments.train_baseline --dataset svhn --model wideresnet --augmentation tiered_curriculum --tier_schedule ets --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 42 --val_split 0.0
echo "Running full tiered_ets seed 123, 200 epochs, cosine..." && python -m experiments.train_baseline --dataset svhn --model wideresnet --augmentation tiered_curriculum --tier_schedule ets --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 123 --val_split 0.0
echo "Running full tiered_ets seed 456, 200 epochs, cosine..." && python -m experiments.train_baseline --dataset svhn --model wideresnet --augmentation tiered_curriculum --tier_schedule ets --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 456 --val_split 0.0
echo "Running full tiered_ets seed 3407, 200 epochs, cosine..." && python -m experiments.train_baseline --dataset svhn --model wideresnet --augmentation tiered_curriculum --tier_schedule ets --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 3407 --val_split 0.0
echo "Running full tiered_ets seed 1024, 200 epochs, cosine..." && python -m experiments.train_baseline --dataset svhn --model wideresnet --augmentation tiered_curriculum --tier_schedule ets --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 1024 --val_split 0.0

# # ── Tiered LPS, 200 epochs ───────────────────────────────────────────────
# echo "Running tiered_lps seed 42, 200 epochs, cosine..." && python -m experiments.train_baseline --dataset svhn --model wideresnet --augmentation tiered_curriculum --tier_schedule lps --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 42
# echo "Running tiered_lps seed 123, 100 epochs, cosine..." && python -m experiments.train_baseline --dataset svhn --model wideresnet --augmentation tiered_curriculum --tier_schedule lps --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 123
# echo "Running tiered_lps seed 456, 100 epochs, cosine..." && python -m experiments.train_baseline --dataset svhn --model wideresnet --augmentation tiered_curriculum --tier_schedule lps --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 456
# echo "Running tiered_lps seed 3407, 100 epochs, cosine..." && python -m experiments.train_baseline --dataset svhn --model wideresnet --augmentation tiered_curriculum --tier_schedule lps --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 3407
# echo "Running tiered_lps seed 1024, 100 epochs, cosine..." && python -m experiments.train_baseline --dataset svhn --model wideresnet --augmentation tiered_curriculum --tier_schedule lps --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 1024


# # ── Tiered EGS 100 epochs ───────────────────────────────────────────────
# echo "Running tiered_egs_v2 seed 42, 100 epochs, cosine..." && python -m experiments.train_baseline --dataset cifar100 --model resnet50 --augmentation tiered_curriculum --tier_schedule egs --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --egs_update_freq 3 --egs_min_epochs_per_tier 10 --egs_max_epochs_per_tier 25 --egs_max_promote_frac 0.10 --egs_mix_threshold 0.50 --egs_mix_min_epoch 45 --mix_alpha 0.2 --label_smoothing 0.1 --use_amp --seed 42 --experiment_name egs_v2_resnet50_19op_100ep_s42

# echo "Running tiered_egs_v2 seed 123, 100 epochs, cosine..." && python -m experiments.train_baseline --dataset cifar100 --model resnet50 --augmentation tiered_curriculum --tier_schedule egs --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --egs_update_freq 3 --egs_min_epochs_per_tier 10 --egs_max_epochs_per_tier 25 --egs_max_promote_frac 0.10 --egs_mix_threshold 0.50 --egs_mix_min_epoch 45 --mix_alpha 0.2 --label_smoothing 0.1 --use_amp --seed 123 --experiment_name egs_v2_resnet50_19op_100ep_s123
  
# echo "Running tiered_egs_v2 seed 456, 100 epochs, cosine..." && python -m experiments.train_baseline --dataset cifar100 --model resnet50 --augmentation tiered_curriculum --tier_schedule egs --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --egs_update_freq 3 --egs_min_epochs_per_tier 10 --egs_max_epochs_per_tier 25 --egs_max_promote_frac 0.10 --egs_mix_threshold 0.50 --egs_mix_min_epoch 45 --mix_alpha 0.2 --label_smoothing 0.1 --use_amp --seed 456 --experiment_name egs_v2_resnet50_19op_100ep_s456

# echo "Running tiered_egs_v2 seed 3407, 100 epochs, cosine..." && python -m experiments.train_baseline --dataset cifar100 --model resnet50 --augmentation tiered_curriculum --tier_schedule egs --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --egs_update_freq 3 --egs_min_epochs_per_tier 10 --egs_max_epochs_per_tier 25 --egs_max_promote_frac 0.10 --egs_mix_threshold 0.50 --egs_mix_min_epoch 45 --mix_alpha 0.2 --label_smoothing 0.1 --use_amp --seed 3407 --experiment_name egs_v2_resnet50_19op_100ep_s3407

# echo "Running tiered_egs_v2 seed 1024, 100 epochs, cosine..." && python -m experiments.train_baseline --dataset cifar100 --model resnet50 --augmentation tiered_curriculum --tier_schedule egs --epochs 200 --scheduler cosine --warmup_epochs 5 --lr 0.1 --egs_update_freq 3 --egs_min_epochs_per_tier 10 --egs_max_epochs_per_tier 25 --egs_max_promote_frac 0.10 --egs_mix_threshold 0.50 --egs_mix_min_epoch 45 --mix_alpha 0.2 --label_smoothing 0.1 --use_amp --seed 1024 --experiment_name egs_v2_resnet50_19op_100ep_s1024
