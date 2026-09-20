#!/bin/bash 
#SBATCH --job-name=sota2_ets_static_cifra100_10_WRN_RESNET
#SBATCH --output=results/logs/%x_%j.log     
#SBATCH --error=results/errors/%x_%j.err
#SBATCH --mail-user=
#SBATCH --mail-type=ALL                                                                              
#SBATCH --partition=STUD                
#SBATCH --gres=gpu:1                                                                                 
                                                                                                       
source ~/miniconda3/etc/profile.d/conda.sh
conda activate curraug

# # ── WRN-28-10 · CIFAR-100 · Static Mixing (3 seeds) ──────────────────────────
# python -m experiments.train_baseline --augmentation static --mix_mode both --dataset cifar100 --model wideresnet --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --seed 42 --use_amp --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
# python -m experiments.train_baseline --augmentation static --mix_mode both --dataset cifar100 --model wideresnet --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --seed 123 --use_amp --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
# python -m experiments.train_baseline --augmentation static --mix_mode both --dataset cifar100 --model wideresnet --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --seed 456 --use_amp --checkpoint_dir checkpoints/sota --log_dir results/logs/sota

#   # ── WRN-28-10 · CIFAR-10 · Static Mixing (3 seeds) ───────────────────────────
# python -m experiments.train_baseline --augmentation static --mix_mode both --dataset cifar10 --model wideresnet --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 42 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
# python -m experiments.train_baseline --augmentation static --mix_mode both --dataset cifar10 --model wideresnet --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 123 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
# python -m experiments.train_baseline --augmentation static --mix_mode both --dataset cifar10 --model wideresnet --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 456 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota

# # ── WRN-28-10 · CIFAR-10 · ETS (5 seeds) ────────────────────────────────────
# python -m experiments.train_baseline --augmentation tiered_curriculum --tier_schedule ets --mix_mode both --dataset cifar10 --model wideresnet --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 42 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
# python -m experiments.train_baseline --augmentation tiered_curriculum --tier_schedule ets --mix_mode both --dataset cifar10 --model wideresnet --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 123 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
# python -m experiments.train_baseline --augmentation tiered_curriculum --tier_schedule ets --mix_mode both --dataset cifar10 --model wideresnet --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 456 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
# python -m experiments.train_baseline --augmentation tiered_curriculum --tier_schedule ets --mix_mode both --dataset cifar10 --model wideresnet --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 3407 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
# python -m experiments.train_baseline --augmentation tiered_curriculum --tier_schedule ets --mix_mode both --dataset cifar10 --model wideresnet --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 1024 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota

# # ── ResNet-50 · CIFAR-100 · Static Mixing (3 seeds) ──────────────────────────
# python -m experiments.train_baseline --augmentation static --mix_mode both --dataset cifar100 --model resnet50 --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 42 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
# python -m experiments.train_baseline --augmentation static --mix_mode both --dataset cifar100 --model resnet50 --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 123 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
# python -m experiments.train_baseline --augmentation static --mix_mode both --dataset cifar100 --model resnet50 --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 456 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota

# # ── ResNet-50 · CIFAR-100 · ETS (5 seeds) ────────────────────────────────────
# python -m experiments.train_baseline --augmentation tiered_curriculum --tier_schedule ets --mix_mode both --dataset cifar100 --model resnet50 --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 42 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
# python -m experiments.train_baseline --augmentation tiered_curriculum --tier_schedule ets --mix_mode both --dataset cifar100 --model resnet50 --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 123 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
# python -m experiments.train_baseline --augmentation tiered_curriculum --tier_schedule ets --mix_mode both --dataset cifar100 --model resnet50 --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 456 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
# python -m experiments.train_baseline --augmentation tiered_curriculum --tier_schedule ets --mix_mode both --dataset cifar100 --model resnet50 --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 3407 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
# python -m experiments.train_baseline --augmentation tiered_curriculum --tier_schedule ets --mix_mode both --dataset cifar100 --model resnet50 --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 1024 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota

# ── ResNet-50 · CIFAR-10 · Static Mixing (3 seeds) ───────────────────────────
# python -m experiments.train_baseline --augmentation static --mix_mode both --dataset cifar10 --model resnet50 --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 42 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
python -m experiments.train_baseline --augmentation static --mix_mode both --dataset cifar10 --model resnet50 --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 123 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
python -m experiments.train_baseline --augmentation static --mix_mode both --dataset cifar10 --model resnet50 --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 456 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
  
# ── ResNet-50 · CIFAR-10 · ETS (5 seeds) ─────────────────────────────────────
python -m experiments.train_baseline --augmentation tiered_curriculum --tier_schedule ets --mix_mode both --dataset cifar10 --model resnet50 --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 42 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
python -m experiments.train_baseline --augmentation tiered_curriculum --tier_schedule ets --mix_mode both --dataset cifar10 --model resnet50 --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 123 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
python -m experiments.train_baseline --augmentation tiered_curriculum --tier_schedule ets --mix_mode both --dataset cifar10 --model resnet50 --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 456 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
python -m experiments.train_baseline --augmentation tiered_curriculum --tier_schedule ets --mix_mode both --dataset cifar10 --model resnet50 --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 3407 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota
python -m experiments.train_baseline --augmentation tiered_curriculum --tier_schedule ets --mix_mode both --dataset cifar10 --model resnet50 --epochs 200 --val_split 0.0 --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 1024 --checkpoint_dir checkpoints/sota --log_dir results/logs/sota