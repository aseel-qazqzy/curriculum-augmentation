#!/bin/bash 
#SBATCH --job-name=lps_wideresnet_model_tiny_imagenet_dev
#SBATCH --output=results/logs/%x_%j.log     
#SBATCH --error=results/errors/%x_%j.err
#SBATCH --mail-user=
#SBATCH --mail-type=ALL                                                                              
#SBATCH --partition=STUD                
#SBATCH --gres=gpu:1                                                                                 
                                                                                                       
source ~/miniconda3/etc/profile.d/conda.sh  
conda activate curraug

# ------- Tiny imagenet static mixing seeds(123, 456) ----------------
# python -m experiments.train_baseline --dataset tiny_imagenet --model wideresnet --augmentation static_mixing --epochs 100 --scheduler cosine --use_amp --seed 123
# python -m experiments.train_baseline --dataset tiny_imagenet --model wideresnet --augmentation static_mixing --epochs 100 --scheduler cosine --use_amp --seed 456
# python -m experiments.train_baseline --dataset tiny_imagenet --model wideresnet --augmentation static_mixing --epochs 100 --scheduler cosine --use_amp --seed 3407
# python -m experiments.train_baseline --dataset tiny_imagenet --model wideresnet --augmentation static_mixing --epochs 100 --scheduler cosine --use_amp --seed 1024
# ------- Tiny imagenet ets seeds(123, 456) ----------------
# python -m experiments.train_baseline --dataset tiny_imagenet --model wideresnet --augmentation tiered_curriculum --tier_schedule ets --epochs 100 --scheduler cosine --use_amp --seed 123
# python -m experiments.train_baseline --dataset tiny_imagenet --model wideresnet --augmentation tiered_curriculum --tier_schedule ets --epochs 100 --scheduler cosine --use_amp --seed 456
# python -m experiments.train_baseline --dataset tiny_imagenet --model wideresnet --augmentation tiered_curriculum --tier_schedule ets --epochs 100 --scheduler cosine --use_amp --seed 3407
# python -m experiments.train_baseline --dataset tiny_imagenet --model wideresnet --augmentation tiered_curriculum --tier_schedule ets --epochs 100 --scheduler cosine --use_amp --seed 1024

# ------- Tiny imagenet lps seeds(123, 456) ----------------
# python -m experiments.train_baseline --dataset tiny_imagenet --model wideresnet --augmentation tiered_curriculum --tier_schedule lps --epochs 100 --scheduler cosine --use_amp --seed 123
# python -m experiments.train_baseline --dataset tiny_imagenet --model wideresnet --augmentation tiered_curriculum --tier_schedule lps --epochs 100 --scheduler cosine --use_amp --seed 456
python -m experiments.train_baseline --dataset tiny_imagenet --model wideresnet --augmentation tiered_curriculum --tier_schedule lps --epochs 100 --scheduler cosine --use_amp --seed 3407
python -m experiments.train_baseline --dataset tiny_imagenet --model wideresnet --augmentation tiered_curriculum --tier_schedule lps --epochs 100 --scheduler cosine --use_amp --seed 1024