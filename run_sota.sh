#!/bin/bash 
#SBATCH --job-name=madaug_main_cifra100_WRN_42
#SBATCH --output=results/logs/%x_%j.log     
#SBATCH --error=results/errors/%x_%j.err
#SBATCH --mail-user=
#SBATCH --mail-type=ALL                                                                              
#SBATCH --partition=STUD                
#SBATCH --gres=gpu:1                                                                                 
                                                                                                       
source ~/miniconda3/etc/profile.d/conda.sh
conda activate curraug

# Run MADAug model on cifar 100 then updated LPS 
# python -m experiments.madaug.train_madaug --seed 42

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m experiments.madaug.train_madaug --seed 42 --epochs 200 --stop_after_epochs 2 --out_dir results/madaug_memtest