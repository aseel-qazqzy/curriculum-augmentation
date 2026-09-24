#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CONTROLLED-PROTOCOL RERUN of Static Mixing / ETS / LPS / EGS for the MADAug comparison.
# NEW runs — they never replace or mix with the original baseline results.
#   split     : fixed 49k/1k/10k (experiments/madaug/splits/cifar100_val1000.json)
#   reporting : final epoch
#   outputs   : results/controlled_baselines/{checkpoints,logs,metrics}/  (run names end in _ctrl49k)
#
# One method x one seed per job:
#   sbatch --export=METHOD=ets,SEED=42 experiments/madaug/controlled_baselines/run_controlled.sbatch
# METHOD in {static_mixing, ets, lps, egs}; SEED in {42, 123, 456, 3407, 1024}
#
# Run protocol_test.py first:  python -m experiments.madaug.controlled_baselines.protocol_test
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=ctrl49k
#SBATCH --output=results/controlled_baselines/logs/slurm_%x_%j.log
#SBATCH --error=results/controlled_baselines/logs/slurm_%x_%j.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate curraug
mkdir -p results/controlled_baselines/logs

: "${METHOD:?set METHOD=static_mixing|ets|lps|egs}"
: "${SEED:?set SEED}"

case "$METHOD" in
    static_mixing) AUG=(--augmentation static_mixing --mix_mode both) ;;
    ets|lps|egs)   AUG=(--augmentation tiered_curriculum --tier_schedule "$METHOD" --mix_mode both) ;;
    *) echo "unknown METHOD=$METHOD" >&2; exit 1 ;;
esac

python -m experiments.madaug.controlled_baselines.train_controlled "${AUG[@]}" \
    --dataset cifar100 --model wideresnet --epochs 200 --scheduler cosine \
    --warmup_epochs 5 --lr 0.1 --use_amp --seed "$SEED"
