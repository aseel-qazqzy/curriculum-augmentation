#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# RandAugment / TrivialAugment external baselines under the controlled CIFAR-100 protocol
# of the MADAug comparison (49k/1k split, WRN-28-10, 200 ep, final-epoch reporting).
# NEW runs — they never replace or mix with existing Static/ETS/LPS/EGS/MADAug results.
#   outputs : results/external_baselines/{checkpoints,logs,metrics,configs}/
#
# One method x one seed per job:
#   sbatch --export=METHOD=randaugment,SEED=42 experiments/external_baselines/run_external.sbatch
# METHOD in {randaugment, trivialaugment}; SEED in {42, 123, 456, 3407, 1024}
# After a time limit, resubmit with RESUME=1:
#   sbatch --export=METHOD=randaugment,SEED=42,RESUME=1 experiments/external_baselines/run_external.sbatch
#
# Run the sanity test first:  python -m experiments.external_baselines.sanity_test
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=extbase
#SBATCH --output=results/external_baselines/logs/slurm_%x_%j.log
#SBATCH --error=results/external_baselines/logs/slurm_%x_%j.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate curraug
mkdir -p results/external_baselines/logs

: "${METHOD:?set METHOD=randaugment|trivialaugment}"
: "${SEED:?set SEED}"
EXTRA=()
if [ "${RESUME:-0}" = "1" ]; then EXTRA+=(--resume); fi

python -m experiments.external_baselines.train_external --method "$METHOD" --seed "$SEED" ${EXTRA[@]+"${EXTRA[@]}"}
