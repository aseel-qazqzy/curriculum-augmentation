#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# MADAug Test 2 — "MADAug under our controlled protocol", full-data GPU smoke test.
#
#   0. sanity_test.py checks 1-4 on the cluster checkout (official copies, protected files,
#      wrapper, instrumented bi-level step)
#   A. seed 42, the REAL 200-epoch schedule, stopped after 2 epochs      (--stop_after_epochs 2)
#   B. resume the same run from its checkpoint and stop after epoch 3    (--resume --stop_after_epochs 3)
#   C. summary of the 11 report items
#
# Outputs go to results/madaug_smoke/ (run name suffix _smoke), never to results/madaug/.
# Submit from the project root:  sbatch experiments/madaug/cluster/test2_integrated_smoke.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=madaug_test2_smoke
#SBATCH --output=results/logs/%x_%j.log
#SBATCH --error=results/errors/%x_%j.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate curraug
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$SLURM_JOB_ID"

OUT=results/madaug_smoke
mkdir -p "$OUT" results/logs results/errors
python -c "import higher" 2>/dev/null || pip install higher==0.2.1 --no-deps

echo "=== environment ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "=== 0. sanity checks 1-4 ==="
python -m experiments.madaug.sanity_test --skip_smoke 2>&1 | grep -E "FAIL|checks passed"

nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 5 > "$OUT/gpu_mem_nvidia_smi.csv" &
SMI=$!

echo "=== A. epochs 0-1 of the 200-epoch schedule ==="
python -m experiments.madaug.train_madaug --seed 42 --epochs 200 --stop_after_epochs 2 --out_dir "$OUT"
echo "A exit code: $?"

echo "=== B. resume, epoch 2 ==="
python -m experiments.madaug.train_madaug --seed 42 --epochs 200 --stop_after_epochs 3 --resume --out_dir "$OUT"
echo "B exit code: $?"

kill $SMI 2>/dev/null

echo "=== C. summary ==="
python -m experiments.madaug.cluster.summarize_smoke --out_dir "$OUT" \
    --name madaug_cifar100_wrn28-10_ep200_s42_smoke --nvidia_smi_csv "$OUT/gpu_mem_nvidia_smi.csv"
