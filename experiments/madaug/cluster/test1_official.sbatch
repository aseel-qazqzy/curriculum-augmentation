#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# MADAug Test 1 — run the UNTOUCHED official repository (software sanity check only).
#
#   1a. official code on CIFAR-100      -> expected to fail (no CIFAR-100 support); the error
#                                          is captured and Test 1 stops there for CIFAR-100.
#   1b. official code on its supported reduced_cifar10 config (main.sh values) for 2 epochs
#       with --bi_epochs 0 (author setting; a CLI argument, not a source edit), so the
#       bi-level step runs in epoch 1.
#
# The official source is cloned at the pinned commit and verified clean before running.
# NOT a thesis number. Submit from the project root:  sbatch experiments/madaug/cluster/test1_official.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=madaug_test1_official
#SBATCH --output=results/logs/%x_%j.log
#SBATCH --error=results/errors/%x_%j.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate curraug

PROJECT_ROOT="$(pwd)"
WORK="$PROJECT_ROOT/results/madaug/test1"
COMMIT=279b60a4d6e338aaa03ef0564a7bf94b8995f0b6
mkdir -p "$WORK" results/logs results/errors

echo "=== environment ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'EOF'
import torch, importlib.metadata as md
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cudnn", torch.backends.cudnn.version())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
for p in ("higher", "seaborn", "opencv-python", "opencv-python-headless", "scikit-learn", "Pillow"):
    try: print(p, md.version(p))
    except md.PackageNotFoundError: print(p, "MISSING")
EOF

# Official dependencies not in the curraug env: higher (bi-level), seaborn (utils.py import),
# opencv (adaptive_augmentor.py `from cv2 import magnitude`). Installed only if missing.
python -c "import higher" 2>/dev/null || pip install higher==0.2.1 --no-deps
python -c "import seaborn" 2>/dev/null || pip install seaborn
python -c "import cv2" 2>/dev/null || pip install opencv-python-headless

echo "=== clone official repository at pinned commit ==="
if [ ! -d "$WORK/MADAug" ]; then
    git clone https://github.com/JackHck/MADAug.git "$WORK/MADAug" || { echo "CLONE FAILED"; exit 1; }
fi
cd "$WORK/MADAug"
git checkout -q "$COMMIT"
echo "HEAD: $(git rev-parse HEAD)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "ERROR: official source is modified"; git status; exit 1
fi
echo "official source clean: yes"

echo "=== Test 1a: official code on CIFAR-100 (expected: unsupported) ==="
python main_higher.py --dataset cifar100 --dataroot "$PROJECT_ROOT/data" --model_name wresnet28_10 \
    --epochs 1 --batch_size 128 --save test1a_cifar100 2>&1 | tail -n 15
echo "Test 1a exit code: ${PIPESTATUS[0]}  (non-zero expected; CIFAR-100 needs an adaptation -> stop)"

echo "=== Test 1b: official reduced_cifar10 config (main.sh), 2 epochs, bi_epochs=0 ==="
nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 5 > "$WORK/test1b_gpu_mem.csv" &
SMI=$!
START=$(date +%s)
# Values from official main.sh run_reduced_cifar10, except --epochs 2 (smoke) and --bi_epochs 0 (author).
python main_higher.py --k_ops 2 --report_freq 10 --num_workers 4 --epochs 2 --batch_size 128 \
    --learning_rate 0.1 --dataset reduced_cifar10 --model_name wresnet28_10 --gpu 0 \
    --weight_decay 0.0005 --proj_learning_rate 0.001 --search_freq 3 --temperature 3 \
    --cutout --cutout_length 16 --bi_epochs 0 --dataroot "$PROJECT_ROOT/data" --save test1b_reduced_cifar10
RC=$?
END=$(date +%s)
kill $SMI 2>/dev/null
echo "Test 1b exit code: $RC | wall clock: $((END-START)) s"
echo "peak GPU memory used (MiB, nvidia-smi):"
tail -n +2 "$WORK/test1b_gpu_mem.csv" | awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print m}'
echo "official source still clean: $([ -z "$(git status --porcelain --untracked-files=no)" ] && echo yes || echo NO)"
echo "log: $(ls -d search/reduced_cifar10/*test1b_reduced_cifar10 2>/dev/null | tail -1)/log.txt"
