#!/bin/bash
#SBATCH --job-name=figs_s42_default
#SBATCH --output=results/logs/%x_%j.log
#SBATCH --error=results/errors/%x_%j.err
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# Regenerate the t-SNE and Grad-CAM figures from the default-config seed-42 models.
# Run after run_s42_default.sh has finished (ETS + LPS in checkpoints_s42_default/).
# The unaffected seed-42 checkpoints (No Aug, Static Mixing, EGS) are linked in from checkpoints/.
#
# Submit from ~/curriculum-augmentation:   sbatch run_figs_s42_default.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate curraug
mkdir -p checkpoints_s42_default

for f in wideresnet_none_sgd_cosine_ep100_cifar100_s42_best.pth \
         wideresnet_static_mixing_sgd_cosine_ep100_cifar100_s42_p19_best.pth \
         egs_v2_100ep_s42_ep100_cifar100_s42_p19_best.pth; do
    ln -sfn "../checkpoints/$f" "checkpoints_s42_default/$f"
done

echo "── Checkpoints used ──"
python analysis/inspect_ckpt_history.py --checkpoint_dir checkpoints_s42_default --pattern "*s42*"

echo "── Grad-CAM ──" && python analysis/grade_cam.py --checkpoint_dir checkpoints_s42_default --skip_tiny_imagenet
echo "── t-SNE ──" && python analysis/tsne_features.py --checkpoint_dir checkpoints_s42_default
