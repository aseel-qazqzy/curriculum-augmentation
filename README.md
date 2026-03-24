
## Project Overview
This thesis implements **Curriculum-Style Data Augmentation** for image classification. Rather than applying random augmentations uniformly, the approach progressively increases augmentation difficulty during training:
- **Early training**: Easy augmentations (horizontal/vertical flips, random crops)
- **Progressive stages**: Intermediate augmentations (color jitter, rotation)
- **Advanced stages**: Complex augmentations (cutout, mixup)

The curriculum can be adapted by class imbalance, per-instance difficulty, or training loss signals. The research evaluates how dynamic augmentation schedules impact model convergence, robustness to corruptions, and generalization on standard benchmarks (CIFAR-10/100, ImageNet).

## Key Research Questions
- How does progressive augmentation difficulty affect training dynamics?
- Which adaptation strategies (class, instance, loss-based) work best?
- What is the trade-off between convergence speed and final model robustness?

## To plot the curves
python analysis/plot_curves.py
python analysis/compare_methods.py
python analysis/visualize_schedule.py

### Recommended Priority
Priority 1 — Finish CL on CIFAR-10     
Priority 2 — Add CIFAR-100               
Priority 3 — Add ResNet-34                
Priority 4 — Tiny ImageNet               

------------------------------
  # 1. NoAugmentation
  python experiments/train_baseline.py \
    --dataset cifar10 --augmentation none \
    --experiment_name resnet18_no_aug_cifar10 \
    --epochs 150 --lr 0.1 --optimizer sgd --scheduler multistep

  # 2. StaticAugmentation
  python experiments/train_baseline.py \
    --dataset cifar10 --augmentation static \
    --experiment_name resnet18_static_aug_cifar10 \
    --epochs 150 --lr 0.1 --optimizer sgd --scheduler multistep

  # 3. RandomAugmentation (RQ2 ablation — your custom baseline)
  python experiments/train_baseline.py \
    --dataset cifar10 --augmentation random \
    --experiment_name resnet18_random_aug_cifar10 \
    --epochs 150 --lr 0.1 --optimizer sgd --scheduler multistep

  # 4. RandAugment (Cubuk et al. — citable external baseline)
  python experiments/train_baseline.py \
    --dataset cifar10 --augmentation randaugment \
    --ra_n 2 --ra_m 9 \
    --experiment_name resnet18_randaugment_N2_M9_cifar10 \
    --epochs 150 --lr 0.1 --optimizer sgd --scheduler multistep

  Smoke test before committing to 150 epochs

  python experiments/train_baseline.py --augmentation randaugment --debug



  Usage examples for train_baseline:
    python experiments/train_baseline.py --augmentation none
    python experiments/train_baseline.py --augmentation static
    python experiments/train_baseline.py --augmentation random
    python experiments/train_baseline.py --augmentation randaugment --ra_n 2 --ra_m 9
    python experiments/train_baseline.py --dataset cifar100 --augmentation randaugment --ra_n 2 --ra_m 14
    python experiments/train_baseline.py --debug