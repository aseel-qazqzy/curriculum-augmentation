# Curriculum-Style Data Augmentation for Image Classification

**Master's Thesis — University of Hildesheim**

---

## Overview

This repository contains the implementation for my thesis investigating whether the *order* in which data augmentations are introduced during training affects model generalization. The central hypothesis is that a curriculum-based schedule — presenting simple augmentations early and progressively introducing harder transformations — provides a more favourable learning signal than applying all augmentations at full strength from the start.

Both the baseline (static augmentation) and the proposed method (tiered curriculum) use the same set of seven augmentation operations at the same fixed strength (0.7). The only variable is *when* each operation enters the training pipeline.

---

## Project Structure

```
curriculum-augmentation/
├── augmentations/
│   ├── primitives.py          — individual augmentation operations (flip, crop, cutout, etc.)
│   ├── policies.py            — static and 3-tier curriculum augmentation policies
│   ├── curriculum.py          — CurriculumDataset (loss-guided variant)
│   ├── schedules.py           — epoch-level difficulty scheduling
│   └── randaugment.py         — RandAugment baseline
├── data/
│   └── datasets.py            — CIFAR-10/100 data loaders with train/val/test split
├── experiments/
│   ├── train_baseline.py      — training script for static aug and tiered curriculum
│   ├── train.py               — training script for loss-guided curriculum
│   └── config.py              — shared hyperparameter configuration
├── training/
│   ├── trainer.py             — training loop (standard and curriculum-aware)
│   └── losses.py              — loss functions and per-sample difficulty scoring
├── models/
│   └── registry.py            — model factory (ResNet-18, ResNet-50, WideResNet, etc.)
├── analysis/
│   ├── plot_curves.py         — validation accuracy and loss curves
│   ├── compare_methods.py     — summary table and bar chart comparisons
│   └── visualize_schedule.py  — augmentation schedule visualiser
├── results/
│   ├── logs/                  — per-run training logs (.log)
│   └── figures/               — exported figures (.png)
└── checkpoints/               — saved model weights and training histories
```

---

## Augmentation Design

### Operation Pool

Seven operations are shared across all methods. The tiered curriculum introduces them in three stages; the static baseline applies all seven from epoch 1.

| Tier   | Operations                              | Introduced at epoch |
|--------|-----------------------------------------|---------------------|
| Easy   | horizontal flip, random crop            | 1                   |
| Medium | + color jitter, rotation, shear         | 34                  |
| Hard   | + grayscale, cutout                     | 67                  |

**Design notes:**
- `FIXED_STRENGTH = 0.7` is applied uniformly so that augmentation strength is not a confound.
- Gaussian blur is excluded: at 32×32 resolution it creates a deterministic distribution shift between training and validation splits that cannot be controlled for cleanly.

---

## Experimental Setup

| Hyperparameter       | Value                                          | Source       |
|----------------------|------------------------------------------------|--------------|
| Dataset              | CIFAR-100                                      | `config.py`  |
| Model                | ResNet-50                                      | `config.py`  |
| Training epochs      | 100                                            | `config.py`  |
| Batch size           | 128                                            | `config.py`  |
| Optimiser            | SGD, lr = 0.1, weight decay = 5 × 10⁻⁴        | `config.py`  |
| LR scheduler         | MultiStepLR, milestones = [33, 66, 83], γ = 0.1 | `config.py` |
| Augmentation strength | 0.7                                           | `policies.py`|
| Validation split     | 10% of training set (5,000 images)             | `config.py`  |
| Random seed          | 42                                             | `config.py`  |

---

## Training

### Static Augmentation (main baseline)

```bash
python experiments/train_baseline.py \
  --dataset cifar100 \
  --model resnet50 \
  --augmentation static \
  --experiment_name resnet50_static_aug_cifar100_v2
```

### Tiered Curriculum Learning (proposed method)

```bash
python experiments/train_baseline.py \
  --dataset cifar100 \
  --model resnet50 \
  --augmentation tiered_curriculum \
  --experiment_name resnet50_tiered_curriculum_cifar100
```

### No Augmentation (lower-bound baseline)

```bash
python experiments/train_baseline.py \
  --dataset cifar100 \
  --model resnet50 \
  --augmentation none \
  --experiment_name resnet50_no_aug_cifar100
```

### Running both conditions sequentially

MPS does not support parallel training runs; use sequential execution:

```bash
python experiments/train_baseline.py \
  --dataset cifar100 --model resnet50 --augmentation static \
  --experiment_name resnet50_static_aug_cifar100_v2 && \
python experiments/train_baseline.py \
  --dataset cifar100 --model resnet50 --augmentation tiered_curriculum \
  --experiment_name resnet50_tiered_curriculum_cifar100
```

### Quick validation (2 epochs — sanity check before a full run)

```bash
python experiments/train_baseline.py \
  --dataset cifar100 \
  --model resnet50 \
  --augmentation static \
  --experiment_name resnet50_static_aug_cifar100_v2 \
  --epochs 2
```

### Debug mode (2 epochs, 512 samples)

```bash
python experiments/train_baseline.py --augmentation static --debug
```

---

## Analysis and Visualisation

Run these scripts after both training runs have completed.

### 1. Validation accuracy and loss curves

```bash
python analysis/plot_curves.py --mode all
```

Output: `results/figures/fig1_val_comparison_all.png`  
Displays static augmentation versus tiered curriculum learning curves with LR decay markers.

### 2. Overfitting analysis (train–validation gap)

```bash
python analysis/plot_curves.py --mode all
```

Output: `results/figures/fig2_overfitting_all.png`  
Tracks the gap between training and validation accuracy over the full training run.

### 3. Generalisation gap (validation loss over epochs)

```bash
python analysis/plot_curves.py --mode all
```

Output: `results/figures/fig3_generalization_all.png`

### 4. Method comparison table and bar chart

```bash
python analysis/compare_methods.py
```

Output: `results/figures/fig_compare_methods.png`  
Prints a summary table of best validation accuracy, test top-1, test top-5, and validation–test gap for each condition.

### 5. Baselines only

```bash
python analysis/plot_curves.py --mode baselines
```

### 6. Augmentation schedule visualiser

```bash
python analysis/visualize_schedule.py
```

Illustrates how augmentation difficulty increases across epochs for each schedule type.

---

## Checkpoints

Each completed run writes two files to `checkpoints/`:

| File                                | Contents                                             |
|-------------------------------------|------------------------------------------------------|
| `{experiment_name}_best.pth`        | Best model weights with val\_acc, test\_top1, test\_top5 |
| `{experiment_name}_history.pt`      | Full per-epoch training history (losses, accuracies) |

---

## Expected Results

Results below are indicative targets for CIFAR-100 with ResNet-50 at augmentation strength 0.7.

| Method                        | Best Val Acc. | Test Top-1   | Val–Test Gap |
|-------------------------------|---------------|--------------|--------------|
| No Augmentation               | %       | %      | %        |
| Static Augmentation           | %       | %      | %        |
| **Tiered Curriculum (proposed)** | **~%** | **~%** | **~%**   |

---

## Research Questions

1. Does progressively introducing augmentation operations outperform applying them all from the first epoch?
2. Is any observed improvement attributable to the curriculum ordering, or to the choice of operations alone?
3. Does curriculum augmentation reduce overfitting as measured by the training–validation accuracy gap?
