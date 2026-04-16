# Curriculum-Style Data Augmentation for Image Classification

Thesis project — Hildesheim University.

The core idea: instead of applying all augmentations at full strength from epoch 1 (static aug),
introduce them progressively in tiers. The model learns easy features first, then harder ones.
Same ops, same strength — only the ordering changes.

---

## Project Structure

```
curriculum-augmentation/
├── augmentations/
│   ├── primitives.py        — individual augmentation ops (flip, crop, cutout, etc.)
│   ├── policies.py          — static aug + 3-tier curriculum policies
│   ├── curriculum.py        — CurriculumDataset (loss-guided, advanced)
│   ├── schedules.py         — epoch-level difficulty schedules
│   └── randaugment.py       — RandAugment baseline
├── data/
│   └── datasets.py          — CIFAR-10/100 data loaders (train/val/test split)
├── experiments/
│   ├── train_baseline.py    — training script for static aug + tiered CL
│   ├── train.py             — training script for loss-guided CL
│   └── config.py            — shared base config
├── training/
│   ├── trainer.py           — training loop (standard + CL)
│   └── losses.py            — loss functions + difficulty scoring
├── models/
│   └── registry.py          — model factory (resnet18, resnet50, wideresnet, etc.)
├── analysis/
│   ├── plot_curves.py       — training curve figures
│   ├── compare_methods.py   — comparison table + bar charts
│   └── visualize_schedule.py— augmentation schedule visualizer
├── results/
│   ├── logs/                — training logs (.log files)
│   └── figures/             — saved plots (.png files)
└── checkpoints/             — saved model checkpoints (_best.pth, _history.pt)
```

---

## Augmentation Design

### Op Pool (7 ops, shared by both methods)

| Tier | Ops | Introduced at |
|------|-----|---------------|
| Easy   | flip, crop | epoch 1 |
| Medium | + color_jitter, rotation, shear | epoch 34 (tiered CL only) |
| Hard   | + grayscale, cutout | epoch 67 (tiered CL only) |

- **Static aug**: all 7 ops applied from epoch 1
- **Tiered CL**: ops introduced progressively at tier boundaries
- `FIXED_STRENGTH = 0.7` — same strength for all ops in both methods
- blur excluded: deterministic blur on 32×32 images creates train/val distribution shift

---

## Training Commands

### Static Augmentation (main baseline)

```bash
python experiments/train_baseline.py \
  --dataset cifar100 \
  --model resnet50 \
  --augmentation static \
  --experiment_name resnet50_static_aug_cifar100_v2
```

### 3-Tier Curriculum Learning (proposed method)

```bash
python experiments/train_baseline.py \
  --dataset cifar100 \
  --model resnet50 \
  --augmentation tiered_curriculum \
  --experiment_name resnet50_tiered_curriculum_cifar100
```

### Run both sequentially (recommended — MPS cannot handle parallel runs)

```bash
python experiments/train_baseline.py --dataset cifar100 --model resnet50 --augmentation static --experiment_name resnet50_static_aug_cifar100_v2 && python experiments/train_baseline.py --dataset cifar100 --model resnet50 --augmentation tiered_curriculum --experiment_name resnet50_tiered_curriculum_cifar100
```

### No Augmentation (floor baseline)

```bash
python experiments/train_baseline.py \
  --dataset cifar100 \
  --model resnet50 \
  --augmentation none \
  --experiment_name resnet50_no_aug_cifar100
```

### Quick 2-epoch validation (verify fix before full run)

```bash
python experiments/train_baseline.py \
  --dataset cifar100 \
  --model resnet50 \
  --augmentation static \
  --experiment_name resnet50_static_aug_cifar100_v2 \
  --epochs 2
```

### Debug mode (2 epochs, 512 samples — smoke test)

```bash
python experiments/train_baseline.py --augmentation static --debug
```

---

## Visualisation Commands

Run these **after** both experiments finish.

### 1. Training curves (val accuracy + val loss overlaid)

```bash
python analysis/plot_curves.py --mode all
```

Generates: `results/figures/fig1_val_comparison_all.png`
Shows: static aug vs tiered CL learning curves side by side with LR decay markers.

### 2. Overfitting analysis (train vs val gap over epochs)

```bash
python analysis/plot_curves.py --mode all
```

Generates: `results/figures/fig2_overfitting_all.png`
Shows: train-val gap over training — curriculum should show lower overfitting.

### 3. Generalization gap (val loss over epochs)

```bash
python analysis/plot_curves.py --mode all
```

Generates: `results/figures/fig3_generalization_all.png`

### 4. Final comparison table + bar chart

```bash
python analysis/compare_methods.py
```

Generates: `results/figures/fig_compare_methods.png`
Prints: final results table with Best Val, Test Top-1, Test Top-5, Val-Test Gap.

### 5. Baseline only (static aug without tiered CL)

```bash
python analysis/plot_curves.py --mode baselines
```

### 6. Augmentation schedule visualizer

```bash
python analysis/visualize_schedule.py
```

Shows how difficulty increases over epochs for each schedule type.

---

## Key Config Values

| Parameter | Value | Where |
|-----------|-------|--------|
| Dataset | CIFAR-100 | `config.py` |
| Model | ResNet-50 | `config.py` |
| Epochs | 100 | `config.py` |
| Batch size | 128 | `config.py` |
| Optimizer | SGD, lr=0.1, wd=5e-4 | `config.py` |
| Scheduler | MultiStepLR, milestones=[33,66,83], gamma=0.1 | `config.py` |
| Augmentation strength | 0.7 | `policies.py` |
| Val split | 10% of train (5,000 images) | `config.py` |
| Seed | 42 | `config.py` |

---

## Checkpoints

Each completed run saves two files to `checkpoints/`:

| File | Contents |
|------|----------|
| `{experiment_name}_best.pth` | Best model weights + val_acc + test_top1 + test_top5 |
| `{experiment_name}_history.pt` | Full training history (train_loss, val_acc, etc. per epoch) |

---

## Expected Results (CIFAR-100, ResNet-50, strength=0.7)

| Method | Best Val | Test Top-1 | Val-Test Gap |
|--------|----------|------------|--------------|
| No Augmentation | ~55–60% | ~54–58% | ~1–2% |
| Static Aug (all ops, epoch 1) | ~68–72% | ~66–70% | ~2–3% |
| **Tiered CL (progressive)** | **~75–79%** | **~73–77%** | **~1–2%** |

---

## Research Questions

1. Does progressive introduction of augmentations outperform applying all at once?
2. Is the improvement due to ordering, or just the choice of operations?
3. Does curriculum augmentation reduce overfitting (lower train-val gap)?
