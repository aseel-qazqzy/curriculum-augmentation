# Curriculum-Style Data Augmentation for Image Classification

**Master's Thesis — University of Hildesheim**

---

## Overview

This repository investigates whether the *order* in which augmentation operations are introduced during training affects model generalisation on image classification. The central hypothesis is that a 3-tier curriculum — starting with simple geometric ops and progressively adding harder photometric ops — provides a better learning signal than applying all operations from epoch 1.

Three scheduling mechanisms are compared for advancing curriculum tiers, alongside a static mixing baseline. All methods use the same 19-op pool at the same fixed strength (0.7); the only variable is *when* each operation enters training.

---

## Project Structure

```
curriculum-augmentation/
├── augmentations/
│   ├── primitives.py          — low-level op implementations (flip, crop, color_jitter, cutout, etc.)
│   ├── policies.py            — NoAug, Static, StaticMixing, RandAugment, ThreeTierCurriculumAugmentation
│   ├── mixing.py              — BatchMixer: CutMix + MixUp (Tier 3 only)
│   ├── curriculum.py          — CurriculumDataset (index-aware dataset wrapper for EGS)
│   └── schedules.py           — tier strength ramp helpers
├── data/
│   └── datasets.py            — CIFAR-10/100 and Tiny ImageNet loaders
├── experiments/
│   ├── train_baseline.py      — main entry point for all runs (baseline + curriculum)
│   ├── config.py              — BASE_CONFIG shared across all scripts
│   ├── utils.py               — set_seed, get_device, build_optimizer, build_scheduler
│   ├── rank_aug_ops.py        — rank ops by val-loss delta; produces aug_op_ranking.json
│   └── compute_entropy.py     — per-sample entropy scoring for EGS
├── training/
│   ├── trainer.py             — train_one_epoch, evaluate, compute_training_entropy
│   └── losses.py              — LabelSmoothingLoss, LossPlateauScheduler
├── models/
│   └── registry.py            — get_model factory (ResNet-18/50, WideResNet-28-10, PyramidNet)
├── analysis/
│   ├── plot_curves.py         — validation accuracy and loss curves
│   ├── compare_methods.py     — summary table and bar chart across methods/seeds
│   ├── visualize_schedule.py  — augmentation schedule visualiser
│   └── wideresnet_cifar100/   — per-run analysis markdown files
├── scripts/
│   └── run_cluster.sh         — SLURM submission script for university cluster
├── results/
│   ├── logs/                  — per-run training logs
│   └── figures/               — exported figures
└── checkpoints/               — saved model weights and training histories
```

---

## Augmentation Design

### 19-Op Pool — 3 Tiers

| Tier | Pool (cumulative) | Ops sampled | Strength | Default epoch range (100 ep) |
|------|-------------------|-------------|----------|------------------------------|
| 1 | flip, crop, translate_x, translate_y | 3 of 4 | 0.28 | 1–20 |
| 2 | + color_jitter, rotation, shear, auto_contrast, equalize, sharpness, perspective | 5 of 11 | 0.49 (ramp 5 ep) | 21–45 |
| 3 | + grayscale, cutout, contrast, brightness, blur, solarize, posterize, invert | 8 of 19 | 0.70 (ramp 5 ep) | 46–100 |

- Strength ceiling: `fixed_strength=0.7`. Tiers 1 and 2 scale to 40% and 70% of ceiling.
- Tier 3 adds **CutMix + MixUp** batch mixing (`mix_mode=both`, alpha=1.0, p=0.5).
- Ops are randomly subsampled each batch — same tier, different subset per image.

### Static Mixing Baseline

Samples 8 ops from the full 19-op pool from epoch 1 at full strength. Identical op set and sample count to Tier 3 — the only variable vs curriculum is the progression, not the operations.

---

## Scheduling Mechanisms

### ETS — Epoch-Threshold Scheduling
Fixed epoch boundaries: `--tier_t1 0.20 --tier_t2 0.45` (fraction of total epochs).  
Deterministic, reproducible. Default for thesis runs.

### LPS — Loss-Plateau Scheduling
Advances tier when validation loss improvement over a sliding window drops below `lps_tau=0.02`.  
Parameters: `--lps_tau 0.02 --lps_window 5 --lps_min_epochs 10`.  
Requires `--val_split 0.1` (never use `--val_split 0.0` with LPS).

### EGS — Entropy-Guided Scheduling
Per-sample advancement based on prediction entropy. Samples with plateauing entropy advance independently through tiers.  
Parameters: `--egs_update_freq 5 --egs_min_epochs_per_tier 20 --egs_max_epochs_per_tier 40`.

---

## Experimental Setup

| Hyperparameter | Value | Source |
|---|---|---|
| Dataset | CIFAR-100 | `config.py` |
| Model | WideResNet-28-10 | `config.py` |
| Training epochs | 100 | `config.py` |
| Batch size | 128 | `config.py` |
| Optimiser | SGD, lr=0.1, wd=5×10⁻⁴ | `config.py` |
| LR scheduler | CosineAnnealingLR + 5-ep linear warmup | `config.py` |
| Augmentation strength | 0.7 | `policies.py` |
| Validation split | 0.1 (45k train / 5k val / 10k test) | `config.py` |
| Seeds | 42, 123, 456 | multi-seed sweep |

---

## Training

### Debug mode (2 epochs, 512 samples — sanity check)

```bash
python -m experiments.train_baseline --augmentation tiered_curriculum \
    --tier_schedule ets --dataset cifar10 --model resnet18 --debug
```

### Full thesis run matrix (WideResNet, CIFAR-100, 3 seeds each)

```bash
# Static mixing baseline
python -m experiments.train_baseline --dataset cifar100 --model wideresnet \
    --augmentation static_mixing --epochs 100 --scheduler cosine \
    --warmup_epochs 5 --lr 0.1 --use_amp --use_wandb --seed 42

# Tiered curriculum — ETS
python -m experiments.train_baseline --dataset cifar100 --model wideresnet \
    --augmentation tiered_curriculum --tier_schedule ets --epochs 100 \
    --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --use_wandb --seed 42

# Tiered curriculum — LPS
python -m experiments.train_baseline --dataset cifar100 --model wideresnet \
    --augmentation tiered_curriculum --tier_schedule lps --epochs 100 \
    --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --use_wandb --seed 42

# Tiered curriculum — EGS
python -m experiments.train_baseline --dataset cifar100 --model wideresnet \
    --augmentation tiered_curriculum --tier_schedule egs --epochs 100 \
    --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --use_wandb --seed 42
```

Replace `--seed 42` with `--seed 123` and `--seed 456` for the full 3-seed sweep.

### Resume from checkpoint

```bash
python -m experiments.train_baseline ... --resume checkpoints/<name>_best.pth
```

### Op ranking (loss-based tier ordering)

```bash
python -m experiments.rank_aug_ops \
    --checkpoint checkpoints/<name>_best.pth \
    --dataset cifar100 \
    --output results/aug_op_ranking.json
```

Pass `--op_ranking_file results/aug_op_ranking.json` to any training run to use loss-ranked op ordering and per-op calibrated strengths instead of the manual tier design.

---

## Analysis

### Validation curves

```bash
python analysis/plot_curves.py --mode all
```

### Method comparison table and bar chart

```bash
python analysis/compare_methods.py
```

Output: `results/figures/fig_compare_methods.png` — best val acc, test top-1, test top-5, val–test gap per method.

### Augmentation schedule visualiser

```bash
python analysis/visualize_schedule.py
```

---

## Checkpoints

Each completed run writes two files to `checkpoints/`:

| File | Contents |
|---|---|
| `{experiment_name}_best.pth` | Best model weights + val_acc, test_top1, test_top5 |
| `{experiment_name}_history.pt` | Full per-epoch history (train_loss, train_acc, val_loss, val_acc, val_top5) |

Experiment names are auto-built: `{model}_{aug}_{optimizer}_{scheduler}_ep{N}_{dataset}_s{seed}_p{pool_size}`.  
Example: `wideresnet_tiered_ets_mix_both_sgd_cosine_ep100_cifar100_s42_p19`

---

## Results

WideResNet-28-10 on CIFAR-100, dev mode (val_split=0.1), seed 42.

| Method | Scheduler | Best Val Top-1 | Test Top-1 | Val–Test Gap |
|---|---|---|---|---|
| Static Mixing | cosine | — | — | — |
| Tiered ETS | cosine | 81.96% | 81.84% | 0.12% |
| Tiered LPS | cosine | — | — | — |
| Tiered EGS | cosine | — | — | — |

Multi-seed results (seeds 42/123/456) in progress.

---

## Research Questions

1. Does progressively introducing augmentation operations outperform applying them all from epoch 1?
2. Which scheduling signal (fixed epochs, loss plateau, entropy plateau) produces the best tier advancement strategy?
3. Does curriculum augmentation reduce overfitting as measured by the training–validation accuracy gap?

---

## Changelog

### 2026-05-15
- **augmentations/policies.py** — `tier_label()` now derives op names and pool sizes from `_tier_ops` at runtime; startup print fully dynamic from `_TIER_OPS`; tier activation printed immediately when ETS crosses a boundary
- **experiments/rank_aug_ops.py** — new script: rank ops by val-loss delta, produce `aug_op_ranking.json` for loss-based tier ordering
- **analysis/wideresnet_cifar100/C1_tiered_ets_cosine_wr_s42.md** — scheduler ablation: cosine_wr vs plain cosine on ETS, WideResNet, CIFAR-100. cosine_wr collapsed 27.88pp at epoch 50 due to LR restart coinciding with Tier 3 activation. cosine_wr ruled out.
