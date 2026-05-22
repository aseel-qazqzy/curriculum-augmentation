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
│   ├── primitives.py          — low-level op implementations (flip, crop, color_jitter, cutout …)
│   ├── policies.py            — NoAug, Static, StaticMixing, RandAugment, ThreeTierCurriculumAugmentation
│   ├── mixing.py              — BatchMixer: CutMix + MixUp (Tier 3 only)
│   ├── curriculum.py          — CurriculumDataset (index-aware dataset wrapper for EGS)
│   ├── schedules.py           — tier strength ramp helpers
│   ├── clip_scorer.py         — CLIPDifficultyScorer: frozen CLIP ViT-B/32 semantic distance
│   ├── clip_calibration.py    — offline script: score all 19 ops + mixing with CLIP
│   └── plot_clip_scores.py    — publication bar chart of CLIP scores grouped by tier
├── data/
│   └── datasets.py            — CIFAR-10/100 and Tiny-ImageNet loaders
├── experiments/
│   ├── train_baseline.py      — main entry point for all runs (baseline + curriculum)
│   ├── config.py              — BASE_CONFIG shared across all scripts
│   ├── utils.py               — set_seed, get_device, build_optimizer, build_scheduler
│   ├── compute_entropy.py     — per-sample entropy scoring for EGS
│   └── rank_aug_ops.py        — rank ops by val-loss delta; produces aug_op_ranking.json
├── training/
│   ├── trainer.py             — train_one_epoch, evaluate, compute_training_entropy
│   └── losses.py              — LabelSmoothingLoss, LossPlateauScheduler, EntropyPlateauScheduler
├── models/
│   └── registry.py            — get_model factory (ResNet-18/50, WideResNet-28-10, PyramidNet)
├── analysis/
│   ├── thesis_results_tables.md  — all experimental results with scientific findings
│   ├── thesis_writing.md         — ready-to-use thesis paragraphs per result section
│   └── related_works.md          — related paper survey with thesis positioning
├── configs/                   — pre-baked YAML configs (aug_difficulty_scores_*.pt generated here)
├── scripts/
│   └── run_cluster.sh         — SLURM submission script for university cluster
├── results/
│   ├── logs/                  — per-run training logs
│   └── figs/                  — exported figures
│       ├── clip_validation/   — CLIP difficulty bar charts
│       ├── training_curves/   — val acc / loss curves
│       ├── ablation/          — mixing, reverse curriculum, etc.
│       ├── architecture/      — ResNet-50 vs WideResNet
│       └── dataset/           — Tiny-ImageNet results
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
- Tier 3 adds **CutMix** batch mixing (`mix_mode=cutmix`, alpha=1.0, p=0.5) — CutMix alone outperforms CutMix+MixUp combined (+0.39pp, ablation confirmed).
- Ops are randomly subsampled each batch — same tier, different subset per image.
- Strength ramps linearly over 5 epochs at each tier boundary.

### CLIP Validation of Tier Design

The manual tier ordering is empirically validated using a frozen **CLIP ViT-B/32** model. All 19 ops receive a ✓ — CLIP's semantic difficulty ranking agrees with the manual assignment. Key finding: CLIP confirms solarize and blur are the hardest ops semantically, while flip and sharpness are the easiest.

```bash
# Run once offline to generate scores
python -m augmentations.clip_calibration --dataset cifar100

# Plot bar chart (grouped by tier, saved to results/figs/clip_validation/)
python -m augmentations.plot_clip_scores --dataset cifar100
```

---

## Scheduling Mechanisms

### ETS — Epoch-Threshold Scheduling
Fixed epoch boundaries: `--tier_t1 0.20 --tier_t2 0.45` (fraction of total epochs).
Deterministic, reproducible. **Best performing** on CIFAR-100: **81.32% ± 0.05%** (19-op, 3 seeds).

### LPS — Loss-Plateau Scheduling
Advances tier when validation loss improvement drops below `lps_tau=0.02` over a sliding window.
Parameters: `--lps_tau 0.02 --lps_window 5 --lps_min_epochs 10`.
Statistically equivalent to ETS: **81.35% ± 0.07%**. Requires `--val_split 0.1`.

### EGS — Entropy-Guided Scheduling (v2)
Per-sample advancement based on prediction entropy from a frozen pass over unaugmented data.
Recommended v2 parameters for 100-epoch runs:

```bash
--egs_update_freq 3 --egs_min_epochs_per_tier 10 --egs_max_epochs_per_tier 25 \
--egs_max_promote_frac 0.10 --egs_mix_threshold 0.50 --egs_mix_min_epoch 45 \
--mix_alpha 0.2 --label_smoothing 0.1
```

EGS v2 result: **80.01% ± 0.27%** (−1.31pp vs ETS, structural gap due to delayed T3 exposure).

### Reverse Curriculum (Ablation)
Hard→Easy ordering to verify the easy→hard direction is required:
```bash
--reverse_curriculum
```
Result: **78.17%** (−3.18pp vs ETS) — confirms ordering matters.

---

## Key Results (WideResNet-28-10 · CIFAR-100 · 19-op · 100 ep)

| Method | Seeds | Test Top-1 | Δ vs Static |
|---|---|---|---|
| No Augmentation | 1 | 72.86% | — |
| Static Mixing | 3 | 77.43% ± 0.44% | — |
| Tiered EGS v2 | 3 | 80.01% ± 0.27% | +2.58pp |
| Tiered ETS | 3 | **81.32% ± 0.05%** | **+3.89pp** |
| Tiered LPS | 3 | **81.35% ± 0.07%** | **+3.92pp** |

### Key Findings

1. **Curriculum advantage scales with op difficulty** — +0.01pp gain with 14 safe ops; +3.89pp with 19-op pool including blur/solarize/invert.
2. **ETS ≈ LPS** — fixed epochs and adaptive loss-plateau produce statistically identical results (Δ = 0.03pp).
3. **Ordering matters** — reverse curriculum drops 3.18pp; near-equivalent to static mixing.
4. **Mixing decomposition** — CutMix alone (+2.45pp) outperforms combined CutMix+MixUp (+2.06pp). Curriculum amplifies mixing: CutMix hurts static (−0.80pp) but helps ETS (+2.45pp).
5. **Architecture-agnostic** — +3.79pp gain on ResNet-50, matching WideResNet's +3.89pp.

---

## Experimental Setup

| Hyperparameter | Value |
|---|---|
| Primary dataset | CIFAR-100 |
| Primary model | WideResNet-28-10 |
| Training epochs | 100 |
| Batch size | 128 |
| Optimiser | SGD, lr=0.1, wd=5×10⁻⁴ |
| LR scheduler | CosineAnnealingLR + 5-ep linear warmup |
| Augmentation strength | 0.7 |
| Validation split | 0.1 (45k/5k/10k) |
| Seeds | 42, 123, 456 |
| Secondary model | ResNet-50 (architecture generalisation) |
| Secondary dataset | Tiny-ImageNet (dataset generalisation) |

---

## Training

### Debug mode (2 epochs, 512 samples)

```bash
python -m experiments.train_baseline --augmentation tiered_curriculum \
    --tier_schedule ets --dataset cifar10 --model resnet18 --debug
```

### Primary run matrix (WideResNet · CIFAR-100 · 3 seeds)

```bash
# Static mixing baseline
python -m experiments.train_baseline --dataset cifar100 --model wideresnet \
    --augmentation static_mixing --epochs 100 --scheduler cosine \
    --warmup_epochs 5 --lr 0.1 --use_amp --seed 42

# ETS
python -m experiments.train_baseline --dataset cifar100 --model wideresnet \
    --augmentation tiered_curriculum --tier_schedule ets --epochs 100 \
    --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 42

# LPS
python -m experiments.train_baseline --dataset cifar100 --model wideresnet \
    --augmentation tiered_curriculum --tier_schedule lps --epochs 100 \
    --scheduler cosine --warmup_epochs 5 --lr 0.1 --use_amp --seed 42

# EGS v2
python -m experiments.train_baseline --dataset cifar100 --model wideresnet \
    --augmentation tiered_curriculum --tier_schedule egs --epochs 100 \
    --scheduler cosine --warmup_epochs 5 --lr 0.1 \
    --egs_update_freq 3 --egs_min_epochs_per_tier 10 --egs_max_epochs_per_tier 25 \
    --egs_max_promote_frac 0.10 --egs_mix_threshold 0.50 --egs_mix_min_epoch 45 \
    --mix_alpha 0.2 --label_smoothing 0.1 --use_amp --seed 42
```

Replace `--seed 42` with `--seed 123` and `--seed 456` for the 3-seed sweep.

### Ablation runs

```bash
# Reverse curriculum
python -m experiments.train_baseline --dataset cifar100 --model wideresnet \
    --augmentation tiered_curriculum --tier_schedule ets --reverse_curriculum \
    --epochs 100 --scheduler cosine --use_amp --seed 42

# ETS no mixing
python -m experiments.train_baseline --dataset cifar100 --model wideresnet \
    --augmentation tiered_curriculum --tier_schedule ets --mix_mode none \
    --epochs 100 --scheduler cosine --use_amp --seed 42

# CutMix only
python -m experiments.train_baseline --dataset cifar100 --model wideresnet \
    --augmentation tiered_curriculum --tier_schedule ets --mix_mode cutmix \
    --epochs 100 --scheduler cosine --use_amp --seed 42
```

---

## Checkpoints

| File | Contents |
|---|---|
| `{name}_best.pth` | Best model weights + val_acc, test_top1, test_top5, cfg |
| `{name}_history.pt` | Full per-epoch history (train_loss, train_acc, val_loss, val_acc, val_top5) |

Auto-built name pattern: `{model}_{aug}_{optimizer}_{scheduler}_ep{N}_{dataset}_s{seed}_p{pool}`

- **Scheduler ablation** — cosine_wr collapsed 27.88pp at epoch 50 due to LR restart coinciding with Tier 3 activation; cosine_wr ruled out
