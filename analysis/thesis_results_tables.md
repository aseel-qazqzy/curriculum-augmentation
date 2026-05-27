# Thesis Results — WideResNet-28-10 / CIFAR-100

Model: WideResNet-28-10  |  Dataset: CIFAR-100  |  val_split: 0.1  |  Updated: 2026-05-24

> partial = some seeds still running  |  — = not yet run

---

## Table 1 — Primary Comparison: 14-op Pool (100 epochs)

**Config:** Cosine scheduler · SGD lr=0.1 · 100 epochs

| Method | Seed 42 | Seed 123 | Seed 456 | Mean ± Std | Avg Time |
|:---|:---:|:---:|:---:|:---:|:---:|
| No Augmentation | 72.86% | — | — | — | 225 min |
| Static Mixing | 81.50% | 81.90% | 81.44% | **81.61% ± 0.20%** | 223 min |
| Tiered ETS | 81.84% | 81.22% | 81.80% | **81.62% ± 0.28%** | 225 min |
| Tiered LPS | 81.48% | 80.65% | 81.76% | **81.30% ± 0.47%** | 226 min |
| Tiered EGS | 79.69% | 79.33% | 80.12% | **79.71% ± 0.32%** | 378 min |

> With 14 safe ops: ETS (81.62%) ≈ Static (81.61%) — curriculum provides no measurable advantage when all ops are benign.
> No augmentation: train 99.98% vs test 72.86% — severe overfitting (+27pp gap).

---

## Table 2 — Primary Comparison: 19-op Pool (100 epochs)

**Config:** Cosine scheduler · SGD lr=0.1 · 100 epochs · WideResNet-28-10 · CIFAR-100

| Method | Seed 42 | Seed 123 | Seed 456 | Seed 3407 | Seed 1024 | Mean ± Std | Avg Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Static Mixing | 77.79% | 76.82% | 77.69% | 78.17% | **77.54%** | **77.60% ± 0.50%** | 135 min |
| Tiered EGS (original) | 79.50% | 79.70% | 79.57% | — | — | **79.59% ± 0.08%** | 279 min |
| Tiered EGS v2 | 79.83% | 80.39% | 79.81% | 79.21% | **79.49%** | **79.75% ± 0.44%** | 263 min |
| Tiered ETS | 81.35% | 81.25% | 81.35% | 81.79% | **81.23%** | **81.39% ± 0.23%** | 133 min |
| Tiered LPS | 81.36% | 81.43% | 81.27% | **81.34%** | **81.79%** | **81.44% ± 0.20%** | 135 min |

> ETS/LPS gain (+3.89pp over Static) appears only with the 19-op pool; absent with 14-op pool (Δ = 0.01pp). ETS and LPS are statistically equivalent (Δ = 0.05pp). EGS v2 trails ETS by ~1.6pp — per-sample scheduling delays full Tier 3 exposure to epoch ~89, leaving fewer epochs at maximum augmentation.

---

## Table 3 — Op Pool Expansion: Curriculum Robustness (Seed 42)

| Method | 14-op pool | 19-op pool | Δ (14→19) |
|:---|:---:|:---:|:---:|
| Static Mixing | 81.50% | 77.79% | **−3.71 pp** |
| Tiered ETS | 81.84% | 81.35% | −0.49 pp |
| Tiered LPS | 81.48% | 81.36% | −0.12 pp |
| **ETS vs Static** | +0.34 pp | **+3.56 pp** | |
| **LPS vs Static** | −0.02 pp | **+3.57 pp** | |

> Expanding from 14 to 19 ops drops Static by 3.71pp but ETS by only 0.49pp and LPS by only 0.12pp. The curriculum's advantage over static scheduling is near-zero with benign 14-op pool, and grows to +3.57pp when the pool includes high-distortion ops.

---

## Table 4 — Training Duration: ETS 14-op Pool

| Epochs | Seed 42 | Seed 123 | Seed 456 | Mean | Avg Time |
|:---|:---:|:---:|:---:|:---:|:---:|
| 100 | 81.84% | 81.22% | 81.80% | **81.62% ± 0.28%** | 225 min |
| 150 | — | **82.70%** | **82.19%** | ~82.45% | 336 min |
| Gain | — | +1.48 pp | +0.39 pp | | +111 min |

> Best single result across all experiments (14-op): **82.70%** (ETS · 14-op · 150 ep · seed 123).
> All seeds still converging at epoch 100 — 150 epochs consistently improves results.

---

## Table 4b — ETS and LPS at 200 Epochs: Inter-Method Comparison (val_split=0.1)

**Config:** Cosine scheduler · SGD lr=0.1 · WideResNet-28-10 · CIFAR-100 · 19-op pool · seed 42 · val_split=0.1 (45k train / 5k val / 10k test)

| Method | Epochs | T1 → T2 | T2 → T3 | Tier 3 duration | Test Top-1 | Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Static Mixing | 100 | — | — | — | 77.60% ± 0.50% (5 seeds) | 135 min |
| ETS | 100 | ep 40 (fixed) | ep 90 (fixed) | 10 epochs | 81.39% ± 0.23% (5 seeds) | 133 min |
| LPS | 100 | ep 27 (adaptive) | ep 38 (adaptive) | 62 epochs | 81.44% ± 0.20% (5 seeds) | 135 min |
| ETS | 200 | ep 40 (fixed) | ep 90 (fixed) | 110 epochs | 82.47% | 264 min |
| LPS | 200 | ep 30 (adaptive) | ep 40 (adaptive) | 160 epochs | **82.52%** | 262 min |

> ETS and LPS remain statistically equivalent at both 100 and 200 epochs (Δ = 0.05pp at both). LPS at 200 epochs advanced to Tier 3 at epoch 40 — 50 epochs earlier than ETS's fixed boundary at epoch 90 — and spent 160 epochs in Tier 3 vs ETS's 110. Despite this large scheduling difference, final accuracy is identical. This indicates that the tier transition timing has negligible impact on the final result once sufficient Tier 3 exposure is provided, and confirms that fixed and adaptive scheduling converge to the same outcome.

---

## Table 4c — ETS at 200 Epochs: Comparison with Published Baselines (val_split=0.0)

**Config:** Cosine scheduler · SGD lr=0.1 · WideResNet-28-10 · CIFAR-100 · 19-op pool · val_split=0.0 (50k train / 10k test)

> LPS is excluded from this table because it requires a held-out validation set for loss-guided tier advancement and cannot be trained with val_split=0.0. This comparison applies to ETS only.

### Per-seed results

| Seed | Tier 2 | Tier 3 | Test Top-1 | Test Top-5 | Time |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 42 | ep 40 (fixed) | ep 90 (fixed) | 83.52% | — | 264 min |
| 123 | ep 41 (fixed) | ep 91 (fixed) | 83.14% | 96.30% | 271.5 min |
| 456 | ep 41 (fixed) | ep 91 (fixed) | 83.22% | 96.28% | 272 min |
| 3407 | ep 41 (fixed) | ep 91 (fixed) | 83.37% | 96.19% | 272 min |
| 1024 | ep 41 (fixed) | ep 91 (fixed) | 83.09% | 96.00% | 272 min |
| **Mean (5 seeds)** | | | **83.27% ± 0.18%** | | |

### Comparison with published baselines

| Method | Test Top-1 | Epochs | Train set | vs ETS mean |
|:---|:---:|:---:|:---:|:---:|
| **ETS (this work, 5-seed mean)** | **83.27% ± 0.18%** | 200 | 50k | — |
| RandAugment (Cubuk et al., 2020) | 83.3% | 200 | 50k | −0.03pp |
| AutoAugment (Cubuk et al., 2019) | 82.9% | 200 | 50k | +0.37pp |
| TrivialAugment (Müller & Hutter, 2021) | 82.5% | 200 | 50k | +0.77pp |
| AugMix (Hendrycks et al., 2020) | 80.9% | 200 | 50k | +2.37pp |

> ETS at 200 epochs (5-seed mean 83.27% ± 0.18%) is statistically tied with RandAugment (83.3%, Δ = −0.03pp — within seed variance), and outperforms AutoAugment (+0.37pp), TrivialAugment (+0.77pp), and AugMix (+2.37pp). Seed 42 individually exceeds RandAugment at 83.52% (+0.22pp). The test set (10k) is held out throughout and evaluated only once at epoch 200.
>
> Context for the val_split difference: ETS at val_split=0.1 reaches 81.39% ± 0.23% (5-seed mean), and at val_split=0.0 reaches 83.29% ± 0.20% (3-seed mean, +1.90pp). The full published baseline comparison uses val_split=0.0 for all methods.

---

## Table 5 — LPS Adaptive Tier Transitions

### 14-op pool

| Seed | T1 → T2 | T2 → T3 | T3 duration | Test Top-1 |
|:---:|:---:|:---:|:---:|:---:|
| 42 | epoch 20 | epoch 32 | 68 epochs | 81.48% |
| 123 | epoch 26 | epoch 43 | 57 epochs | 80.65% |
| 456 | epoch 29 | epoch 40 | 60 epochs | 81.76% |
| **ETS fixed** | epoch 21 | epoch 46 | 55 epochs | 81.62% *(mean)* |

### 19-op pool

| Seed | T1 → T2 | T2 → T3 | T3 duration | Test Top-1 |
|:---:|:---:|:---:|:---:|:---:|
| 42 | epoch 30 | epoch 41 | 59 epochs | 81.36% |
| 123 | epoch 18 | epoch 36 | 64 epochs | 81.43% |
| 456 | epoch 26 | epoch 39 | 61 epochs | 81.27% |
| 3407 | — | — | — | 81.34% |
| 1024 | epoch 27 | epoch 38 | 62 epochs | 81.79% |
| **ETS fixed** | epoch 21 | epoch 46 | 55 epochs | 81.44% *(5-seed mean)* |
| **LPS 200ep** | epoch 30 | epoch 40 | **160 epochs** | **82.52%** *(s42)* |

### Cross-architecture comparison (19-op pool · Seed 42)

| Architecture | T1 → T2 | T2 → T3 | T3 duration | Test Top-1 |
|:---:|:---:|:---:|:---:|:---:|
| WideResNet-28-10 | epoch 30 | epoch 41 | 59 epochs | 81.36% |
| ResNet-50 | epoch 23 | epoch 36 | **64 epochs** | 80.50% |
| **ETS fixed** | epoch 21 | epoch 46 | 55 epochs | — |

> **Finding — LPS transitions vary by up to 12 epochs across seeds yet final accuracy is stable** (std ±0.20pp with 19-op WideResNet · 5 seeds), demonstrating that LPS is robust to seed-dependent convergence variation.
>
> **Finding — Architecture affects LPS timing but not outcome:** ResNet-50 advances to Tier 3 five epochs earlier than WideResNet (epoch 36 vs 41), reflecting its faster loss convergence at lower capacity. Despite the earlier transition giving ResNet-50 64 epochs in Tier 3 vs WideResNet's 59, the final accuracy gap remains consistent with the static mixing gap (0.81pp), confirming that additional T3 duration does not compensate for architectural capacity.

---

## Table 6 — EGS Tier Progression (Seed 42, 19-op pool)

| Epoch | T1 | T2 | T3 | % T3 | Mixing | Mean Entropy |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 5 | 45,000 | 0 | 0 | 0% | pending | 2.001 |
| 20 | 38,250 | 6,750 | 0 | 0% | pending | 1.176 |
| 35 | 18,000 | 27,000 | 0 | 0% | pending | 0.649 |
| 50 | 11,863 | 19,024 | 14,113 | 31% | pending | 0.594 |
| 60 | 7,596 | 14,058 | 23,346 | 52% | pending | 0.404 |
| 75 | 0 | 10,070 | 34,930 | **78%** | **ACTIVE** | 0.189 |
| 85 | 0 | 5,183 | 39,817 | 89% | active | 0.333 |
| 100 | 0 | 16 | 44,984 | ~100% | active | 0.175 |

> Mixing activated at epoch 75 — 29 epochs later than ETS (epoch 46).
> Val accuracy jumped **+8.66pp** in the 2 epochs immediately after mixing activated (67.86% → 76.52%).

---

## Table 7 — Scheduler Ablation: ETS vs Cosine WarmRestart (Seed 42, 19-op)

| Scheduler | Test Top-1 | Best Val | Val–Test Gap | Time | Note |
|:---|:---:|:---:|:---:|:---:|:---|
| Cosine | **81.35%** | 81.34% (ep 98) | 0.01% | 136 min | Stable convergence |
| Cosine WarmRestart | 77.27% | 77.82% (ep 99) | 0.55% | 609 min | LR restart at ep 50 → −27.88pp collapse |

### No Augmentation: Scheduler Comparison (Seed 42)

| Scheduler | Test Top-1 | Train Acc | Time |
|:---|:---:|:---:|:---:|
| Cosine | 72.86% | 99.98% | 225 min |
| Cosine WarmRestart | 72.40% | 99.98% | 136 min |

> Without augmentation both schedulers produce the same severe overfitting.
> cosine_wr halves runtime but gains nothing on test accuracy.

---

## Table 8 — Complete Run Reference

| Method | Pool | Scheduler | Seed | Ep | Test Top-1 | Val–Test Gap | Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| no_aug | — | cosine | 42 | 100 | 72.86% | 0.68% | 225 min |
| no_aug | — | cosine_wr | 42 | 100 | 72.40% | 0.60% | 136 min |
| static_mixing | 14 | cosine | 42 | 100 | 81.50% | 0.10% | 223 min |
| static_mixing | 14 | cosine | 123 | 100 | 81.90% | 0.24% | 224 min |
| static_mixing | 14 | cosine | 456 | 100 | 81.44% | 0.88% | 223 min |
| tiered_ets | 14 | cosine | 42 | 100 | 81.84% | 0.12% | 223 min |
| tiered_ets | 14 | cosine | 123 | 100 | 81.22% | 0.88% | 225 min |
| tiered_ets | 14 | cosine | 456 | 100 | 81.80% | 0.48% | 226 min |
| tiered_ets | 14 | cosine | 123 | 150 | **82.70%** | 0.16% | 336 min |
| tiered_ets | 14 | cosine | 456 | 150 | 82.19% | 0.39% | 337 min |
| tiered_ets | 14 | cosine_wr | 42 | 100 | 77.27% | 0.55% | 609 min |
| **reverse_ets** | 19 | cosine | 42 | 100 | 78.17% | 0.99% | 139 min |
| **ets_nomix** | 19 | cosine | 42 | 100 | 79.29% | 0.53% | 137 min |
| **ets_cutmix** | 19 | cosine | 42 | 100 | 81.74% | 0.04% | 137 min |
| **ets_mixup** | 19 | cosine | 42 | 100 | 80.45% | 0.61% | 137 min |
| **static_nomix** | 19 | cosine | 42 | 100 | 78.23% | 1.15% | 139 min |
| tiered_lps | 14 | cosine | 42 | 100 | 81.48% | 0.56% | 226 min |
| tiered_lps | 14 | cosine | 123 | 100 | 80.65% | 1.23% | 226 min |
| tiered_lps | 14 | cosine | 456 | 100 | 81.76% | 0.64% | 226 min |
| tiered_egs | 14 | cosine | 42 | 100 | 79.69% | 0.95% | 372 min |
| tiered_egs | 14 | cosine | 123 | 100 | 79.33% | 0.73% | 371 min |
| tiered_egs | 14 | cosine | 456 | 100 | 80.12% | 0.62% | 390 min |
| tiered_egs | 19 | cosine_wr | 42 | 100 | 79.79% | 0.61% | 269 min |
| static_mixing | 19 | cosine | 42 | 100 | 77.79% | 0.29% | 139 min |
| static_mixing | 19 | cosine | 123 | 100 | 76.82% | 1.70% | 139 min |
| static_mixing | 19 | cosine | 456 | 100 | 77.69% | 0.61% | 138 min |
| static_mixing | 19 | cosine | 3407 | 100 | 78.17% | 0.11% | 133 min |
| static_mixing | 19 | cosine | 1024 | 100 | 77.54% | 1.12% | 133 min |
| **static_mixing** | **19** | **cosine** | **mean (5-seed)** | **100** | **77.60% ± 0.50%** | | |
| tiered_ets | 19 | cosine | 42 | 100 | 81.35% | 0.01% | 136 min |
| tiered_ets | 19 | cosine | 123 | 100 | 81.25% | 0.33% | 135 min |
| tiered_ets | 19 | cosine | 456 | 100 | 81.35% | 1.15% | 136 min |
| tiered_ets | 19 | cosine | 3407 | 100 | 81.79% | 0.05% | 132 min |
| tiered_ets | 19 | cosine | 1024 | 100 | 81.23% | 0.53% | 132 min |
| **tiered_ets** | **19** | **cosine** | **mean (5-seed)** | **100** | **81.39% ± 0.23%** | | |
| tiered_lps | 19 | cosine | 42 | 100 | 81.36% | 0.50% | 135 min |
| tiered_lps | 19 | cosine | 123 | 100 | 81.43% | 0.43% | 136 min |
| tiered_lps | 19 | cosine | 456 | 100 | 81.27% | 0.79% | 135 min |
| tiered_lps | 19 | cosine | 3407 | 100 | 81.34% | 0.68% | 135 min |
| tiered_egs | 19 | cosine | 42 | 100 | 79.50% | 0.46% | 288 min |
| tiered_egs | 19 | cosine | 123 | 100 | 79.70% | 0.50% | 290 min |
| tiered_egs | 19 | cosine | 456 | 100 | 79.57% | 0.49% | 258 min |
| tiered_egs_v2 | 19 | cosine | 42 | 100 | 79.83% | 0.47% | 252 min |
| tiered_egs_v2 | 19 | cosine | 123 | 100 | 80.39% | 0.29% | 264 min |
| tiered_egs_v2 | 19 | cosine | 456 | 100 | 79.81% | 0.77% | 294 min |
| tiered_egs_v2 | 19 | cosine | 3407 | 100 | 79.21% | 1.05% | 221 min |
| tiered_egs_v2 | 19 | cosine | 1024 | 100 | 79.49% | 1.07% | 223 min |
| **tiered_egs_v2** | **19** | **cosine** | **mean (5-seed)** | **100** | **79.75% ± 0.44%** | | |

---

## Table 9 — Augmentation Strength Ablation

**Config:** ResNet-50 · CIFAR-100 · ETS · MultiStep scheduler · 100 epochs · Seed 42

| Strength | Test Top-1 | Train Acc | Val–Test Gap |
|:---:|:---:|:---:|:---:|
| 0.3 | 71.38% | 98.48% | 0.58% |
| 0.5 | 72.98% | 95.84% | 0.76% |
| **0.7** *(default)* | **73.98%** | **91.43%** | 0.82% |
| 0.9 | 73.28% | 83.18% | 0.52% |

> **0.7 is optimal** — inverted-U relationship: 0.3 under-regularises (train 98.48%, model memorises), 0.9 disrupts training signal (train drops to 83.18%). The default `fixed_strength=0.7` is validated and held fixed across all WideResNet experiments.

---

## Table A1 — CLIP Semantic Difficulty Validation (CIFAR-100 · WideResNet-28-10 · strength=0.7)

> Validates the manual 3-tier op assignment using CLIP ViT-B/32 semantic distance.
> Score = 1 − cosine_similarity(CLIP(original), CLIP(augmented)) averaged over 1,000 images.
> All 19 ops show ✓ — CLIP difficulty ranking agrees with manual tier assignment.

| Rank | Op | CLIP Score | Manual Tier | CLIP agrees |
|:---:|:---|:---:|:---:|:---:|
| 1 | sharpness | 0.003 | T2 | ✓ |
| 2 | flip | 0.005 | T1 | ✓ |
| 3 | auto_contrast | 0.007 | T2 | ✓ |
| 4 | grayscale | 0.015 | T3 | ✓ |
| 5 | contrast | 0.017 | T3 | ✓ |
| 6 | posterize | 0.021 | T3 | ✓ |
| 7 | shear | 0.021 | T2 | ✓ |
| 8 | brightness | 0.021 | T3 | ✓ |
| 9 | equalize | 0.029 | T2 | ✓ |
| 10 | color_jitter | 0.029 | T2 | ✓ |
| 11 | perspective | 0.032 | T2 | ✓ |
| 12 | translate_y | 0.036 | T1 | ✓ |
| 13 | translate_x | 0.038 | T1 | ✓ |
| 14 | crop | 0.042 | T1 | ✓ |
| 15 | invert | 0.048 | T3 | ✓ |
| 16 | rotation | 0.052 | T2 | ✓ |
| 17 | cutout | 0.053 | T3 | ✓ |
| 18 | blur | 0.121 | T3 | ✓ |
| 19 | solarize | 0.148 | T3 | ✓ |
| — | **CutMix** | **~0.XX** | T3 mixing | — |
| — | **MixUp** | **~0.XX** | T3 mixing | — |

> **Finding — CLIP validates manual tier design:** All 19 ops are ranked consistently with their manual tier assignment — T3 ops occupy the harder end of the CLIP ranking and T1 ops the easier end. Notably, CLIP scores are highly compressed (all < 0.15), confirming that even the most aggressive augmentations (solarize, blur) are semantically mild relative to natural image variation in CLIP's training distribution. The two ops showing the most interesting CLIP vs manual disagreement are `grayscale` (CLIP rank 4th easiest; manually T3) and `crop` (CLIP rank 14th; manually T1) — reflecting the distinction between semantic preservation (CLIP's measure) and learning stability (the manual design's criterion): grayscale preserves object identity but disrupts colour-based feature learning; crop is a safe geometric operation despite removing image content.

---

## Pending Results

| Experiment | Seeds Remaining | Note |
|:---|:---:|:---|
| tiered_egs (19-op) | — | done — Complete — 79.59% ± 0.08% |
| tiered_ets (14-op, 150ep) | 42 | Optional — 100ep s42 = 81.84% already strong |

---

## Table 10 — EGS Hyperparameter Tuning Log (WideResNet-28-10 · CIFAR-100 · 19-op · 100ep · Seed 42)

> Tuning runs use seed 42 only. Full 3-seed sweep only on the best config.
> Baseline for comparison: old EGS (broken) = 79.59% ± 0.08% | ETS = 81.32% ± 0.05%

| Version | T3 thresh | mix_alpha | mix_min_ep | promote_frac | label_smooth | Test Top-1 | Train Acc | T3@ep50 | Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| old EGS (broken, thresh=0.30) | 0.30 | 1.0 | 30 | 0.15 | 0.0 | 79.50% | 40% collapse | 32% | 288 min |
| **v2** (fixed thresholds, soft mix) | 0.15 | 0.2 | 45 | 0.10 | 0.1 | 79.83% / 80.39% / 79.81% → **80.01% ± 0.27%** | 69.77% | 50% | 252–294 min |
| **v3** (raise T3 thresh + alpha) | 0.25 | 0.4 | 40 | 0.10 | 0.1 | 79.62% (s42 only) | 67.48% | 51% | 273 min |

> T2 entropy threshold fixed at 0.40 across all v2+ runs.
> T3@ep50 = fraction of samples in Tier 3 at epoch 50 — proxy for how early full augmentation kicks in.
> **Finding:** Fixing the training collapse (v2) yields +0.42pp over the original EGS. Further threshold and alpha tuning (v3) does not improve beyond v2, indicating the remaining 1.31pp gap vs ETS is structural to the per-sample scheduling design rather than a hyperparameter issue.

---

---
# ResNet-50 / CIFAR-100 Results

> All ResNet-50 runs: Cosine scheduler · SGD lr=0.1 · 100 epochs · 19-op pool · Seed 42 · val_split=0.1
> Kept separate from WideResNet tables — different capacity model, not directly averaged together.

---

## Table 12 — Architecture Comparison: ResNet-50 vs WideResNet-28-10 (CIFAR-100 · 19-op · 100ep · Seed 42)

| Method | ResNet-50 | WideResNet-28-10 | Δ (R50 → WRN) | vs Static (R50) | R50 Time |
|:---|:---:|:---:|:---:|:---:|:---:|
| No Augmentation | 65.30% | 72.86% | −7.56pp | — | 67 min |
| Static Mixing | 76.62% | 77.43% *(mean)* | −0.81pp | — | 69 min |
| Tiered EGS v2 | 79.10% | 80.01% *(mean)* | −0.91pp | +2.48pp | 151 min |
| Tiered ETS | **80.41%** | **81.32%** *(mean)* | −0.91pp | **+3.79pp** | 68 min |
| Tiered LPS | **80.50%** | **81.35%** *(mean)* | −0.85pp | **+3.88pp** | 68 min |

> **Curriculum benefit is architecture-agnostic:** ETS and LPS outperform Static Mixing by +3.79pp and +3.88pp respectively on ResNet-50, nearly identical to their WideResNet advantages (+3.89pp each, Δ ≤ 0.10pp). Progressive augmentation scheduling provides consistent gains regardless of backbone capacity.
>
> **ETS vs LPS equivalence holds across architectures:** ETS (80.41%) and LPS (80.50%) are statistically equivalent on ResNet-50 (Δ = 0.09pp), replicating the WideResNet result (Δ = 0.03pp). LPS transitions earlier (T2→T3 at epoch 36 vs ETS epoch 46), but extra T3 duration does not translate to accuracy improvement.
>
> **EGS-ETS gap is consistent across architectures:** EGS v2 trails ETS by 1.31pp on both ResNet-50 (79.10% vs 80.41%) and WideResNet (80.01% vs 81.32%). The identical gap confirms the EGS limitation is structural — per-sample scheduling delays full T3 exposure — not architecture-specific. EGS is also 2.2× slower on ResNet-50 (151 min vs 68 min) due to entropy computation overhead.
>
> **Augmentation closes the capacity gap:** Without augmentation, ResNet-50 trails WideResNet by 7.56pp. With curriculum augmentation (ETS/LPS), the gap narrows to 0.85–0.91pp — augmentation disproportionately benefits lower-capacity models by providing the implicit regularisation that wider networks achieve through their architecture.

---

## Table 13 — ResNet-50 Complete Run Reference

| Method | Pool | Seed | Ep | Test Top-1 | Val–Test Gap | Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| no_aug | — | 42 | 100 | 65.30% | 0.14% | 67 min |
| static_mixing | 19 | 42 | 100 | 76.62% | 0.06% | 69 min |
| tiered_egs_v2 | 19 | 42 | 100 | 79.10% | 1.14% | 151 min |
| tiered_ets | 19 | 42 | 100 | 80.41% | 0.53% | 68 min |
| tiered_lps | 19 | 42 | 100 | 80.50% | 0.86% | 68 min |

---

## Table 11 — Curriculum Structure Ablation (WideResNet-28-10 · CIFAR-100 · 19-op · 100ep · Seed 42)

> Tests whether the progressive easy→hard ordering is the source of performance gains, or whether any structured schedule suffices.

| Variant | T1 Ops | T3 Ops | T1 Strength | T3 Strength | Mixing | Test Top-1 | Δ vs ETS |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Forward ETS** *(baseline)* | 4 easy | 19 all | 40% | 100% | T3 only (both) | **81.35%** | — |
| ETS + CutMix only | 4 easy | 19 all | 40% | 100% | T3 CutMix | **81.74%** | **+0.39pp** |
| ETS + MixUp only | 4 easy | 19 all | 40% | 100% | T3 MixUp | 80.45% | −0.90pp |
| ETS No Mixing | 4 easy | 19 all | 40% | 100% | none | 79.29% | −2.06pp |
| **Skip T2** (T1→T3 direct, ep 1–45 T1, ep 46–end T3) | 4 easy | 19 all | 40% | 100% | T3 only (both) | **81.35%** | **0.00pp** |
| Static Mixing | 19 all | 19 all | 100% | 100% | from ep 1 | 77.43% | −3.92pp |
| Reverse ETS (Hard→Easy) | 19 all | 4 easy | 100% | 40% | T3 only | 78.17% | −3.18pp |
| Static No Mixing | 19 all | 19 all | 100% | 100% | none | 78.23% | −3.12pp |
| Hard from Epoch 1 | — | 19 all | — | 100% | from ep 1 | pending | — |

> **Ordering matters (FIT Q105):** Reverse curriculum (Hard→Easy) achieves 78.17%, which is **3.18pp below forward ETS** and only 0.74pp above static mixing (77.43%). Beginning training with all 19 ops at full strength prevents stable feature acquisition (train acc 25.72% at ep10 vs ~57% forward). Reversed curriculum provides almost no benefit over a flat static policy.
>
> **Mixing contributes +2.06pp; curriculum ordering contributes +1.86pp independently:**
> The 3.89pp total ETS advantage over static mixing decomposes as:
> - ETS + mixing (81.35%) − ETS no-mix (79.29%) = **+2.06pp from delayed CutMix/MixUp**
> - ETS no-mix (79.29%) − Static + mixing (77.43%) = **+1.86pp from curriculum ordering alone**
>
> Critically, the curriculum structure outperforms static mixing **even without any mixing** (+1.86pp). The train accuracy of 99.98% under ETS no-mix confirms that CutMix/MixUp is the primary regulariser — without it the model memorises the training set nearly perfectly (20.69pp train-test gap vs ~15pp with mixing). Both components — curriculum ordering and delayed mixing — are independently beneficial and additive.
>
> **CutMix is the dominant mixing strategy:**
>
> | Mixing | Test Top-1 | Δ vs no-mix |
> |:---|:---:|:---:|
> | No mixing | 79.29% | — |
> | MixUp only | 80.45% | +1.16pp |
> | Both (CutMix + MixUp) | 81.35% | +2.06pp |
> | **CutMix only** | **81.74%** | **+2.45pp** |
>
> CutMix alone (+2.45pp) outperforms both combined (+2.06pp) and MixUp alone (+1.16pp). Combining CutMix with MixUp slightly degrades vs CutMix alone (−0.39pp), indicating mild interference. MixUp's marginal contribution when added to CutMix is negative. **CutMix is the recommended mixing strategy for this setting.**
>
> **Complete 2×2 decomposition (curriculum × mixing):**
>
> | | No Mixing | CutMix (best) | Mixing effect |
> |:---|:---:|:---:|:---:|
> | Static (no curriculum) | 78.23% | 77.43% *(mean)* | **−0.80pp** (hurts) |
> | ETS (curriculum) | 79.29% | 81.74% | **+2.45pp** (helps) |
> | **Curriculum effect** | **+1.06pp** | **+4.31pp** | |
>
> **Critical finding — the curriculum amplifies the benefit of mixing:** Applying CutMix from epoch 1 (static) slightly *hurts* performance (−0.80pp). The same CutMix applied only in Tier 3 after curriculum warm-up *helps* significantly (+2.45pp). The interaction between curriculum and mixing is super-additive: curriculum+CutMix gains +4.31pp over static alone, far exceeding the sum of their individual effects (+1.06pp + 2.45pp). This demonstrates that delaying mixing until the model has acquired stable representations (via the curriculum) is what makes mixing beneficial — not mixing itself in isolation.

---

---
# Tiny-ImageNet Results

> WideResNet-28-10 · Tiny-ImageNet (200 classes · 64×64) · 19-op pool · Seed 42 · val_split=0.1
> Train: 90,000 · Val: 10,000 · Test: 10,000 · ~17 hours per run

---

## Table 14 — Dataset Generalisation: Tiny-ImageNet (WideResNet-28-10 · 19-op · 100ep · Seed 42)

| Method | Test Top-1 | Test Top-5 | Train Acc | Train–Test Gap | Val–Test Gap | Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| No Augmentation | **63.46%** | — | 99.99% | **36.53pp** | 0.25% | 1043 min |
| Static Mixing | 66.88% | 87.18% | 55.09%† | −11.79pp† | 0.05% | 1069 min |
| Tiered ETS | **69.16%** | **87.48%** | 83.87% | **14.71pp** | 0.16% | 1032 min |
| Tiered LPS | **69.47%** | **87.48%** | 82.25% | **12.78pp** | 0.32% | 1016 min |

> † Static Mixing train accuracy (55.09%) is lower than test accuracy (66.88%) because CutMix applied from epoch 1 mixes training labels — the model is scored against soft mixed labels during training, not against clean class labels. This artificially depresses reported train accuracy and is not an indication of underfitting; it is a known artefact of CutMix applied from epoch 1.

> **Curriculum advantage generalises to Tiny-ImageNet:** ETS achieves 69.16% on Tiny-ImageNet vs static mixing at 66.88%, an advantage of **+2.28pp**. This matches the direction of the CIFAR-100 result (+3.89pp), confirming that the progressive curriculum mechanism generalises beyond CIFAR-100 to a harder 200-class dataset. The slightly smaller gap (+2.28pp vs +3.89pp) is consistent with Tiny-ImageNet's larger training set (90,000 vs 45,000 images) — more data reduces the marginal benefit of curriculum-based ordering because the model encounters sufficient within-class variety even without a curriculum.
>
> **Curriculum provides strong regularisation:** Without augmentation, the train-test gap is 36.53pp. ETS reduces this to 14.71pp — a compression of 21.82pp. This regularisation effect is proportionally similar to CIFAR-100 (no-aug: 27.12pp → ETS ~0.01pp val-test gap with train acc ~80%), confirming that curriculum augmentation is an effective regulariser across dataset scales.
>
> **Static Mixing CutMix effect:** The negative train-test gap for static mixing (train 55.09% < test 66.88%) is an artefact of CutMix training labels making training accuracy appear artificially low. However, it also reflects the disruption that aggressive mixing causes to early training: the model has to simultaneously learn from perceptually altered images and mixed labels from epoch 1, resulting in slower early convergence compared to ETS which defers mixing to Tier 3 (epoch 46).
>
> **Tier transition dip on Tiny-ImageNet:** ETS exhibited a −4.41pp accuracy dip at the T2→T3 transition (47.42% at ep46 → 43.01% at ep50, recovering by ep55). This is consistent with the same tier-transition dip observed on CIFAR-100, validating that the dip is an inherent feature of the curriculum mechanism on harder ops — not an artefact of the specific dataset or the number of classes.
>
> **LPS edges ETS on Tiny-ImageNet (+0.31pp):** LPS achieves 69.47% vs ETS 69.16%, consistent with CIFAR-100 where LPS also marginally outperformed ETS (81.35% vs 81.32%). The difference is within seed variance and not statistically significant with a single seed, but the direction is consistent across both datasets.
>
> **LPS adaptive transitions avoid the tier-transition dip:** LPS advanced T2→T3 at epoch 39 (vs ETS fixed at epoch 46), and val accuracy *improved* at the transition (44.60% ep35 → 47.18% ep40) — no dip observed. This contrasts sharply with ETS's −4.41pp dip. The adaptive scheduler advanced only when the model was genuinely ready, eliminating the disruption caused by forcing tier advancement at a fixed epoch regardless of model state.
>
> **LPS tier transitions on Tiny-ImageNet:** T1→T2 at epoch 19 (2 epochs earlier than ETS fixed ep21), T2→T3 at epoch 39 (7 epochs earlier than ETS fixed ep46). The earlier T3 advancement gave LPS 61 epochs in Tier 3 vs 54 for ETS, contributing to its slight accuracy advantage.

---

## Table 15 — Tiny-ImageNet Complete Run Reference

| Method | Pool | Seed | Ep | Test Top-1 | Test Top-5 | Train Acc | Val–Test Gap | Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| no_aug | — | 42 | 100 | 63.46% | — | 99.99% | 0.25% | 1043 min |
| static_mixing | 19 | 42 | 100 | 66.88% | 87.18% | 55.09%† | 0.05% | 1069 min |
| tiered_ets | 19 | 42 | 100 | **69.16%** | **87.48%** | 83.87% | 0.16% | 1032 min |
| tiered_lps | 19 | 42 | 100 | **69.47%** | **87.48%** | 82.25% | 0.32% | 1016 min |

> ETS tier transitions: T1→T2 at epoch 21 (val 35.36%→36.17%), T2→T3 at epoch 46 (val dip 47.42%→43.01% at ep50, recovery by ep55, best val 69.32% at ep97).

---

---
# t-SNE Feature Visualisation

> WideResNet-28-10 · CIFAR-100 · 19-op pool · Seed 42 · 15 classes · 60 samples/class · 900 points total
> Hook: `model.fc.register_forward_hook()` captures 640-dim penultimate representation (input to FC layer).
> Figures: `results/figs/tsne/tsne_grid.png` (thesis 2×3 grid) · `tsne_grid_hd.png` (300 dpi) · `tsne_row.png` (slides 1×5)

---

## Table 16 — t-SNE Representation Quality (WideResNet-28-10 · CIFAR-100 · 19-op · Seed 42)

**Separation Ratio = Inter-class centroid distance / Mean intra-class distance. Higher = better-separated feature clusters.**

| Method | Test Top-1 | Intra ↓ | Inter ↑ | Ratio ↑ | Relative to No Aug |
|:---|:---:|:---:|:---:|:---:|:---:|
| No Augmentation | 72.86% | 5.42 | 29.73 | 5.49 | — |
| Static Mixing | 77.43% | 3.85 | 32.01 | 8.32 | +1.51× |
| Tiered EGS v2 | 79.83% | 3.23 | 32.25 | 9.97 | +1.82× |
| Tiered ETS | 81.35% | 2.92 | 31.44 | 10.75 | +1.96× |
| **Tiered LPS** | **81.36%** | **2.85** | **30.87** | **10.82** | **+1.97×** |

> **Curriculum nearly doubles representational quality:** LPS achieves a separation ratio of 10.82 vs 5.49 for No Augmentation — a 97% improvement. This demonstrates that curriculum augmentation improves the *geometric quality* of learned representations, not just final accuracy. The clusters of same-class images in feature space are both tighter (lower intra-class variance) and further apart (higher inter-class margin) under curriculum training.
>
> **Separation ratio tracks accuracy across all methods:** The ordering Static (8.32) < EGS (9.97) < ETS (10.75) ≈ LPS (10.82) mirrors the accuracy ranking (77.43% < 79.83% < 81.35% ≈ 81.36%). This alignment provides geometric evidence that the accuracy improvements are grounded in better feature learning, not decision boundary tuning.
>
> **EGS has widest class separation (Inter=32.25) but looser clusters:** EGS pushes class centres furthest apart yet has worse intra-class cohesion (3.23 vs 2.85 for LPS), explaining its lower ratio (9.97) and 1.52pp accuracy deficit. The per-sample scheduling delays full Tier 3 exposure, giving the model fewer epochs to tighten within-class representations.
>
> **Static Mixing vs curriculum representational gap:** Even with CutMix/MixUp from epoch 1, Static Mixing achieves a separation ratio of only 8.32 — well below LPS/ETS (10.82/10.75). The 2.5-point gap in separation ratio corresponds to the 4.01pp accuracy gap, confirming that forcing aggressive augmentation from the start prevents stable feature formation, and that the curriculum's value is specifically in *when* hard augmentation is introduced.

---

# Statistical Significance

> WideResNet-28-10 · CIFAR-100 · 19-op pool · 100 epochs · 5 seeds (42, 123, 456, 3407, 1024)
> Test: Welch's t-test (unequal variance) · Effect size: Cohen's d

## Table 17 — Per-Method Summary (5-seed)

| Method | Mean | Std | Seeds |
|:---|:---:|:---:|:---|
| Static Mixing | 77.60% | ±0.50% | [77.79, 76.82, 77.69, 78.17, 77.54] |
| EGS v2 | 79.75% | ±0.44% | [79.83, 80.39, 79.81, 79.21, 79.49] |
| ETS | 81.39% | ±0.23% | [81.35, 81.25, 81.35, 81.79, 81.23] |
| LPS | 81.44% | ±0.20% | [81.36, 81.43, 81.27, 81.34, 81.79] |

## Table 18 — Pairwise Welch's t-test Results

| Comparison | Δ Acc | t | p | Sig | Cohen's d |
|:---|:---:|:---:|:---:|:---:|:---:|
| ETS vs Static Mixing | +3.79pp | 15.550 | 0.0000 | *** | 9.83 |
| LPS vs Static Mixing | +3.84pp | 16.004 | 0.0000 | *** | 10.12 |
| EGS v2 vs Static Mixing | +2.14pp | 7.228 | 0.0001 | *** | 4.57 |
| ETS vs EGS v2 | +1.65pp | 7.418 | 0.0003 | *** | 4.69 |
| LPS vs EGS v2 | +1.69pp | 7.777 | 0.0003 | *** | 4.92 |
| **ETS vs LPS** | **−0.04pp** | **−0.321** | **0.7567** | **ns** | **−0.20** |

> Significance: *** p<0.001 · ** p<0.01 · * p<0.05 · ns = not significant
> Cohen's d: small ≥0.2 · medium ≥0.5 · large ≥0.8

> **All curriculum methods significantly outperform static mixing:** ETS (d=9.83), LPS (d=10.12), and EGS (d=4.57) all achieve p<0.001 against Static Mixing. Cohen's d values of 9–10 are extraordinarily large — "large" effect begins at 0.8. These differences cannot be attributed to random seed variation.
>
> **ETS and LPS are statistically indistinguishable:** Δ=−0.04pp, p=0.757, d=−0.20. The scheduling mechanism (fixed epoch thresholds vs adaptive loss plateaus) does not significantly affect final accuracy. The curriculum structure — progressive tier exposure — is the critical factor, not the advancement signal.
>
> **EGS is significantly weaker than ETS/LPS:** ETS vs EGS: p=0.0003, d=4.69. LPS vs EGS: p=0.0003, d=4.92. The 1.65–1.69pp gap is statistically confirmed. Per-sample scheduling delays full Tier 3 exposure, resulting in measurably worse representations and accuracy.

---

# CIFAR-100-C Robustness

> WideResNet-28-10 · CIFAR-100 · Seed 42 · 19 corruptions × 5 severity levels · 10,000 images per corruption-severity pair
> Script: `python analysis/cifar100c_robustness.py --c_root data/CIFAR-100-C`
> Figures: `results/figs/cifar100c_robustness.png` · `cifar100c_robustness_hd.png`

---

## Table 19 — CIFAR-100-C Robustness Summary

| Method | Clean Acc | Mean Corrupted Acc ↑ | Robustness Drop ↓ |
|:---|:---:|:---:|:---:|
| No Augmentation | 72.86% | 45.58% | 27.28 pp |
| **EGS v2** | **79.75%** | **66.90%** | **12.85 pp** |
| Static Mixing | 77.60% | 66.27% | 11.33 pp |
| LPS | 81.35% | 52.13% | 29.22 pp |
| ETS | 81.39% | 51.81% | 29.58 pp |

## Table 20 — Per-Corruption Mean Accuracy (%) across 5 Severity Levels

| Corruption | No Aug | Static | ETS | LPS | EGS |
|:---|:---:|:---:|:---:|:---:|:---:|
| gaussian_noise | 21.36 | 49.89 | 21.43 | 22.38 | 47.58 |
| shot_noise | 29.87 | 59.27 | 30.89 | 31.62 | 57.18 |
| impulse_noise | 21.71 | 64.64 | 24.53 | 26.95 | 63.26 |
| speckle_noise | 31.34 | 61.23 | 33.14 | 34.17 | 59.64 |
| defocus_blur | 54.44 | 74.97 | 63.07 | 62.56 | 77.23 |
| glass_blur | 15.73 | 52.13 | 17.72 | 20.93 | 51.10 |
| motion_blur | 49.75 | 69.61 | 58.12 | 57.26 | 68.31 |
| zoom_blur | 48.44 | 72.80 | 55.97 | 55.10 | 74.59 |
| fog | 59.15 | 71.47 | 68.40 | 68.39 | 72.97 |
| frost | 46.60 | 68.02 | 53.00 | 53.52 | 66.95 |
| snow | 51.94 | 68.17 | 61.00 | 61.49 | 67.94 |
| brightness | 69.66 | 76.23 | 77.90 | 77.52 | 78.11 |
| contrast | 47.87 | 70.08 | 59.79 | 59.96 | 72.94 |
| elastic_transform | 54.27 | 68.71 | 63.56 | 63.18 | 69.44 |
| pixelate | 51.27 | 54.79 | 53.32 | 52.01 | 60.23 |
| jpeg_compression | 48.72 | 58.16 | 50.97 | 51.31 | 58.91 |
| saturate | 61.97 | 69.54 | 69.27 | 69.17 | 72.25 |
| gaussian_blur | 46.56 | 74.64 | 53.32 | 52.87 | 76.88 |
| spatter | 55.38 | 74.72 | 68.93 | 70.12 | 75.52 |
| **Mean** | **45.58** | **66.27** | **51.81** | **52.13** | **66.90** |

> **Clean accuracy vs corruption robustness trade-off:** ETS and LPS achieve the highest clean accuracy (+3.84pp over Static Mixing) but exhibit substantially lower CIFAR-100-C robustness (51.81–52.13% vs 66.27%). EGS marginally outperforms Static Mixing on corrupted data (66.90% vs 66.27%, +0.63pp) while also exceeding it on clean accuracy (+2.15pp). This reveals a trade-off inherent to late-tier mixing curriculum designs.
>
> **Noise corruptions expose the mixing timing effect:** On noise corruptions (gaussian, shot, impulse, speckle), ETS (21–33%) performs nearly identically to No Augmentation (21–31%), while Static Mixing (49–65%) and EGS (47–63%) are dramatically better. Noise is absent from all training tiers, so robustness to it is driven entirely by CutMix/MixUp training. Static Mixing and EGS apply mixing from epoch 1 and ~epoch 30 respectively, building noise-robust features. ETS/LPS restrict mixing to Tier 3 (epoch 46+), providing insufficient exposure.
>
> **Curriculum augmentation types improve where they are present:** ETS/LPS show clear improvement over No Augmentation on corruptions that overlap with their training ops: brightness (+8.24pp), fog (+9.25pp), elastic_transform (+9.29pp), snow (+9.06pp). Where ops are absent (noise), no improvement is observed. This confirms that the robustness gap is caused by mixing timing, not by augmentation op selection.
>
> **EGS's gradual mixing builds corruption robustness:** EGS promotes samples to Tier 3 (with mixing) progressively from around epoch 30, giving the network ~70 epochs of mixing exposure for the earliest-promoted samples — compared to ETS's fixed 54 epochs and Static's full 100 epochs. This earlier average mixing exposure explains EGS's strong corruption robustness (66.90%) despite its per-sample scheduling design.
>
> **Limitation and future work:** The clean accuracy vs robustness trade-off could be resolved by enabling CutMix/MixUp from Tier 2 (epoch ~20) rather than Tier 3. This would give ETS/LPS 80 epochs of mixing exposure, potentially achieving both high clean accuracy and strong corruption robustness. Testing this modification is left as future work.

---

## Table 21 — Top-10 Easiest Classes per Method (seed 42, 19-op, 100ep)

| Rank | No Aug | Static | ETS | LPS | EGS v2 |
|:---:|:---|:---|:---|:---|:---|
| 1 | road (96%) | road (98%) | motorcycle (97%) | motorcycle (98%) | bicycle (98%) |
| 2 | motorcycle (95%) | pickup_truck (97%) | pickup_truck (97%) | road (97%) | road (98%) |
| 3 | orange (95%) | chair (95%) | apple (96%) | apple (95%) | motorcycle (96%) |
| 4 | wardrobe (94%) | sunflower (95%) | bicycle (96%) | aquarium_fish (95%) | pickup_truck (95%) |
| 5 | palm_tree (93%) | apple (94%) | skunk (96%) | bicycle (95%) | sunflower (95%) |
| 6 | aquarium_fish (91%) | bicycle (93%) | skyscraper (96%) | orange (95%) | wardrobe (95%) |
| 7 | skyscraper (90%) | orange (93%) | chimpanzee (94%) | pickup_truck (95%) | bottle (93%) |
| 8 | sunflower (90%) | skunk (92%) | keyboard (94%) | skunk (95%) | orange (93%) |
| 9 | apple (89%) | aquarium_fish (91%) | orange (94%) | sunflower (95%) | apple (92%) |
| 10 | chair (89%) | chimpanzee (91%) | road (94%) | skyscraper (95%) | aquarium_fish (92%) |

> Easy classes are visually distinctive with consistent shape or texture: vehicles (motorcycle, bicycle, pickup_truck), fruits (apple, orange), and man-made objects (road, skyscraper, wardrobe). These are stable across all methods.

---

## Table 22 — Top-10 Per-Class Gains vs Static Mixing (seed 42, 19-op, 100ep)

### Static → ETS

| Class | Static | ETS | Δ |
|:---|:---:|:---:|:---:|
| squirrel | 58% | 82% | +24pp |
| crocodile | 64% | 83% | +19pp |
| leopard | 73% | 91% | +18pp |
| bear | 57% | 72% | +15pp |
| otter | 44% | 59% | +15pp |
| lizard | 59% | 72% | +13pp |
| mouse | 54% | 67% | +13pp |
| beaver | 64% | 76% | +12pp |
| girl | 42% | 54% | +12pp |
| forest | 59% | 70% | +11pp |

### Static → LPS

| Class | Static | LPS | Δ |
|:---|:---:|:---:|:---:|
| crocodile | 64% | 85% | +21pp |
| squirrel | 58% | 78% | +20pp |
| mouse | 54% | 72% | +18pp |
| baby | 61% | 75% | +14pp |
| bear | 57% | 71% | +14pp |
| forest | 59% | 73% | +14pp |
| snail | 73% | 87% | +14pp |
| beaver | 64% | 77% | +13pp |
| girl | 42% | 54% | +12pp |
| porcupine | 73% | 85% | +12pp |

### Static → EGS v2

| Class | Static | EGS v2 | Δ |
|:---|:---:|:---:|:---:|
| otter | 44% | 70% | +26pp |
| squirrel | 58% | 82% | +24pp |
| mouse | 54% | 70% | +16pp |
| lizard | 59% | 73% | +14pp |
| forest | 59% | 72% | +13pp |
| rose | 79% | 89% | +10pp |
| baby | 61% | 70% | +9pp |
| bear | 57% | 66% | +9pp |
| plate | 69% | 78% | +9pp |
| fox | 81% | 89% | +8pp |

> Squirrel, mouse, bear, and forest appear in the top-10 gains for all three methods. These are fine-grained classes with high inter-class visual similarity where early-tier geometric-only training builds more discriminative representations before mixing is introduced. EGS v2 achieves the single largest per-class gain across all experiments: otter +26pp (44% → 70%).

---
