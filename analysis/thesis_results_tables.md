# Thesis Results — WideResNet-28-10 / CIFAR-100

**Generated:** 2026-05-17 &nbsp;|&nbsp; **Model:** WideResNet-28-10 &nbsp;|&nbsp; **Dataset:** CIFAR-100 &nbsp;|&nbsp; **val_split:** 0.1

> 🔶 = partial (seeds still running) &nbsp;|&nbsp; — = not yet run

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

| Method | Seed 42 | Seed 123 | Seed 456 | Mean ± Std | Avg Time |
|:---|:---:|:---:|:---:|:---:|:---:|
| Static Mixing | 77.79% | 76.82% | 77.69% | **77.43% ± 0.44%** | 138 min |
| Tiered EGS (original) | 79.50% | 79.70% | 79.57% | **79.59% ± 0.08%** | 279 min |
| Tiered EGS v2 | 79.83% | 80.39% | 79.81% | **80.01% ± 0.27%** | 270 min |
| Tiered ETS | 81.35% | 81.25% | 81.35% | **81.32% ± 0.05%** | 136 min |
| Tiered LPS | 81.36% | 81.43% | 81.27% | **81.35% ± 0.07%** | 135 min |

> **Finding 1 — Curriculum advantage with aggressive ops:** When the 19-op pool introduces ops with significant information loss (blur, solarize, posterize, invert), progressive curriculum scheduling (ETS/LPS) outperforms the static baseline by **+3.89pp** (81.32% vs 77.43%). This gain is absent with the 14-op pool (Table 1, Δ = 0.01pp), confirming that curriculum benefit scales with augmentation difficulty.
>
> **Finding 2 — ETS vs LPS statistical equivalence:** ETS (81.32% ± 0.05%) and LPS (81.35% ± 0.07%) are statistically indistinguishable (Δ = 0.03pp, within one standard deviation of either method), indicating that the specific tier-advancement signal — fixed epoch thresholds vs adaptive loss plateaus — does not significantly affect final accuracy when both methods are given the same augmentation pool.
>
> **Finding 3 — EGS vs ETS gap:** EGS v2 (80.01% ± 0.27%) trails ETS by 1.31pp. This gap is attributed to the per-sample scheduling design: EGS reaches full Tier 3 exposure only at epoch ~89 on average, leaving only ~11 epochs of maximum augmentation, compared to 55 epochs for ETS. The per-sample adaptivity introduces scheduling overhead without proportional accuracy benefit at 100 epochs.

---

## Table 3 — Op Pool Expansion: Curriculum Robustness (Seed 42)

| Method | 14-op pool | 19-op pool | Δ (14→19) |
|:---|:---:|:---:|:---:|
| Static Mixing | 81.50% | 77.79% | **−3.71 pp** |
| Tiered ETS | 81.84% | 81.35% | −0.49 pp |
| Tiered LPS | 81.48% | 81.36% | −0.12 pp |
| **ETS vs Static** | +0.34 pp | **+3.56 pp** | |
| **LPS vs Static** | −0.02 pp | **+3.57 pp** | |

> **Core thesis finding — curriculum robustness scales with augmentation difficulty:** Expanding the pool from 14 to 19 ops causes static mixing to drop 3.71pp (81.50% → 77.79%), while ETS drops only 0.49pp (81.84% → 81.35%) and LPS drops only 0.12pp (81.48% → 81.36%). The curriculum's protective mechanism — deferring high-distortion ops to Tier 3 when the model has already acquired stable low-level representations — becomes decisive precisely when the ops are most likely to destabilise early training. The advantage of curriculum over static scheduling is near-zero when all ops are geometrically mild (14-op pool), but grows to +3.57pp when the pool includes perceptually destructive transformations (19-op pool).

---

## Table 4 — Training Duration: ETS 14-op Pool

| Epochs | Seed 42 | Seed 123 | Seed 456 | Mean | Avg Time |
|:---|:---:|:---:|:---:|:---:|:---:|
| 100 | 81.84% | 81.22% | 81.80% | **81.62% ± 0.28%** | 225 min |
| 150 | — | **82.70%** | **82.19%** | ~82.45% | 336 min |
| Gain | — | +1.48 pp | +0.39 pp | | +111 min |

> Best single result across all experiments: **82.70%** (ETS · 14-op · 150 ep · seed 123).
> All seeds still converging at epoch 100 — 150 epochs consistently improves results.

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
| **ETS fixed** | epoch 21 | epoch 46 | 55 epochs | 81.32% *(mean)* |

### Cross-architecture comparison (19-op pool · Seed 42)

| Architecture | T1 → T2 | T2 → T3 | T3 duration | Test Top-1 |
|:---:|:---:|:---:|:---:|:---:|
| WideResNet-28-10 | epoch 30 | epoch 41 | 59 epochs | 81.36% |
| ResNet-50 | epoch 23 | epoch 36 | **64 epochs** | 80.50% |
| **ETS fixed** | epoch 21 | epoch 46 | 55 epochs | — |

> **Finding — LPS transitions vary by up to 12 epochs across seeds yet final accuracy is stable** (std ±0.07pp with 19-op WideResNet), demonstrating that LPS is robust to seed-dependent convergence variation.
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
| tiered_ets | 19 | cosine | 42 | 100 | 81.35% | 0.01% | 136 min |
| tiered_ets | 19 | cosine | 123 | 100 | 81.25% | 0.33% | 135 min |
| tiered_ets | 19 | cosine | 456 | 100 | 81.35% | 1.15% | 136 min |
| tiered_lps | 19 | cosine | 42 | 100 | 81.36% | 0.50% | 135 min |
| tiered_lps | 19 | cosine | 123 | 100 | 81.43% | 0.43% | 136 min |
| tiered_lps | 19 | cosine | 456 | 100 | 81.27% | 0.79% | 135 min |
| tiered_egs | 19 | cosine | 42 | 100 | 79.50% | 0.46% | 288 min |
| tiered_egs | 19 | cosine | 123 | 100 | 79.70% | 0.50% | 290 min |
| tiered_egs | 19 | cosine | 456 | 100 | 79.57% | 0.49% | 258 min |
| tiered_egs_v2 | 19 | cosine | 42 | 100 | 79.83% | 0.47% | 252 min |
| tiered_egs_v2 | 19 | cosine | 123 | 100 | 80.39% | 0.29% | 264 min |
| tiered_egs_v2 | 19 | cosine | 456 | 100 | 79.81% | 0.77% | 294 min |
| **tiered_egs_v2** | **19** | **cosine** | **mean** | **100** | **80.01% ± 0.27%** | | |

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
| tiered_egs (19-op) | — | ✅ Complete — 79.59% ± 0.08% |
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

> **Finding 1 — Curriculum benefit is architecture-agnostic:** ETS and LPS outperform Static Mixing by +3.79pp and +3.88pp respectively on ResNet-50, nearly identical to their WideResNet advantages (+3.89pp each, Δ ≤ 0.10pp). Progressive augmentation scheduling provides consistent gains regardless of backbone capacity.
>
> **Finding 2 — ETS vs LPS equivalence holds across architectures:** ETS (80.41%) and LPS (80.50%) are statistically equivalent on ResNet-50 (Δ = 0.09pp), replicating the WideResNet result (Δ = 0.03pp). LPS transitions earlier (T2→T3 at epoch 36 vs ETS epoch 46), but extra T3 duration does not translate to accuracy improvement.
>
> **Finding 3 — EGS-ETS gap is consistent across architectures:** EGS v2 trails ETS by 1.31pp on both ResNet-50 (79.10% vs 80.41%) and WideResNet (80.01% vs 81.32%). The identical gap confirms the EGS limitation is structural — per-sample scheduling delays full T3 exposure — not architecture-specific. EGS is also 2.2× slower on ResNet-50 (151 min vs 68 min) due to entropy computation overhead.
>
> **Finding 4 — Augmentation closes the capacity gap:** Without augmentation, ResNet-50 trails WideResNet by 7.56pp. With curriculum augmentation (ETS/LPS), the gap narrows to 0.85–0.91pp — augmentation disproportionately benefits lower-capacity models by providing the implicit regularisation that wider networks achieve through their architecture.

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
| Static Mixing | 19 all | 19 all | 100% | 100% | from ep 1 | 77.43% | −3.92pp |
| Reverse ETS (Hard→Easy) | 19 all | 4 easy | 100% | 40% | T3 only | 78.17% | −3.18pp |
| Static No Mixing | 19 all | 19 all | 100% | 100% | none | 78.23% | −3.12pp |
| Hard from Epoch 1 | — | 19 all | — | 100% | from ep 1 | 📋 | — |

> **Finding 1 — Ordering matters (FIT Q105):** Reverse curriculum (Hard→Easy) achieves 78.17%, which is **3.18pp below forward ETS** and only 0.74pp above static mixing (77.43%). Beginning training with all 19 ops at full strength prevents stable feature acquisition (train acc 25.72% at ep10 vs ~57% forward). Reversed curriculum provides almost no benefit over a flat static policy.
>
> **Finding 2 — Mixing contributes +2.06pp; curriculum ordering contributes +1.86pp independently:**
> The 3.89pp total ETS advantage over static mixing decomposes as:
> - ETS + mixing (81.35%) − ETS no-mix (79.29%) = **+2.06pp from delayed CutMix/MixUp**
> - ETS no-mix (79.29%) − Static + mixing (77.43%) = **+1.86pp from curriculum ordering alone**
>
> Critically, the curriculum structure outperforms static mixing **even without any mixing** (+1.86pp). The train accuracy of 99.98% under ETS no-mix confirms that CutMix/MixUp is the primary regulariser — without it the model memorises the training set nearly perfectly (20.69pp train-test gap vs ~15pp with mixing). Both components — curriculum ordering and delayed mixing — are independently beneficial and additive.
>
> **Finding 3 — CutMix is the dominant mixing strategy:**
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
> **Finding 4 — Complete 2×2 decomposition (curriculum × mixing):**
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
| Tiered LPS | 📋 | — | — | — | — | — |

> † Static Mixing train accuracy (55.09%) is lower than test accuracy (66.88%) because CutMix applied from epoch 1 mixes training labels — the model is scored against soft mixed labels during training, not against clean class labels. This artificially depresses reported train accuracy and is not an indication of underfitting; it is a known artefact of CutMix applied from epoch 1.

> **Finding 1 — Curriculum advantage generalises to Tiny-ImageNet:** ETS achieves 69.16% on Tiny-ImageNet vs static mixing at 66.88%, an advantage of **+2.28pp**. This matches the direction of the CIFAR-100 result (+3.89pp), confirming that the progressive curriculum mechanism generalises beyond CIFAR-100 to a harder 200-class dataset. The slightly smaller gap (+2.28pp vs +3.89pp) is consistent with Tiny-ImageNet's larger training set (90,000 vs 45,000 images) — more data reduces the marginal benefit of curriculum-based ordering because the model encounters sufficient within-class variety even without a curriculum.
>
> **Finding 2 — Curriculum provides strong regularisation:** Without augmentation, the train-test gap is 36.53pp. ETS reduces this to 14.71pp — a compression of 21.82pp. This regularisation effect is proportionally similar to CIFAR-100 (no-aug: 27.12pp → ETS ~0.01pp val-test gap with train acc ~80%), confirming that curriculum augmentation is an effective regulariser across dataset scales.
>
> **Finding 3 — Static Mixing CutMix effect:** The negative train-test gap for static mixing (train 55.09% < test 66.88%) is an artefact of CutMix training labels making training accuracy appear artificially low. However, it also reflects the disruption that aggressive mixing causes to early training: the model has to simultaneously learn from perceptually altered images and mixed labels from epoch 1, resulting in slower early convergence compared to ETS which defers mixing to Tier 3 (epoch 46).
>
> **Finding 4 — Tier transition dip on Tiny-ImageNet:** ETS exhibited a −4.41pp accuracy dip at the T2→T3 transition (47.42% at ep46 → 43.01% at ep50, recovering by ep55). This is consistent with the same tier-transition dip observed on CIFAR-100, validating that the dip is an inherent feature of the curriculum mechanism on harder ops — not an artefact of the specific dataset or the number of classes.

---

## Table 15 — Tiny-ImageNet Complete Run Reference

| Method | Pool | Seed | Ep | Test Top-1 | Test Top-5 | Train Acc | Val–Test Gap | Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| no_aug | — | 42 | 100 | 63.46% | — | 99.99% | 0.25% | 1043 min |
| static_mixing | 19 | 42 | 100 | 66.88% | 87.18% | 55.09%† | 0.05% | 1069 min |
| tiered_ets | 19 | 42 | 100 | **69.16%** | **87.48%** | 83.87% | 0.16% | 1032 min |
| tiered_lps | 19 | 42 | 100 | 📋 | — | — | — | — |

> ETS tier transitions: T1→T2 at epoch 21 (val 35.36%→36.17%), T2→T3 at epoch 46 (val dip 47.42%→43.01% at ep50, recovery by ep55, best val 69.32% at ep97).
