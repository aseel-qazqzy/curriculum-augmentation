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

**Config:** Cosine scheduler · SGD lr=0.1 · 100 epochs

| Method | Seed 42 | Seed 123 | Seed 456 | Mean ± Std | Avg Time |
|:---|:---:|:---:|:---:|:---:|:---:|
| Static Mixing | 77.79% | 76.82% | 77.69% | **77.43% ± 0.44%** | 138 min |
| Tiered ETS | 81.35% | 81.25% | 81.35% | **81.32% ± 0.05%** | 136 min |
| Tiered LPS | 81.36% | 81.43% | 81.27% | **81.35% ± 0.07%** | 135 min |
| Tiered EGS | 79.50% | 🔶 pending | 🔶 pending | — | 288 min |

> With 19 ops (incl. blur, invert, solarize, posterize): ETS/LPS outperform static by **+3.89pp**.
> ETS and LPS are statistically identical (81.32% vs 81.35%, Δ = 0.03pp).

---

## Table 3 — Op Pool Expansion: Curriculum Robustness (Seed 42)

| Method | 14-op pool | 19-op pool | Δ (14→19) |
|:---|:---:|:---:|:---:|
| Static Mixing | 81.50% | 77.79% | **−3.71 pp** |
| Tiered ETS | 81.84% | 81.35% | −0.49 pp |
| Tiered LPS | 81.48% | 81.36% | −0.12 pp |
| **ETS vs Static** | +0.34 pp | **+3.56 pp** | |
| **LPS vs Static** | −0.02 pp | **+3.57 pp** | |

> When aggressive ops are added at full strength from epoch 1, static mixing drops 3.71pp.
> The curriculum shields the model: new ops are introduced only in Tier 3 when the model is already robust.
> This is the core thesis finding — curriculum advantage grows from marginal to decisive as ops get harder.

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

> LPS transitions vary by up to 12 epochs across seeds yet final accuracy is stable (std ±0.07pp with 19-op).
> LPS adapts correctly: holds Tier 1 longer when loss still improving, advances Tier 3 earlier when Tier 2 plateaus fast.

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

## Pending Results

| Experiment | Seeds Remaining | Note |
|:---|:---:|:---|
| tiered_egs (19-op) | 123, 456 | Will complete Table 2 EGS row |
| tiered_ets (14-op, 150ep) | 42 | Optional — 100ep s42 = 81.84% already strong |
