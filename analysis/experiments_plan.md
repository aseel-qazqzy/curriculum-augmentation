# Experiments Plan

**Generated:** 2026-05-17 &nbsp;|&nbsp; **Sources:** FIT presentation · FIT Q&A (105 questions) · session discussions

---

### Legend

| Symbol | Meaning |
|:---|:---|
| **E / L / G** | ETS · LPS · EGS |
| **W / R** | WideResNet-28-10 · ResNet-50 |
| **3 / 1** | Full 3-seed sweep · Single ablation seed |
| ✅ | Done |
| 🔶 | Partial |
| 📋 | TODO |
| 🔍 | Analysis only — no new training |

---

## Group A — Core Experiments

> **MVT** = Minimum Viable Thesis — required for defense

| Experiment | Apply | Seeds | Arch | Pool | Dataset | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| No Augmentation | — | 1 | W | — | CIFAR-100 | ✅ |
| Static Mixing (14-op) | — | 3 | W | 14 | CIFAR-100 | ✅ |
| Static Mixing (19-op) | — | 3 | W | 19 | CIFAR-100 | ✅ |
| Tiered ETS (14-op) | E | 3 | W | 14 | CIFAR-100 | ✅ |
| Tiered ETS (19-op) | E | 3 | W | 19 | CIFAR-100 | ✅ |
| Tiered LPS (14-op) | L | 3 | W | 14 | CIFAR-100 | ✅ |
| Tiered LPS (19-op) | L | 3 | W | 19 | CIFAR-100 | ✅ |
| Tiered EGS (14-op) | G | 3 | W | 14 | CIFAR-100 | ✅ |
| Tiered EGS (19-op, old code) | G | 3 | W | 19 | CIFAR-100 | ✅ 79.59% ± 0.08% |
| Tiered EGS (19-op, _TIER_OPS fix, thresh050, 100ep) | G | 1 | W | 19 | CIFAR-100 | ✅ 80.12% s42 — best ep99, needs 150ep |
| Tiered EGS (19-op, _TIER_OPS fix, thresh050, 150ep) | G | 3 | W | 19 | CIFAR-100 | 📋 run next |
| EGS opt Config 3 (old code, s42) | G | 1 | W | 19 | CIFAR-100 | ✅ 80.68% — fixed epoch, old ops |
| ETS 150 epochs (14-op) | E | 3 | W | 14 | CIFAR-100 | 🔶 |
| **RandAugment N=2, M=9** ⚠️ MVT | — | 1 | W | — | CIFAR-100 | 📋 |
| **Random Aug (same pool, no ordering)** ⚠️ MVT | — | 1 | W | 19 | CIFAR-100 | 📋 |

**Answers:** Does the proposed curriculum outperform published baselines (RandAugment)? Is the improvement from ordering or just the ops chosen?

```bash
# RandAugment
--augmentation randaugment --ra_n 2 --ra_m 9 --scheduler cosine --use_amp

# Random (same pool, no ordering)
--augmentation random --scheduler cosine --use_amp
```

---

## Group B — Scheduler Ablation

| Experiment | Apply | Seeds | Arch | Pool | Dataset | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Static — MultiStep | — | 1 | W | 19 | CIFAR-100 | 📋 |
| ETS — MultiStep | E | 1 | W | 19 | CIFAR-100 | 📋 |
| LPS — MultiStep | L | 1 | W | 19 | CIFAR-100 | 📋 |
| ETS — Cosine WarmRestart | E | 1 | W | 19 | CIFAR-100 | ✅ |

**Answers:** Does the LR scheduler choice affect the curriculum benefit? Is cosine better than the traditional MultiStep used in WideResNet papers?

> EGS is excluded — it is scheduler-independent by design.

```bash
--scheduler multistep
```

---

## Group C — Mixing Ablation

| Experiment | Apply | Seeds | Arch | Pool | Dataset | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ETS — No mixing** ⚠️ MVT | E | 1 | W | 19 | CIFAR-100 | ✅ 79.29% s42 · 137 min |
| ETS — CutMix only | E | 1 | W | 19 | CIFAR-100 | ✅ 81.74% s42 · 137 min |
| ETS — MixUp only | E | 1 | W | 19 | CIFAR-100 | ✅ 80.45% s42 · 137 min |
| ETS — Both CutMix + MixUp *(current)* | E | 3 | W | 19 | CIFAR-100 | ✅ |
| Static — No mixing | — | 1 | W | 19 | CIFAR-100 | ✅ 78.23% s42 · 139 min |

**Answers:** Is the accuracy gain driven by the curriculum ordering of ops, or simply by the delayed introduction of CutMix/MixUp? Which mixing strategy contributes more?

```bash
--mix_mode none      # ETS no-mix — most critical
--mix_mode cutmix
--mix_mode mixup
```

---

## Group D — Curriculum Structure Ablation

> Needs code changes for Reverse / Skip / 2-tier variants.
> "Hard from epoch 1" works today with existing CLI.

| Experiment | Apply | Seeds | Arch | Pool | Dataset | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Reverse: Hard → Medium → Easy** ⭐ FIT Q105 | E | 1 | W | 19 | CIFAR-100 | ✅ 78.17% s42 |
| **Hard from epoch 1 (t1=0, t2=0)** ⚠️ MVT | E | 1 | W | 19 | CIFAR-100 | 📋 |
| T1 + T3 only (skip T2) | E | 1 | W | 19 | CIFAR-100 | 📋 |
| T2 only (photometric ops, all 100 epochs) | E | 1 | W | 19 | CIFAR-100 | 📋 |
| 2-tier only: Easy → Hard | E | 1 | W | 19 | CIFAR-100 | 📋 |

**Answers:** Does the progressive 3-tier structure itself drive the gains, or just the final set of ops? Is the middle tier necessary? FIT Q105: *"If reverse performs worse, it directly confirms that order is what matters."*

```bash
# Hard from epoch 1 — no code change needed
--tier_t1 0.0 --tier_t2 0.0

# Reverse / Skip / 2-tier — needs code: _TIER_OPS_REVERSE in policies.py
```

---

## Group E — Augmentation Strength Ablation

| Experiment | Apply | Seeds | Arch | Pool | Dataset | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Strength = 0.3 | E | 1 | R | old | CIFAR-100 | ✅ *(ResNet-50, MultiStep, Apr-26)* |
| Strength = 0.5 | E | 1 | R | old | CIFAR-100 | ✅ *(ResNet-50, MultiStep, Apr-26)* |
| Strength = 0.7 *(default)* | E | 1+3 | R+W | old+19 | CIFAR-100 | ✅ |
| Strength = 0.9 | E | 1 | R | old | CIFAR-100 | ✅ *(ResNet-50, MultiStep, Apr-26)* |
| Strength = 0.5 on WideResNet | E | 1 | W | 19 | CIFAR-100 | 📋 *(optional)* |
| Strength = 0.9 on WideResNet | E | 1 | W | 19 | CIFAR-100 | 📋 *(optional)* |
| No strength ramp (hard jump at tier boundary) | E | 1 | W | 19 | CIFAR-100 | 📋 |

**Answers:** Is augmentation strength an independent variable from the curriculum? Is the 5-epoch strength ramp at tier transitions important for smooth convergence?

**Result (ResNet-50):** 0.7 is optimal — 73.98% Test Top-1. Inverted-U: 0.3=71.38%, 0.5=72.98%, 0.9=73.28%. Default 0.7 validated. → See `thesis_results_tables.md` Table 9.

```bash
--fixed_strength 0.5
--fixed_strength 0.9
```

---

## Group F — Tier Boundary Timing Ablation

| Experiment | Apply | Seeds | Arch | Pool | Dataset | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Early: T1=10%, T2=30% | E | 1 | W | 19 | CIFAR-100 | 📋 |
| Default: T1=20%, T2=45% | E | 3 | W | 19 | CIFAR-100 | ✅ |
| Late: T1=30%, T2=60% | E | 1 | W | 19 | CIFAR-100 | 📋 |
| Very late: T1=40%, T2=70% | E | 1 | W | 19 | CIFAR-100 | 📋 |

**Answers:** How sensitive is the curriculum to when tier transitions happen? Are the default 20%/45% thresholds specifically tuned, or does the method work robustly across a range of timings?

```bash
--tier_t1 0.10 --tier_t2 0.30
--tier_t1 0.30 --tier_t2 0.60
--tier_t1 0.40 --tier_t2 0.70
```

---

## Group G — EGS Hyperparameter Sensitivity

| Experiment | Apply | Seeds | Arch | Pool | Dataset | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| min_epochs_per_tier = 10 *(default)* | G | 3 | W | 19 | CIFAR-100 | ✅ (v2) |
| min_epochs_per_tier = 20 *(old default)* | G | 3 | W | 19 | CIFAR-100 | ✅ 79.59% |
| max_promote_frac = 0.25 *(faster advancement)* | G | 1 | W | 19 | CIFAR-100 | 📋 |
| update_freq = 10 *(less frequent)* | G | 1 | W | 19 | CIFAR-100 | 📋 |

**Answers:** How sensitive is EGS to its hyperparameters? Would faster tier advancement (lower min_epochs) activate mixing earlier and improve results? Does update frequency matter?

```bash
--egs_min_epochs_per_tier 10
--egs_max_promote_frac 0.25
--egs_update_freq 10
```

---

## Group G2 — EGS v2/v3 Tuning Runs *(seed 42 only — find best config, then 3-seed sweep)*

> See `thesis_results_tables.md` Table 10 for full tuning log.

| Version | T3 thresh | mix_alpha | Seeds | Test Top-1 | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| v2 — fixed thresholds, soft mix (α=0.2) | 0.15 | 0.2 | 3 | s42=79.83% · s123=80.39% · s456=79.81% → **80.01% ± 0.27%** | ✅ complete |
| v3 — raised T3 thresh (0.25), stronger mix (α=0.4) | 0.25 | 0.4 | 1 | 79.62% | ✅ worse than v2 |

```bash
# v3 command
python -m experiments.train_baseline --dataset cifar100 --model wideresnet \
  --augmentation tiered_curriculum --tier_schedule egs --epochs 100 \
  --scheduler cosine --warmup_epochs 5 --lr 0.1 \
  --egs_update_freq 3 --egs_min_epochs_per_tier 10 --egs_max_epochs_per_tier 25 \
  --egs_max_promote_frac 0.10 --egs_mix_threshold 0.50 --egs_mix_min_epoch 40 \
  --egs_t3_entropy_thresh 0.25 --mix_alpha 0.4 --label_smoothing 0.1 \
  --use_amp --seed 42 --experiment_name egs_v3_100ep_s42
```

---

## Group H — Dataset Generalization

> FIT Slide 11, Q54. CIFAR-10 **not needed** (FIT Q89: *"already solved, not meaningful"*).

| Experiment | Apply | Seeds | Arch | Pool | Dataset | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| No Augmentation | — | 1 | W | — | Tiny-ImageNet | ✅ 63.46% s42 · 1043 min |
| Static Mixing | — | 1 | W | 19 | Tiny-ImageNet | ✅ 66.88% s42 · 1069 min |
| Tiered ETS | E | 1 | W | 19 | Tiny-ImageNet | ✅ 69.16% s42 · 1032 min (+2.28pp vs static) |
| Tiered LPS | L | 1 | W | 19 | Tiny-ImageNet | 📋 |

**Answers:** Does the curriculum benefit hold on a harder dataset (200 classes, 64×64)? Does the method scale beyond CIFAR-100?

```bash
--dataset tiny_imagenet --model wideresnet --epochs 100 --scheduler cosine --use_amp
```

---

## Group I — Architecture Comparison

> FIT Q43, Q55: ResNet-50 was the **primary** backbone in the FIT. Committee expects results.

| Experiment | Apply | Seeds | Arch | Pool | Dataset | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| No Augmentation | — | 1 | R | — | CIFAR-100 | ✅ 65.30% s42 · 67 min |
| Static Mixing | — | 1 | R | 19 | CIFAR-100 | ✅ 76.62% s42 · 69 min |
| Tiered EGS v2 | G | 1 | R | 19 | CIFAR-100 | ✅ 79.10% s42 · 151 min |
| Tiered ETS | E | 1 | R | 19 | CIFAR-100 | ✅ 80.41% s42 · 68 min |
| Tiered LPS | L | 1 | R | 19 | CIFAR-100 | ✅ 80.50% s42 · T1→T2 ep23 · T2→T3 ep36 · 68 min |

**Answers:** Is the curriculum benefit specific to WideResNet or does it generalise across architectures? Do larger capacity models benefit more or less from progressive augmentation?

```bash
--model resnet50 --scheduler cosine --use_amp
```

---

## Group J — Analysis Tasks *(no new training)*

> ⭐ = grade-boosting addition · ⚠️ MVT = minimum viable thesis

| Task | Uses | Answers | Status |
|:---|:---|:---|:---:|
| **Statistical significance (t-test / Wilcoxon)** ⚠️ MVT | 3-seed results | Are observed differences statistically significant? | 📋 |
| **t-SNE feature visualisation** ⭐ | WRN checkpoints (all 5 methods) | Do curriculum models learn better-separated representations? | 📋 |
| **CIFAR-100-C robustness (mCE)** ⭐ | WRN checkpoints | Is the model more robust to natural corruptions? | 📋 |
| Convergence speed (epochs to 70/75/80%) | History files | Does curriculum reach target accuracy faster? | 📋 |
| ECE — Expected Calibration Error ⭐ | WRN checkpoints | Does curriculum reduce model overconfidence? | 📋 |
| Per-class accuracy analysis (top/bottom 10 classes) | WRN checkpoints | Which classes benefit most from curriculum? | 📋 |
| EGS force-promoted sample analysis | EGS logs | Do stuck samples concentrate in specific semantic categories? | 📋 |
| Tier transition dip quantification | ETS/LPS histories | How large are accuracy dips at tier boundaries and how fast is recovery? | 📋 |
| Entropy trajectory plot | EGS logs | How does model confidence evolve across training? | 📋 |
| Train–val gap inversion analysis | All histories | Does curriculum reduce overfitting (train acc ≈ val acc)? | 📋 |
| MADAug comparison vs published numbers | Their paper | How does the method compare to the closest prior work? | 📋 |

### Group J Commands

```bash
# t-SNE feature visualisation (all 5 methods — run on cluster)
python analysis/tsne_features.py --data_root data --val_split 0.1
# Output: results/figs/tsne/tsne_grid.png (thesis) + tsne_row.png (slides)

# Statistical significance — Welch's t-test between ETS and Static Mixing
# (3-seed results already in thesis_results_tables.md — just run this script)
python - << 'EOF'
from scipy import stats
import numpy as np
# WideResNet · CIFAR-100 · 19-op · 100ep (seed 42, 123, 456)
ets     = [81.35, 81.25, 81.35]
lps     = [81.36, 81.43, 81.27]
egs     = [79.83, 80.39, 79.81]
static  = [77.79, 76.82, 77.69]
pairs = [("ETS vs Static", ets, static), ("LPS vs Static", lps, static),
         ("ETS vs EGS",   ets, egs),    ("ETS vs LPS",   ets, lps)]
for name, a, b in pairs:
    t, p = stats.ttest_ind(a, b, equal_var=False)  # Welch's t-test
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    d = (np.mean(a) - np.mean(b)) / np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
    print(f"{name:<20}  t={t:.3f}  p={p:.4f}  {stars}  Cohen's d={d:.2f}")
EOF

# Convergence speed — epochs to reach 70% / 75% / 80% (from history files)
python - << 'EOF'
import torch, sys
from pathlib import Path
sys.path.insert(0, '.')
CKPT = Path("checkpoints")
thresholds = [0.70, 0.75, 0.80]
runs = {
    "Static":  "wideresnet_static_mixing_mix_both_sgd_cosine_ep100_cifar100_s42_p19",
    "ETS":     "wideresnet_tiered_ets_mix_both_sgd_cosine_ep100_cifar100_s42_p19",
    "LPS":     "wideresnet_tiered_lps_mix_both_sgd_cosine_ep100_cifar100_s42_p19",
    "EGS":     "egs_v2_100ep_s42_ep100_cifar100_s42_p19",
}
print(f"{'Method':<12}", "  ".join(f"{int(t*100)}%@ep" for t in thresholds))
for name, stem in runs.items():
    h = torch.load(CKPT / f"{stem}_history.pt", map_location="cpu", weights_only=False)
    accs = h["val_acc"]
    epochs = [next((i+1 for i,a in enumerate(accs) if a >= t), None) for t in thresholds]
    print(f"{name:<12}", "  ".join(str(e) if e else "—" for e in epochs))
EOF

# RandAugment (cluster — ~135 min)
python -m experiments.train_baseline --dataset cifar100 --model wideresnet \
    --augmentation randaugment --ra_n 2 --ra_m 9 \
    --epochs 100 --scheduler cosine --warmup_epochs 5 --lr 0.1 \
    --use_amp --seed 42

# Random augmentation same pool (cluster — ~135 min)
python -m experiments.train_baseline --dataset cifar100 --model wideresnet \
    --augmentation random --epochs 100 --scheduler cosine \
    --warmup_epochs 5 --lr 0.1 --use_amp --seed 42

# LPS Tiny-ImageNet (cluster — ~1069 min)
python -m experiments.train_baseline --dataset tiny_imagenet --model wideresnet \
    --augmentation tiered_curriculum --tier_schedule lps \
    --epochs 100 --scheduler cosine --warmup_epochs 5 --lr 0.1 \
    --use_amp --seed 42
```

---

## Priority Order

> ✅ = done · 📋 = pending

| Rank | Experiment / Task | Est. Time | Why | Status |
|:---:|:---|:---:|:---|:---:|
| 1 | EGS 19-op seeds 123 + 456 | ~576 min | Completes primary Table 2 | ✅ |
| 2 | **RandAugment** | ~135 min | MVT — named in FIT, must be in main table | 📋 |
| 3 | ETS no-mix | ~135 min | MVT — direct answer to "is it mixing or curriculum?" | ✅ |
| 4 | Reverse curriculum | ~135 min | FIT Q105 — single most important ablation | ✅ |
| 5 | Hard from epoch 1 | ~135 min | MVT — answers "does order matter?" | 📋 |
| 6 | Random augmentation | ~135 min | Required baseline | 📋 |
| 7 | **Statistical significance (t-test)** | analysis | MVT — FIT Q50 | 📋 |
| 8 | **t-SNE feature visualisation** ⭐ | ~30 min | Shows better representations visually | 📋 |
| 9 | **CIFAR-100-C robustness** ⭐ | ~2 hrs | Proves generalisation beyond CIFAR-100 test set | 📋 |
| 10 | CutMix only + MixUp only | ~270 min | FIT Q33 — mixing decomposition | ✅ |
| 11 | Tiny-ImageNet LPS | ~1069 min | Completes Group H | 📋 |
| 12 | ResNet-50 × 4 | ~540 min | FIT primary backbone | ✅ |
| 13 | MultiStep scheduler × 3 | ~405 min | Committee will ask why cosine was chosen | 📋 |
| 14 | Tier boundary timing × 3 | ~405 min | FIT Q16, Q30 — sensitivity | 📋 |
| 15 | Strength ablation × 3 | ~405 min | FIT Q71-72 | 📋 |
| 16 | EGS sensitivity × 3 | ~576 min | FIT Q24 | 📋 |
| 17 | T1+T3 skip, 2-tier | ~270 min | Structure ablation | 📋 |
| 18 | Convergence speed analysis | analysis | FIT Q9, Q69 | 📋 |
| 19 | ECE calibration ⭐ | analysis | Shows curriculum improves model confidence | 📋 |
| 20 | Per-class accuracy analysis | analysis | FIT Q51-52 | 📋 |

---

## Outstanding Training Runs Summary

| Group | Runs Remaining | Est. Time |
|:---|:---:|:---:|
| A — Core | 2 | ~270 min |
| B — Scheduler | 3 | ~405 min |
| C — Mixing | 0 | ✅ complete |
| D — Curriculum structure | 4 | ~540 min |
| E — Strength | 4 | ~540 min |
| F — Tier boundaries | 3 | ~405 min |
| G — EGS sensitivity | 4 | ~576 min |
| H — Tiny-ImageNet | 1 | ~1069 min |
| I — ResNet-50 | 0 | ✅ complete |
| **Total** | **21 runs** | **~64 hours** |

---

## Code Changes Still Needed

| Feature | Implementation |
|:---|:---|
| Reverse curriculum (Hard→Med→Easy) | `_TIER_OPS_REVERSE` dict in `policies.py` |
| T1 + T3 skip T2 | `--tier_structure` flag or custom pool |
| 2-tier (Easy → Hard) | Same as above |
| T2 only | Same as above |
| No strength ramp | `--strength_ramp_epochs 0` flag |
| Hard from epoch 1 | ✅ Already works: `--tier_t1 0.0 --tier_t2 0.0` |

---

## What the FIT Explicitly Said Is NOT Needed

| Item | Reason |
|:---|:---|
| CIFAR-10 | Q89: *"Already solved — not meaningful differentiation"* |
| ImageNet | Q46: *"Not feasible at master's thesis scale"* |
| AutoAugment (rerun) | Too expensive; compare against published numbers only |
| MADAug (rerun) | Q48: *"Compare against their published 100-epoch numbers"* |
