# Project Explanation: Loss-Guided Curriculum Augmentation

## What This Code Does — Detailed Explanation

---

### The Core Idea: Loss-Guided Curriculum Augmentation

The project implements **Curriculum Learning for image classification**, where instead of applying the same data augmentation throughout training, augmentation difficulty is **progressively increased** based on how well the model is learning each sample.

The central thesis question is:
> *Does training with easy augmentations first, then hard ones later, outperform always using the same fixed augmentation?*

---

### The Problem With Standard Training

In typical image classification training (e.g., on CIFAR-10), you apply the same augmentation pipeline to every image at every epoch — random crop, flip, color jitter, etc. This is called **static augmentation**.

The problem: early in training, the model barely understands the data. Showing it heavily distorted images (cutout, solarize, strong rotation) when it can't even classify basic examples yet adds noise that slows or hurts learning.

---

### The Curriculum Solution

The code implements three interacting mechanisms:

```
Epoch schedule  ──┐
                   ├──► Final per-sample difficulty ──► augmentation strength
Per-sample loss ──┘
```

---

#### Mechanism 1: Augmentation Primitives (`augmentations/primitives.py`)

Ten augmentation operations are defined, each accepting a `strength` parameter from 0.0 to 1.0:

| Level  | Augmentations                              | Activates at difficulty |
|--------|--------------------------------------------|------------------------|
| Easy   | flip, crop                                 | 0.0 (always)           |
| Medium | color jitter, rotation, shear              | 0.25–0.40              |
| Hard   | grayscale, blur, cutout, solarize, posterize | 0.55–0.85            |

At difficulty=0.3: only flip + crop + mild jitter are applied.
At difficulty=1.0: all 10 augmentations at maximum strength.

The `strength` within each active augmentation also scales linearly. When cutout first activates at difficulty=0.70, it starts very mild and grows to full strength at 1.0 — so there are no jarring jumps.

---

#### Mechanism 2: Epoch-Level Schedule (`augmentations/schedules.py` + `training/losses.py`)

A global difficulty score (0→1) controls how much augmentation the whole dataset receives as training progresses. Four schedule shapes are supported:

```
Sigmoid (default)  ─── slow start, fast ramp in the middle, plateau at end
Linear             ─── constant ramp from epoch 1 to final epoch
Cosine             ─── smooth S-curve
Step               ─── discrete jumps at 33%, 66%, 83% of training
```

There is also a **warmup period** (default: first 5 epochs) where difficulty is locked at 0.0. The model first learns the basic structure of the data before augmentation kicks in.

This aligns deliberately with the **MultiStepLR scheduler** milestones (at epochs 33%, 66%, 83% of training). The learning rate drops coincide with the difficulty phases — the model stabilizes on easy data, the LR drops, then harder augmentation pushes generalization further.

---

#### Mechanism 3: Per-Sample Loss Difficulty (`training/losses.py`)

This is the **thesis contribution**. Instead of every sample getting the same epoch-level difficulty, each sample gets an individual difficulty score based on how hard the model currently finds it:

```
High loss sample → model is struggling → LOW difficulty → mild augmentation
Low loss sample  → model has learned it → HIGH difficulty → heavy augmentation
```

This is the `compute_sample_difficulty` function in `inverse` mode:
```
normalized_loss = (loss - min_loss) / (max_loss - min_loss)
difficulty      = 1.0 - normalized_loss
```

A sample the model finds easy gets more augmentation to prevent overfitting on it.
A sample the model finds hard gets less augmentation so it can focus on the underlying pattern without extra noise.

The **final per-sample difficulty** blends both signals:
```
final = 0.7 × epoch_difficulty + 0.3 × sample_difficulty
```
The `blend=0.7` default means the epoch schedule mostly drives difficulty, with the loss signal fine-tuning within each epoch.

The `LossTracker` applies exponential moving average (momentum=0.9) to smooth the loss signal across epochs, preventing single noisy batches from swinging the difficulty wildly.

---

#### Mechanism 4: CurriculumDataset (`augmentations/curriculum.py`)

This wraps the CIFAR-10 dataset so each sample carries its own difficulty score:

```
sample index → look up difficulty[idx] → call CurriculumTransform → augmented tensor
```

After each epoch, `trainer.py` scatters the freshly computed per-sample difficulties back into the dataset for the next epoch. The flow per epoch:

```
1. Forward pass on batch
2. Compute per-sample CE loss
3. Blend with epoch schedule → per-sample difficulty
4. Store difficulties in LossTracker (EMA smoothed)
5. Backward pass → update weights
6. After epoch: write new difficulties into CurriculumDataset
7. Next epoch: DataLoader picks up the new difficulties
```

---

### The Experimental Setup

The code compares four methods, all using **ResNet-18 on CIFAR-10**, trained for **150 epochs** with **SGD + Nesterov + MultiStepLR**:

| Method               | Description                                              |
|----------------------|----------------------------------------------------------|
| `NoAugmentation`     | Absolute floor — no transforms at all                    |
| `StaticAugmentation` | Standard fixed pipeline — the main baseline to beat      |
| `RandomAugmentation` | All augmentations randomly, no schedule                  |
| `CurriculumLearning` | Loss-guided progressive augmentation (thesis method)     |

The difference between **Random** and **Curriculum** is the critical comparison:
- If Random ≈ Curriculum → the *order* doesn't matter
- If Curriculum > Random → the progressive schedule is a genuine contribution

---

### The Ablation Studies (`experiments/ablation.py`)

Six ablations systematically isolate each component:

| Ablation | Variable            | Options tested                              |
|----------|---------------------|---------------------------------------------|
| A1       | Schedule shape      | sigmoid, linear, cosine, step               |
| A2       | Loss→difficulty mode| inverse, direct, normalized                 |
| A3       | Blend ratio         | 0.0 (pure loss), 0.7 (mixed), 1.0 (pure epoch) |
| A4       | Warmup epochs       | 0, 5, 10                                    |
| A5       | Label smoothing     | 0.0, 0.05, 0.1 (CIFAR-100)                  |

---

### The Analysis Pipeline (`analysis/`)

Three analysis scripts generate publication-quality figures:

| Script                  | Output                                                                 |
|-------------------------|------------------------------------------------------------------------|
| `plot_curves.py`        | Training/val loss & accuracy curves, overfitting gap, LR schedule fig  |
| `visualize_schedule.py` | Difficulty ramp over epochs, individual augmentation parameter curves  |
| `compare_methods.py`    | Bar chart + table: val acc, test acc, generalization gap per method    |

---

### Architecture of the Codebase

```
curriculum-augmentation/
├── augmentations/
│   ├── primitives.py        ← 10 augmentation ops with strength 0→1
│   ├── curriculum.py        ← CurriculumTransform + CurriculumDataset wrapper
│   ├── schedules.py         ← Epoch-level difficulty curves
│   └── policies.py          ← Static baseline policies (NoAug, Static, Random)
├── training/
│   ├── losses.py            ← Core: loss→difficulty mapping + LossTracker EMA
│   ├── trainer.py           ← CL training loop (per-sample difficulty updates)
│   └── evaluate.py          ← Robustness eval on CIFAR-10-C (19 corruption types)
├── experiments/
│   ├── train.py             ← Main CL experiment entry point
│   ├── train_baseline.py    ← Baseline experiments entry point
│   └── ablation.py          ← Automated ablation study runner
├── models/
│   ├── baseline_cnn.py      ← Custom 6-layer CNN
│   ├── baseline_resnet18.py ← ResNet-18
│   └── registry.py          ← Model factory
├── data/
│   └── datasets.py          ← CIFAR-10/100 loaders with train/val/test splits
└── analysis/
    ├── plot_curves.py        ← Training curves + overfitting figures
    ├── compare_methods.py    ← Final comparison table + bar chart
    └── visualize_schedule.py ← Curriculum schedule figures
```

---

### Data Flow (End-to-End)

```
CIFAR-10 raw images
        │
        ▼
random_split(seed=42) ──► train set (45,000) | val set (5,000) | test set (10,000)
        │
        ▼ (train only)
CurriculumDataset
  ├── difficulties[45000]  ← updated each epoch by trainer.py
  └── CurriculumTransform
        ├── get_active_augmentations(difficulty)
        └── apply_augmentations(img, difficulty)
              ├── flip        (always active)
              ├── crop        (always active)
              ├── color_jitter (difficulty > 0.25)
              ├── rotation    (difficulty > 0.35)
              ├── shear       (difficulty > 0.40)
              ├── grayscale   (difficulty > 0.55)
              ├── blur        (difficulty > 0.60)
              ├── cutout      (difficulty > 0.70)
              ├── solarize    (difficulty > 0.80)
              └── posterize   (difficulty > 0.85)
        │
        ▼
Normalized tensor ──► ResNet-18 ──► Cross-Entropy Loss
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                       Backprop               compute_sample_difficulty()
                     (update weights)                 │
                                             blend with epoch_difficulty()
                                                      │
                                             LossTracker.update() (EMA)
                                                      │
                                       CurriculumDataset.set_difficulties()
                                         (ready for next epoch)
```

---

### Why This Matters for the Thesis

The hypothesis is grounded in the original Curriculum Learning paper (Bengio et al., 2009): *a meaningful order from easy to hard examples helps generalization*. This work applies that idea specifically to **data augmentation strength** rather than sample ordering, using the **model's own loss signal** as the measure of difficulty.

This is adaptive — the curriculum adjusts to what the specific model is finding hard, rather than following a fixed schedule blind to training dynamics.

**Key claim:** Loss-guided curriculum augmentation should outperform static augmentation on both clean accuracy and robustness (CIFAR-10-C), while reducing the generalization gap (train acc − val acc).

---

### Known Issues in the Codebase (as of code review)

1. **`experiments/train.py:208`** — Debug mode crashes: `Subset` replaces `CurriculumDataset`, losing `.difficulties` and `.set_difficulties()`.
2. **`analysis/compare_methods.py:227`** — Wrong key `"test_acc"` should be `"test_top1"`; test column always blank.
3. **`experiments/train_baseline.py:223`** — `--model cnn` silently ignored; always uses ResNet-18.
4. **`training/evaluate.py:316-317`** — Bar chart crashes if any corruption file is missing (length mismatch).
5. **`training/trainer.py:265`** — Test accuracy evaluated on last-epoch model, not best checkpoint.
6. **`augmentations/primitives.py:31`** — `random_flip` never fires at difficulty=0 (p = 0.5 × 0 = 0), contradicting the "flip+crop at difficulty=0" design.
7. **`augmentations/curriculum.py:19`** — `get_schedule` imported but never used.
8. **`schedules.py` vs `losses.py`** — Duplicate schedule implementations; training only uses `losses.py`.
