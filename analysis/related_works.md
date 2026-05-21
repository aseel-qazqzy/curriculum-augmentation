# Related Works — Curriculum Augmentation Survey
**Thesis:** Progressive Data Augmentation Scheduling for Image Classification
**Author:** Aseel Al-Qazqzy · University of Hildesheim · 2026

---

## Overview

Seven papers reviewed and evaluated for inclusion in the thesis related work section.
Grouped by methodology, ordered by relevance to the thesis contribution.

---

## Group 1 — Search-Based Static Policies

> Papers that search for an optimal augmentation policy offline or online.
> The policy, once found, is applied uniformly from epoch 1 with no curriculum.
> **Thesis positioning:** These methods require search infrastructure; your work requires none.

---

### Paper 1 — AutoAugment

| Field | Details |
|:---|:---|
| **Title** | AutoAugment: Learning Augmentation Policies from Data |
| **Authors** | Cubuk, Zoph, Mane, Vasudevan, Le (Google Brain) |
| **Venue** | CVPR 2019 |
| **arXiv** | 1805.09501 |
| **Method** | Reinforcement learning searches for the best combination of ops, magnitudes, and probabilities across 25 sub-policies |
| **Curriculum** | None — static policy from epoch 1 |
| **Per-sample** | No |
| **Search cost** | ~5,000 GPU hours |
| **CIFAR-100 WRN-28-10** | ~82.9% |
| **Relevance** | Foundational baseline; shows what learned policies achieve |
| **Thesis use** | Related work — contrast: your method requires no search |

**Summary for thesis:**
AutoAugment establishes that augmentation policies can be discovered automatically via RL search, achieving 82.9% on CIFAR-100 WRN-28-10. However, the 5,000 GPU-hour search cost makes it impractical without large-scale compute. Your work avoids this entirely by using a manually designed 3-tier curriculum that requires only standard training time.

---

### Paper 2 — Fast AutoAugment

| Field | Details |
|:---|:---|
| **Title** | Fast AutoAugment |
| **Authors** | Lim, Kim, Kim, Kim, Kim |
| **Venue** | NeurIPS 2019 |
| **arXiv** | 1905.00397 |
| **Method** | Density matching replaces RL search; finds policy by matching augmented data distributions |
| **Curriculum** | None — static policy from epoch 1 |
| **Per-sample** | No |
| **Search cost** | ~2 GPU hours (orders of magnitude faster than AutoAugment) |
| **CIFAR-100 WRN-28-10** | ~82.7% |
| **Relevance** | Low — same category as AutoAugment, just faster search |
| **Thesis use** | One-line mention alongside AutoAugment |

**Summary for thesis:**
Fast AutoAugment reduces AutoAugment search time via density matching while maintaining comparable accuracy. The contribution is search efficiency, not augmentation strategy — no curriculum, no progressive difficulty.

---

### Paper 3 — OHL-Auto-Aug

| Field | Details |
|:---|:---|
| **Title** | Online Hyper-parameter Learning for Auto-Augmentation Strategy |
| **Authors** | Lin, Guo, Li, Xin, Wu, Lin, Ouyang, Yan |
| **Venue** | ICCV 2019 |
| **arXiv** | 1905.07373 |
| **Method** | Formulates augmentation policy as a parameterised probability distribution optimised jointly with network weights — no separate search phase |
| **Curriculum** | None |
| **Per-sample** | No |
| **Search cost** | ~0 (online, integrated into training) |
| **CIFAR-100 WRN-28-10** | ~82–83% (competitive with AutoAugment) |
| **Relevance** | Low — search efficiency paper, not curriculum |
| **Thesis use** | One-line mention in search-based group |

**Summary for thesis:**
OHL-Auto-Aug integrates policy optimisation into standard training (60× faster than AutoAugment) by treating augmentation probabilities as differentiable parameters. Like AutoAugment and Fast AutoAugment, the resulting policy is static per epoch — no progressive difficulty or curriculum structure.

---

## Group 2 — Simple Random Policies

> Papers that apply random augmentation uniformly with no search and no curriculum.
> **Thesis positioning:** Strong baselines that challenge complexity; your Random Aug baseline directly tests this philosophy.

---

### Paper 4 — TrivialAugment

| Field | Details |
|:---|:---|
| **Title** | TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation |
| **Authors** | Müller, Hutter |
| **Venue** | ICCV 2021 |
| **arXiv** | 2103.10158 |
| **Method** | Applies exactly one randomly selected augmentation per image, sampled uniformly from all ops and magnitudes. Zero tuning, zero search |
| **Curriculum** | None |
| **Per-sample** | Random, not adaptive |
| **Search cost** | None |
| **CIFAR-100 WRN-28-10** | ~83.4% (200 epochs) |
| **Relevance** | High — challenges your accuracy numbers directly |
| **Thesis use** | Related work + honest comparison; your Random Aug baseline addresses this |

**Summary for thesis:**
TrivialAugment demonstrates that a single random augmentation per image, with no tuning, achieves state-of-the-art results. This challenges complexity-based approaches. However, TrivialAugment addresses a different question than your thesis — it maximises accuracy with minimal engineering, while your work investigates *when* and *why* progressive scheduling provides an advantage. Your Table 3 shows that the curriculum advantage (+3.89pp) emerges precisely when augmentation ops are most distorting — the regime TrivialAugment applies indiscriminately.

---

## Group 3 — Adaptive Learned Policies

> Papers that adapt augmentation to the model's current state during training.
> **Thesis positioning:** Most competitive group; share the adaptivity principle with EGS but use learned policies vs your manual tiers.

---

### Paper 5 — Adversarial AutoAugment

| Field | Details |
|:---|:---|
| **Title** | Adversarial AutoAugment |
| **Authors** | Zhang, Wang, Zhang, Zhong |
| **Venue** | ICLR 2020 |
| **arXiv** | 1912.11188 |
| **Method** | Policy network generates augmentations designed to maximise training loss (anti-curriculum). Target network learns to be robust against hardest possible augmentations at every step |
| **Curriculum** | Anti-curriculum — always maximises difficulty |
| **Per-sample** | No — global policy per batch |
| **Search cost** | ~12× less than AutoAugment |
| **CIFAR-100 WRN-28-10** | ~82.5% |
| **Relevance** | Medium — adaptive to model state but opposite direction to curriculum |
| **Thesis use** | Contrast to EGS; supports why easy→hard matters (Table 11) |

**Summary for thesis:**
Adversarial AutoAugment adapts augmentation to the model's current state, always finding the hardest augmentations. This is the conceptual opposite of curriculum learning. The performance gap vs your reverse curriculum (Table 11, −3.18pp) supports the interpretation that starting with hard augmentations is counterproductive — consistent with what Adversarial AutoAugment does from epoch 1.

---

### Paper 6 — MADAug

| Field | Details |
|:---|:---|
| **Title** | When to Learn What: Model-Adaptive Data Augmentation Curriculum |
| **Authors** | Hou, Zhang, Zhou |
| **Venue** | ICCV 2023 |
| **arXiv** | 2309.04747 |
| **Method** | Trains a policy network alongside target model via bi-level optimisation. Policy conditioned on intermediate features generates per-image augmentation probabilities. Curriculum formed via tanh schedule: p(t) = tanh(t/τ) |
| **Curriculum** | Yes — tanh-based progressive curriculum (global, not per-sample) |
| **Per-sample** | Yes — policy conditioned on per-image features |
| **Search cost** | None — integrated into training (~1.8 hrs total) |
| **CIFAR-100 WRN-28-10** | **83.9% ± 0.10%** |
| **Relevance** | HIGHEST — most directly related paper; explicitly uses curriculum augmentation |
| **Thesis use** | Primary related work; acknowledge accuracy gap; defend interpretability and simplicity |

**Summary for thesis:**
MADAug is the closest prior work. It explicitly forms a data augmentation curriculum using a learned policy network and bi-level optimisation, achieving 83.9% on CIFAR-100 WRN-28-10 vs your 81.35% (−2.55pp). Key differences: (1) MADAug learns the curriculum implicitly — the tanh schedule and policy network are not human-interpretable; your 3-tier design is fully explicit. (2) MADAug requires an additional policy network and bi-level training loop; your ETS/LPS add near-zero overhead. (3) MADAug's curriculum is continuous; yours is discrete and analysable. The accuracy gap is acknowledged honestly; the thesis contribution is interpretability and simplicity, not raw accuracy maximisation.

---

### Paper 7 — AdaAugment

| Field | Details |
|:---|:---|
| **Title** | AdaAugment: A Tuning-Free and Adaptive Approach to Enhance Data Augmentation |
| **Authors** | Yang, Li, Xiong, Shen, Zhao (Nanjing University) |
| **Venue** | 2024 |
| **arXiv** | 2405.11467 |
| **Method** | RL policy network dynamically controls per-sample augmentation magnitudes (not op selection). Scheduling parameter λ decreases from 1→0: early training = weak augmentation, later = strong. 14-op fixed pool. Dual-model: policy + target |
| **Curriculum** | Yes — progressive weak→strong via λ schedule (explicit curriculum) |
| **Per-sample** | Yes — magnitudes adapted per sample |
| **Search cost** | None — integrated into training |
| **CIFAR-100 WRN-28-10** | **83.23%** |
| **CIFAR-100 ResNet-50** | **81.46%** |
| **Relevance** | Very high — uses same 14-op pool, progressive curriculum, per-sample adaptation |
| **Thesis use** | Direct related work; shares curriculum principle; higher accuracy via magnitude adaptation |

**Summary for thesis:**
AdaAugment shares three key properties with your work: (1) progressive curriculum (weak→strong), (2) per-sample adaptation, (3) same 14-op pool. The key difference is mechanism: AdaAugment learns magnitude schedules via RL; your work uses manually designed discrete tiers. AdaAugment achieves 83.23% on WRN-28-10 (−1.88pp vs yours), suggesting magnitude adaptation provides additional gains beyond op-level curriculum. On ResNet-50, the gap narrows to 0.96pp (81.46% vs 80.50%).

---

## Group 4 — Narrow Curriculum Augmentation

> Papers that apply curriculum to a single augmentation operation rather than a full pipeline.
> **Thesis positioning:** Supports your curriculum motivation but narrow scope; your work is the first to apply curriculum across the full pipeline with explicit tier structure.

---

### Paper 8 — Colorful Cutout

| Field | Details |
|:---|:---|
| **Title** | Colorful Cutout: Enhancing Image Data Augmentation with Curriculum Learning |
| **Authors** | Choi, Kim |
| **Venue** | ICLR 2024 Tiny Papers |
| **arXiv** | 2403.20012 |
| **Method** | Applies curriculum to Cutout only. Erasure box divided into 2^N_epoch sub-regions — exponential difficulty increase per epoch |
| **Curriculum** | Yes — single-operation curriculum (Cutout only) |
| **Per-sample** | No — global schedule |
| **Search cost** | None |
| **CIFAR-100 ResNet-50** | 81.57% (+0.42pp from curriculum ablation) |
| **CIFAR-100 WRN-28-10** | Not reported |
| **Relevance** | Medium — validates curriculum principle; narrow scope |
| **Thesis use** | Supporting citation; confirms curriculum helps even at single-op level |

**Summary for thesis:**
Colorful Cutout validates the curriculum principle at the operation level: progressively increasing Cutout complexity improves ResNet-50 CIFAR-100 accuracy by +0.42pp over static Cutout (ablation result). However, the method applies curriculum only to a single operation in isolation. Your work is more comprehensive: curriculum is applied across the full 19-operation pipeline with explicit tier boundaries, multiple advancement signals, and mixing strategy.

---

## Summary Comparison Table

| Paper | Year | Venue | CIFAR-100 WRN | Curriculum | Per-sample | Search | Relevance |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AutoAugment | 2019 | CVPR | ~82.9% | No | No | ~5000 hrs | Medium |
| Fast AutoAugment | 2019 | NeurIPS | ~82.7% | No | No | ~2 hrs | Low |
| OHL-Auto-Aug | 2019 | ICCV | ~82-83% | No | No | ~0 hrs | Low |
| TrivialAugment | 2021 | ICCV | ~83.4% | No | No | None | High |
| Adversarial AutoAugment | 2020 | ICLR | ~82.5% | Anti | No | Moderate | Medium |
| MADAug | 2023 | ICCV | **83.9%** | Yes (tanh) | Yes | None | **Highest** |
| AdaAugment | 2024 | — | **83.23%** | Yes (λ) | Yes | None | **Very High** |
| Colorful Cutout | 2024 | ICLR TP | N/A | Yes (1 op) | No | None | Medium |
| **Your ETS/LPS** | **2025** | **Thesis** | **81.35%** | **Yes (3-tier)** | **No** | **None** | — |
| **Your EGS** | **2025** | **Thesis** | **80.01%** | **Yes (3-tier)** | **Yes (entropy)** | **None** | — |

---

## Your Unique Position

Your work is the **only approach** that:

1. **Applies curriculum across the full augmentation pipeline** (not a single op like Colorful Cutout)
2. **Uses an explicit, interpretable 3-tier structure** with defined op sets per tier (vs learned black-box policies in MADAug/AdaAugment)
3. **Provides three interchangeable advancement signals** (ETS, LPS, EGS) analysed under a unified framework
4. **Adds zero overhead for ETS/LPS** (vs policy networks in MADAug/AdaAugment)
5. **Demonstrates the benefit scales with augmentation difficulty** (the +3.89pp finding in Table 3 — not shown in any prior work)
6. **Proves ordering matters** via reverse curriculum ablation (−3.18pp, Table 11)

---

## Recommended Related Work Structure

### 2.1 Automated Augmentation Policy Search
> AutoAugment → Fast AutoAugment → OHL-Auto-Aug (one paragraph)

### 2.2 Random and Tuning-Free Augmentation
> TrivialAugment (one paragraph)

### 2.3 Adaptive Augmentation
> Adversarial AutoAugment → MADAug → AdaAugment (two paragraphs)

### 2.4 Curriculum Learning in Augmentation
> Colorful Cutout + your work positioning (one paragraph)

### 2.5 Curriculum Learning (General)
> Bengio et al. (2009) — foundational CL paper
> Weinshall et al. (2018) — transfer learning CL
> (brief, 3–4 sentences)

---

*Generated: 2026-05-21 | curriculum-augmentation thesis project*
