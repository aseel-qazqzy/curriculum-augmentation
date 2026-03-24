# Thesis Proposal

**Title:** Loss-Guided Curriculum Augmentation for Image Classification:
A Progressive Difficulty Framework for Data Augmentation in Deep Neural Networks

**Author:** Aseel Ahmed Al-Qazqzy
**Institution:** University of Hildesheim
**Program:** [Your Program Name]
**Supervisor:** [Supervisor Name]
**Date:** March 2026

---

## Abstract

Data augmentation is a widely used regularization technique in deep learning for image classification. Conventional approaches apply a fixed augmentation pipeline uniformly across all training samples and all training epochs. This thesis proposes and evaluates **Loss-Guided Curriculum Augmentation (LGCA)**, a method that dynamically adjusts augmentation difficulty over the course of training using the model's own per-sample cross-entropy loss as a real-time difficulty signal. The core hypothesis is that starting training with mild augmentations and progressively introducing harder transformations — while further adapting per sample based on the model's current performance — leads to better generalization than applying a static augmentation policy. Experiments are conducted on CIFAR-10 and CIFAR-100 using ResNet-18, with systematic ablation studies isolating the contribution of each component of the framework.

---

## 1. Introduction and Motivation

### 1.1 The Problem

Deep neural networks for image classification are highly prone to overfitting, particularly when training data is limited. Data augmentation — artificially expanding the training set through image transformations such as cropping, flipping, color jittering, and erasing — is the standard countermeasure. However, the predominant practice is to apply the same augmentation pipeline at every epoch, to every sample, at full strength from the first epoch to the last.

This one-size-fits-all approach has a fundamental mismatch with the learning dynamics of neural networks. In the early stages of training, the model has not yet learned even coarse features. Presenting heavily distorted images at this stage introduces noise that competes with the signal the model needs to form stable representations. In the late stages of training, when the model has largely converged, applying the same mild augmentations that were used at the start may be insufficient to push generalization further.

### 1.2 The Curriculum Learning Insight

Curriculum Learning, introduced by Bengio et al. (2009), proposes that training on examples ordered from easy to hard improves convergence and generalization — analogous to how humans learn. Prior work has applied this idea to sample selection (training on simpler examples first), but few works have applied it systematically to **augmentation difficulty** itself.

The key insight motivating this thesis is:

> *If the model finds a sample easy (low loss), it has already learned that sample — apply harder augmentation to prevent memorization and push robustness. If the model finds a sample hard (high loss), it is still struggling — protect it with mild augmentation so the model can first learn the underlying pattern.*

This is a form of **adaptive, loss-driven curriculum** applied at the augmentation level, operating per-sample and updated dynamically each epoch.

### 1.3 Why This Has Not Been Fully Explored

Existing strong baselines such as RandAugment (Cubuk et al., 2020), AutoAugment (Cubuk et al., 2019), and TrivialAugment (Müller & Hein, 2021) focus on **which** augmentations to apply and **how strongly**, optimizing policies over the space of possible transformations. None of them explicitly model the **temporal dimension** of training or use the model's own loss signal to adapt augmentation strength per sample. This thesis fills that gap.

---

## 2. Research Questions

This thesis addresses the following research questions:

**RQ1 (Primary):**
Does progressively increasing augmentation difficulty from easy to hard over the course of training improve classification accuracy on CIFAR-10 and CIFAR-100 compared to static augmentation?

**RQ2 (Curriculum vs. Random):**
Is the *order* of augmentation difficulty important, or does applying all augmentations randomly (without a schedule) achieve equivalent performance? If Curriculum > Random, the progressive ordering is the genuine contribution.

**RQ3 (Loss-Guided Adaptation):**
Does using the model's per-sample loss to fine-tune individual augmentation difficulty provide additional benefit beyond a purely epoch-level schedule?

**RQ4 (Schedule Shape):**
Which epoch-level difficulty schedule (sigmoid, linear, cosine, step) best supports the curriculum, and does this interact with the learning rate schedule?

**RQ5 (Robustness):**
Does curriculum augmentation improve not just clean accuracy but also **corruption robustness** on CIFAR-10-C (Hendrycks & Dietterich, 2019), measured by mean Corruption Error (mCE)?

---

## 3. Related Work

### 3.1 Static Data Augmentation

Standard augmentation pipelines for CIFAR-scale datasets consist of random horizontal flip, random crop with padding, and color jitter. These are fixed from the first to the last epoch and applied identically to every sample (Krizhevsky et al., 2012; He et al., 2016). While effective, they treat augmentation as a static regularizer rather than an adaptive training strategy.

### 3.2 Automated Augmentation Search

**AutoAugment** (Cubuk et al., 2019) uses reinforcement learning to search for optimal augmentation policies on the validation set. While powerful, it requires expensive search computation and learns a fixed policy for the entire training run.

**RandAugment** (Cubuk et al., 2020) simplifies this by sampling augmentations uniformly from a predefined set with two hyperparameters (N operations, magnitude M). Magnitude is fixed throughout training.

**TrivialAugment** (Müller & Hein, 2021) further simplifies by sampling a single random augmentation with random strength per image, achieving surprisingly competitive results with minimal overhead.

All three methods are **static** in the temporal sense — the augmentation policy does not change as training progresses, and they do not use the model's learning signal.

### 3.3 Curriculum Learning

**Bengio et al. (2009)** introduced Curriculum Learning, showing that presenting training examples in a meaningful order (easy to hard) improves convergence and generalization. Subsequent work has applied curriculum ideas to:
- Sample weighting and selection (Jiang et al., 2018 — MentorNet)
- Self-paced learning (Kumar et al., 2010)
- Mixup scheduling (Guo et al., 2019)
- Loss-driven sample reweighting (Ren et al., 2018)

The application of curriculum principles specifically to **augmentation strength scheduling** has been explored partially in:
- **Curriculum by Smoothing** (Sinha et al., 2020): gradually reduces label smoothing over training
- **Progressive Augmentation** (linearscaling of magnitudes): informal practice in some pipelines

However, no published work combines (a) a progressive epoch-level augmentation schedule with (b) per-sample adaptation driven by the model's live loss signal in a unified framework.

### 3.4 Loss-Based Sample Difficulty

**Forgetting events** (Toneva et al., 2019) and **loss dynamics** (Swamynathan et al., 2022) have been used to characterize sample difficulty. High loss samples are consistently harder across training. This thesis leverages this observation in the opposite direction from most prior work: instead of filtering hard samples out, it reduces augmentation noise for them so the model can learn them more effectively.

---

## 4. Methodology

### 4.1 Framework Overview

The proposed framework, **Loss-Guided Curriculum Augmentation (LGCA)**, has four components:

```
┌─────────────────────────────────────────────────────────────┐
│                   LGCA Framework                            │
│                                                             │
│  1. Augmentation Primitives  (10 ops, strength 0→1)         │
│  2. Epoch-Level Schedule     (global difficulty 0→1)         │
│  3. Per-Sample Loss Signal   (individual difficulty fine-tune)│
│  4. Blended Difficulty       (weighted combination)          │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Augmentation Primitives

Ten augmentation operations are organized into three difficulty tiers, each parameterized by a `strength` ∈ [0, 1]:

| Tier   | Operations                                        | Activation Threshold |
|--------|---------------------------------------------------|----------------------|
| Easy   | Horizontal Flip, Random Crop                      | difficulty ≥ 0.00    |
| Medium | Color Jitter, Rotation, Shear                     | difficulty ≥ 0.25    |
| Hard   | Grayscale, Gaussian Blur, Cutout, Solarize, Posterize | difficulty ≥ 0.55 |

When a new augmentation first activates, its strength begins near zero and increases linearly to 1.0 as difficulty approaches 1.0, preventing discontinuous jumps in augmentation intensity.

### 4.3 Epoch-Level Difficulty Schedule

A monotonically increasing function maps the current epoch to a global difficulty score d_epoch ∈ [0, 1]:

**Sigmoid (default):**
```
d_epoch = 1 / (1 + exp(-((p - 0.5) / 0.15)))
```
where p = (epoch − warmup) / (total_epochs − warmup)

This produces a slow start, rapid ramp in the middle third of training, and a plateau near the end. A warmup period (default: 5 epochs) locks d_epoch = 0, allowing the model to form basic representations before any augmentation is applied.

Four schedule variants are compared in ablation: sigmoid, linear, cosine, step.

### 4.4 Per-Sample Loss-Guided Difficulty

After each forward pass, the per-sample cross-entropy loss l_i is computed for each sample i in the batch. Sample difficulty is derived as:

**Inverse mode (default):**
```
l_norm_i  = (l_i − l_min) / (l_max − l_min)   [normalize within batch]
d_sample_i = 1 − l_norm_i                       [invert: high loss → low difficulty]
```

A LossTracker maintains an Exponential Moving Average (EMA) of per-sample losses across epochs (momentum = 0.9), providing a smoothed signal that is robust to batch-level noise:
```
ema_i ← 0.9 × ema_i + 0.1 × l_i
```

### 4.5 Blended Final Difficulty

The per-sample final difficulty is a weighted blend of the epoch-level global signal and the per-sample loss signal:

```
d_final_i = α × d_epoch + (1 − α) × d_sample_i
```

where α = 0.7 (blend factor). This means the epoch schedule provides the coarse trajectory (easy→hard over training), while the loss signal provides fine-grained per-sample adjustment within each epoch.

The final difficulty score is clamped to [0, 1] and used to select active augmentations and their strengths for that sample at that epoch.

### 4.6 Training Protocol

All experiments use the following fixed protocol to ensure fair comparison:

| Hyperparameter      | Value                              |
|--------------------|------------------------------------|
| Model              | ResNet-18 (standard architecture)  |
| Dataset            | CIFAR-10 / CIFAR-100               |
| Train/Val/Test     | 45,000 / 5,000 / 10,000            |
| Epochs             | 150                                |
| Batch size         | 128                                |
| Optimizer          | SGD, momentum=0.9, Nesterov=True   |
| Learning rate      | 0.1 initial                        |
| LR schedule        | MultiStepLR, γ=0.1 at epochs 49, 99, 124 |
| Weight decay       | 5×10⁻⁴                            |
| Label smoothing    | 0.1 (CIFAR-100 only)               |
| Random seed        | 42                                 |
| Warmup epochs      | 5                                  |
| CL blend (α)       | 0.7                                |

---

## 5. Experimental Design

### 5.1 Baseline Comparisons

Four methods are compared on CIFAR-10 and CIFAR-100:

| Method                  | Description                                               |
|-------------------------|-----------------------------------------------------------|
| **NoAugmentation**      | No transforms — absolute floor, shows raw augmentation benefit |
| **StaticAugmentation**  | Fixed pipeline (crop + flip + color jitter) — main baseline to beat |
| **RandomAugmentation**  | All 10 augmentations randomly applied, no schedule — tests if order matters |
| **LGCA (ours)**         | Loss-guided curriculum augmentation — thesis method       |

### 5.2 Ablation Studies

Six ablation groups systematically isolate each design choice:

| Group | Variable              | Conditions                              | Purpose                                      |
|-------|-----------------------|-----------------------------------------|----------------------------------------------|
| A1    | Schedule shape        | sigmoid, linear, cosine, step           | Which difficulty ramp works best?            |
| A2    | Loss mapping mode     | inverse, direct, normalized             | Should hard samples get easy or hard augmentation? |
| A3    | Blend factor α        | 0.0, 0.3, 0.7, 1.0                      | How much should loss signal influence difficulty? |
| A4    | Warmup duration       | 0, 5, 10 epochs                         | Does initial protection improve learning?    |
| A5    | Label smoothing       | 0.0, 0.05, 0.1                          | Interaction with curriculum (CIFAR-100)      |

### 5.3 Robustness Evaluation

The best LGCA checkpoint is evaluated on **CIFAR-10-C** (Hendrycks & Dietterich, 2019), which contains 19 corruption types (Gaussian noise, blur, weather, digital distortions) at 5 severity levels. The metric is **mean Corruption Error (mCE)**, normalized against the AlexNet baseline:

```
mCE = mean over corruptions { model_error_c / alexnet_error_c }
```

Lower mCE = better corruption robustness. The hypothesis is that curriculum augmentation, having exposed the model to progressively harder transformations, produces more robust representations than static augmentation.

### 5.4 Evaluation Metrics

| Metric                | Description                                              |
|-----------------------|----------------------------------------------------------|
| Top-1 Val Accuracy    | Best validation accuracy during training                  |
| Top-1 Test Accuracy   | Accuracy on held-out test set (evaluated at best checkpoint) |
| Top-5 Test Accuracy   | Top-5 accuracy (meaningful for CIFAR-100)                |
| Generalization Gap    | Final train accuracy − best val accuracy (lower = less overfit) |
| mCE                   | Mean Corruption Error on CIFAR-10-C (lower = more robust)|
| Convergence Speed     | First epoch to reach 90% val accuracy                    |

---

## 6. Expected Results and Hypotheses

### 6.1 Primary Hypothesis

LGCA will achieve higher Top-1 test accuracy than StaticAugmentation on CIFAR-10, validating the benefit of progressive augmentation difficulty.

| Method              | Expected CIFAR-10 Acc | Expected CIFAR-100 Acc |
|---------------------|-----------------------|------------------------|
| NoAugmentation      | ~78%                  | ~55%                   |
| StaticAugmentation  | ~84%                  | ~63%                   |
| RandomAugmentation  | ~85%                  | ~64%                   |
| **LGCA (ours)**     | **~87%**              | **~66%**               |

### 6.2 Order Hypothesis (RQ2)

If LGCA > RandomAugmentation, the progressive schedule is the key contribution (not just the set of augmentations). This is the most critical comparison for the thesis claim.

### 6.3 Generalization Gap Hypothesis

LGCA should show a smaller generalization gap (train acc − val acc) than StaticAugmentation, indicating that loss-adaptive difficulty acts as a more effective regularizer.

### 6.4 Robustness Hypothesis

LGCA should achieve lower mCE than StaticAugmentation on CIFAR-10-C, because the curriculum progressively introduces corruption-like transformations (blur, noise-like cutout, solarize) at controlled intensity, providing richer exposure to distribution shifts.

### 6.5 Ablation Hypotheses

- **Schedule**: Sigmoid should outperform step, as its smooth ramp avoids abrupt difficulty jumps at milestone boundaries.
- **Mode**: Inverse mode (protect hard samples) should outperform direct mode, consistent with the core curriculum principle.
- **Blend**: A balanced blend (α=0.7) should outperform pure epoch schedule (α=1.0) or pure sample schedule (α=0.0), as both signals provide complementary information.
- **Warmup**: 5 epochs of warmup should outperform 0, as the model benefits from learning basic features before augmentation noise is introduced.

---

## 7. Contributions

This thesis makes the following contributions to the field:

1. **A unified loss-guided curriculum augmentation framework** that combines a global epoch-level difficulty schedule with per-sample adaptation driven by the model's live cross-entropy loss, applied specifically to augmentation strength rather than sample selection or weighting.

2. **A structured augmentation primitive library** with 10 operations organized into three difficulty tiers, each continuously parameterized by strength, enabling smooth curriculum transitions without discrete jumps.

3. **Systematic ablation studies** isolating the contribution of schedule shape, loss mapping mode, blend factor, and warmup duration, providing design guidance for practitioners.

4. **Empirical evaluation** on CIFAR-10 and CIFAR-100 against four baselines with a controlled training protocol, and robustness evaluation on CIFAR-10-C.

5. **A reproducible open codebase** implementing the full framework, all baselines, ablation runner, and analysis scripts.

---

## 8. Timeline

| Period         | Milestone                                                  |
|----------------|------------------------------------------------------------|
| Month 1        | Literature review finalization; codebase complete and tested |
| Month 2        | Run all baseline experiments (NoAug, Static, Random) on CIFAR-10 |
| Month 3        | Run LGCA experiments on CIFAR-10; run full ablation suite  |
| Month 4        | Run CIFAR-100 experiments; robustness evaluation on CIFAR-10-C |
| Month 5        | Analysis, figure generation, results interpretation        |
| Month 6        | Thesis writing: Introduction, Related Work, Methodology    |
| Month 7        | Thesis writing: Experiments, Results, Discussion, Conclusion |
| Month 8        | Revision, proofreading, supervisor feedback incorporation  |
| Month 9        | Final submission                                           |

---

## 9. References

Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *Proceedings of the 26th Annual International Conference on Machine Learning (ICML)*, 41–48.

Cubuk, E. D., Zoph, B., Mane, D., Vasudevan, V., & Le, Q. V. (2019). AutoAugment: Learning augmentation strategies from data. *Proceedings of CVPR 2019*, 113–123.

Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020). RandAugment: Practical automated data augmentation with a reduced search space. *Advances in Neural Information Processing Systems (NeurIPS) 2020*.

DeVries, T., & Taylor, G. W. (2017). Improved regularization of convolutional neural networks with cutout. *arXiv preprint arXiv:1708.04552*.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of CVPR 2016*, 770–778.

Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. *Proceedings of ICLR 2019*.

Jiang, L., Zhou, Z., Leung, T., Li, L. J., & Fei-Fei, L. (2018). MentorNet: Learning data-driven curriculum for very deep neural networks. *Proceedings of ICML 2018*.

Kumar, M. P., Packer, B., & Koller, D. (2010). Self-paced learning for latent variable models. *Advances in Neural Information Processing Systems (NeurIPS) 2010*.

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems (NeurIPS) 2012*.

Müller, S. G., & Hein, M. (2021). TrivialAugment: Tuning-free yet state-of-the-art data augmentation. *Proceedings of ICCV 2021*.

Ren, M., Zeng, W., Yang, B., & Urtasun, R. (2018). Learning to reweight examples for robust deep learning. *Proceedings of ICML 2018*.

Sinha, S., Garg, A., & Larochelle, H. (2020). Curriculum by smoothing. *Advances in Neural Information Processing Systems (NeurIPS) 2020*.

Toneva, M., Sordoni, A., des Combes, R. T., Trischler, A., Bengio, Y., & Gordon, G. J. (2019). An empirical study of example forgetting during deep neural network learning. *Proceedings of ICLR 2019*.

---

## Appendix A: Key Design Decisions and Rationale

### Why MultiStepLR over CosineAnnealingLR?

MultiStepLR produces discrete, predictable learning rate drops that align with the phase boundaries of the curriculum. When the LR drops at epoch 49 (from 0.1 to 0.01), the model has been trained on easy augmentation and has stabilized — the lower LR then fine-tunes within that regime before the next difficulty phase begins. CosineAnnealingLR decays continuously, reaching near-zero LR in the final epochs precisely when hard augmentation samples arrive — wasting the curriculum's hardest phase on a nearly frozen model.

### Why Inverse Mode as the Default?

The core curriculum principle is that hard samples (high loss) need *less* noise to learn their underlying pattern, while easy samples (low loss) need *more* noise to prevent memorization. Inverse mode directly implements this: high loss → low difficulty → mild augmentation. Direct mode does the opposite (focus hardest augmentation on samples already causing the most confusion), which is contrary to the curriculum principle and expected to perform worse in ablation.

### Why a Blend Factor of 0.7?

A pure epoch schedule (α=1.0) ignores all per-sample information and reduces to a coarser form of static augmentation. A pure sample schedule (α=0.0) is highly volatile — a single batch can swing all difficulties based on transient loss values. The blend at α=0.7 provides stable epoch-level progression as the backbone, with the loss signal providing adaptive fine-tuning within each epoch. This is validated in ablation study A3.

### Why ResNet-18?

ResNet-18 is a well-understood, reproducible architecture with known accuracy ranges on CIFAR-10/100. It is deep enough to benefit meaningfully from augmentation and CL, but light enough to run 150 full ablation experiments without requiring GPU clusters. It provides a clean, controlled experimental unit for isolating the effect of augmentation strategy.
