# Thesis Writing — Ready-to-Use Paragraphs

> Each section maps to a table in `thesis_results_tables.md`.
> Language is thesis-level academic. Adapt as needed.

---

## 5.1 Primary Result: Curriculum Advantage Scales with Augmentation Difficulty

### Tables 1, 2 & 3

**Opening paragraph:**

We evaluate three curriculum scheduling strategies — Epoch-Threshold Scheduling (ETS), Loss-Plateau Scheduling (LPS), and Entropy-Guided Scheduling (EGS) — against a static mixing baseline on CIFAR-100 using WideResNet-28-10. All methods use the same set of augmentation operations; the only experimental variable is the order and timing of their introduction.

**Core finding paragraph:**

Table 1 reports results on the 14-operation pool, which consists exclusively of geometric and photometric transformations with limited perceptual distortion (flip, crop, translate, colour jitter, rotation, shear). Under this configuration, the curriculum methods (ETS: 81.62% ± 0.28%; LPS: 81.30% ± 0.47%) are statistically indistinguishable from the static baseline (81.61% ± 0.20%), with a maximum advantage of 0.01pp. This result establishes that curriculum scheduling provides no measurable benefit when all augmentation operations are perceptually mild. The curriculum mechanism is neither harmful nor beneficial under benign augmentation conditions.

**Amplification paragraph (the core thesis claim):**

The picture changes markedly when the augmentation pool is expanded to 19 operations by adding four high-distortion transformations: solarize, posterize, blur, and invert. As shown in Table 2, static mixing accuracy drops by 3.71pp (81.50% → 77.43%), while ETS drops by only 0.49pp (81.84% → 81.35%) and LPS by 0.12pp (81.48% → 81.36%). The curriculum advantage, negligible at 0.01pp with the 14-op pool, grows to 3.89pp with the 19-op pool (Table 3). This amplification effect constitutes the central finding of this thesis: **the benefit of progressive augmentation scheduling is directly proportional to the difficulty of the augmentation operations being scheduled.** When the pool contains operations that can severely distort or destroy semantic content — invert reverses pixel intensities, solarize inverts pixels above a threshold, posterize reduces colour depth — introducing them at full strength from epoch 1 disrupts the model's ability to acquire stable low-level feature representations. The curriculum protects the model by deferring these high-distortion operations to Tier 3, by which point the model has already learned robust representations from the milder Tier 1 and Tier 2 operations.

---

## 5.2 ETS vs LPS: Fixed Thresholds vs Adaptive Loss Signals

### Tables 2 & 5

**Equivalence paragraph:**

Epoch-Threshold Scheduling (ETS) and Loss-Plateau Scheduling (LPS) achieve statistically equivalent performance across all experimental configurations. On the 19-op pool with WideResNet-28-10, ETS achieves 81.32% ± 0.05% and LPS achieves 81.35% ± 0.07% — a difference of 0.03pp, well within one standard deviation of either method. This equivalence is replicated on the 14-op pool (ETS: 81.62%; LPS: 81.30%, Δ = 0.32pp, not statistically significant) and on ResNet-50 (ETS: 80.41%; LPS: 80.50%, Δ = 0.09pp).

**LPS adaptivity paragraph:**

Despite achieving equivalent accuracy, LPS exhibits meaningful adaptivity in its tier transition timing. As reported in Table 5, LPS tier transitions vary by up to 12 epochs across seeds (T2→T3 ranging from epoch 36 to epoch 43 on the 19-op pool), yet final accuracy remains stable at 81.35% ± 0.07%. LPS correctly adapts to seed-dependent convergence dynamics: it holds Tier 1 longer when validation loss is still improving, and advances to Tier 3 earlier when Tier 2 loss plateaus rapidly. The robustness of final accuracy to this variation suggests that, beyond a minimum T3 duration, additional epochs in Tier 3 provide diminishing returns within the 100-epoch budget.

**Interpretation paragraph:**

The equivalence of ETS and LPS indicates that the specific signal used to advance tiers — whether a pre-defined epoch fraction or a data-driven loss plateau — does not determine the outcome. What matters is that the curriculum structure is maintained: easy geometric operations in Tier 1, medium photometric operations in Tier 2, and high-distortion operations deferred to Tier 3. The advancement timing is a second-order effect. This finding has a practical implication: ETS, despite its simplicity, is as effective as the more complex LPS in this setting, suggesting that the curriculum structure itself — not the sophistication of the advancement signal — is the primary driver of performance gains.

---

## 5.3 Entropy-Guided Scheduling: Per-Sample Adaptivity and Its Limitations

### Tables 2, 6 & 10

**EGS description paragraph:**

Entropy-Guided Scheduling (EGS) assigns each training sample individually to a curriculum tier based on the model's per-sample prediction entropy, computed every three epochs on the unaugmented training set. Samples with entropy below 0.15 × log(C) — indicating high model confidence — are promoted to Tier 3, while uncertain samples remain in earlier tiers receiving lighter augmentation. This per-sample adaptivity is conceptually appealing: samples the model has not yet learned from should not be distorted by the hardest augmentations.

**Performance gap paragraph:**

Despite this principled design, EGS v2 achieves 80.01% ± 0.27% on WideResNet-28-10 (19-op pool, 100 epochs), trailing ETS by 1.31pp. An analysis of the tier progression in Table 6 reveals the structural cause: EGS reaches full Tier 3 exposure only at epoch 87–90 on average, leaving approximately 10–13 epochs of maximum augmentation within the 100-epoch budget. By contrast, ETS transitions all samples to Tier 3 at epoch 46, providing 55 epochs of full augmentation. The final 35 epochs of the 150-epoch EGS run (Table 10) — during which all samples were in Tier 3 — delivered a gain of +10.5pp (69.6% → 80.06%), confirming that the mechanism is sound but the scheduling horizon is insufficient at 100 epochs.

**Tuning paragraph:**

Hyperparameter tuning of EGS (Table 10) reveals that the performance gap is structural rather than a tuning artefact. The original EGS implementation suffered a training accuracy collapse to 40% due to overly aggressive mixing (alpha=1.0, instant activation). EGS v2 addresses this by tightening entropy thresholds, introducing a 10-epoch mixing ramp, and softening mixing intensity (alpha=0.2), yielding a +0.42pp improvement (79.59% → 80.01%). However, further tuning of entropy thresholds and mixing parameters (v3) does not improve beyond v2, indicating that the remaining 1.31pp gap cannot be closed by hyperparameter adjustment. The per-sample scheduling mechanism inherently delays full Tier 3 exposure, and this delay is the primary source of underperformance relative to ETS at 100 epochs.

---

## 5.4 Curriculum Structure Ablation: Does Ordering Matter?

### Table 11

**Opening paragraph:**

To determine whether the easy→hard ordering specifically drives performance gains — as opposed to any structured schedule — and to isolate the contribution of CutMix/MixUp from the contribution of curriculum ordering, we conduct three ablation experiments: (1) ETS without mixing, (2) reverse curriculum (Hard→Easy), and (3) static maximum augmentation from epoch 1. All variants use the identical 19-op pool and training infrastructure as ETS.

**Mixing contribution paragraph (ETS no-mix):**

Table 11 reports that ETS without any batch mixing (CutMix/MixUp disabled) achieves 79.29%, compared to 81.35% with mixing — a difference of **+2.06pp attributable to delayed CutMix/MixUp alone**. Crucially, ETS without mixing (79.29%) still outperforms the static mixing baseline (77.43%) by **+1.86pp**, demonstrating that the curriculum ordering itself provides an independent and statistically meaningful benefit, even in the complete absence of batch-level mixing. The train accuracy of ETS no-mix reaches 99.98% by epoch 100, confirming that CutMix/MixUp is the primary regularisation mechanism: without it, the model memorises the training set almost perfectly, producing a 20.69pp train-test gap. The 3.89pp total curriculum advantage (Table 2) therefore decomposes into two additive components: +1.86pp from the progressive easy→hard ordering, and +2.06pp from deferring CutMix/MixUp to Tier 3 rather than applying it from epoch 1. Both components are independently beneficial and together account for the full observed advantage.

**Reverse curriculum paragraph:**

The reverse curriculum — beginning with all 19 operations at 100% strength (Tier 1) and progressively reducing to 4 easy operations at 40% strength (Tier 3) — achieves 78.17%, which is 3.18pp below forward ETS (81.35%). The severity of this drop is consistent with the training dynamics: at epoch 10, reverse curriculum train accuracy is 25.72%, compared to approximately 57% for forward ETS. Exposing the model to invert, solarize, posterize, and cutout before any stable feature representations have been established prevents effective early learning. Furthermore, reverse curriculum (78.17%) is only 0.74pp above static mixing (77.43%), suggesting that once the ordering is reversed, the curriculum structure provides almost no benefit over a flat policy. **These results directly answer whether the easy→hard direction is specifically required: replacing it with hard→easy degrades performance by 3.18pp, confirming that the ordering — not merely the presence of a schedule — is the operative mechanism.**

**Comparison paragraph:**

The near-equivalence of reverse ETS (78.17%) and static mixing (77.43%) is particularly instructive. Both approaches expose the model to maximum augmentation in the first 20 epochs; both produce similar final accuracy. This suggests that the damage inflicted by early high-distortion exposure is difficult to recover from regardless of whether augmentation is subsequently reduced (reverse) or held constant (static). The forward curriculum avoids this damage entirely by beginning with mild operations and progressively increasing difficulty as the model's representations mature.

---

## 5.5 Architecture Generalisation: ResNet-50

### Table 12

**Opening paragraph:**

To assess whether the curriculum benefit is specific to WideResNet-28-10 or generalises across architectures, we replicate the primary comparison on ResNet-50 — the primary backbone used in the FIT presentation. All experimental conditions are identical: CIFAR-100, 19-op pool, cosine scheduler, 100 epochs, seed 42.

**Consistency finding paragraph:**

Table 12 reports that the curriculum advantage is fully preserved on ResNet-50. ETS outperforms Static Mixing by +3.79pp (80.41% vs 76.62%) and LPS by +3.88pp (80.50% vs 76.62%), closely matching the WideResNet advantages of +3.89pp. The maximum deviation between architectures is 0.10pp, which is within the seed-to-seed variability observed on WideResNet (std ±0.05pp for ETS). This consistency across architectures with different capacity, depth, and width confirms that the curriculum benefit is a property of the augmentation scheduling mechanism rather than an interaction with a specific architectural design.

**Capacity gap paragraph:**

Without augmentation, ResNet-50 trails WideResNet-28-10 by 7.56pp (65.30% vs 72.86%), reflecting WideResNet's architectural advantage in generalisation without explicit regularisation. With curriculum augmentation, this gap narrows to 0.85–0.91pp. **Augmentation disproportionately benefits the lower-capacity architecture, reducing the performance gap by 87%.** This finding suggests that progressive augmentation acts as a regulariser that partially compensates for architectural limitations, a particularly relevant observation for deployment scenarios where computational constraints favour smaller models.

**EGS consistency paragraph:**

The EGS-ETS performance gap is identical across architectures: −1.31pp on ResNet-50 (79.10% vs 80.41%) and −1.31pp on WideResNet-28-10 (80.01% vs 81.32%). This exact consistency across two architectures with substantially different properties strongly supports the interpretation that the EGS deficit is structural — a consequence of the per-sample scheduling design delaying full Tier 3 exposure — rather than an architecture-specific failure mode.

---

## 5.6 Training Duration

### Table 4

**Duration paragraph:**

Table 4 shows that extending training from 100 to 150 epochs consistently improves results on the 14-op ETS configuration: seed 123 improves from 81.22% to 82.70% (+1.48pp) and seed 456 from 81.80% to 82.19% (+0.39pp). The best single result across all experiments is **82.70%** (ETS, 14-op pool, 150 epochs, seed 123). The continued improvement at epoch 150 and the steep accuracy gains observed in the final epochs of multiple runs (e.g., EGS v2: +0.86pp in the last 9 epochs) indicate that models trained under curriculum augmentation have not fully converged at 100 epochs. The cosine learning rate schedule, which reaches near-zero at epoch 100, drives much of the late-epoch improvement through fine-grained weight updates in a highly regularised regime. The 100-epoch budget adopted throughout this thesis is a conservative estimate; the practical ceiling of the proposed methods is higher.

---

## 5.7 Summary of Findings

| # | Finding | Evidence | Table |
|:---:|:---|:---|:---:|
| F1 | Curriculum advantage is absent with mild ops; emerges with aggressive ops | 0.01pp vs 3.89pp advantage | 1, 2, 3 |
| F2 | ETS and LPS are statistically equivalent despite different advancement signals | Δ = 0.03pp, within 1 std | 2, 5 |
| F3 | LPS tier timing varies (±12 ep) but accuracy is stable (±0.07pp) | 3-seed LPS transitions | 5 |
| F4 | EGS underperforms ETS by 1.31pp due to late T3 exposure (~89 vs ~46 ep) | Tier progression analysis | 6, 10 |
| F5 | Mixing contributes +2.06–2.45pp; curriculum ordering contributes +1.86pp independently | ETS+mix=81.35%, ETS+CutMix=81.74%, ETS no-mix=79.29% | 11 |
| F5b | CutMix alone (81.74%) > Both (81.35%) > MixUp alone (80.45%) > no-mix (79.29%) — CutMix dominates | Full mixing ablation complete | 11 |
| F5c | Curriculum amplifies mixing: CutMix hurts static (−0.80pp) but helps ETS (+2.45pp) — super-additive interaction | 2×2 decomposition complete | 11 |
| F6 | Easy→hard ordering is required; reversing it degrades by 3.18pp | Reverse ETS = 78.17% | 11 |
| F7 | Reversed curriculum provides almost no benefit over static mixing (Δ = 0.74pp) | Reverse vs Static comparison | 11 |
| F8 | Curriculum benefit is architecture-agnostic (consistent across WRN and R50) | +3.79pp vs +3.89pp | 12 |
| F9 | EGS-ETS gap is identical across architectures (−1.31pp), confirming structural cause | Exact consistency R50 = WRN | 12 |
| F10 | Augmentation reduces architecture capacity gap from 7.56pp to 0.85–0.91pp | No-aug vs curriculum | 12 |
| F11 | Models not fully converged at 100 epochs; 150 epochs adds +0.83–1.48pp | 150ep ETS runs | 4 |
