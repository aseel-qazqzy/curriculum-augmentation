"""
augmentations/policies.py
Baseline Augmentation Policies for Comparison.

These are the established methods your thesis compares against.
All are wrapped to use the same interface as your curriculum method.

Policies included:
    1. NoAugmentation    — absolute floor baseline
    2. StaticAugmentation — standard fixed pipeline (main baseline)
    3. RandomAugmentation — all 10 primitives at random strengths, no schedule
                            Critical RQ2 baseline: same ops as LGCA, but no ordering.

Reference papers:
    RandAugment:    https://arxiv.org/abs/1909.13719
    AutoAugment:    https://arxiv.org/abs/1805.09501
    TrivialAugment: https://arxiv.org/abs/2103.10158
"""

import random

import torch
import torchvision.transforms as T
from PIL import Image

from augmentations.primitives import AUGMENTATION_REGISTRY
from data.datasets import CIFAR_STATS as STATS


# ── Shared constants ──────────────────────────────────────────────────────────
# Both static and 3-tier curriculum use the same per-op strength.
# The ONLY difference between them is WHEN each op is introduced.
# This isolates the curriculum contribution cleanly.
FIXED_STRENGTH = 0.7

# Op groups by tier — matches the difficulty levels in primitives.py
# blur removed: deterministic blur on 32x32 CIFAR images every epoch
# creates a training/val distribution shift that harms all methods equally.
# 7 ops total — flip, crop, color_jitter, rotation, shear, grayscale, cutout.
_TIER_OPS = {
    1: ["flip", "crop"],                                                   # Easy
    2: ["flip", "crop", "color_jitter", "rotation", "shear"],             # +Medium
    3: ["flip", "crop", "color_jitter", "rotation", "shear",
        "grayscale", "cutout"],                                            # +Hard (7 ops, no blur)
}
# ─────────────────────────────────────────────────────────────────────────────


# POLICY BASE CLASS


class AugmentationPolicy:
    """
    Base class for all augmentation policies.
    All policies implement get_train_transform() and get_val_transform().
    """

    def __init__(self, dataset: str = "cifar10"):
        self.dataset   = dataset
        self.mean      = STATS[dataset]["mean"]
        self.std       = STATS[dataset]["std"]
        self.normalize = T.Normalize(self.mean, self.std)

    def get_train_transform(self) -> T.Compose:
        raise NotImplementedError

    def get_val_transform(self) -> T.Compose:
        """Validation transform is always the same — no augmentation."""
        return T.Compose([
            T.ToTensor(),
            self.normalize,
        ])

    def __repr__(self):
        return f"{self.__class__.__name__}(dataset={self.dataset})"


# 1. NO AUGMENTATION — Absolute floor baseline
class NoAugmentation(AugmentationPolicy):
    """
    No augmentation at all.
    This is your absolute floor — every method should beat this.
    Shows the raw benefit of augmentation.
    """

    def get_train_transform(self) -> T.Compose:
        return T.Compose([
            T.ToTensor(),
            self.normalize,
        ])



# 2. STATIC AUGMENTATION — Main baseline

class _FullStaticTransform:
    """
    All 7 ops applied at FIXED_STRENGTH from epoch 1 — no schedule, no ordering.

    Ops (same 7 as the 3-tier curriculum, same FIXED_STRENGTH):
        1. flip         — horizontal flip
        2. crop         — random crop with padding
        3. color_jitter — brightness / contrast / saturation / hue
        4. rotation     — random rotation
        5. shear        — random affine shear
        6. grayscale    — random grayscale conversion
        7. cutout       — erase a square patch

    blur excluded: deterministic blur on 32x32 images every epoch creates
    a training/val distribution shift that harms learning regardless of method.

    The ONLY difference vs ThreeTierCurriculumTransform: static applies all
    7 ops from epoch 1. Strength is identical (FIXED_STRENGTH) in both methods.
    """

    def __init__(self, dataset: str = "cifar10", strength: float = FIXED_STRENGTH):
        mean = STATS[dataset]["mean"]
        std  = STATS[dataset]["std"]
        self.normalize = T.Normalize(mean, std)
        self.to_tensor = T.ToTensor()
        self.strength  = strength

    def __call__(self, img):
        for name in _TIER_OPS[3]:   # all 7 ops, every epoch
            fn, _, _ = AUGMENTATION_REGISTRY[name]
            img = fn(img, self.strength)
        return self.normalize(self.to_tensor(img))


class StaticAugmentation(AugmentationPolicy):
    """
    All 8 augmentation primitives (difficulty=0.70) applied at full strength
    from epoch 1 — no schedule, no loss signal.

    Matches the max_difficulty=0.70 cap used in all CL experiments, so the
    only variable between static and CL is the progressive ordering —
    not the set of operations.

    Your curriculum method must beat this to prove that ordering matters.
    """

    def __init__(self, dataset: str = "cifar10", strength: float = FIXED_STRENGTH):
        super().__init__(dataset=dataset)
        self.strength = strength

    def get_train_transform(self) -> "_FullStaticTransform":
        return _FullStaticTransform(dataset=self.dataset, strength=self.strength)



# 3-TIER CURRICULUM AUGMENTATION

class ThreeTierCurriculumTransform:
    """
    Introduces augmentation ops in three progressive tiers aligned with the
    MultiStepLR milestones [33, 66]:

        Tier 1 — epochs   1–t1  : flip, crop                          (easy)
        Tier 2 — epochs t1+1–t2 : + color_jitter, rotation, shear     (+medium)
        Tier 3 — epochs t2+1–end: + grayscale, blur, cutout           (+hard, all 8)

    All ops use FIXED_STRENGTH throughout — identical to _FullStaticTransform.
    The ONLY variable vs static aug is WHEN each op is introduced.

    Call set_epoch(epoch) at the start of every training epoch so the
    transform knows which tier is active.
    """

    TIER_LABELS = {
        1: "Tier 1 [flip, crop]",
        2: "Tier 2 [+color_jitter, rotation, shear]",
        3: "Tier 3 [+grayscale, cutout]",
    }

    def __init__(self, dataset: str = "cifar10", t1: int = 33, t2: int = 66,
                 strength: float = FIXED_STRENGTH):
        mean = STATS[dataset]["mean"]
        std  = STATS[dataset]["std"]
        self.normalize = T.Normalize(mean, std)
        self.to_tensor = T.ToTensor()
        self.t1       = t1
        self.t2       = t2
        self.epoch    = 1
        self.strength = strength

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def tier(self) -> int:
        if self.epoch <= self.t1:
            return 1
        elif self.epoch <= self.t2:
            return 2
        return 3

    def tier_label(self) -> str:
        return self.TIER_LABELS[self.tier()]

    def __call__(self, img):
        for name in _TIER_OPS[self.tier()]:
            fn, _, _ = AUGMENTATION_REGISTRY[name]
            img = fn(img, self.strength)
        return self.normalize(self.to_tensor(img))


class ThreeTierCurriculumAugmentation(AugmentationPolicy):
    """
    3-tier curriculum augmentation policy.

    Tier boundaries default to [33, 66] to match MultiStepLR milestones —
    each LR drop coincides with the introduction of the next op tier:
        Tier 1 (easy)   → first LR drop at epoch 33 → Tier 2 (medium)
        Tier 2 (medium) → second LR drop at epoch 66 → Tier 3 (hard)

    Use with train_baseline.py --augmentation tiered_curriculum.
    The training loop must call train_transform.set_epoch(epoch) each epoch.
    """

    def __init__(self, dataset: str = "cifar10", t1: int = 33, t2: int = 66,
                 strength: float = FIXED_STRENGTH):
        super().__init__(dataset=dataset)
        self.t1       = t1
        self.t2       = t2
        self.strength = strength

    def get_train_transform(self) -> ThreeTierCurriculumTransform:
        return ThreeTierCurriculumTransform(dataset=self.dataset, t1=self.t1, t2=self.t2,
                                            strength=self.strength)


# RANDOM AUGMENT TRANSFORM

class RandomAugmentTransform:
    """
    Applies all 10 primitives from AUGMENTATION_REGISTRY with independently
    sampled random strengths in [0, 1].

    This is the RQ2 controlled baseline for the thesis:
        Same op pool as LGCA, same capacity, but no ordering.
        If LGCA > RandomAugment, the progressive curriculum schedule is the
        genuine contribution — not just the choice of augmentation operations.

    Design:
        - All 10 ops are applied on every forward call
        - Each op gets strength ~ Uniform(0, 1) sampled independently per image
        - Low strength values produce near-identity transforms, so the sampler
          naturally creates a soft "which ops are active" distribution without
          hard thresholds
        - No epoch counter, no loss tracker, no difficulty score

    Interface matches CurriculumTransform: callable on a PIL Image,
    returns a normalized torch.Tensor.
    """

    def __init__(self, dataset: str = "cifar10"):
        mean = STATS[dataset]["mean"]
        std  = STATS[dataset]["std"]
        self.normalize = T.Normalize(mean, std)
        self.to_tensor = T.ToTensor()
        # Ordered list of (name, fn) — same registry as LGCA, same ops
        self.ops = [(name, fn) for name, (fn, _, _) in AUGMENTATION_REGISTRY.items()]

    def __call__(self, img: Image.Image) -> torch.Tensor:
        for _, fn in self.ops:
            strength = random.random()   # Uniform[0, 1], independent per op
            img = fn(img, strength)
        return self.normalize(self.to_tensor(img))

    def __repr__(self) -> str:
        names = [name for name, _ in self.ops]
        return f"RandomAugmentTransform(ops={names})"



# 3. RANDOM AUGMENTATION — No schedule baseline
class RandomAugmentation(AugmentationPolicy):
    """
    All 10 primitives (same pool as LGCA) applied with independently sampled
    random strengths — no epoch schedule, no loss signal.

    This is a critical baseline for the thesis (RQ2):
        If RandomAugmentation ≈ LGCA → the ORDER of augmentation does not matter.
        If LGCA > RandomAugmentation → the progressive curriculum schedule is the
        genuine contribution, not just the set of operations used.

    Uses RandomAugmentTransform, which matches the interface of CurriculumTransform
    so both can be passed to the same DataLoader pipeline.
    """

    def get_train_transform(self) -> RandomAugmentTransform:
        return RandomAugmentTransform(dataset=self.dataset)

    def get_val_transform(self) -> T.Compose:
        return T.Compose([T.ToTensor(), self.normalize])



# ─────────────────────────────────────────────────────────────
# 4. RANDAUGMENT — Cubuk et al. (NeurIPS 2020)
# ─────────────────────────────────────────────────────────────

class RandAugmentPolicy(AugmentationPolicy):
    """
    RandAugment (Cubuk et al., NeurIPS 2020).

    Wraps RandAugmentTransform from augmentations/randaugment.py
    in the standard AugmentationPolicy interface.

    Paper hyperparameters for CIFAR-10:  N=2, M=9
    Paper hyperparameters for CIFAR-100: N=2, M=14

    Reference:
        Cubuk et al. (2020). RandAugment: Practical automated data augmentation
        with a reduced search space. NeurIPS 2020.
        https://arxiv.org/abs/1909.13719
    """

    def __init__(self, dataset: str = "cifar10", N: int = 2, M: int = 9):
        super().__init__(dataset=dataset)
        self.N = N
        self.M = M

    def get_train_transform(self):
        from augmentations.randaugment import RandAugmentTransform
        return RandAugmentTransform(N=self.N, M=self.M, dataset=self.dataset)

    def get_val_transform(self) -> T.Compose:
        return T.Compose([T.ToTensor(), self.normalize])

    def __repr__(self):
        return f"RandAugmentPolicy(dataset={self.dataset}, N={self.N}, M={self.M})"


POLICY_REGISTRY = {
    "none":              NoAugmentation,
    "static":            StaticAugmentation,
    "tiered_curriculum": ThreeTierCurriculumAugmentation,
    "random":            RandomAugmentation,
    "randaugment":       RandAugmentPolicy,
}


def get_policy(name: str, dataset: str = "cifar10") -> AugmentationPolicy:
    """
    Get an augmentation policy by name.

    Args:
        name:    policy name (see POLICY_REGISTRY keys)
        dataset: "cifar10" or "cifar100"

    Returns:
        AugmentationPolicy instance

    Usage:
        policy          = get_policy("randaugment", dataset="cifar10")
        train_transform = policy.get_train_transform()
        val_transform   = policy.get_val_transform()
    """
    name = name.lower().strip()
    if name not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy: '{name}'. "
            f"Available: {list(POLICY_REGISTRY.keys())}"
        )
    return POLICY_REGISTRY[name](dataset=dataset)


def get_all_baseline_policies(dataset: str = "cifar10") -> dict:
    """
    Returns all baseline policies — used in ablation/comparison scripts.

    Returns:
        dict of {name: AugmentationPolicy}
    """
    return {
        name: cls(dataset=dataset)
        for name, cls in POLICY_REGISTRY.items()
    }


# THESIS COMPARISON TABLE (reference)


THESIS_BASELINES = {
    # name               description                             expected_acc
    "none":          ("No augmentation — absolute floor",        "~78%"),
    "static":        ("Standard fixed pipeline — main baseline", "~84%"),
    "random":        ("All augs randomly, no schedule",          "~85%"),
    # "randaugment":   ("RandAugment N=2, M=9 (SOTA baseline)",    "~87%"),
    # "autoaugment":   ("AutoAugment CIFAR-10 policy",             "~87%"),
    # "trivialaugment":("TrivialAugment (surprisingly strong)",    "~86%"),
    # My methods:
#     "curriculum_loss":     ("Curriculum loss-guided ()",     "~89%?"),
#     "curriculum_instance": ("Curriculum instance-guided ()", "~88%?"),
#     "curriculum_class":    ("Curriculum class-guided ()",    "~88%?"),
 }



if __name__ == "__main__":
    import numpy as np

    print("Testing all augmentation policies...\n")

    img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))

    for name, PolicyClass in POLICY_REGISTRY.items():
        policy    = PolicyClass(dataset="cifar10")
        transform = policy.get_train_transform()
        try:
            result = transform(img)
            print(f"  {name:<18} → output shape: {result.shape}  transform: {type(transform).__name__}")
        except Exception as e:
            print(f"  {name:<18} → ERROR: {e}")

    print("\nRandomAugmentTransform ops (must match LGCA primitive pool):")
    rt = RandomAugmentTransform(dataset="cifar10")
    for name, _ in rt.ops:
        print(f"  {name}")

    print("\nSanity check — 3 calls should differ (random strengths per call):")
    for i in range(3):
        t = rt(img)
        print(f"  call {i+1}: mean={t.mean():.4f}  std={t.std():.4f}")

    print("\nThesis Baseline Reference:")
    print(f"  {'Method':<22} {'Expected Acc':<15} Description")
    print("  " + "─" * 65)
    for name, (desc, acc) in THESIS_BASELINES.items():
        print(f"  {name:<22} {acc:<15} {desc}")
