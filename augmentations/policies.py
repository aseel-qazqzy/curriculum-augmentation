"""augmentations/policies.py — augmentation policies used in experiments."""

import random

import torch
import torchvision.transforms as T
from PIL import Image

from augmentations.primitives import AUGMENTATION_REGISTRY
from data.datasets import CIFAR_STATS as STATS


FIXED_STRENGTH = 0.7

# blur excluded: deterministic blur on 32x32 images every epoch creates a
# training/val distribution shift that harms all methods equally.
_TIER_OPS = {
    1: ["flip", "crop"],
    2: ["flip", "crop", "color_jitter", "rotation", "shear"],
    3: ["flip", "crop", "color_jitter", "rotation", "shear", "grayscale", "cutout"],
}


class AugmentationPolicy:
    def __init__(self, dataset: str = "cifar10"):
        self.dataset   = dataset
        self.mean      = STATS[dataset]["mean"]
        self.std       = STATS[dataset]["std"]
        self.normalize = T.Normalize(self.mean, self.std)

    def get_train_transform(self) -> T.Compose:
        raise NotImplementedError

    def get_val_transform(self) -> T.Compose:
        return T.Compose([T.ToTensor(), self.normalize])

    def __repr__(self):
        return f"{self.__class__.__name__}(dataset={self.dataset})"


class NoAugmentation(AugmentationPolicy):
    """Lower bound baseline — no augmentation."""

    def get_train_transform(self) -> T.Compose:
        return T.Compose([T.ToTensor(), self.normalize])


class _FullStaticTransform:
    """All 7 ops at FIXED_STRENGTH from epoch 1."""

    def __init__(self, dataset: str = "cifar10", strength: float = FIXED_STRENGTH):
        mean = STATS[dataset]["mean"]
        std  = STATS[dataset]["std"]
        self.normalize = T.Normalize(mean, std)
        self.to_tensor = T.ToTensor()
        self.strength  = strength

    def __call__(self, img):
        for name in _TIER_OPS[3]:
            fn, _, _ = AUGMENTATION_REGISTRY[name]
            img = fn(img, self.strength)
        return self.normalize(self.to_tensor(img))


class StaticAugmentation(AugmentationPolicy):
    """All 7 ops at FIXED_STRENGTH from epoch 1 — no schedule."""

    def __init__(self, dataset: str = "cifar10", strength: float = FIXED_STRENGTH):
        super().__init__(dataset=dataset)
        self.strength = strength

    def get_train_transform(self) -> "_FullStaticTransform":
        return _FullStaticTransform(dataset=self.dataset, strength=self.strength)


class ThreeTierCurriculumTransform:
    """
    Introduces ops in three tiers aligned with MultiStepLR milestones [33, 66].
    Call set_epoch(epoch) at the start of every training epoch.
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
    """3-tier curriculum: ops introduced progressively at epochs t1 and t2."""

    def __init__(self, dataset: str = "cifar10", t1: int = 33, t2: int = 66,
                 strength: float = FIXED_STRENGTH):
        super().__init__(dataset=dataset)
        self.t1       = t1
        self.t2       = t2
        self.strength = strength

    def get_train_transform(self) -> ThreeTierCurriculumTransform:
        return ThreeTierCurriculumTransform(dataset=self.dataset, t1=self.t1, t2=self.t2,
                                            strength=self.strength)


class RandomAugmentTransform:
    """All ops applied with independently sampled random strengths."""

    def __init__(self, dataset: str = "cifar10"):
        mean = STATS[dataset]["mean"]
        std  = STATS[dataset]["std"]
        self.normalize = T.Normalize(mean, std)
        self.to_tensor = T.ToTensor()
        self.ops = [(name, fn) for name, (fn, _, _) in AUGMENTATION_REGISTRY.items()]

    def __call__(self, img: Image.Image) -> torch.Tensor:
        for _, fn in self.ops:
            strength = random.random()
            img = fn(img, strength)
        return self.normalize(self.to_tensor(img))

    def __repr__(self) -> str:
        names = [name for name, _ in self.ops]
        return f"RandomAugmentTransform(ops={names})"


class RandomAugmentation(AugmentationPolicy):
    """Same op pool as curriculum but random strengths with no ordering."""

    def get_train_transform(self) -> RandomAugmentTransform:
        return RandomAugmentTransform(dataset=self.dataset)

    def get_val_transform(self) -> T.Compose:
        return T.Compose([T.ToTensor(), self.normalize])


class RandAugmentPolicy(AugmentationPolicy):
    """RandAugment (Cubuk et al., NeurIPS 2020). N=2, M=9 for CIFAR-10; N=2, M=14 for CIFAR-100."""

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
    name = name.lower().strip()
    if name not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy: '{name}'. "
            f"Available: {list(POLICY_REGISTRY.keys())}"
        )
    return POLICY_REGISTRY[name](dataset=dataset)


def get_all_baseline_policies(dataset: str = "cifar10") -> dict:
    return {name: cls(dataset=dataset) for name, cls in POLICY_REGISTRY.items()}


THESIS_BASELINES = {
    "none":   ("No augmentation — absolute floor",        "~78%"),
    "static": ("Standard fixed pipeline — main baseline", "~84%"),
    "random": ("All augs randomly, no schedule",          "~85%"),
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

    print("\nRandomAugmentTransform ops:")
    rt = RandomAugmentTransform(dataset="cifar10")
    for name, _ in rt.ops:
        print(f"  {name}")

    print("\nSanity check — 3 calls should differ (random strengths per call):")
    for i in range(3):
        t = rt(img)
        print(f"  call {i+1}: mean={t.mean():.4f}  std={t.std():.4f}")
