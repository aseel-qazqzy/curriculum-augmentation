"""augmentations/policies.py — augmentation policies used in experiments."""

import random

import torch
import torchvision.transforms as T
from PIL import Image

from augmentations.primitives import AUGMENTATION_REGISTRY
from data.datasets import CIFAR_STATS as STATS


FIXED_STRENGTH = 0.7

_TIER_OPS = {
    1: ["flip", "crop", "translate_x", "translate_y"],
    2: [
        "flip",
        "crop",
        "translate_x",
        "translate_y",
        "color_jitter",
        "rotation",
        "shear",
        "auto_contrast",
        "equalize",
        "sharpness",
        "perspective",
    ],
    3: [
        "flip",
        "crop",
        "translate_x",
        "translate_y",
        "color_jitter",
        "rotation",
        "shear",
        "auto_contrast",
        "equalize",
        "sharpness",
        "perspective",
        "grayscale",
        "cutout",
        "contrast",
        "brightness",
        "blur",
        "solarize",
        "posterize",
        "invert",
    ],
}

# How many ops are randomly sampled per tier (subsampling adds within-tier diversity).
_TIER_N_OPS = {1: 3, 2: 5, 3: 8}

# Strength as a fraction of the ceiling (self.strength).
# Tier 3 always equals the ceiling; lower tiers scale down proportionally.
_TIER_STRENGTH_FRACS = {1: 0.40, 2: 0.70, 3: 1.0}

_STRENGTH_RAMP_EPOCHS = 5  # epochs to linearly ramp strength at each tier boundary


class AugmentationPolicy:
    def __init__(self, dataset: str = "cifar100"):
        self.dataset = dataset
        self.mean = STATS[dataset]["mean"]
        self.std = STATS[dataset]["std"]
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
    """Samples _TIER_N_OPS[3] ops from the full Tier 3 pool at ceiling strength from epoch 1.

    Mirrors exactly what ThreeTierCurriculumTransform does in Tier 3 — same pool,
    same sample size, same strength — so the only experimental variable is the
    curriculum progression, not the number of ops per image.
    """

    def __init__(self, dataset: str = "cifar100", strength: float = FIXED_STRENGTH):
        mean = STATS[dataset]["mean"]
        std = STATS[dataset]["std"]
        self.normalize = T.Normalize(mean, std)
        self.to_tensor = T.ToTensor()
        self.strength = strength

    def __call__(self, img):
        active = random.sample(_TIER_OPS[3], _TIER_N_OPS[3])
        for name in active:
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
    Three-tier progressive augmentation curriculum.
    Call set_epoch(epoch) at the start of every training epoch.

    Enhancements over the original design:
      - Strength grows across tiers (40% → 70% → 100% of ceiling).
      - Strength ramps smoothly over _STRENGTH_RAMP_EPOCHS at each boundary
        instead of jumping instantly, decoupling the op-set transition from
        the intensity transition.
      - Ops are randomly subsampled within each tier (_TIER_N_OPS) so each
        image sees a different subset, adding diversity without changing tiers.
    """

    TIER_LABELS = {
        1: "Tier 1 [flip, crop, translate_x/y]",
        2: "Tier 2 [+color_jitter, rotation, shear, auto_contrast, equalize, sharpness, perspective]",
        3: "Tier 3 [+grayscale, cutout, contrast, brightness, blur, solarize, posterize, invert]",
    }

    def __init__(
        self,
        dataset: str = "cifar10",
        t1: int = 33,
        t2: int = 66,
        strength: float = FIXED_STRENGTH,
    ):
        mean = STATS[dataset]["mean"]
        std = STATS[dataset]["std"]
        self.normalize = T.Normalize(mean, std)
        self.to_tensor = T.ToTensor()
        self.t1 = t1
        self.t2 = t2
        self.epoch = 1
        self.strength = strength  # ceiling — reached at Tier 3
        self._forced_tier = None  # set by loss/entropy schedulers; None = time-based
        # Records the epoch at which each tier was activated via set_tier().
        # Used by _current_strength() and mix_scale() so ramp boundaries are
        # correct under LPS/EGS (which advance tiers at different epochs than t1/t2).
        self._tier_start_epoch: dict[int, int] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_tier(self, tier: int, epoch: int = None) -> None:
        """Override tier directly — used by loss-guided and entropy-guided schedulers.

        Pass epoch (the detection epoch) so _current_strength() and mix_scale()
        use the actual transition point rather than the ETS t1/t2 boundaries.
        The new tier becomes active on epoch+1 (set_epoch is called first each epoch).
        """
        if tier is not None and tier not in (1, 2, 3):
            raise ValueError(f"tier must be 1, 2, or 3, got {tier}")
        if tier != self._forced_tier and epoch is not None:
            self._tier_start_epoch[tier] = epoch
        self._forced_tier = tier

    def tier(self) -> int:
        if self._forced_tier is not None:  # loss/entropy path
            return self._forced_tier
        if self.epoch <= self.t1:  # time-based path — unchanged
            return 1
        if self.epoch <= self.t2:
            return 2
        return 3

    def tier_label(self) -> str:
        return self.TIER_LABELS[self.tier()]

    def _tier_strength(self, tier: int) -> float:
        return self.strength * _TIER_STRENGTH_FRACS[tier]

    def _current_strength(self) -> float:
        """Linearly ramp strength over _STRENGTH_RAMP_EPOCHS after a tier boundary.

        Under ETS the boundary is the configured t1/t2 epoch. Under LPS/EGS the
        boundary is the actual detection epoch stored by set_tier(), so the ramp
        always starts from the real transition point regardless of scheduler.
        """
        tier = self.tier()
        s_curr = self._tier_strength(tier)
        if tier == 1:
            return s_curr
        if self._forced_tier is not None and tier in self._tier_start_epoch:
            boundary = self._tier_start_epoch[tier]
        else:
            boundary = self.t1 if tier == 2 else self.t2
        epochs_in = self.epoch - boundary
        if 0 < epochs_in <= _STRENGTH_RAMP_EPOCHS:
            s_prev = self._tier_strength(tier - 1)
            return s_prev + (s_curr - s_prev) * (epochs_in / _STRENGTH_RAMP_EPOCHS)
        return s_curr

    def mix_scale(self) -> float:
        """Returns 0.0 before Tier 3, then ramps linearly 0→1 over _STRENGTH_RAMP_EPOCHS.

        Uses the actual Tier 3 start epoch (LPS/EGS) or the configured t2 (ETS)
        as the ramp origin, so mixing always activates on the first Tier 3 epoch.
        """
        if self.tier() < 3:
            return 0.0
        if self._forced_tier is not None and 3 in self._tier_start_epoch:
            boundary = self._tier_start_epoch[3]
        else:
            boundary = self.t2
        epochs_in = self.epoch - boundary
        if epochs_in <= 0:
            return 0.0
        if epochs_in <= _STRENGTH_RAMP_EPOCHS:
            return epochs_in / _STRENGTH_RAMP_EPOCHS
        return 1.0

    def __call__(self, img):
        tier = self.tier()
        pool = _TIER_OPS[tier]
        n = _TIER_N_OPS[tier]
        active = random.sample(pool, n)
        s = self._current_strength()
        for name in active:
            fn, _, _ = AUGMENTATION_REGISTRY[name]
            img = fn(img, s)
        return self.normalize(self.to_tensor(img))


class ThreeTierCurriculumAugmentation(AugmentationPolicy):
    """3-tier curriculum: ops introduced progressively at epochs t1 and t2."""

    def __init__(
        self,
        dataset: str = "cifar10",
        t1: int = 33,
        t2: int = 66,
        strength: float = FIXED_STRENGTH,
    ):
        super().__init__(dataset=dataset)
        self.t1 = t1
        self.t2 = t2
        self.strength = strength

    def get_train_transform(self) -> ThreeTierCurriculumTransform:
        return ThreeTierCurriculumTransform(
            dataset=self.dataset, t1=self.t1, t2=self.t2, strength=self.strength
        )


class RandomAugmentTransform:
    """All ops applied with independently sampled random strengths."""

    def __init__(self, dataset: str = "cifar10"):
        mean = STATS[dataset]["mean"]
        std = STATS[dataset]["std"]
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
    "none": NoAugmentation,
    "static": StaticAugmentation,
    "tiered_curriculum": ThreeTierCurriculumAugmentation,
    "random": RandomAugmentation,
    "randaugment": RandAugmentPolicy,
}


def get_policy(name: str, dataset: str = "cifar10") -> AugmentationPolicy:
    name = name.lower().strip()
    if name not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy: '{name}'. Available: {list(POLICY_REGISTRY.keys())}"
        )
    return POLICY_REGISTRY[name](dataset=dataset)


def get_all_baseline_policies(dataset: str = "cifar10") -> dict:
    return {name: cls(dataset=dataset) for name, cls in POLICY_REGISTRY.items()}


THESIS_BASELINES = {
    "none": ("No augmentation — absolute floor", "~78%"),
    "static": ("Standard fixed pipeline — main baseline", "~84%"),
    "random": ("All augs randomly, no schedule", "~85%"),
}


if __name__ == "__main__":
    import numpy as np

    print("Testing all augmentation policies...\n")

    img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))

    for name, PolicyClass in POLICY_REGISTRY.items():
        policy = PolicyClass(dataset="cifar10")
        transform = policy.get_train_transform()
        try:
            result = transform(img)
            print(
                f"  {name:<18} → output shape: {result.shape}  transform: {type(transform).__name__}"
            )
        except Exception as e:
            print(f"  {name:<18} → ERROR: {e}")

    print("\nRandomAugmentTransform ops:")
    rt = RandomAugmentTransform(dataset="cifar10")
    for name, _ in rt.ops:
        print(f"  {name}")

    print("\nSanity check — 3 calls should differ (random strengths per call):")
    for i in range(3):
        t = rt(img)
        print(f"  call {i + 1}: mean={t.mean():.4f}  std={t.std():.4f}")
