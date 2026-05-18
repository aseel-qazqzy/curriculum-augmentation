"""augmentations/curriculum.py — curriculum transform and dataset wrapper."""

import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from data.datasets import CIFAR_STATS as STATS


class CurriculumTransform:
    """Per-sample augmentation transform using _TIER_OPS pools.

    Accepts a tier integer (1, 2, or 3) per sample and applies the same
    op pool and random sampling as ThreeTierCurriculumTransform — ensuring
    EGS uses identical augmentations to ETS/LPS.
    """

    def __init__(
        self,
        dataset: str = "cifar10",
        base_difficulty: float = 0.0,
        strength: float = 0.7,
    ):
        self.dataset = dataset
        self.base_difficulty = base_difficulty
        self.strength = strength
        self.mean = STATS[dataset]["mean"]
        self.std = STATS[dataset]["std"]
        self.normalize = T.Normalize(self.mean, self.std)
        self.to_tensor = T.ToTensor()

    def __call__(self, img: Image.Image, difficulty: int = 1) -> torch.Tensor:
        from augmentations.policies import (
            _TIER_OPS,
            _TIER_N_OPS,
            _TIER_STRENGTH_FRACS,
        )
        from augmentations.primitives import AUGMENTATION_REGISTRY

        tier = int(difficulty)  # difficulty now carries tier integer (1/2/3)
        tier = max(1, min(3, tier))

        pool = _TIER_OPS[tier]
        n = min(_TIER_N_OPS[tier], len(pool))
        active = random.sample(pool, n)
        op_strength = self.strength * _TIER_STRENGTH_FRACS[tier]

        for name in active:
            fn, _, _ = AUGMENTATION_REGISTRY[name]
            img = fn(img, op_strength)

        return self.normalize(self.to_tensor(img))

    def get_val_transform(self):
        return T.Compose([T.ToTensor(), self.normalize])

    def __repr__(self):
        return f"CurriculumTransform(dataset={self.dataset}, strength={self.strength})"


class CurriculumDataset(torch.utils.data.Dataset):
    """Wraps a dataset to apply per-sample difficulty via CurriculumTransform."""

    def __init__(
        self,
        base_dataset,
        transform: CurriculumTransform,
        default_difficulty: float = 0.0,
    ):
        self.base_dataset = base_dataset
        self.transform = transform
        # Store tier integers (1/2/3); default_difficulty=0.0 → start all at Tier 1
        self.difficulties = torch.ones(len(base_dataset), dtype=torch.long)

    def set_difficulties(self, difficulties: torch.Tensor):
        assert len(difficulties) == len(self.base_dataset)
        self.difficulties = difficulties.clamp(1, 3).long()

    def set_global_difficulty(self, difficulty: float):
        tier = max(1, min(3, int(difficulty))) if difficulty > 0 else 1
        self.difficulties = torch.full(
            (len(self.base_dataset),), tier, dtype=torch.long
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        difficulty = float(self.difficulties[idx])

        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)

        img_tensor = self.transform(img, difficulty=difficulty)
        return img_tensor, label, idx  # return idx for LossTracker updates


if __name__ == "__main__":
    import numpy as np

    print("Testing CurriculumTransform...\n")

    img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
    transform = CurriculumTransform(dataset="cifar10")

    for diff in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tensor = transform(img, difficulty=diff)
        active = transform.describe(diff)
        print(f"  difficulty={diff:.2f}  shape={tensor.shape}  active={active}")

    print("\nDone.")
