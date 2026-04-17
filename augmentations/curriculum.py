"""augmentations/curriculum.py — curriculum transform and dataset wrapper."""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from augmentations.primitives import apply_augmentations, get_active_augmentations
from data.datasets import CIFAR_STATS as STATS

class CurriculumTransform:
    """Augmentation transform that scales with a difficulty score [0, 1]."""

    def __init__(self, dataset: str = "cifar10", base_difficulty: float = 0.0):
        self.dataset         = dataset
        self.base_difficulty = base_difficulty
        self.mean            = STATS[dataset]["mean"]
        self.std             = STATS[dataset]["std"]
        self.normalize       = T.Normalize(self.mean, self.std)
        self.to_tensor       = T.ToTensor()

    def __call__(self, img: Image.Image, difficulty: float = 0.5) -> torch.Tensor:
        d = max(self.base_difficulty, difficulty)
        img = apply_augmentations(img, difficulty=d)
        return self.normalize(self.to_tensor(img))

    def get_val_transform(self):
        return T.Compose([T.ToTensor(), self.normalize])

    def describe(self, difficulty: float) -> list:
        return [name for name, _, _ in get_active_augmentations(difficulty)]

    def __repr__(self):
        return (f"CurriculumTransform(dataset={self.dataset}, "
                f"base_difficulty={self.base_difficulty})")

class CurriculumDataset(torch.utils.data.Dataset):
    """Wraps a dataset to apply per-sample difficulty via CurriculumTransform."""

    def __init__(self, base_dataset, transform: CurriculumTransform, default_difficulty: float = 0.0):
        self.base_dataset = base_dataset
        self.transform    = transform
        self.difficulties = torch.full((len(base_dataset),), default_difficulty)

    def set_difficulties(self, difficulties: torch.Tensor):
        assert len(difficulties) == len(self.base_dataset)
        self.difficulties = difficulties.clamp(0.0, 1.0)

    def set_global_difficulty(self, difficulty: float):
        self.difficulties = torch.full((len(self.base_dataset),), difficulty)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        difficulty = float(self.difficulties[idx])

        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)

        img_tensor = self.transform(img, difficulty=difficulty)
        return img_tensor, label, idx   # return idx for LossTracker updates


if __name__ == "__main__":
    import numpy as np

    print("Testing CurriculumTransform...\n")

    img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
    transform = CurriculumTransform(dataset="cifar10")

    for diff in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tensor = transform(img, difficulty=diff)
        active = transform.describe(diff)
        print(f"  difficulty={diff:.2f}  shape={tensor.shape}  "
              f"active={active}")

    print("\nDone.")
