"""augmentations/mixing.py — CutMix and MixUp batch-level augmentations.

Stage 2 of the two-stage Tier 3 pipeline (per-image ops happen in the
DataLoader; these run in the training loop after the batch is assembled).
"""

import random
import numpy as np
import torch


def mixup(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
):
    """Linear interpolation of two images.

    lambda (lam) ~ Beta(alpha, alpha).  Returns mixed images plus both label sets
    so the caller can compute: lam * loss(out, label_a) + (1-lam) * loss(out, label_b).
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
):
    """Paste a random rectangular patch from one image into another.

    Patch area ≈ (1 - lam) of the image.  lam is recomputed from the
    actual pixel area so the loss weighting matches the real mix ratio.
    """
    lam = float(np.random.beta(alpha, alpha))
    B, C, H, W = images.shape
    idx = torch.randperm(B, device=images.device)

    cut_h = int(H * np.sqrt(1.0 - lam))
    cut_w = int(W * np.sqrt(1.0 - lam))

    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]

    lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)   # recompute from actual area
    return mixed, labels, labels[idx], lam


class BatchMixer:
    """Applies MixUp or CutMix to a batch with probability p.

    Intended for Tier 3 only — the caller (train_one_epoch) receives None
    for earlier tiers so no mixing happens there.

    Args:
        mode:  'cutmix' | 'mixup' | 'both' (randomly picks one per batch)
        alpha: Beta(alpha, alpha) shape param — 1.0 gives uniform mix ratio
        p:     probability of applying mixing to any given batch (default 0.5)
    """

    def __init__(self, mode: str = "both", alpha: float = 1.0, p: float = 0.5):
        if mode not in ("cutmix", "mixup", "both"):
            raise ValueError(f"mode must be 'cutmix', 'mixup', or 'both', got '{mode}'")
        self.mode  = mode
        self.alpha = alpha
        self.p     = p

    def __call__(self, images: torch.Tensor, labels: torch.Tensor):
        """Returns (mixed_images, label_a, label_b, lam).

        When skipped by probability returns (images, labels, labels, 1.0)
        so the caller's loss path is identical regardless.
        """
        if random.random() > self.p:
            return images, labels, labels, 1.0

        mode = random.choice(["cutmix", "mixup"]) if self.mode == "both" else self.mode
        if mode == "cutmix":
            return cutmix(images, labels, self.alpha)
        return mixup(images, labels, self.alpha)

    def __repr__(self):
        return f"BatchMixer(mode={self.mode}, alpha={self.alpha}, p={self.p})"


if __name__ == "__main__":
    B, C, H, W = 4, 3, 32, 32
    images = torch.rand(B, C, H, W)
    labels = torch.randint(0, 10, (B,))
    mixer  = BatchMixer(mode="both", alpha=1.0, p=1.0)

    for _ in range(3):
        mixed, la, lb, lam = mixer(images, labels)
        print(f"  lam={lam:.3f}  mixed shape={mixed.shape}  label_a={la.tolist()}  label_b={lb.tolist()}")
