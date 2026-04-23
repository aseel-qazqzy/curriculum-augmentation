"""training/losses.py — loss-guided difficulty scoring for curriculum learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def compute_sample_difficulty(
    loss_per_sample: torch.Tensor,
    mode: str = "inverse",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Convert per-sample losses to difficulty scores in [0, 1]."""
    if mode == "inverse":
        l_min = loss_per_sample.min()
        l_max = loss_per_sample.max()
        if (l_max - l_min) < 1e-8:
            return torch.full_like(loss_per_sample, 0.5)
        norm = (loss_per_sample - l_min) / (l_max - l_min)
        return 1.0 - norm

    elif mode == "direct":
        l_min = loss_per_sample.min()
        l_max = loss_per_sample.max()
        if (l_max - l_min) < 1e-8:
            return torch.full_like(loss_per_sample, 0.5)
        return (loss_per_sample - l_min) / (l_max - l_min)

    elif mode == "normalized":
        scaled = loss_per_sample / temperature
        return torch.sigmoid(scaled - scaled.mean())

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'inverse', 'direct', or 'normalized'.")


def epoch_difficulty(
    epoch: int,
    total_epochs: int,
    schedule: str = "sigmoid",
    warmup_epochs: int = 5,
    aug_milestones: list = None,
    max_difficulty: float = 1.0,
) -> float:
    """Global difficulty for the current epoch, rising from 0 to max_difficulty."""
    if schedule == "milestone":
        milestones = aug_milestones or [(20, 0.20), (60, 0.45)]
        for end_epoch, difficulty in sorted(milestones, key=lambda x: x[0]):
            if epoch <= end_epoch:
                return min(float(difficulty), max_difficulty)
        return max_difficulty

    if epoch <= warmup_epochs:
        return 0.0

    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    progress = min(progress, 1.0)

    if schedule == "sigmoid":
        t      = (progress - 0.5) / 0.15
        result = float(1.0 / (1.0 + np.exp(-t)))

    elif schedule == "linear":
        result = float(progress)

    elif schedule == "cosine":
        result = float(0.5 * (1 - np.cos(np.pi * progress)))

    elif schedule == "step":
        if progress < 0.33:   result = 0.0
        elif progress < 0.66: result = 0.33
        elif progress < 0.83: result = 0.66
        else:                 result = 1.0

    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return min(result, max_difficulty)


def get_batch_difficulties(
    loss_per_sample: torch.Tensor,
    epoch: int,
    total_epochs: int,
    schedule: str  = "sigmoid",
    mode: str      = "inverse",
    warmup_epochs: int = 5,
    blend: float   = 0.7,
    aug_milestones: list = None,
    max_difficulty: float = 1.0,
) -> torch.Tensor:
    """
    Per-sample difficulty = blend * epoch_level + (1 - blend) * sample_level.
    blend=1.0 → pure epoch schedule; blend=0.0 → fully loss-driven.
    """
    ep_diff        = epoch_difficulty(epoch, total_epochs, schedule, warmup_epochs,
                                      aug_milestones=aug_milestones,
                                      max_difficulty=max_difficulty)
    ep_diff_tensor = torch.full_like(loss_per_sample, ep_diff)
    sample_diff    = compute_sample_difficulty(loss_per_sample, mode=mode)
    final          = blend * ep_diff_tensor + (1.0 - blend) * sample_diff
    return final.clamp(0.0, 1.0)


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing. Use reduction='none' for per-sample loss."""

    def __init__(self, num_classes: int, smoothing: float = 0.1,
                 reduction: str = "mean"):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing   = smoothing
        self.reduction   = reduction
        self.confidence  = 1.0 - smoothing

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs,
                                             self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            return loss.sum()


class LossTracker:
    """Exponential moving average of per-sample losses for stable difficulty signals."""

    def __init__(self, n_samples: int, momentum: float = 0.9):
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        self.n_samples = n_samples
        self.momentum  = momentum
        self.ema_loss  = np.zeros(n_samples, dtype=np.float32)
        self.n_updates = np.zeros(n_samples, dtype=np.int32)

    def reset(self):
        self.ema_loss  = np.zeros(self.n_samples, dtype=np.float32)
        self.n_updates = np.zeros(self.n_samples, dtype=np.int32)

    def update(self, indices: torch.Tensor, losses: torch.Tensor):
        idx        = indices.cpu().numpy()
        lss        = losses.detach().cpu().numpy()
        first_seen = self.n_updates[idx] == 0
        self.ema_loss[idx] = np.where(
            first_seen,
            lss,
            self.momentum * self.ema_loss[idx] + (1.0 - self.momentum) * lss,
        )
        self.n_updates[idx] += 1

    def get_difficulty(self, indices: torch.Tensor, mode: str = "inverse") -> torch.Tensor:
        idx    = indices.cpu().numpy()
        losses = torch.tensor(self.ema_loss[idx], dtype=torch.float32)
        return compute_sample_difficulty(losses, mode=mode)

    def mean_loss(self) -> float:
        seen = self.n_updates > 0
        return float(self.ema_loss[seen].mean()) if seen.any() else 0.0


if __name__ == "__main__":
    print("Testing losses.py...\n")

    losses = torch.tensor([0.1, 0.5, 1.2, 2.5, 4.0])
    diff   = compute_sample_difficulty(losses, mode="inverse")
    print("Per-sample difficulty (inverse mode):")
    for l, d in zip(losses, diff):
        print(f"  loss={l:.1f} → difficulty={d:.3f}")

    print("\nEpoch difficulty (sigmoid, 150 epochs):")
    for ep in [1, 25, 49, 50, 75, 99, 100, 125, 150]:
        d = epoch_difficulty(ep, 150, schedule="sigmoid")
        bar = "█" * int(d * 20)
        print(f"  ep={ep:>3}  {d:.3f}  {bar}")

    print("\nLabel smoothing loss:")
    criterion = LabelSmoothingLoss(num_classes=10, smoothing=0.1, reduction="none")
    logits  = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    loss    = criterion(logits, targets)
    print(f"  Per-sample losses: {loss.tolist()}")

    print("\nDone.")
