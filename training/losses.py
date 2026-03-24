"""
training/losses.py
Loss-guided curriculum difficulty scoring.

This is the CORE of your thesis contribution.

The key idea:
    Instead of scheduling augmentation difficulty by epoch (static schedule),
    we use the model's own loss to decide how hard each sample should be.

    High loss sample  → model is struggling → keep augmentation mild (easy)
    Low loss sample   → model has learned it → apply hard augmentation
    
    This is loss-guided curriculum learning:
    "Make easy samples harder, protect hard samples from extra noise"

Reference:
    Bengio et al. (2009) - Curriculum Learning
    Curriculum by Smoothing: https://arxiv.org/abs/2003.01367
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

# CORE: PER-SAMPLE DIFFICULTY FROM LOSS

def compute_sample_difficulty(
    loss_per_sample: torch.Tensor,
    mode: str = "inverse",
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Convert per-sample losses into difficulty scores [0, 1].

    Args:
        loss_per_sample: shape (B,) — cross-entropy loss per sample
        mode:            how to convert loss → difficulty
            "inverse"   → high loss = LOW difficulty  (protect struggling samples)
            "direct"    → high loss = HIGH difficulty  (focus on hard samples)
            "normalized"→ normalize within batch
        temperature:     controls sharpness of the conversion

    Returns:
        difficulty: shape (B,) — values in [0, 1]

    Thesis note:
        "inverse" mode implements the core CL idea:
        samples the model finds hard (high loss) get LESS augmentation
        so the model can focus on learning the underlying pattern first.
        Once loss drops, augmentation increases.
    """
    if mode == "inverse":
        # High loss → low difficulty (protect hard samples)
        # Normalize loss to [0,1] within batch first
        l_min = loss_per_sample.min()
        l_max = loss_per_sample.max()
        if (l_max - l_min) < 1e-8:
            return torch.full_like(loss_per_sample, 0.5)
        norm = (loss_per_sample - l_min) / (l_max - l_min)
        return 1.0 - norm   # invert: high loss → low difficulty

    elif mode == "direct":
        # High loss → high difficulty (focus augmentation on hard samples)
        l_min = loss_per_sample.min()
        l_max = loss_per_sample.max()
        if (l_max - l_min) < 1e-8:
            return torch.full_like(loss_per_sample, 0.5)
        return (loss_per_sample - l_min) / (l_max - l_min)

    elif mode == "normalized":
        # Softmax-like normalization with temperature
        scaled = loss_per_sample / temperature
        return torch.sigmoid(scaled - scaled.mean())

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'inverse', 'direct', or 'normalized'.")

# GLOBAL DIFFICULTY SCHEDULE
def epoch_difficulty(
    epoch: int,
    total_epochs: int,
    schedule: str = "sigmoid",
    warmup_epochs: int = 5,
) -> float:
    """
    Global difficulty level for the current epoch.
    This is the EPOCH-LEVEL schedule (0.0 → 1.0 over training).

    Combined with per-sample difficulty:
        final_difficulty = epoch_difficulty * sample_difficulty

    Args:
        epoch:         current epoch (1-indexed)
        total_epochs:  total training epochs
        schedule:      "sigmoid" | "linear" | "cosine" | "step"
        warmup_epochs: epochs before difficulty starts increasing

    Returns:
        float in [0.0, 1.0]
    """
    if epoch <= warmup_epochs:
        return 0.0

    # Adjust progress to account for warmup
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    progress = min(progress, 1.0)

    if schedule == "sigmoid":
        # Slow start, fast middle, plateau at end
        # Centred at 50% of (post-warmup) training
        t = (progress - 0.5) / 0.15
        return float(1.0 / (1.0 + np.exp(-t)))

    elif schedule == "linear":
        return float(progress)

    elif schedule == "cosine":
        # Cosine ramp from 0 to 1
        return float(0.5 * (1 - np.cos(np.pi * progress)))

    elif schedule == "step":
        # Discrete steps at 33%, 66%, 83%
        if progress < 0.33: return 0.0
        if progress < 0.66: return 0.33
        if progress < 0.83: return 0.66
        return 1.0

    else:
        raise ValueError(f"Unknown schedule: {schedule}")


# COMBINED DIFFICULTY

def get_batch_difficulties(
    loss_per_sample: torch.Tensor,
    epoch: int,
    total_epochs: int,
    schedule: str  = "sigmoid",
    mode: str      = "inverse",
    warmup_epochs: int = 5,
    blend: float   = 0.7,
) -> torch.Tensor:
    """
    Final per-sample difficulty combining epoch schedule + loss signal.

    final = blend * epoch_level + (1 - blend) * sample_level

    Args:
        loss_per_sample: (B,) per-sample CE losses
        epoch:           current epoch
        total_epochs:    total epochs
        schedule:        epoch-level schedule type
        mode:            how loss maps to difficulty
        warmup_epochs:   warmup before CL starts
        blend:           weight of epoch-level vs sample-level
                         0.0 = pure sample-level (fully adaptive)
                         1.0 = pure epoch-level (ignores loss)
                         0.7 = recommended (epoch guides, loss fine-tunes)

    Returns:
        difficulties: (B,) tensor of per-sample difficulties in [0, 1]
    """
    # Epoch-level: same for all samples in batch
    ep_diff = epoch_difficulty(epoch, total_epochs, schedule, warmup_epochs)
    ep_diff_tensor = torch.full_like(loss_per_sample, ep_diff)

    # Sample-level: per-sample based on loss
    sample_diff = compute_sample_difficulty(loss_per_sample, mode=mode)

    # Blend
    final = blend * ep_diff_tensor + (1.0 - blend) * sample_diff
    return final.clamp(0.0, 1.0)


# LABEL SMOOTHING LOSS
class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing.
    Helps with CIFAR-100 (+0.3-0.5% typically).

    Args:
        num_classes:  number of output classes
        smoothing:    label smoothing factor (0.1 is standard)
        reduction:    'mean' | 'none' (use 'none' for per-sample loss)
    """
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

        # Smooth targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs,
                                             self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss          # shape (B,) — for per-sample difficulty
        else:
            return loss.sum()



# LOSS TRACKER — tracks per-sample loss over time

class LossTracker:
    """
    Tracks exponential moving average of per-sample losses.
    Used to get a smoother difficulty signal across epochs.

    Usage:
        tracker = LossTracker(n_samples=45000, momentum=0.9)

        # Each epoch, update with new losses:
        tracker.update(indices, losses)

        # Get smoothed difficulty for a batch:
        difficulty = tracker.get_difficulty(indices)
    """
    def __init__(self, n_samples: int, momentum: float = 0.9):
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        self.n_samples = n_samples
        self.momentum  = momentum
        self.ema_loss  = np.zeros(n_samples, dtype=np.float32)
        self.n_updates = np.zeros(n_samples, dtype=np.int32)

    def reset(self):
        """Clear all tracked loss history (e.g. when resuming from checkpoint)."""
        self.ema_loss  = np.zeros(self.n_samples, dtype=np.float32)
        self.n_updates = np.zeros(self.n_samples, dtype=np.int32)

    def update(self, indices: torch.Tensor, losses: torch.Tensor):
        idx = indices.cpu().numpy()
        lss = losses.detach().cpu().numpy()
        for i, l in zip(idx, lss):
            if self.n_updates[i] == 0:
                self.ema_loss[i] = l
            else:
                self.ema_loss[i] = (self.momentum * self.ema_loss[i]
                                    + (1 - self.momentum) * l)
            self.n_updates[i] += 1

    def get_difficulty(self, indices: torch.Tensor, mode: str = "inverse") -> torch.Tensor:
        idx    = indices.cpu().numpy()
        losses = torch.tensor(self.ema_loss[idx], dtype=torch.float32)
        return compute_sample_difficulty(losses, mode=mode)

    def mean_loss(self) -> float:
        seen = self.n_updates > 0
        return float(self.ema_loss[seen].mean()) if seen.any() else 0.0



# QUICK TEST
if __name__ == "__main__":
    print("Testing losses.py...\n")

    # Test per-sample difficulty
    losses = torch.tensor([0.1, 0.5, 1.2, 2.5, 4.0])
    diff   = compute_sample_difficulty(losses, mode="inverse")
    print("Per-sample difficulty (inverse mode):")
    for l, d in zip(losses, diff):
        print(f"  loss={l:.1f} → difficulty={d:.3f}")

    # Test epoch schedule
    print("\nEpoch difficulty (sigmoid, 150 epochs):")
    for ep in [1, 25, 49, 50, 75, 99, 100, 125, 150]:
        d = epoch_difficulty(ep, 150, schedule="sigmoid")
        bar = "█" * int(d * 20)
        print(f"  ep={ep:>3}  {d:.3f}  {bar}")

    # Test label smoothing
    print("\nLabel smoothing loss:")
    criterion = LabelSmoothingLoss(num_classes=10, smoothing=0.1, reduction="none")
    logits  = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    loss    = criterion(logits, targets)
    print(f"  Per-sample losses: {loss.tolist()}")

    print("\n✅ losses.py working correctly")