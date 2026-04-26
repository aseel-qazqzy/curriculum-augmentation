"""
augmentations/schedules.py
Epoch-level difficulty schedules — thin wrappers around training/losses.py.

The canonical implementation lives in training/losses.py:epoch_difficulty().
These wrappers keep the original per-schedule API for backwards compatibility.
"""
from training.losses import epoch_difficulty, LossPlateauScheduler


def sigmoid_schedule(epoch: int, total: int, warmup: int = 5) -> float:
    """Slow start, fast middle, plateau. Recommended default."""
    return epoch_difficulty(epoch, total, schedule="sigmoid", warmup_epochs=warmup)

def linear_schedule(epoch: int, total: int, warmup: int = 5) -> float:
    """Constant ramp from 0 to 1."""
    return epoch_difficulty(epoch, total, schedule="linear", warmup_epochs=warmup)

def cosine_schedule(epoch: int, total: int, warmup: int = 5) -> float:
    """Smooth cosine ramp."""
    return epoch_difficulty(epoch, total, schedule="cosine", warmup_epochs=warmup)

def step_schedule(epoch: int, total: int, warmup: int = 5) -> float:
    """Discrete steps at 33%, 66%, 83% — aligns with MultiStepLR."""
    return epoch_difficulty(epoch, total, schedule="step", warmup_epochs=warmup)

def milestone_schedule(epoch: int, total: int, warmup: int = 0,
                       aug_milestones: list = None) -> float:
    """Fixed difficulty stages at explicit epoch boundaries.
    Default: easy (0.20) → medium (0.45) at epoch 20/60 → hard (1.0) after."""
    return epoch_difficulty(epoch, total, schedule="milestone",
                            warmup_epochs=0, aug_milestones=aug_milestones)


SCHEDULE_REGISTRY = {
    "sigmoid":   sigmoid_schedule,
    "linear":    linear_schedule,
    "cosine":    cosine_schedule,
    "step":      step_schedule,
    "milestone": milestone_schedule,
}

def get_schedule(name: str):
    if name not in SCHEDULE_REGISTRY:
        raise ValueError(f"Unknown schedule '{name}'. "
                         f"Available: {list(SCHEDULE_REGISTRY.keys())}")
    return SCHEDULE_REGISTRY[name]


if __name__ == "__main__":
    print("Schedule comparison (150 epochs):\n")
    print(f"  {'Epoch':>5}  {'sigmoid':>8}  {'linear':>8}  {'cosine':>8}  {'step':>8}")
    print("  " + "─" * 45)
    for ep in [1, 10, 25, 49, 50, 75, 99, 100, 124, 125, 150]:
        vals = {n: f(ep, 150) for n, f in SCHEDULE_REGISTRY.items()}
        print(f"  {ep:>5}  {vals['sigmoid']:>8.3f}  {vals['linear']:>8.3f}  "
              f"{vals['cosine']:>8.3f}  {vals['step']:>8.3f}")
