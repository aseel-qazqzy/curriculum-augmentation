"""
experiments/config.py
Shared base configuration for all experiment scripts.

Each script extends BASE_CONFIG with its own specific defaults.
"""

from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = str(PROJECT_ROOT / "checkpoints")

BASE_CONFIG = {
    # Data
    "dataset":         "cifar10",
    "data_root":       str(PROJECT_ROOT / "data"),
    "val_split":       0.1,

    # Model
    "model":           "resnet18",

    # Training
    "epochs":          150,
    "batch_size":      128,
    "lr":              0.1,
    "optimizer":       "sgd",
    "weight_decay":    5e-4,
    "scheduler":       "multistep",

    # Logging
    "checkpoint_dir":  CHECKPOINT_DIR,
    "log_every":       10,
    "seed":            42,
    "use_wandb":       False,
}
