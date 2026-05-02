"""
experiments/config.py
Shared base configuration for all experiment scripts.

Each script extends BASE_CONFIG with its own specific defaults.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = str(PROJECT_ROOT / "checkpoints")

BASE_CONFIG = {
    # Data
    "dataset": "cifar100",
    "data_root": str(PROJECT_ROOT / "data"),
    # 0.1 = development mode (45k train / 5k val / 10k test)
    # 0.0 = full-train mode  (50k train / 10k test) — use for final thesis numbers
    "val_split": 0.1,
    # Model
    "model": "resnet50",
    # Training
    "epochs": 100,
    "batch_size": 128,
    "fixed_strength": 0.7,  # augmentation op strength (0.0–1.0); used by static & tiered CL
    "lr": 0.1,
    "optimizer": "sgd",
    "weight_decay": 5e-4,
    "scheduler": "multistep",
    # Tiered curriculum — scheduling signal
    # ets = Epoch-Threshold Scheduling  (implemented)
    # lps = Loss-Plateau Scheduling     (implemented)
    # egs = Entropy-Guided Scheduling   (June)
    "tier_schedule": "ets",
    # ETS tier boundaries — fraction of total epochs (0.0–1.0) or absolute epoch int
    "tier_t1": 0.20,  # Tier 1 ends at 20% of epochs
    "tier_t2": 0.45,  # Tier 2 ends at 45% of epochs
    # Batch-level mixing — active only in Tier 3 of tiered_curriculum
    "mix_mode": "both",  # cutmix | mixup | both | none
    "mix_alpha": 1.0,  # Beta(alpha, alpha) — 1.0 = uniform mix ratio
    "mix_prob": 0.5,  # probability of mixing any given batch
    # Logging
    "checkpoint_dir": CHECKPOINT_DIR,
    "log_every": 10,
    "seed": 42,
    "use_wandb": False,
    # Loss-based CL
    "lps_tau": 0.02,
    "lps_window": 5,
    "lps_min_epochs": 10,
    # Label smoothing (0.0 = disabled — plain CrossEntropy; existing ETS/LPS runs unchanged)
    "label_smoothing": 0.0,
    # Entropy-Guided Scheduling (EGS)
    "egs_update_freq": 10,  # recompute per-sample entropy every N epochs
    "egs_min_epochs_per_tier": 20,  # min epochs in a tier before a sample can advance
    "egs_max_epochs_per_tier": 40,  # epochs stuck in a tier before force-bump (0=disabled)
    "egs_mix_threshold": 0.75,  # fraction of samples in T3 required to activate mixing
    "egs_mix_min_epoch": 30,  # earliest epoch mixing can activate regardless of tier counts
}
