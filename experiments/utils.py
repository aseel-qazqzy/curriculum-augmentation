"""
experiments/utils.py
Shared utilities for all experiment scripts.
"""

import sys
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


class _Tee:
    """
    Mirrors all writes to sys.stdout into a log file simultaneously.
    """

    def __init__(self, log_path: Path):
        self._console = sys.stdout
        self._log_file = open(log_path, "w", buffering=1)
        sys.stdout = self

    def write(self, msg: str):
        self._console.write(msg)
        self._log_file.write(msg)

    def flush(self):
        self._console.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return False

    def fileno(self):
        return self._console.fileno()

    def close(self):
        sys.stdout = self._console
        self._log_file.close()


def setup_logging(cfg: dict) -> _Tee:
    """
    Creates results/logs/<experiment_name>_<timestamp>.log and starts
    mirroring all stdout output into it.
    """
    log_dir = (
        Path(cfg.get("checkpoint_dir", "./checkpoints")).parent / "results" / "logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{cfg['experiment_name']}_{ts}.log"

    tee = _Tee(log_path)
    print(f"  Log file    : {log_path}")
    return tee


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only=True avoids crashes on ops without deterministic kernels (e.g. scatter_add on CUDA)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _mps_is_stable() -> bool:
    """Return False if the MPS backend crashes on a conv2d probe (PyTorch < 2.1 is unreliable)."""
    try:
        import subprocess, sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch.nn as nn; "
                "x = torch.randn(2, 3, 32, 32, device='mps'); "
                "y = nn.Conv2d(3, 16, 3, padding=1).to('mps')(x); "
                "assert y.shape == (2, 16, 32, 32)",
            ],
            timeout=15,
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device      : CUDA | {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and _mps_is_stable():
        device = torch.device("mps")
        print(f"  Device      : MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"  Device      : CPU")
    return device


def build_optimizer(model: nn.Module, cfg: dict):
    """Returns (optimizer, lr)."""
    name = cfg["optimizer"].lower()
    lr = float(cfg["lr"])
    wd = cfg["weight_decay"]

    if name == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True
        )
    elif name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Use 'sgd' or 'adam'.")

    print(f"  Optimizer   : {name.upper()} | lr={lr} | wd={wd}")
    return optimizer, lr


def build_scheduler(optimizer, cfg: dict):
    """Returns (scheduler, milestones)."""
    epochs = cfg["epochs"]
    name = cfg["scheduler"].lower()

    # Accept common aliases
    _aliases = {
        "cosine_annealing": "cosine",
        "multisteplr": "multistep",
    }
    name = _aliases.get(name, name)

    if name == "multistep":
        milestones = [int(epochs * f) for f in (0.33, 0.66, 0.83)]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1
        )
        print(f"  Scheduler   : MultiStepLR | milestones={milestones} | gamma=0.1")
    elif name == "cosine":
        warmup_epochs = cfg.get("warmup_epochs", 5)
        eta_min = cfg.get("eta_min", 1e-6)
        milestones = []
        if warmup_epochs > 0:
            warmup_sched = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=eta_min
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
            print(
                f"  Scheduler   : CosineAnnealingLR + LinearWarmup "
                f"| warmup={warmup_epochs} ep | T_max={epochs - warmup_epochs} | eta_min={eta_min}"
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=eta_min
            )
            print(
                f"  Scheduler   : CosineAnnealingLR | T_max={epochs} | eta_min={eta_min}"
            )
    elif name == "none":
        scheduler = None
        milestones = []
        print("  Scheduler   : None")
    else:
        raise ValueError(
            f"Unknown scheduler '{name}'. Use 'multistep', 'cosine', or 'none'."
        )

    return scheduler, milestones
