"""
experiments/train.py
Loss-guided curriculum augmentation training.

Usage:
    python experiments/train.py --dataset cifar10 --model resnet18
    python experiments/train.py --dataset cifar100 --model resnet50
    python experiments/train.py --dataset cifar10 --cl_schedule linear
    python experiments/train.py --debug
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.registry import get_model

from experiments.utils import (
    set_seed,
    get_device,
    build_optimizer,
    build_scheduler,
    setup_logging,
)
from experiments.config import BASE_CONFIG
from data.datasets import CIFAR_STATS
from augmentations.curriculum import CurriculumTransform, CurriculumDataset
from training.losses import LabelSmoothingLoss, LossTracker
from training.trainer import run_training, evaluate


class ValDataset(torch.utils.data.Dataset):
    """Wraps a base dataset with a fixed transform (no curriculum). Module-level for pickle."""

    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if not isinstance(img, torch.Tensor):
            img = self.transform(img)
        return img, label


DEFAULT_CONFIG = {
    **BASE_CONFIG,
    "cl_schedule": "cosine",
    "cl_mode": "inverse",
    "cl_blend": 1.0,
    "warmup_epochs": 10,
    "max_difficulty": 0.70,
    "label_smoothing": 0.1,
    "aug_milestones": None,
    "experiment_name": "resnet18_cl_loss_sgd_multistep",
    "model": "resnet18",
}


def main(cfg: dict):
    dataset = cfg["dataset"]
    if not cfg["experiment_name"].endswith(f"_{dataset}"):
        cfg["experiment_name"] = f"{cfg['experiment_name']}_{dataset}"

    tee = setup_logging(cfg)
    set_seed(cfg["seed"])
    device = get_device()

    print(f"\n{'=' * 60}")
    print(f"  Experiment  : {cfg['experiment_name']}")
    print(f"  Dataset     : {cfg['dataset'].upper()}")
    print(f"  Model       : {cfg['model']}")
    print(f"  Method      : Loss-Guided Curriculum Learning")
    print(f"  CL Schedule : {cfg['cl_schedule']}")
    print(f"  CL Mode     : {cfg['cl_mode']}")
    print(f"  CL Blend    : {cfg['cl_blend']} (epoch/sample mix)")
    print(f"  Max Diff    : {cfg.get('max_difficulty', 1.0)}")
    if cfg["cl_schedule"] == "milestone":
        milestones = cfg.get("aug_milestones") or [(20, 0.20), (60, 0.45)]
        print(f"  Milestones  : {milestones}  (beyond last → max_difficulty)")
    else:
        print(f"  Warmup      : {cfg['warmup_epochs']} epochs")
    print(f"  Epochs      : {cfg['epochs']}")
    print(f"  Batch Size  : {cfg['batch_size']}")

    if cfg.get("use_wandb"):
        import wandb

        wandb.init(
            project="curriculum-augmentation", name=cfg["experiment_name"], config=cfg
        )

    dataset_name = cfg["dataset"]
    num_classes = 100 if dataset_name == "cifar100" else 10

    import torchvision.transforms as T

    stats = CIFAR_STATS[dataset_name]
    val_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(stats["mean"], stats["std"]),
        ]
    )

    DS = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100

    full_train = DS(cfg["data_root"], train=True, download=True, transform=None)
    test_ds = DS(cfg["data_root"], train=False, download=True, transform=val_transform)

    val_size = int(len(full_train) * cfg["val_split"])
    train_size = len(full_train) - val_size
    train_base, val_base = torch.utils.data.random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    if cfg.get("debug"):
        train_base = torch.utils.data.Subset(train_base, range(512))
        val_base = torch.utils.data.Subset(val_base, range(128))
        test_ds = torch.utils.data.Subset(test_ds, range(128))

    cl_transform = CurriculumTransform(dataset=dataset_name)
    cl_dataset = CurriculumDataset(train_base, cl_transform, default_difficulty=0.0)
    loss_tracker = LossTracker(n_samples=len(train_base), momentum=0.9)

    val_dataset = ValDataset(val_base, val_transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        cl_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )

    print(
        f"\n  Train: {len(cl_dataset):,} | Val: {len(val_dataset):,} | "
        f"Test: {len(test_ds):,}"
    )

    model = get_model(cfg["model"], num_classes=num_classes).to(device)

    smoothing = cfg.get("label_smoothing", 0.0)
    if smoothing > 0:
        criterion = LabelSmoothingLoss(
            num_classes=num_classes, smoothing=smoothing, reduction="mean"
        )
        print(f"  Loss        : LabelSmoothing (smoothing={smoothing})")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"  Loss        : CrossEntropy")

    optimizer, effective_lr = build_optimizer(model, cfg)
    scheduler, milestones = build_scheduler(optimizer, cfg)
    cfg["effective_lr"] = effective_lr
    cfg["milestones"] = milestones
    print(f"{'=' * 60}\n")

    history, best_val, test_top1, test_top5 = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        cfg=cfg,
        cl_dataset=cl_dataset,
        loss_tracker=loss_tracker,
    )

    if cfg.get("use_wandb"):
        import wandb

        wandb.finish()

    tee.close()
    return history, best_val, test_top1


def parse_args():
    parser = argparse.ArgumentParser(description="Train CL experiment")
    parser.add_argument(
        "--dataset", type=str, default=None, choices=["cifar10", "cifar100"]
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "resnet18",
            "resnet50",
            "wideresnet",
            "wrn16_8",
            "pyramidnet",
            "pyramidnet272",
            "baseline_cnn",
        ],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=["sgd", "adam"])
    parser.add_argument(
        "--scheduler", type=str, default=None, choices=["multistep", "cosine", "none"]
    )
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument(
        "--cl_schedule",
        type=str,
        default=None,
        choices=["sigmoid", "linear", "cosine", "step", "milestone"],
    )
    parser.add_argument(
        "--cl_mode", type=str, default=None, choices=["inverse", "direct", "normalized"]
    )
    parser.add_argument("--cl_blend", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--max_difficulty", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument(
        "--aug_milestone_epochs", type=int, nargs="+", default=None, metavar="EPOCH"
    )
    parser.add_argument(
        "--aug_milestone_difficulties",
        type=float,
        nargs="+",
        default=None,
        metavar="DIFF",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = DEFAULT_CONFIG.copy()

    for key in [
        "dataset",
        "model",
        "epochs",
        "lr",
        "batch_size",
        "optimizer",
        "scheduler",
        "experiment_name",
        "checkpoint_dir",
        "cl_schedule",
        "cl_mode",
        "cl_blend",
        "warmup_epochs",
        "max_difficulty",
        "label_smoothing",
        "resume",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    if args.aug_milestone_epochs is not None:
        epochs_list = args.aug_milestone_epochs
        diffs_list = args.aug_milestone_difficulties or [0.20, 0.45][: len(epochs_list)]
        if len(diffs_list) != len(epochs_list):
            raise ValueError(
                f"--aug_milestone_epochs ({len(epochs_list)} values) and "
                f"--aug_milestone_difficulties ({len(diffs_list)} values) must have the same length."
            )
        cfg["aug_milestones"] = list(zip(epochs_list, diffs_list))
        if cfg.get("cl_schedule") != "milestone":
            cfg["cl_schedule"] = "milestone"

    if args.use_wandb:
        cfg["use_wandb"] = True
    if args.debug:
        cfg["debug"] = True
        cfg["epochs"] = 2
        print("⚠️  DEBUG MODE: 2 epochs, tiny dataset")

    main(cfg)
