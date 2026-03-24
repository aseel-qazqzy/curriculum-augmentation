"""
train_baseline.py - baseline training, no curriculum
augmentation: none / static / random / randaugment
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Project root on sys.path so all internal imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from data.datasets import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_static_transforms,
    get_no_augmentation_transforms,
)
from models.registry import get_model
from experiments.utils import set_seed, get_device, build_optimizer, build_scheduler, setup_logging
from experiments.config import BASE_CONFIG


#
# DEFAULT CONFIG — extends shared BASE_CONFIG with baseline-specific keys
DEFAULT_CONFIG = {
    **BASE_CONFIG,
    "augmentation":    "static",
    "ra_n":            2,   # RandAugment: ops per image   (paper default)
    "ra_m":            9,   # RandAugment: magnitude 0-30  (paper default for CIFAR-10)
    "experiment_name": "resnet18_static_aug_sgd_multistep",
}



# AUGMENTATION
def build_transforms(cfg: dict):
    """Return (train_transform, val_transform) for the chosen augmentation strategy."""
    aug     = cfg["augmentation"]
    dataset = cfg["dataset"]

    if aug == "none":
        return get_no_augmentation_transforms(dataset)

    elif aug == "static":
        return get_static_transforms(dataset)

    elif aug == "random":
        from augmentations.policies import RandomAugmentation
        policy = RandomAugmentation(dataset=dataset)
        return policy.get_train_transform(), policy.get_val_transform()

    elif aug == "randaugment":
        from augmentations.policies import RandAugmentPolicy
        policy = RandAugmentPolicy(dataset=dataset, N=cfg["ra_n"], M=cfg["ra_m"])
        return policy.get_train_transform(), policy.get_val_transform()

    else:
        raise ValueError(
            f"Unknown augmentation '{aug}'. "
            f"Choose from: none, static, random, randaugment"
        )


# TRAIN / EVAL

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += outputs.argmax(dim=1).eq(labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct1, correct5, total = 0.0, 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs     = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        correct1   += outputs.argmax(dim=1).eq(labels).sum().item()
        k           = min(5, outputs.size(1))
        correct5   += outputs.topk(k, dim=1).indices.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct1 / total, correct5 / total


# MAIN
def main(cfg: dict):
    # Always suffix experiment name with dataset so checkpoints/logs are unambiguous
    dataset = cfg["dataset"]
    if not cfg["experiment_name"].endswith(f"_{dataset}"):
        cfg["experiment_name"] = f"{cfg['experiment_name']}_{dataset}"

    tee = setup_logging(cfg)
    set_seed(cfg["seed"])
    device = get_device()

    print(f"\n{'='*60}")
    print(f"  Experiment  : {cfg['experiment_name']}")
    print(f"  Dataset     : {cfg['dataset']}")
    print(f"  Model       : {cfg['model']}")
    print(f"  Augmentation: {cfg['augmentation']}"
          + (f"  (N={cfg['ra_n']}, M={cfg['ra_m']})" if cfg['augmentation'] == 'randaugment' else ""))
    print(f"  Epochs      : {cfg['epochs']} | Batch: {cfg['batch_size']}")

    if cfg.get("use_wandb"):
        import wandb
        wandb.init(project="curriculum-augmentation", name=cfg["experiment_name"], config=cfg)

    # ── Data 
    train_transform, val_transform = build_transforms(cfg)

    loader_fn = get_cifar100_loaders if cfg["dataset"] == "cifar100" else get_cifar10_loaders
    train_loader, val_loader, test_loader = loader_fn(
        root            = cfg["data_root"],
        batch_size      = cfg["batch_size"],
        val_split       = cfg["val_split"],
        train_transform = train_transform,
        test_transform  = val_transform,
        debug           = cfg.get("debug", False),
    )

    # ── Model 
    num_classes = 100 if cfg["dataset"] == "cifar100" else 10
    model       = get_model(cfg["model"], num_classes=num_classes).to(device)
    criterion   = nn.CrossEntropyLoss()

    # ── Optimizer & Scheduler 
    optimizer, _         = build_optimizer(model, cfg)
    scheduler, milestones = build_scheduler(optimizer, cfg)
    print(f"{'='*60}\n")

    # ── Setup 
    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    ckpt_path    = os.path.join(cfg["checkpoint_dir"], f"{cfg['experiment_name']}_best.pth")
    history_path = os.path.join(cfg["checkpoint_dir"], f"{cfg['experiment_name']}_history.pt")

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "val_top5":   [],
    }
    best_val_acc = 0.0
    start_time   = time.time()

    # ── Training Loop 
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc       = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_top5 = evaluate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_top5"].append(val_top5)
        torch.save(history, history_path)

        if cfg.get("use_wandb"):
            import wandb
            wandb.log({
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "val_top5": val_top5,
                "lr": optimizer.param_groups[0]["lr"],
            })

        if epoch % cfg["log_every"] == 0 or epoch == 1 or cfg.get("debug"):
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch:>3}/{cfg['epochs']}] "
                f"Train: {train_loss:.4f} / {train_acc*100:.2f}% | "
                f"Val: {val_loss:.4f} / {val_acc*100:.2f}% | "
                f"Top-5: {val_top5*100:.2f}% | "
                f"LR: {optimizer.param_groups[0]['lr']:.5f} | "
                f"Time: {elapsed:.0f}s"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "val_acc":          val_acc,
                "history":          history,
                "cfg":              {**cfg, "milestones": milestones},
            }, ckpt_path)
            print(f"Best saved (epoch={epoch}, val_acc={val_acc*100:.2f}%)")

    # ── Final Test Evaluation 
    # Load best checkpoint weights before eval on test set
    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_top1, test_top5 = evaluate(model, test_loader, criterion, device)
    total_time = time.time() - start_time

    best_ckpt["test_top1"]     = test_top1
    best_ckpt["test_top5"]     = test_top5
    best_ckpt["total_minutes"] = total_time / 60
    torch.save(best_ckpt, ckpt_path)

    print(f"\n── FINAL RESULTS: {cfg['experiment_name']} ──")
    print(f"  Best Val Top-1 : {best_val_acc*100:.2f}%")
    print(f"  Test Top-1     : {test_top1*100:.2f}%")
    print(f"  Test Top-5     : {test_top5*100:.2f}%")
    print(f"  Val-Test Gap   : {abs(best_val_acc - test_top1)*100:.2f}%")
    print(f"  Total Time     : {total_time/60:.1f} min")

    if cfg.get("use_wandb"):
        import wandb
        wandb.finish()

    tee.close()
    return history, best_val_acc, test_top1


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description="Baseline training — CIFAR-10/100")

    parser.add_argument("--dataset",         type=str,   default=None,
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--model",           type=str,   default=None,
                        choices=["resnet18", "resnet50", "wideresnet", "wrn16_8",
                                 "pyramidnet", "pyramidnet272", "baseline_cnn"])
    parser.add_argument("--augmentation",    type=str,   default=None,
                        choices=["none", "static", "random", "randaugment"])
    parser.add_argument("--ra_n",            type=int,   default=None,
                        help="RandAugment: ops per image (paper default: 2)")
    parser.add_argument("--ra_m",            type=int,   default=None,
                        help="RandAugment: magnitude in [0-30] (paper default: 9 for CIFAR-10, 14 for CIFAR-100)")
    parser.add_argument("--epochs",          type=int,   default=None)
    parser.add_argument("--batch_size",      type=int,   default=None)
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--optimizer",       type=str,   default=None,
                        choices=["sgd", "adam"])
    parser.add_argument("--scheduler",       type=str,   default=None,
                        choices=["multistep", "cosine", "none"])
    parser.add_argument("--experiment_name", type=str,   default=None)
    parser.add_argument("--checkpoint_dir",  type=str,   default=None)
    parser.add_argument("--use_wandb",       action="store_true")
    parser.add_argument("--debug",           action="store_true",
                        help="2 epochs on 512 samples — quick smoke test")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = DEFAULT_CONFIG.copy()

    for key in ["dataset", "model", "augmentation", "ra_n", "ra_m",
                "epochs", "batch_size", "lr", "optimizer", "scheduler",
                "experiment_name", "checkpoint_dir"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    # if args.use_wandb:
    #     cfg["use_wandb"] = True

    # if args.debug:
    #     cfg["debug"]  = True
    #     cfg["epochs"] = 2
    #     print("DEBUG MODE: 2 epochs, 512 samples")

    main(cfg)
