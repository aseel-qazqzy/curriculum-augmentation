"""
train_baseline.py - baseline training, no curriculum
augmentation: none / static / random / randaugment
"""

import os
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from data.datasets import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_tiny_imagenet_loaders,
    get_static_transforms,
    get_no_augmentation_transforms,
)
from models.registry import get_model
from experiments.utils import set_seed, get_device, build_optimizer, build_scheduler, setup_logging
from experiments.config import BASE_CONFIG


DEFAULT_CONFIG = {
    **BASE_CONFIG,
    "augmentation":    "static",
    "ra_n":            2,
    "ra_m":            9,
    "experiment_name": "resnet18_static_aug_sgd_multistep",
}


def build_transforms(cfg: dict):
    aug     = cfg["augmentation"]
    dataset = cfg["dataset"]

    if aug == "none":
        return get_no_augmentation_transforms(dataset)

    elif aug == "static":
        from augmentations.policies import StaticAugmentation
        policy = StaticAugmentation(dataset=dataset, strength=cfg.get("fixed_strength", 0.7))
        return policy.get_train_transform(), policy.get_val_transform()

    elif aug == "random":
        from augmentations.policies import RandomAugmentation
        policy = RandomAugmentation(dataset=dataset)
        return policy.get_train_transform(), policy.get_val_transform()

    elif aug == "randaugment":
        from augmentations.policies import RandAugmentPolicy
        policy = RandAugmentPolicy(dataset=dataset, N=cfg["ra_n"], M=cfg["ra_m"])
        return policy.get_train_transform(), policy.get_val_transform()

    elif aug == "tiered_curriculum":
        from augmentations.policies import ThreeTierCurriculumAugmentation
        policy = ThreeTierCurriculumAugmentation(dataset=dataset, strength=cfg.get("fixed_strength", 0.7))
        return policy.get_train_transform(), policy.get_val_transform()

    else:
        raise ValueError(
            f"Unknown augmentation '{aug}'. "
            f"Choose from: none, static, random, randaugment, tiered_curriculum"
        )


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


def main(cfg: dict):
    dataset = cfg["dataset"]
    if cfg["experiment_name"] == DEFAULT_CONFIG["experiment_name"]:
        cfg["experiment_name"] = (
            f"{cfg['model']}_{cfg['augmentation']}_aug"
            f"_{cfg['optimizer']}_{cfg['scheduler']}"
        )

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
    if cfg['augmentation'] == 'tiered_curriculum':
        print("  Tier 1 (ep  1-33): flip, crop")
        print("  Tier 2 (ep 34-66): + color_jitter, rotation, shear")
        print("  Tier 3 (ep 67-end): + grayscale, cutout")
        print(f"  Strength (all ops): {cfg.get('fixed_strength', 0.5)}")
    print(f"  Epochs      : {cfg['epochs']} | Batch: {cfg['batch_size']}")

    if cfg.get("use_wandb"):
        import wandb
        wandb.init(project="curriculum-augmentation", name=cfg["experiment_name"], config=cfg)

    train_transform, val_transform = build_transforms(cfg)

    if cfg["dataset"] == "cifar100":
        loader_fn = get_cifar100_loaders
    elif cfg["dataset"] == "tiny_imagenet":
        loader_fn = get_tiny_imagenet_loaders
    else:
        loader_fn = get_cifar10_loaders
    train_loader, val_loader, test_loader = loader_fn(
        root            = cfg["data_root"],
        batch_size      = cfg["batch_size"],
        val_split       = cfg["val_split"],
        train_transform = train_transform,
        test_transform  = val_transform,
        debug           = cfg.get("debug", False),
    )

    num_classes = {"cifar100": 100, "tiny_imagenet": 200}.get(cfg["dataset"], 10)
    model       = get_model(cfg["model"], num_classes=num_classes).to(device)
    criterion   = nn.CrossEntropyLoss()

    optimizer, _         = build_optimizer(model, cfg)
    scheduler, milestones = build_scheduler(optimizer, cfg)
    print(f"{'='*60}\n")

    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    ckpt_path    = os.path.join(cfg["checkpoint_dir"], f"{cfg['experiment_name']}_best.pth")
    history_path = os.path.join(cfg["checkpoint_dir"], f"{cfg['experiment_name']}_history.pt")

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "val_top5":   [],
    }
    best_val_acc = 0.0
    start_epoch  = 1
    start_time   = time.time()
    es_patience  = cfg.get("early_stopping_patience", 0)
    es_counter   = 0

    resume_path = cfg.get("resume")
    if resume_path:
        print(f"  Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler:
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            else:
                scheduler.last_epoch = ckpt["epoch"]
        history      = ckpt["history"]
        best_val_acc = ckpt["val_acc"]
        start_epoch  = ckpt["epoch"] + 1
        print(f"  Resuming from epoch {start_epoch} | best val acc so far: {best_val_acc*100:.2f}%\n")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        if hasattr(train_transform, "set_epoch"):
            train_transform.set_epoch(epoch)

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
            elapsed  = time.time() - start_time
            tier_str = f" | {train_transform.tier_label()}" if hasattr(train_transform, "tier_label") else ""
            print(
                f"Epoch [{epoch:>3}/{cfg['epochs']}] "
                f"Train: {train_loss:.4f} / {train_acc*100:.2f}% | "
                f"Val: {val_loss:.4f} / {val_acc*100:.2f}% | "
                f"Top-5: {val_top5*100:.2f}% | "
                f"LR: {optimizer.param_groups[0]['lr']:.5f}{tier_str} | "
                f"Time: {elapsed:.0f}s"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            es_counter   = 0
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "scheduler_state":  scheduler.state_dict() if scheduler else None,
                "val_acc":          val_acc,
                "history":          history,
                "cfg":              {**cfg, "milestones": milestones},
            }, ckpt_path)
            print(f"Best saved (epoch={epoch}, val_acc={val_acc*100:.2f}%)")
        else:
            es_counter += 1
            if es_patience > 0 and es_counter >= es_patience:
                print(f"Early stopping at epoch {epoch} — no improvement for {es_patience} epochs.")
                break

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


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline training — CIFAR-10/100")

    parser.add_argument("--dataset",         type=str,   default=None,
                        choices=["cifar10", "cifar100", "tiny_imagenet"])
    parser.add_argument("--model",           type=str,   default=None,
                        choices=["resnet18", "resnet50", "wideresnet", "wrn16_8",
                                 "pyramidnet", "pyramidnet272", "baseline_cnn"])
    parser.add_argument("--augmentation",    type=str,   default=None,
                        choices=["none", "static", "random", "randaugment", "tiered_curriculum"])
    parser.add_argument("--ra_n",            type=int,   default=None)
    parser.add_argument("--ra_m",            type=int,   default=None)
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
    parser.add_argument("--resume",          type=str, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--fixed_strength",  type=float, default=None,
                        help="Augmentation op strength 0.0–1.0 (default: 0.7)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = DEFAULT_CONFIG.copy()

    for key in ["dataset", "model", "augmentation", "ra_n", "ra_m",
                "epochs", "batch_size", "lr", "optimizer", "scheduler",
                "experiment_name", "checkpoint_dir", "resume",
                "early_stopping_patience", "fixed_strength"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    main(cfg)
