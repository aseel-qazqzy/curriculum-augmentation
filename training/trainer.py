"""training/trainer.py — training loop for curriculum learning experiments."""

import os
import time
import torch
import torch.nn as nn
from pathlib import Path

from training.losses import (
    LabelSmoothingLoss,
    LossTracker,
    get_batch_difficulties,
    epoch_difficulty,
)
from augmentations.curriculum import CurriculumDataset


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


def train_one_epoch_cl(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch: int,
    total_epochs: int,
    cl_dataset: CurriculumDataset,
    loss_tracker: LossTracker,
    schedule: str  = "sigmoid",
    mode: str      = "inverse",
    blend: float   = 0.7,
    warmup_epochs: int = 5,
    aug_milestones: list = None,
    max_difficulty: float = 1.0,
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_indices      = []
    all_difficulties = []

    criterion_per_sample = nn.CrossEntropyLoss(reduction="none")

    for images, labels, indices in loader:
        images  = images.to(device)
        labels  = labels.to(device)
        indices = indices.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            per_sample_loss = criterion_per_sample(outputs, labels)

        loss_tracker.update(indices, per_sample_loss)

        difficulties = get_batch_difficulties(
            per_sample_loss,
            epoch=epoch,
            total_epochs=total_epochs,
            schedule=schedule,
            mode=mode,
            warmup_epochs=warmup_epochs,
            blend=blend,
            aug_milestones=aug_milestones,
            max_difficulty=max_difficulty,
        )

        all_indices.append(indices.cpu())
        all_difficulties.append(difficulties.cpu())

        total_loss += loss.item() * images.size(0)
        correct    += outputs.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)

    if all_indices:
        idx_tensor  = torch.cat(all_indices)
        diff_tensor = torch.cat(all_difficulties)
        new_diffs   = cl_dataset.difficulties.clone()
        new_diffs[idx_tensor] = diff_tensor
        cl_dataset.set_difficulties(new_diffs)

    mean_diff = float(torch.cat(all_difficulties).mean()) if all_difficulties else 0.0
    return total_loss / total, correct / total, mean_diff


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct1, correct5, total = 0.0, 0, 0, 0

    for batch in loader:
        # Handle both (images, labels) and (images, labels, indices)
        images, labels = batch[0], batch[1]
        images, labels = images.to(device), labels.to(device)

        outputs     = model(images)
        loss        = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)

        pred1    = outputs.argmax(dim=1)
        correct1 += pred1.eq(labels).sum().item()

        k        = min(5, outputs.size(1))
        top5     = outputs.topk(k, dim=1).indices
        correct5 += top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

        total += labels.size(0)

    return total_loss / total, correct1 / total, correct5 / total


def run_training(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    cfg: dict,
    cl_dataset: CurriculumDataset = None,
    loss_tracker: LossTracker     = None,
):
    """Full training loop. Curriculum mode if cl_dataset and loss_tracker are provided."""
    is_cl      = cl_dataset is not None and loss_tracker is not None
    epochs     = cfg["epochs"]
    log_every  = cfg.get("log_every", 10)
    ckpt_dir   = cfg["checkpoint_dir"]
    exp_name   = cfg["experiment_name"]

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path    = os.path.join(ckpt_dir, f"{exp_name}_best.pth")
    history_path = os.path.join(ckpt_dir, f"{exp_name}_history.pt")

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "val_top5":   [],
        "difficulty": [],
    }

    best_val_acc = 0.0
    best_epoch   = 0
    start_epoch  = 1
    start_time   = time.time()
    milestones   = cfg.get("milestones", [])
    effective_lr = cfg.get("effective_lr", cfg.get("lr", 0.1))

    resume_path = cfg.get("resume")
    if resume_path:
        print(f"  Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler:
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            else:
                scheduler.last_epoch = ckpt["epoch"]
        history      = ckpt["history"]
        best_val_acc = ckpt["val_acc"]
        best_epoch   = ckpt["epoch"]
        start_epoch  = ckpt["epoch"] + 1
        print(f"  Resuming from epoch {start_epoch} | best val acc so far: {best_val_acc*100:.2f}%\n")

    for epoch in range(start_epoch, epochs + 1):

        if is_cl:
            train_loss, train_acc, mean_diff = train_one_epoch_cl(
                model, train_loader, optimizer, criterion, device,
                epoch=epoch, total_epochs=epochs,
                cl_dataset=cl_dataset, loss_tracker=loss_tracker,
                schedule=cfg.get("cl_schedule", "sigmoid"),
                mode=cfg.get("cl_mode", "inverse"),
                blend=cfg.get("cl_blend", 0.7),
                warmup_epochs=cfg.get("warmup_epochs", 5),
                aug_milestones=cfg.get("aug_milestones", None),
                max_difficulty=cfg.get("max_difficulty", 1.0),
            )
        else:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            mean_diff = 0.0

        val_loss, val_acc, val_top5 = evaluate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_top5"].append(val_top5)
        history["difficulty"].append(mean_diff)
        torch.save(history, history_path)

        if epoch % log_every == 0 or epoch == 1 or cfg.get("debug"):
            elapsed    = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]
            diff_str   = f" | Diff: {mean_diff:.3f}" if is_cl else ""
            print(
                f"Epoch [{epoch:>3}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                f"Top-5: {val_top5*100:.2f}% | "
                f"LR: {current_lr:.5f}{diff_str} | Time: {elapsed:.0f}s"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "scheduler_state":  scheduler.state_dict() if scheduler else None,
                "val_acc":          val_acc,
                "history":          history,
                "cfg":              {**cfg, "effective_lr": effective_lr, "milestones": milestones},
            }, ckpt_path)
            print(f"  Best saved (epoch={epoch}, val_acc={val_acc*100:.2f}%)")

    test_loss, test_top1, test_top5 = evaluate(model, test_loader, criterion, device)
    total_time = time.time() - start_time

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt["test_top1"]     = test_top1
    ckpt["test_top5"]     = test_top5
    ckpt["total_minutes"] = total_time / 60
    torch.save(ckpt, ckpt_path)

    print(f"\n── FINAL RESULTS: {exp_name} ──────────────────────")
    print(f"  Best Val Top-1  : {best_val_acc*100:.2f}%  (epoch {best_epoch})")
    print(f"  Test Top-1      : {test_top1*100:.2f}%")
    print(f"  Test Top-5      : {test_top5*100:.2f}%")
    print(f"  Val-Test Gap    : {abs(best_val_acc - test_top1)*100:.2f}%")
    print(f"  Total Time      : {total_time/60:.1f} minutes")
    print(f"  History saved   : {history_path}")

    return history, best_val_acc, test_top1, test_top5
