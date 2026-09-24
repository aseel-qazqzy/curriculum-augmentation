"""
experiments/madaug/train_madaug.py

Official MADAug (JackHck/MADAug @ 279b60a, main_higher.py) under the controlled CIFAR-100
protocol. Every line of the training procedure maps to official main_higher.py; the
[madaug-adapt] tags mark the only places where the shared protocol replaces an official
setting (see docs/madaug_reproduction_notes.md, Section 9).

Usage:
    python -m experiments.madaug.train_madaug --seed 42
    python -m experiments.madaug.train_madaug --seed 42 --resume          # continue after time limit
    python -m experiments.madaug.train_madaug --seed 42 --debug_n_train 256 --batch_size 32 --epochs 2
"""

import argparse
import copy
import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import higher
import numpy as np
import random
import torch
import torch.nn as nn
from torchvision import transforms

from experiments.madaug.core import adaptive_augmentor
from experiments.madaug.core.adaptive_augmentor import MDAAug
from experiments.madaug.core.projection import Projection
from experiments.madaug.data import get_madaug_cifar100_loaders
from experiments.madaug.make_split import DEFAULT_OUT as DEFAULT_SPLIT
from experiments.madaug.wrn_fg import WRNFeatureClassifier
from experiments.utils import build_optimizer, build_scheduler, set_seed
from models.registry import get_model
from training.trainer import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_COMMIT = "279b60a4d6e338aaa03ef0564a7bf94b8995f0b6"


def get_args(argv=None):
    p = argparse.ArgumentParser("madaug_controlled")
    # ── shared controlled protocol (Protocol B, same values as the other methods) ──
    p.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--split_file", default=str(DEFAULT_SPLIT))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--eta_min", type=float, default=1e-6)
    p.add_argument("--num_workers", type=int, default=4)
    # ── official MADAug settings (main.sh reduced_cifar10 / main_higher.py / author) ──
    p.add_argument("--k_ops", type=int, default=2)  # main.sh KOS=2
    p.add_argument("--temperature", type=float, default=3.0)  # main.sh tem=3
    p.add_argument("--threshold", type=int, default=40)  # main_higher.py default (tau)
    p.add_argument("--search_freq", type=float, default=3)  # main.sh SF=3
    p.add_argument("--bi_epochs", type=int, default=0)  # author, GitHub issue #1
    p.add_argument(
        "--proj_learning_rate", type=float, default=1e-3
    )  # main.sh SLR=0.001
    p.add_argument(
        "--proj_weight_decay", type=float, default=1e-3
    )  # main_higher.py default
    p.add_argument("--n_proj_layer", type=int, default=0)  # main_higher.py default
    p.add_argument("--grad_clip", type=float, default=5)  # main_higher.py default
    p.add_argument("--cutout_length", type=int, default=16)  # main.sh CUTOUT=16
    # ── run control / infrastructure ──
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "results" / "madaug"))
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--resume", action="store_true", help="resume from <run>_last.pth")
    p.add_argument(
        "--detect_anomaly",
        action="store_true",
        help="official utils.reproducibility enables it; debugging only, no numeric effect",
    )
    p.add_argument(
        "--debug_n_train",
        type=int,
        default=0,
        help="train on first N images (smoke tests)",
    )
    p.add_argument(
        "--debug_n_eval",
        type=int,
        default=0,
        help="evaluate on first N test/val images (smoke tests)",
    )
    p.add_argument(
        "--stop_after_epochs",
        type=int,
        default=0,
        help="smoke tests: keep the full --epochs schedule but stop after N completed epochs",
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="stop each epoch after N steps (smoke tests)",
    )
    return p.parse_args(argv)


def _pkg_version(name):
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def git_info() -> dict:
    def run(*cmd):
        try:
            return (
                subprocess.check_output(
                    cmd, cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except Exception:
            return None

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def run_name(args) -> str:
    name = f"madaug_cifar100_wrn28-10_ep{args.epochs}_s{args.seed}"
    if args.debug_n_train or args.debug_n_eval or args.max_steps:
        name += "_debug"
    if args.stop_after_epochs:
        name += "_smoke"
    return name


# ── official main_higher.py::train (lines 177-234) ────────────────────────────
def train(
    train_queue,
    valid_queue,
    gf_model,
    mdaaug,
    criterion,
    gf_optimizer,
    grad_clip,
    h_optimizer,
    epoch,
    search_freq,
    split_rate,
    bi_epochs,
    batch_size,
    device,
    stats,
    max_steps=0,
):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        if max_steps and step >= max_steps:
            break
        target = target.to(
            device, non_blocking=True
        )  # [madaug-adapt] .cuda() -> device
        if epoch > bi_epochs and step % search_freq == 0:
            h_optimizer.zero_grad()
            with higher.innerloop_ctx(gf_model, gf_optimizer) as (meta_model, diffopt):
                mdaaug.gf_model = meta_model
                aug_image = mdaaug(input, mode="explore")
                logits = meta_model.g(aug_image)
                loss = criterion(logits, target)
                nn.utils.clip_grad_norm_(meta_model.parameters(), grad_clip)
                diffopt.step(loss)

                if device.type == "cuda":
                    torch.cuda.empty_cache()
                input_search, target_search = next(iter(valid_queue))
                input_search = input_search.to(device, non_blocking=True)
                target_search = target_search.to(device, non_blocking=True)
                logits = meta_model(input_search)
                loss = criterion(logits, target_search)
                loss.backward()

            # diagnostics only (read-only)
            stats["h_updates"] += 1
            gn = float(sum(p.grad.norm() ** 2 for p in mdaaug.h_model.parameters() if p.grad is not None) ** 0.5)
            stats["epoch_h_grad_norms"].append(gn)
            stats["last_val_loss_meta"] = loss.item()
            stats["nonfinite"] += int(not math.isfinite(gn)) + int(not math.isfinite(stats["last_val_loss_meta"]))
            h_optimizer.step()

            mdaaug.gf_model = copy.deepcopy(gf_model)

        if split_rate < 1.0:
            train_split = torch.split(
                input,
                [
                    int(split_rate * batch_size),
                    batch_size - int(split_rate * batch_size),
                ],
                dim=0,
            )
            aug_image = mdaaug(train_split[0], mode="exploit")
            trans_images = []
            for i, image in enumerate(train_split[1]):
                pil_image = transforms.ToPILImage()(image)
                trans_image = mdaaug.after_transforms(pil_image)
                trans_images.append(trans_image)
            aug_imgs = torch.stack(trans_images, dim=0).to(
                device
            )  # [madaug-adapt] .cuda() -> device
            aug_image = torch.cat((aug_image, aug_imgs), dim=0)
            stats["policy_images"] += int(split_rate * batch_size)
        else:
            aug_image = mdaaug(input, mode="exploit")
            stats["policy_images"] += input.size(0)
        stats["train_images"] += input.size(0)
        gf_model.train()
        gf_optimizer.zero_grad()
        logits = gf_model(aug_image)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(gf_model.parameters(), grad_clip)
        gf_optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.detach().item(), n)
        stats["nonfinite"] += int(not math.isfinite(loss.detach().item()))
        top1.update(prec1.detach().item(), n)
        top5.update(prec5.detach().item(), n)

    return top1.avg, objs.avg


# ── official utils.py: AvgrageMeter, accuracy (verbatim) ──────────────────────
class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# ── checkpoint / resume (infrastructure; official code saves nothing) ─────────
def rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path,
    epoch,
    gf_model,
    h_model,
    mdaaug,
    gf_optimizer,
    h_optimizer,
    scheduler,
    history,
    stats,
):
    snapshot_is_model = mdaaug.gf_model is gf_model
    torch.save(
        {
            "epoch": epoch,  # last completed epoch
            "gf_model": gf_model.state_dict(),
            "h_model": h_model.state_dict(),
            "snapshot_is_model": snapshot_is_model,
            "snapshot": None if snapshot_is_model else mdaaug.gf_model.state_dict(),
            "gf_optimizer": gf_optimizer.state_dict(),
            "h_optimizer": h_optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "stats": stats,
            "rng": rng_state(),
        },
        path,
    )


def main(argv=None):
    args = get_args(argv)
    device = pick_device(args.device)
    adaptive_augmentor.DEVICE = device

    name = run_name(args)
    out = Path(args.out_dir)
    for sub in ("logs", "checkpoints", "metrics", "configs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    ckpt_last = out / "checkpoints" / f"{name}_last.pth"
    ckpt_final = out / "checkpoints" / f"{name}_final.pth"
    metrics_path = out / "metrics" / f"{name}.json"

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(out / "logs" / f"{name}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger().addHandler(fh)

    # [madaug-adapt] shared seeding (experiments/utils.py::set_seed) instead of official
    # utils.reproducibility; both seed random/np/torch/cuda and set cudnn deterministic.
    set_seed(args.seed)
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    train_queue, valid_queue, test_queue, val_eval_queue, (train_idx, val_idx) = (
        get_madaug_cifar100_loaders(
            args.data_root,
            args.split_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cutout_length=args.cutout_length,
            debug_n_train=args.debug_n_train,
            debug_n_eval=args.debug_n_eval,
        )
    )
    n_class = 100

    # [madaug-adapt] backbone: this project's WRN-28-10 (dropout 0.3) via f()/g() wrapper
    gf_model = WRNFeatureClassifier(get_model("wideresnet", num_classes=n_class)).to(
        device
    )
    h_model = Projection(
        in_features=gf_model.fc.in_features, n_layers=args.n_proj_layer, n_hidden=128
    ).to(device)

    # [madaug-adapt] shared optimizer + schedule (SGD nesterov, linear warmup 5 ep + cosine)
    proto = {
        "optimizer": "sgd",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "scheduler": "cosine",
        "warmup_epochs": args.warmup_epochs,
        "eta_min": args.eta_min,
    }
    gf_optimizer, _ = build_optimizer(gf_model, proto)
    scheduler, _ = build_scheduler(gf_optimizer, proto)

    h_optimizer = torch.optim.Adam(
        h_model.parameters(),
        lr=args.proj_learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.proj_weight_decay,
    )

    # [madaug-adapt] plain cross-entropy (shared protocol; paper Eq. 3) instead of LabelSmoothingCrossEntropy
    criterion = nn.CrossEntropyLoss().to(device)

    after_transforms = train_queue.dataset.after_transforms
    mdaaug_config = {
        "sampling": "prob",
        "k_ops": args.k_ops,
        "delta": 0.3,
        "temp": args.temperature,
        "search_d": 32,
        "target_d": 32,
    }
    mdaaug = MDAAug(
        after_transforms=after_transforms,
        n_class=n_class,
        gf_model=gf_model,
        h_model=h_model,
        save_dir=str(out / "logs"),
        config=mdaaug_config,
    )

    history = {
        k: []
        for k in (
            "epoch",
            "lr",
            "split_rate",
            "train_acc",
            "train_loss",
            "val_acc",
            "val_loss",
            "test_acc",
            "test_top5",
            "test_loss",
            "h_updates",
            "h_grad_norm_mean",
            "h_grad_norm_min",
            "h_grad_norm_max",
            "nonfinite",
            "policy_frac",
            "epoch_time_s",
            "gpu_mem_peak_alloc_gib",
            "gpu_mem_peak_reserved_gib",
        )
    }
    stats = {
        "h_updates": 0,
        "policy_images": 0,
        "train_images": 0,
        "last_val_loss_meta": None,
        "epoch_h_grad_norms": [],
        "nonfinite": 0,
    }
    start_epoch = 0

    if args.resume and ckpt_last.exists():
        ck = torch.load(ckpt_last, map_location=device, weights_only=False)
        gf_model.load_state_dict(ck["gf_model"])
        h_model.load_state_dict(ck["h_model"])
        gf_optimizer.load_state_dict(ck["gf_optimizer"])
        h_optimizer.load_state_dict(ck["h_optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if not ck["snapshot_is_model"]:
            mdaaug.gf_model = copy.deepcopy(gf_model)
            mdaaug.gf_model.load_state_dict(ck["snapshot"])
        history, stats = ck["history"], ck["stats"]
        set_rng_state(ck["rng"])
        start_epoch = ck["epoch"] + 1
        logging.info(f"Resumed from {ckpt_last} at epoch {start_epoch}")

    config = {
        "run_name": name,
        "method": "madaug",
        "dataset": "cifar100",
        "model": "wrn28-10 (models/wideresnet.py, dropout 0.3)",
        "args": vars(args),
        "device": str(device),
        "amp": False,
        "optimizer": "SGD(momentum=0.9, nesterov=True) via experiments.utils.build_optimizer",
        "scheduler": "LinearLR(0.1->1, warmup_epochs) + CosineAnnealingLR(T_max=epochs-warmup, eta_min) via build_scheduler",
        "loss": "CrossEntropyLoss",
        "split": {
            "file": args.split_file,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_queue.dataset),
        },
        "madaug": {
            **mdaaug_config,
            "ops": mdaaug.ops_names,
            "threshold_tau": args.threshold,
            "search_freq": args.search_freq,
            "bi_epochs": args.bi_epochs,
            "policy_optimizer": "Adam(betas=(0.9,0.999))",
            "proj_learning_rate": args.proj_learning_rate,
            "proj_weight_decay": args.proj_weight_decay,
            "grad_clip": args.grad_clip,
            "policy_val_batch": valid_queue.batch_size,
        },
        "official_repo": {
            "url": "https://github.com/JackHck/MADAug",
            "commit": OFFICIAL_COMMIT,
        },
        "git": git_info(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "higher": getattr(higher, "__version__", None) or _pkg_version("higher"),
            "python": sys.version.split()[0],
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_total_mem_gib": (torch.cuda.get_device_properties(device).total_memory / 2**30)
            if device.type == "cuda" else None,
        },
    }
    (out / "configs" / f"{name}.json").write_text(
        json.dumps(config, indent=2, default=str)
    )
    logging.info(
        f"Run {name} | device {device} | train {len(train_idx)} / policy-val {len(val_idx)} / test {len(test_queue.dataset)}"
    )

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr = scheduler.get_last_lr()[0]
        logging.info("epoch %d lr %e", epoch, lr)
        split_rate = min(
            torch.tanh(torch.FloatTensor([(epoch) / args.threshold])).item() + 0.01, 1.0
        )
        h_before, pol_before, img_before = (
            stats["h_updates"],
            stats["policy_images"],
            stats["train_images"],
        )
        stats["epoch_h_grad_norms"] = []
        nonfinite_before = stats["nonfinite"]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_acc, train_obj = train(
            train_queue,
            valid_queue,
            gf_model,
            mdaaug,
            criterion,
            gf_optimizer,
            args.grad_clip,
            h_optimizer,
            epoch,
            args.search_freq,
            split_rate,
            args.bi_epochs,
            args.batch_size,
            device,
            stats,
            args.max_steps,
        )
        logging.info(f"train_acc {train_acc} train_obj {train_obj}")

        # [madaug-adapt] evaluation for logging only; nothing below feeds back into training.
        val_loss, val_acc, _ = evaluate(gf_model, val_eval_queue, criterion, device)
        test_loss, test_acc, test_top5 = evaluate(
            gf_model, test_queue, criterion, device
        )
        scheduler.step()

        for k, v in (
            ("epoch", epoch),
            ("lr", lr),
            ("split_rate", split_rate),
            ("train_acc", train_acc),
            ("train_loss", train_obj),
            ("val_acc", val_acc * 100),
            ("val_loss", val_loss),
            ("test_acc", test_acc * 100),
            ("test_top5", test_top5 * 100),
            ("test_loss", test_loss),
            ("h_updates", stats["h_updates"] - h_before),
            ("h_grad_norm_mean", float(np.mean(stats["epoch_h_grad_norms"])) if stats["epoch_h_grad_norms"] else None),
            ("h_grad_norm_min", min(stats["epoch_h_grad_norms"], default=None)),
            ("h_grad_norm_max", max(stats["epoch_h_grad_norms"], default=None)),
            ("nonfinite", stats["nonfinite"] - nonfinite_before),
            (
                "policy_frac",
                (stats["policy_images"] - pol_before)
                / max(1, stats["train_images"] - img_before),
            ),
            ("epoch_time_s", time.time() - t0),
            ("gpu_mem_peak_alloc_gib", torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None),
            ("gpu_mem_peak_reserved_gib", torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None),
        ):
            history[k].append(v)
        logging.info(
            f"val_acc {val_acc * 100:.2f} test_acc {test_acc * 100:.2f} (logging only) | "
            f"split_rate {split_rate:.3f} h_updates {history['h_updates'][-1]} "
            f"h_grad_norm mean {history['h_grad_norm_mean'][-1]} | nonfinite {history['nonfinite'][-1]} | "
            f"peak mem {history['gpu_mem_peak_alloc_gib'][-1]} GiB | {history['epoch_time_s'][-1]:.0f}s"
        )

        save_checkpoint(
            ckpt_last,
            epoch,
            gf_model,
            h_model,
            mdaaug,
            gf_optimizer,
            h_optimizer,
            scheduler,
            history,
            stats,
        )
        metrics_path.write_text(
            json.dumps(
                {"run_name": name, "complete": False, "history": history}, indent=2
            )
        )
        if args.stop_after_epochs and epoch + 1 >= args.stop_after_epochs and epoch + 1 < args.epochs:
            logging.info(f"Smoke test: stopping after epoch {epoch} (schedule is for {args.epochs} epochs)")
            return {"run_name": name, "complete": False, "stopped_after_epoch": epoch, "history": history}

    final = {
        "run_name": name,
        "complete": True,
        "seed": args.seed,
        "epochs": args.epochs,
        "final_epoch_top1": history["test_acc"][-1],
        "final_epoch_top1_error": 100 - history["test_acc"][-1],
        "final_epoch_top5": history["test_top5"][-1],
        "reported": "final epoch (no test-based selection)",
        "total_h_updates": stats["h_updates"],
        "history": history,
    }
    metrics_path.write_text(json.dumps(final, indent=2))
    torch.save(
        {"gf_model": gf_model.state_dict(), "h_model": h_model.state_dict()}, ckpt_final
    )
    logging.info(
        f"FINAL top-1 {final['final_epoch_top1']:.2f}  error {final['final_epoch_top1_error']:.2f}"
    )
    return final


if __name__ == "__main__":
    main()
