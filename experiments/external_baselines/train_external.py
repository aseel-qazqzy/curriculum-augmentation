"""
experiments/external_baselines/train_external.py

RandAugment / TrivialAugment external baselines under the controlled CIFAR-100 protocol of the
MADAug comparison. NEW runs; nothing here touches the Static/ETS/LPS/EGS or MADAug code or results.

Reference protocol = experiments/madaug/train_madaug.py (controlled MADAug). Shared settings are
imported, not copied: split (make_split.load_split), WRN-28-10 (models.registry via the same
WRNFeatureClassifier wrapper), optimizer/scheduler (experiments.utils.build_optimizer /
build_scheduler with the same proto dict), seeding (experiments.utils.set_seed), evaluation
(training.trainer.evaluate), meters (train_madaug.AvgrageMeter / accuracy).
The training step is train_madaug.train() without the MADAug policy branch:
    forward -> CrossEntropy -> backward -> clip_grad_norm_(5) -> SGD step
Only the training augmentation differs (experiments/external_baselines/data.py).

Usage:
    python -m experiments.external_baselines.train_external --method randaugment --seed 42
    python -m experiments.external_baselines.train_external --method trivialaugment --seed 42
    ... --resume                                  # continue after a time limit
    ... --stop_after_epochs 2                     # 2-epoch check on the full schedule
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision

from experiments.external_baselines.data import METHODS, get_external_cifar100_loaders
from experiments.madaug.make_split import DEFAULT_OUT as DEFAULT_SPLIT
from experiments.madaug.train_madaug import (
    AvgrageMeter,
    accuracy,
    git_info,
    pick_device,
    rng_state,
    set_rng_state,
)
from experiments.madaug.wrn_fg import WRNFeatureClassifier
from experiments.utils import build_optimizer, build_scheduler, set_seed
from models.registry import get_model
from training.trainer import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TA_REPO = "https://github.com/automl/trivialaugment"
TA_COMMIT = "e6545d80affcc18b7024185a096e93e21010023e"


def get_args(argv=None):
    p = argparse.ArgumentParser("external_baselines_controlled")
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    # ── shared controlled protocol (identical defaults to train_madaug.py) ──
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
    p.add_argument("--grad_clip", type=float, default=5)
    p.add_argument("--cutout_length", type=int, default=16)
    # ── run control / infrastructure (same flags as train_madaug.py) ──
    p.add_argument(
        "--out_dir", default=str(PROJECT_ROOT / "results" / "external_baselines")
    )
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--resume", action="store_true", help="resume from <run>_last.pth")
    p.add_argument("--debug_n_train", type=int, default=0)
    p.add_argument("--debug_n_eval", type=int, default=0)
    p.add_argument("--stop_after_epochs", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=0)
    return p.parse_args(argv)


def run_name(args) -> str:
    # Same pattern as MADAug's controlled runs: madaug_cifar100_wrn28-10_ep200_s42
    tag = {
        "randaugment": "randaugment_n{n}m{m}".format(**METHODS["randaugment"]),
        "trivialaugment": "trivialaugment_wide",
    }[args.method]
    name = f"{tag}_cifar100_wrn28-10_ep{args.epochs}_s{args.seed}"
    if args.debug_n_train or args.debug_n_eval or args.max_steps:
        name += "_debug"
    if args.stop_after_epochs:
        name += "_smoke"
    return name


def train(model, train_queue, criterion, optimizer, grad_clip, device, max_steps=0):
    """train_madaug.train() minus the MADAug policy branch (no bi-level step, no policy ops)."""
    objs, top1, nonfinite = AvgrageMeter(), AvgrageMeter(), 0
    for step, (input, target) in enumerate(train_queue):
        if max_steps and step >= max_steps:
            break
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        model.train()
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        prec1, _ = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.detach().item(), n)
        nonfinite += int(not np.isfinite(loss.detach().item()))
        top1.update(prec1.detach().item(), n)
    return top1.avg, objs.avg, nonfinite


def main(argv=None):
    args = get_args(argv)
    device = pick_device(args.device)

    name = run_name(args)
    out = Path(args.out_dir)
    for sub in ("logs", "checkpoints", "metrics", "configs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    ckpt_last = out / "checkpoints" / f"{name}_last.pth"
    ckpt_final = out / "checkpoints" / f"{name}_final.pth"
    metrics_path = out / "metrics" / f"{name}.json"
    if metrics_path.exists() and not args.resume:
        if json.loads(metrics_path.read_text()).get("complete"):
            raise FileExistsError(
                f"Completed run already exists: {metrics_path}. Refusing to overwrite."
            )

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(out / "logs" / f"{name}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger().addHandler(fh)

    # Same call order as train_madaug.main(): seed -> loaders -> model, so a given seed gives
    # the same initial WRN-28-10 weights as the controlled MADAug run.
    set_seed(args.seed)

    train_queue, test_queue, val_eval_queue, (train_idx, val_idx) = (
        get_external_cifar100_loaders(
            args.method,
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
    model = WRNFeatureClassifier(get_model("wideresnet", num_classes=n_class)).to(
        device
    )

    proto = {
        "optimizer": "sgd",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "scheduler": "cosine",
        "warmup_epochs": args.warmup_epochs,
        "eta_min": args.eta_min,
    }
    optimizer, _ = build_optimizer(model, proto)
    scheduler, _ = build_scheduler(optimizer, proto)
    criterion = nn.CrossEntropyLoss().to(device)

    history = {
        k: []
        for k in (
            "epoch",
            "lr",
            "train_acc",
            "train_loss",
            "val_acc",
            "val_loss",
            "test_acc",
            "test_top5",
            "test_loss",
            "nonfinite",
            "epoch_time_s",
            "gpu_mem_peak_alloc_gib",
        )
    }
    start_epoch = 0
    if args.resume and ckpt_last.exists():
        ck = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        history = ck["history"]
        set_rng_state(ck["rng"])
        start_epoch = ck["epoch"] + 1
        logging.info(f"Resumed from {ckpt_last} at epoch {start_epoch}")

    train_tf = train_queue.dataset.dataset.transform
    config = {
        "run_name": name,
        "method": args.method,
        "protocol": "controlled_49k1k_final_epoch (reference: experiments/madaug/train_madaug.py)",
        "dataset": "cifar100",
        "model": "wrn28-10 (models/wideresnet.py, dropout 0.3)",
        "args": vars(args),
        "device": str(device),
        "amp": False,
        "optimizer": "SGD(momentum=0.9, nesterov=True) via experiments.utils.build_optimizer",
        "scheduler": "LinearLR(0.1->1, warmup_epochs) + CosineAnnealingLR(T_max=epochs-warmup, eta_min) via build_scheduler",
        "loss": "CrossEntropyLoss",
        "grad_clip": args.grad_clip,
        "split": {
            "file": args.split_file,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_queue.dataset),
        },
        "augmentation": {
            **METHODS[args.method],
            "train_transform": repr(train_tf),
            "implementation": "experiments/external_baselines/vendor/aug_lib.py (official, 1-line py3.11 patch)",
            "official_repo": {"url": TA_REPO, "commit": TA_COMMIT},
        },
        "git": git_info(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": PIL.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
    }
    (out / "configs" / f"{name}.json").write_text(
        json.dumps(config, indent=2, default=str)
    )
    logging.info(
        f"Run {name} | device {device} | train {len(train_idx)} / val {len(val_idx)} / test {len(test_queue.dataset)}"
    )
    logging.info(f"Train transform: {train_tf}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr = scheduler.get_last_lr()[0]
        logging.info("epoch %d lr %e", epoch, lr)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_acc, train_obj, nonfinite = train(
            model,
            train_queue,
            criterion,
            optimizer,
            args.grad_clip,
            device,
            args.max_steps,
        )
        logging.info(f"train_acc {train_acc} train_obj {train_obj}")

        # Evaluation for logging only; nothing below feeds back into training.
        val_loss, val_acc, _ = evaluate(model, val_eval_queue, criterion, device)
        test_loss, test_acc, test_top5 = evaluate(model, test_queue, criterion, device)
        scheduler.step()

        for k, v in (
            ("epoch", epoch),
            ("lr", lr),
            ("train_acc", train_acc),
            ("train_loss", train_obj),
            ("val_acc", val_acc * 100),
            ("val_loss", val_loss),
            ("test_acc", test_acc * 100),
            ("test_top5", test_top5 * 100),
            ("test_loss", test_loss),
            ("nonfinite", nonfinite),
            ("epoch_time_s", time.time() - t0),
            (
                "gpu_mem_peak_alloc_gib",
                torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda"
                else None,
            ),
        ):
            history[k].append(v)
        logging.info(
            f"val_acc {val_acc * 100:.2f} test_acc {test_acc * 100:.2f} (logging only) | "
            f"nonfinite {nonfinite} | {history['epoch_time_s'][-1]:.0f}s"
        )

        torch.save(
            {
                "epoch": epoch,  # last completed epoch
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "rng": rng_state(),
            },
            ckpt_last,
        )
        metrics_path.write_text(
            json.dumps(
                {"run_name": name, "complete": False, "history": history}, indent=2
            )
        )
        if (
            args.stop_after_epochs
            and epoch + 1 >= args.stop_after_epochs
            and epoch + 1 < args.epochs
        ):
            logging.info(
                f"Smoke test: stopping after epoch {epoch} (schedule is for {args.epochs} epochs)"
            )
            return {
                "run_name": name,
                "complete": False,
                "stopped_after_epoch": epoch,
                "history": history,
            }

    best = int(np.argmax(history["val_acc"]))
    final = {
        "run_name": name,
        "method": args.method,
        "complete": True,
        "seed": args.seed,
        "epochs": args.epochs,
        "final_epoch_top1": history["test_acc"][-1],
        "final_epoch_top1_error": 100 - history["test_acc"][-1],
        "final_epoch_top5": history["test_top5"][-1],
        "reported": "final epoch (no test-based selection)",
        "max_val_acc_logging_only": history["val_acc"][best],
        "max_val_acc_epoch_logging_only": history["epoch"][best],
        "total_minutes": sum(history["epoch_time_s"]) / 60,
        "history": history,
    }
    metrics_path.write_text(json.dumps(final, indent=2))
    torch.save({"model": model.state_dict()}, ckpt_final)
    logging.info(
        f"FINAL top-1 {final['final_epoch_top1']:.2f}  error {final['final_epoch_top1_error']:.2f}"
    )
    return final


if __name__ == "__main__":
    main()
