"""
Evaluate a saved checkpoint on the test set and print Top-1 / Top-5 accuracy.

Usage:
    python experiments/eval_checkpoint.py --checkpoint <path>
    python experiments/eval_checkpoint.py --checkpoint checkpoints/my_run_best.pth
    python experiments/eval_checkpoint.py --checkpoint checkpoints/my_run_best.pth --dataset cifar10
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_tiny_imagenet_loaders,
)
from models.registry import get_model
from training.trainer import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .pth checkpoint file"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset override (cifar10 / cifar100 / tiny_imagenet). "
        "Auto-detected from checkpoint config if omitted.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model override. Auto-detected from checkpoint config if omitted.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device     : {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    dataset = args.dataset or cfg.get("dataset", "cifar100")
    model_name = args.model or cfg.get("model", "wideresnet")
    epoch = ckpt.get("epoch", "?")

    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Dataset    : {dataset}")
    print(f"  Model      : {model_name}")
    print(f"  Epoch      : {epoch}")

    # load test set only
    loaders = {
        "cifar10": get_cifar10_loaders,
        "cifar100": get_cifar100_loaders,
        "tiny_imagenet": get_tiny_imagenet_loaders,
    }
    if dataset not in loaders:
        print(
            f"ERROR: unknown dataset '{dataset}'. Choose cifar10 / cifar100 / tiny_imagenet"
        )
        sys.exit(1)

    _, _, test_loader = loaders[dataset](
        batch_size=args.batch_size, val_split=0.0, num_workers=args.num_workers
    )

    num_classes = {"cifar10": 10, "cifar100": 100, "tiny_imagenet": 200}[dataset]
    model = get_model(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion = torch.nn.CrossEntropyLoss()
    loss, top1, top5 = evaluate(model, test_loader, criterion, device)

    # evaluate() returns accuracy as fraction (0–1); convert to percentage
    if top1 <= 1.0:
        top1 *= 100
        top5 *= 100

    print(f"\n{'─' * 44}")
    print(f"  {'Metric':<22} {'Value':>10}")
    print(f"  {'─' * 40}")
    print(f"  {'Test Top-1':<22} {top1:>9.2f}%")
    print(f"  {'Test Top-5':<22} {top5:>9.2f}%")
    print(f"  {'Test Error (Top-1)':<22} {100 - top1:>9.2f}%")
    print(f"  {'Test Error (Top-5)':<22} {100 - top5:>9.2f}%")
    print(f"  {'Test Loss':<22} {loss:>10.4f}")
    print(f"{'─' * 44}\n")


if __name__ == "__main__":
    main()
